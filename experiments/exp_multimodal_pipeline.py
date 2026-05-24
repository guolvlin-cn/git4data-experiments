"""Capstone: end-to-end multimodal data-versioning + labeling + training loop,
lakeFS x MatrixOne git4data, with continuous iteration.

This is the recommended combined architecture (COMPARISON.md §8/§11) run as a
living pipeline:

  * raw multimodal bytes (image/video/doc) -> versioned in lakeFS (commit per drop)
  * MatrixOne catalog references each asset (lakeFS commit + path + content hash),
    holds its embedding (vecf32), annotated label and split
  * each training round = a catalog SNAPSHOT (a dataset version) that PINS the
    lakeFS commit; a model is trained on the labeled embeddings and registered
  * iteration: new data drops, label cleaning, and a raw re-export each create a
    new version on the right layer; DATA BRANCH DIFF shows what changed; any past
    dataset version is byte-for-byte reproducible.

Deterministic (ground truth = data_stream.TRUE_W; eval on the fixed clean
holdout) so the whole thing is reproducible. Requires .lakefs.env + lakeFS up.
Run:  python3 -m experiments.exp_multimodal_pipeline
"""
import hashlib

import numpy as np
import boto3
from botocore.config import Config

import config
import lakefs
from common import data_stream, model as ml
from lakefs_demo import lk_config as lk
from matrixone.mo_client import MO

REPO = "ml-multimodal"
DB = "mld_mmpipe"
NOISE = 0.18            # annotation error rate
MODALITIES = ["image", "video", "doc"]
HOLDOUT = data_stream.make_holdout()   # fixed clean eval set (ground-truth labels)


def hr(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


# ---------- raw asset synthesis (embedding + bytes + labels) ----------
def gen_assets(ids):
    out = []
    for aid in ids:
        rng = np.random.default_rng(10_000 + aid)
        emb = rng.normal(size=config.FEATURE_DIM)
        true = int(emb @ data_stream.TRUE_W + data_stream.TRUE_B > 0)
        annotated = true if rng.random() >= NOISE else 1 - true   # annotator noise
        modality = MODALITIES[aid % 3]
        out.append(dict(aid=aid, modality=modality, emb=emb, true=true,
                        label=annotated, split="train"))
    return out


def asset_bytes(a):
    return (f"{a['modality']}|" + ",".join(f"{x:.6f}" for x in a["emb"])).encode()


def asset_path(a):
    return f"{a['modality']}/a{a['aid']:05d}.bin"


# ---------- lakeFS (byte versioning) ----------
def fresh_repo(clt):
    env = lk.load_env()
    s3 = _oss(env)
    repo = lakefs.Repository(REPO, client=clt)
    try:
        repo.delete()
    except Exception:
        pass
    _wipe(s3, env["OSS_BUCKET"], f"{REPO}/")
    return lakefs.Repository(REPO, client=clt).create(
        storage_namespace=f"s3://{env['OSS_BUCKET']}/{REPO}", exist_ok=True)


def commit_assets(repo, assets, msg):
    main = repo.branch("main")
    for a in assets:
        main.object(asset_path(a)).upload(data=asset_bytes(a), mode="wb")
    return main.commit(message=msg).get_commit().id


def read_lakefs_bytes(repo, commit, path):
    with repo.ref(commit).object(path).reader(mode="rb") as r:
        return r.read()


def _oss(env):
    return boto3.client("s3", endpoint_url=env["OSS_ENDPOINT"], region_name=env["OSS_REGION"],
                        aws_access_key_id=env["OSS_ACCESS_KEY_ID"],
                        aws_secret_access_key=env["OSS_ACCESS_KEY_SECRET"],
                        config=Config(s3={"addressing_style": "virtual"},
                                      request_checksum_calculation="when_required"))


def _wipe(s3, bucket, prefix):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=o["Key"])


# ---------- MatrixOne catalog (labels + training versioning) ----------
def catalog_insert(mo, assets, commit):
    rows = [(a["aid"], a["modality"], asset_path(a), commit,
             hashlib.sha256(asset_bytes(a)).hexdigest()[:16],
             str([round(float(x), 6) for x in a["emb"]]),
             a["true"], a["label"], a["split"]) for a in assets]
    mo.executemany(
        f"INSERT INTO {DB}.catalog (asset_id,modality,lakefs_path,lakefs_commit,"
        f"content_hash,emb,true_label,label,split) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)


def load_train_xy(mo, snapshot=None):
    snap = f" {{snapshot='{snapshot}'}}" if snapshot else ""
    rows = mo.query(
        f"SELECT emb, label FROM {DB}.catalog{snap} WHERE split='train' ORDER BY asset_id")
    X = np.array([[float(v) for v in e.strip("[]").split(",")] for e, _ in rows])
    y = np.array([int(l) for _, l in rows])
    return X, y


def train_round(mo, snapshot=None):
    X, y = load_train_xy(mo, snapshot)
    m = ml.train_from_scratch([(X, y)], seed=config.GLOBAL_SEED)
    return m.evaluate(*HOLDOUT), len(y)


def diff(mo, snap_b, snap_a):
    return {r[0]: int(r[1]) for r in mo.query(
        f"DATA BRANCH DIFF {DB}.catalog {{snapshot='{snap_b}'}} "
        f"AGAINST {DB}.catalog {{snapshot='{snap_a}'}} OUTPUT SUMMARY")}


def register(mo, version, snap, commit, n, metrics, note):
    mo.execute(
        f"INSERT INTO {DB}.model_registry (version,catalog_snapshot,lakefs_commit,"
        f"n_train,accuracy,note) VALUES (%s,%s,%s,%s,%s,%s)",
        (version, snap, commit[:12], n, metrics["accuracy"], note))


def main():
    clt = lk.client()
    repo = fresh_repo(clt)
    commits = {}
    with MO() as mo:
        acct = config.mo_account_name()
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")
        mo.execute(
            f"CREATE TABLE {DB}.catalog (asset_id INT PRIMARY KEY, modality VARCHAR(8), "
            f"lakefs_path VARCHAR(128), lakefs_commit VARCHAR(64), content_hash VARCHAR(32), "
            f"emb vecf32({config.FEATURE_DIM}), true_label INT, label INT, split VARCHAR(8))")
        mo.execute(
            f"CREATE TABLE {DB}.model_registry (version VARCHAR(16) PRIMARY KEY, "
            f"catalog_snapshot VARCHAR(32), lakefs_commit VARCHAR(16), n_train INT, "
            f"accuracy DOUBLE, note VARCHAR(64))")
        for v in ("v1", "v2", "v3", "v4"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS mld_mm_{v}")

        # ---------- Round 1: bootstrap data drop ----------
        hr("Round 1  raw data drop -> lakeFS commit -> catalog+label -> train")
        a1 = gen_assets(range(0, 60))
        commits["C1"] = commit_assets(repo, a1, "raw drop #1 (60 assets)")
        catalog_insert(mo, a1, commits["C1"])
        mo.execute(f"CREATE SNAPSHOT mld_mm_v1 FOR TABLE {DB} catalog")
        m1, n1 = train_round(mo)
        register(mo, "m1", "mld_mm_v1", commits["C1"], n1, m1, "bootstrap (noisy labels)")
        print(f"  lakeFS C1={commits['C1'][:12]}; catalog snapshot v1; trained on {n1} "
              f"labeled assets -> holdout acc={m1['accuracy']}")

        # ---------- Round 2: more data arrives ----------
        hr("Round 2  more data arrives (continuous ingestion)")
        a2 = gen_assets(range(60, 120))
        commits["C2"] = commit_assets(repo, a2, "raw drop #2 (60 assets)")
        catalog_insert(mo, a2, commits["C2"])
        mo.execute(f"CREATE SNAPSHOT mld_mm_v2 FOR TABLE {DB} catalog")
        m2, n2 = train_round(mo)
        register(mo, "m2", "mld_mm_v2", commits["C2"], n2, m2, "+60 assets")
        d = diff(mo, "mld_mm_v2", "mld_mm_v1")
        print(f"  lakeFS C2={commits['C2'][:12]}; DIFF v2 vs v1 INSERTED={d.get('INSERTED',0)}; "
              f"trained on {n2} -> acc={m2['accuracy']}")

        # ---------- Round 3: label cleaning on a branch (MatrixOne side) ----------
        hr("Round 3  fix noisy labels on a branch -> diff -> merge (no new bytes)")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.catalog_clean")
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.catalog_clean FROM {DB}.catalog")
        mo.execute(f"UPDATE {DB}.catalog_clean SET label = true_label WHERE label <> true_label")
        fixed = {r[0]: int(r[1]) for r in mo.query(
            f"DATA BRANCH DIFF {DB}.catalog_clean AGAINST {DB}.catalog OUTPUT SUMMARY")}
        mo.execute(f"DATA BRANCH MERGE {DB}.catalog_clean INTO {DB}.catalog WHEN CONFLICT ACCEPT")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.catalog_clean")
        mo.execute(f"CREATE SNAPSHOT mld_mm_v3 FOR TABLE {DB} catalog")
        m3, n3 = train_round(mo)
        register(mo, "m3", "mld_mm_v3", commits["C2"], n3, m3, "label cleaning")
        print(f"  relabeled {fixed.get('UPDATED',0)} mislabeled assets (DATA BRANCH DIFF/MERGE); "
              f"snapshot v3; acc {m2['accuracy']} -> {m3['accuracy']} (cleaner labels)")

        # ---------- Round 4: a raw asset is re-exported (bytes change -> new lakeFS commit) ----------
        hr("Round 4  raw re-export of one asset (byte-level change tracked by lakeFS)")
        re = gen_assets([5])[0]
        re["emb"] = np.random.default_rng(999_005).normal(size=config.FEATURE_DIM)  # new crop
        re["true"] = int(re["emb"] @ data_stream.TRUE_W + data_stream.TRUE_B > 0)
        re["label"] = re["true"]
        commits["C4"] = commit_assets(repo, [re], "re-export asset 5 (better crop)")
        mo.execute(
            f"UPDATE {DB}.catalog SET lakefs_commit=%s, content_hash=%s, emb=%s, "
            f"true_label=%s, label=%s WHERE asset_id=5",
            (commits["C4"], hashlib.sha256(asset_bytes(re)).hexdigest()[:16],
             str([round(float(x), 6) for x in re["emb"]]), re["true"], re["label"]))
        mo.execute(f"CREATE SNAPSHOT mld_mm_v4 FOR TABLE {DB} catalog")
        m4, n4 = train_round(mo)
        register(mo, "m4", "mld_mm_v4", commits["C4"], n4, m4, "asset#5 re-export")
        print(f"  asset#5 bytes -> lakeFS C4={commits['C4'][:12]}; catalog row repinned to C4; "
              f"snapshot v4; acc={m4['accuracy']}")

        # ---------- Finale A: reproduce a past model exactly ----------
        hr("Finale  reproducibility + byte-level lineage")
        snap, cmt, acc = mo.query_one(
            f"SELECT catalog_snapshot, lakefs_commit, accuracy FROM {DB}.model_registry "
            f"WHERE version='m2'")
        repro_m, _ = train_round(mo, snapshot=snap)
        print(f"  reproduce m2: retrain at catalog {snap} -> acc={repro_m['accuracy']} "
              f"(registry={acc}) reproducible={repro_m['accuracy'] == acc}")

        # ---------- Finale B: byte-level time travel for asset#5 via the combo ----------
        for snap in ("mld_mm_v2", "mld_mm_v4"):
            commit, path = mo.query_one(
                f"SELECT lakefs_commit, lakefs_path FROM {DB}.catalog {{snapshot='{snap}'}} "
                f"WHERE asset_id=5")
            b = read_lakefs_bytes(repo, commit, path)
            print(f"  asset#5 @ catalog {snap} -> lakeFS {commit[:12]} -> bytes head: {b[:32]!r}")

        # ---------- Finale C: lineage registry ----------
        print("\n  model lineage (version -> catalog snapshot -> lakeFS commit -> acc):")
        print(f"    {'ver':<5}{'catalog_snap':<14}{'lakefs':<14}{'n':>5} {'acc':>7}  note")
        for v, s, c, n, a, note in mo.query(
            f"SELECT version,catalog_snapshot,lakefs_commit,n_train,accuracy,note "
            f"FROM {DB}.model_registry ORDER BY version"):
            print(f"    {v:<5}{s:<14}{c:<14}{n:>5} {a:>7}  {note}")

        for v in ("v1", "v2", "v3", "v4"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS mld_mm_{v}")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")

    # cleanup lakeFS repo + OSS prefix
    try:
        lakefs.Repository(REPO, client=clt).delete()
    except Exception:
        pass
    env = lk.load_env()
    _wipe(_oss(env), env["OSS_BUCKET"], f"{REPO}/")
    print("\nDone. Raw bytes versioned by lakeFS, labels/embeddings/dataset versions by "
          "MatrixOne; each model pins (catalog snapshot + lakeFS commit) => continuous "
          "iteration with byte-level reproducibility and full lineage. (cleaned up)")


if __name__ == "__main__":
    main()
