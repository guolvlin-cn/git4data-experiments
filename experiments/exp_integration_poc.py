"""PoC: lakeFS (byte versioning) + MatrixOne git4data (catalog versioning) together.

The recommended platform architecture (COMPARISON.md §8/§11), end to end:

  * raw bytes live in lakeFS (OSS-backed), byte-level versioned by commits;
  * a MatrixOne catalog table records, per asset, the lakeFS commit-id + content
    hash + size + label + split + embedding, and is row-level versioned by
    git4data snapshots;
  * a "dataset version" = a MatrixOne catalog snapshot, which PINS a lakeFS
    commit per asset -> the two layers compose into byte-level time travel that
    neither does alone.

Demo: catalog v1 (assets @ lakeFS commit C1) -> one image's bytes change
(lakeFS commit C2) -> catalog v2. Then resolve the SAME asset at v1 vs v2 and
show v1 reads the OLD bytes (via C1) and v2 the NEW bytes (via C2); DATA BRANCH
DIFF pinpoints which asset changed; and we resolve a direct byte-readable URL.

Requires .lakefs.env + lakeFS up (lakefs_demo/start_lakefs.sh).
Run:  python3 -m experiments.exp_integration_poc
"""
import hashlib

import boto3
from botocore.config import Config

import lakefs
from lakefs_demo import lk_config as lk
from matrixone.mo_client import MO

REPO = "ml-assets-poc"
DB = "mld_integ"
ASSETS = [  # (asset_id, modality, logical_path, label, split)
    (1, "image", "images/img_a.jpg", 1, "train"),
    (2, "image", "images/img_b.jpg", 0, "train"),
    (3, "doc", "docs/doc_c.txt", 1, "val"),
]


def emb_of(b: bytes):
    return [round((len(b) % 97) / 97, 4), round((sum(b) % 97) / 97, 4),
            round(b[0] / 255, 4), round(b[-1] / 255, 4)]


def fresh_repo(clt):
    env = lk.load_env()
    s3 = boto3.client("s3", endpoint_url=env["OSS_ENDPOINT"], region_name=env["OSS_REGION"],
                      aws_access_key_id=env["OSS_ACCESS_KEY_ID"],
                      aws_secret_access_key=env["OSS_ACCESS_KEY_SECRET"],
                      config=Config(s3={"addressing_style": "virtual"}))
    repo = lakefs.Repository(REPO, client=clt)
    try:
        repo.delete()
    except Exception:
        pass
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=env["OSS_BUCKET"], Prefix=f"{REPO}/"):
        for o in page.get("Contents", []):
            s3.delete_object(Bucket=env["OSS_BUCKET"], Key=o["Key"])
    return lakefs.Repository(REPO, client=clt).create(
        storage_namespace=f"s3://{env['OSS_BUCKET']}/{REPO}", exist_ok=True)


def put_and_commit(repo, files: dict, msg):
    """Upload {path: bytes} to lakeFS main and commit; return commit id."""
    main = repo.branch("main")
    for path, data in files.items():
        main.object(path).upload(data=data, mode="wb")
    return main.commit(message=msg).get_commit().id


def stat(repo, commit, path):
    s = repo.ref(commit).object(path).stat()
    return s.physical_address, s.checksum, s.size_bytes


def read_bytes(repo, commit, path):
    with repo.ref(commit).object(path).reader(mode="rb") as r:
        return r.read()


def catalog_upsert(mo, asset_id, modality, path, commit, checksum, size, label, split, emb):
    mo.execute(f"DELETE FROM {DB}.catalog WHERE asset_id = %s", (asset_id,))
    mo.execute(
        f"INSERT INTO {DB}.catalog (asset_id, modality, logical_path, lakefs_repo, "
        f"lakefs_commit, content_hash, size_bytes, label, split, emb) "
        f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (asset_id, modality, path, REPO, commit, checksum, size, label, split, str(emb)))


def main():
    clt = lk.client()
    repo = fresh_repo(clt)
    with MO() as mo:
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")
        mo.execute(
            f"CREATE TABLE {DB}.catalog (asset_id INT PRIMARY KEY, modality VARCHAR(8), "
            f"logical_path VARCHAR(128), lakefs_repo VARCHAR(64), lakefs_commit VARCHAR(64), "
            f"content_hash VARCHAR(128), size_bytes BIGINT, label INT, split VARCHAR(8), "
            f"emb vecf32(4))")
        for s in ("mld_integ_v1", "mld_integ_v2"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS {s}")

        # ---- dataset v1: bytes -> lakeFS commit C1 -> catalog -> snapshot ----
        v1_bytes = {p: f"{p} -- ORIGINAL v1 content".encode() for _, _, p, _, _ in ASSETS}
        c1 = put_and_commit(repo, v1_bytes, "ingest v1")
        for aid, mod, path, label, split in ASSETS:
            _, chk, size = stat(repo, c1, path)
            catalog_upsert(mo, aid, mod, path, c1, chk, size, label, split, emb_of(v1_bytes[path]))
        mo.execute(f"CREATE SNAPSHOT mld_integ_v1 FOR TABLE {DB} catalog")
        print(f"dataset_v1: lakeFS commit C1={c1[:12]}  catalog snapshot mld_integ_v1 "
              f"({int(mo.scalar(f'SELECT COUNT(*) FROM {DB}.catalog'))} assets)")

        # ---- a single image's BYTES change -> lakeFS commit C2 (old bytes kept) ----
        new_a = b"images/img_a.jpg -- EDITED v2 content (relabeled crop)"
        c2 = put_and_commit(repo, {"images/img_a.jpg": new_a}, "img_a re-export v2")
        _, chk2, size2 = stat(repo, c2, "images/img_a.jpg")
        catalog_upsert(mo, 1, "image", "images/img_a.jpg", c2, chk2, size2, 1, "train", emb_of(new_a))
        mo.execute(f"CREATE SNAPSHOT mld_integ_v2 FOR TABLE {DB} catalog")
        print(f"dataset_v2: img_a bytes changed -> lakeFS commit C2={c2[:12]}  "
              f"catalog snapshot mld_integ_v2")

        # ---- payoff: byte-level time travel via the COMBINED system ----
        print("\nResolve asset #1 (img_a) at each dataset version -> read its EXACT bytes:")
        for snap in ("mld_integ_v1", "mld_integ_v2"):
            commit, path = mo.query_one(
                f"SELECT lakefs_commit, logical_path FROM {DB}.catalog {{snapshot='{snap}'}} "
                f"WHERE asset_id=1")
            data = read_bytes(repo, commit, path)
            print(f"  catalog @ {snap} -> lakeFS commit {commit[:12]} -> bytes: {data.decode()!r}")

        # ---- which assets changed between dataset versions (row-level) ----
        d = {r[0]: int(r[1]) for r in mo.query(
            f"DATA BRANCH DIFF {DB}.catalog {{snapshot='mld_integ_v2'}} "
            f"AGAINST {DB}.catalog {{snapshot='mld_integ_v1'}} OUTPUT SUMMARY")}
        print(f"\nDATA BRANCH DIFF v2 vs v1 catalog: UPDATED={d.get('UPDATED',0)} "
              f"(exactly the asset whose bytes/commit changed)")

        # ---- a byte-readable handle for a training dataloader (pin a version) ----
        print("\nByte-readable handles for dataset_v2 (what a dataloader would consume):")
        for aid, path in mo.query(
            f"SELECT asset_id, logical_path FROM {DB}.catalog {{snapshot='mld_integ_v2'}} ORDER BY asset_id"):
            commit = mo.scalar(
                f"SELECT lakefs_commit FROM {DB}.catalog {{snapshot='mld_integ_v2'}} WHERE asset_id={aid}")
            phys, _, _ = stat(repo, commit, path)
            print(f"  asset#{aid}: lakefs://{REPO}/{commit[:12]}/{path}  ->  {phys}")

        for s in ("mld_integ_v1", "mld_integ_v2"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS {s}")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")

    try:
        lakefs.Repository(REPO, client=clt).delete()
    except Exception:
        pass
    env = lk.load_env()
    s3 = boto3.client("s3", endpoint_url=env["OSS_ENDPOINT"], region_name=env["OSS_REGION"],
                      aws_access_key_id=env["OSS_ACCESS_KEY_ID"],
                      aws_secret_access_key=env["OSS_ACCESS_KEY_SECRET"],
                      config=Config(s3={"addressing_style": "virtual"}))
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=env["OSS_BUCKET"], Prefix=f"{REPO}/"):
        for o in page.get("Contents", []):
            s3.delete_object(Bucket=env["OSS_BUCKET"], Key=o["Key"])
    print("\nResult: bytes versioned by lakeFS, catalog versioned by MatrixOne; the catalog "
          "snapshot pins the lakeFS commit per asset -> byte-level time travel + row-level "
          "'what changed' + a direct byte URL, with end-to-end lineage. (cleaned up)")


if __name__ == "__main__":
    main()
