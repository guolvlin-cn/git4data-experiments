"""End-to-end lakeFS demo for ML continuous learning — the lakeFS counterpart
to matrixone/run_demo.py, act for act, so the two can be compared directly.

Run:  python3 -m lakefs_demo.run_demo   (needs lakeFS up: lakefs_demo/start_lakefs.sh)

Same data, same model, same five acts. The point of contrast is the verbs:
where MatrixOne needed manual EXCEPT/replay, lakeFS has native diff / revert /
merge.
  ACT 1  inflow + incremental train, one COMMIT per batch (+ a tag per model)
  ACT 2  native diff between two commits
  ACT 3  reproduce a historical model by reading data at its commit/tag
  ACT 4  poisoned commit -> native REVERT -> recover
  ACT 5  branch -> cleaning experiment -> native diff -> native MERGE
"""
import sys

import numpy as np

import config
from common import data_stream, model as ml
from lakefs_demo import lk_config as lk


def hr(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def log(msg):
    print(f"  {msg}")


def setup_repo():
    import lakefs

    clt = lk.client()
    repo = lakefs.Repository(lk.REPO, client=clt)
    try:
        repo.delete()  # drop repo metadata if it exists (idempotent re-run)
    except Exception:
        pass
    lk.clean_namespace()  # wipe leftover lakeFS objects in the OSS prefix
    repo = lakefs.Repository(lk.REPO, client=clt).create(
        storage_namespace=lk.storage_namespace(), default_branch="main",
        exist_ok=True,
    )
    return repo


def upload_batch(branch, batch_id, X, y):
    branch.object(lk.batch_path(batch_id)).upload(
        data=lk.encode(batch_id, X, y), mode="wb"
    )


def load_batches_at(repo, ref):
    """Ordered (X, y) per batch object as of a ref — for reproducing training."""
    out = []
    objs = sorted(
        (o.path for o in repo.ref(ref).objects(prefix=lk.DATA_PREFIX)),
    )
    for path in objs:
        with repo.ref(ref).object(path).reader(mode="rb") as r:
            _, X, y = lk.decode(r.read())
        out.append((X, y))
    return out


def act1_stream(repo, holdout, registry):
    hr("ACT 1  Continuous inflow + incremental learning (commit per batch)")
    main = repo.branch("main")
    running = ml.IncrementalModel(seed=config.GLOBAL_SEED)
    total = 0
    for b in range(config.N_BATCHES):
        X, y = data_stream.make_batch(b)
        total += len(y)
        upload_batch(main, b, X, y)
        commit = main.commit(message=f"ingest batch {b}",
                             metadata={"batch": str(b), "rows": str(total)})
        cid = commit.get_commit().id
        repo.tag(f"v{b}").create(cid)  # model version -> data commit
        running.update(X, y)
        m = running.evaluate(*holdout)
        registry.append((f"v{b}", cid[:12], total, m, "streaming incremental"))
        log(f"batch {b}: +{len(y)} rows (total={total})  commit={cid[:12]}  "
            f"tag=v{b}  acc={m['accuracy']} f1={m['f1']}")
    return running


def act2_diff(repo):
    hr("ACT 2  Native diff between two data versions (lakectl-style diff)")
    # base.diff(other_ref=newer) => what `newer` changed relative to `base`
    changes = list(repo.ref("v2").diff(other_ref="v5", prefix=lk.DATA_PREFIX))
    kinds = {}
    for c in changes:
        kinds[c.type] = kinds.get(c.type, 0) + 1
    log(f"diff v2 -> v5: {dict(kinds)} ({len(changes)} object-level changes)")
    for c in changes:
        log(f"  {c.type:8s} {c.path}")
    log("NOTE: lakeFS diffs at OBJECT granularity (whole files); row-level diff "
        "needs a table format (Iceberg/Delta) or content compare.")


def act3_reproduce(repo, holdout, registry):
    hr("ACT 3  Reproduce a historical model by reading data at its commit")
    target = "v3"
    orig = next(r for r in registry if r[0] == target)
    log(f"registry says {target} -> commit {orig[1]} "
        f"(acc={orig[3]['accuracy']}, f1={orig[3]['f1']})")
    batches = load_batches_at(repo, target)
    repro = ml.train_from_scratch(batches, seed=config.GLOBAL_SEED)
    m = repro.evaluate(*holdout)
    match = (m == orig[3])
    log(f"read data at tag {target} ({len(batches)} batches), retrain -> "
        f"acc={m['accuracy']} f1={m['f1']}")
    log(f"reproducible == {match}")
    if not match:
        raise SystemExit("reproduction mismatch — determinism broken")


def act4_poison_revert(repo, holdout, running, registry):
    hr("ACT 4  Poisoned commit -> native REVERT -> recovery")
    main = repo.branch("main")
    pb = config.N_BATCHES
    X, y = data_stream.make_batch(pb, poison=True)
    upload_batch(main, pb, X, y)
    bad_commit = main.commit(message=f"ingest batch {pb} (POISONED)").get_commit()
    running.update(X, y)
    bad = running.evaluate(*holdout)
    log(f"committed POISONED batch {pb} ({len(y)} rows)  commit={bad_commit.id[:12]}")
    log(f"  model acc dropped to {bad['accuracy']} (f1={bad['f1']})  <-- regression")

    main.revert(reference=bad_commit.id, parent_number=0)
    n_objs = len(list(repo.ref("main").objects(prefix=lk.DATA_PREFIX)))
    log(f"native REVERT of {bad_commit.id[:12]}  ->  data objects now={n_objs}")

    batches = load_batches_at(repo, "main")
    recovered = ml.train_from_scratch(batches, seed=config.GLOBAL_SEED)
    rec = recovered.evaluate(*holdout)
    registry.append(("v6_recovered", "main@revert", n_objs * config.BATCH_SIZE,
                     rec, "retrained after revert"))
    log(f"retrain on reverted data -> acc={rec['accuracy']} f1={rec['f1']} (recovered)")


def _relabel_branch_to_ground_truth(repo, branch):
    """Cleaning pipeline on a branch: rewrite each batch object with corrected
    labels (re-derived from the known synthetic boundary). Returns rows fixed."""
    fixed = 0
    paths = sorted(o.path for o in repo.ref(branch.id).objects(prefix=lk.DATA_PREFIX))
    for path in paths:
        with repo.ref(branch.id).object(path).reader(mode="rb") as r:
            ids, X, y = lk.decode(r.read())
        true = (X @ data_stream.TRUE_W + data_stream.TRUE_B > 0).astype(np.int64)
        fixed += int((true != y).sum())
        batch_id = int(ids[0]) // config.BATCH_SIZE
        branch.object(path).upload(data=lk.encode(batch_id, X, true), mode="wb")
    return fixed


def act5_branch_merge(repo, holdout, registry):
    hr("ACT 5  Branch -> cleaning experiment -> native diff -> native MERGE")
    main = repo.branch("main")
    main_model = ml.train_from_scratch(load_batches_at(repo, "main"),
                                       seed=config.GLOBAL_SEED)
    main_m = main_model.evaluate(*holdout)

    clean = repo.branch("cleaning").create(source_reference="main")
    fixed = _relabel_branch_to_ground_truth(repo, clean)
    clean.commit(message=f"relabel {fixed} noisy rows to ground truth")
    exp_model = ml.train_from_scratch(load_batches_at(repo, "cleaning"),
                                      seed=config.GLOBAL_SEED)
    exp_m = exp_model.evaluate(*holdout)

    changes = list(repo.ref("main").diff(other_ref="cleaning", prefix=lk.DATA_PREFIX))
    log(f"main     (noisy):            acc={main_m['accuracy']} f1={main_m['f1']}")
    log(f"branch   (relabelled {fixed} rows): acc={exp_m['accuracy']} f1={exp_m['f1']}")
    log(f"native diff main..cleaning: {len(changes)} objects changed")
    if exp_m["accuracy"] <= main_m["accuracy"]:
        log("experiment did not beat main -> delete branch (main untouched)")
        clean.delete()
        return
    log(f"experiment WINS (+{round(exp_m['accuracy'] - main_m['accuracy'], 4)} acc)"
        " -> native MERGE cleaning into main")

    merge_commit = clean.merge_into(main)
    repo.tag("v_merged").create(merge_commit)
    merged_model = ml.train_from_scratch(load_batches_at(repo, "main"),
                                         seed=config.GLOBAL_SEED)
    mm = merged_model.evaluate(*holdout)
    registry.append(("v_merged", merge_commit[:12], config.N_BATCHES * config.BATCH_SIZE,
                     mm, "merged cleaning branch"))
    log(f"merged via clean.merge_into(main)  commit={merge_commit[:12]}  "
        f"acc={mm['accuracy']} f1={mm['f1']}")


def print_registry(registry):
    hr("Model registry (lineage: model_version -> data commit)")
    print(f"  {'version':<14}{'commit':<14}{'rows':>7}  {'acc':>6} {'f1':>6}  note")
    for v, c, n, m, note in registry:
        print(f"  {v:<14}{c:<14}{n:>7}  {m['accuracy']:>6} {m['f1']:>6}  {note}")


def main():
    holdout = data_stream.make_holdout()
    registry = []
    repo = setup_repo()
    running = act1_stream(repo, holdout, registry)
    act2_diff(repo)
    act3_reproduce(repo, holdout, registry)
    act4_poison_revert(repo, holdout, running, registry)
    act5_branch_merge(repo, holdout, registry)
    print_registry(registry)
    hr("Done. lakeFS exercised: commit / tag / diff / read-at-ref / revert / merge")


if __name__ == "__main__":
    sys.exit(main())
