"""End-to-end MatrixOne git4data demo for ML continuous learning.

Run:  python3 -m matrixone.run_demo

Narrative (each act maps a real ML-ops need to a git4data primitive):
  ACT 1  continuous inflow + incremental training, one snapshot per batch
  ACT 2  native row-level diff between two data versions (DATA BRANCH DIFF)
  ACT 3  reproduce a historical model from its data snapshot
  ACT 4  a poisoned batch tanks accuracy -> rollback (RESTORE) -> recover
  ACT 5  branch (DATA BRANCH CREATE) -> clean -> diff -> cherry-pick -> merge
"""
import sys

import numpy as np

import config
from common import data_stream, model as ml
from matrixone import git4data as g4d
from matrixone.mo_client import MO
from matrixone import repo

EXP_TABLE = config.SAMPLES_TABLE + "_exp"   # branch lives in the same database
EXP_DB = config.DB + "_exp"                 # legacy artifact, cleaned if present


def hr(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def log(msg):
    print(f"  {msg}")


def snap_name(batch_id, suffix=""):
    return f"mld_b{batch_id}{suffix}"


def cleanup(mo):
    """Make the demo idempotent: drop our snapshots + the experiment db."""
    for row in g4d.list_snapshots(mo):
        name = row[0]
        if name.startswith("mld_"):
            g4d.drop_snapshot(mo, name)
    mo.execute(f"DROP DATABASE IF EXISTS {EXP_DB}")


def act1_stream(mo, holdout):
    hr("ACT 1  Continuous inflow + incremental learning (snapshot per batch)")
    running = ml.IncrementalModel(seed=config.GLOBAL_SEED)
    total = 0
    for b in range(config.N_BATCHES):
        X, y = data_stream.make_batch(b)
        total += repo.insert_batch(mo, b, X, y)
        snap = g4d.snapshot(mo, snap_name(b), config.DB, config.SAMPLES_TABLE)
        running.update(X, y)
        metrics = running.evaluate(*holdout)
        repo.register_model(
            mo, f"v{b}", snap, total, metrics, note="streaming incremental"
        )
        log(
            f"batch {b}: +{len(y)} rows (total={total})  snapshot={snap}  "
            f"acc={metrics['accuracy']}  f1={metrics['f1']}"
        )
    return running


def act2_diff(mo):
    hr("ACT 2  Native row-level diff between two data versions (DATA BRANCH DIFF)")
    a, b = snap_name(2), snap_name(config.N_BATCHES - 1)
    summary = g4d.branch_diff_summary(
        mo, config.DB, config.SAMPLES_TABLE, config.SAMPLES_TABLE,
        target_snap=b, base_snap=a,
    )
    log(f"DATA BRANCH DIFF {config.SAMPLES_TABLE}@{b} AGAINST {config.SAMPLES_TABLE}@{a}:")
    log(f"  INSERTED={summary.get('INSERTED', 0)}  "
        f"UPDATED={summary.get('UPDATED', 0)}  "
        f"DELETED={summary.get('DELETED', 0)}  (row-level, native)")


def act3_reproduce(mo, holdout):
    hr("ACT 3  Reproduce a historical model from its data snapshot")
    target = "v3"
    row = mo.query_one(
        f"SELECT data_snapshot, accuracy, f1 FROM {config.DB}.{config.REGISTRY_TABLE} "
        f"WHERE model_version = %s",
        (target,),
    )
    snap, orig_acc, orig_f1 = row
    log(f"registry says {target} was trained on snapshot '{snap}' "
        f"(acc={orig_acc}, f1={orig_f1})")
    batches = repo.load_batches(mo, snapshot=snap)
    repro = ml.train_from_scratch(batches, seed=config.GLOBAL_SEED)
    m = repro.evaluate(*holdout)
    match = (m["accuracy"] == orig_acc) and (m["f1"] == orig_f1)
    log(f"time-travel retrain over {len(batches)} batches -> "
        f"acc={m['accuracy']} f1={m['f1']}")
    log(f"reproducible == {match}")
    if not match:
        raise SystemExit("reproduction mismatch — determinism broken")


def act4_poison_rollback(mo, holdout, running):
    hr("ACT 4  Poisoned batch -> rollback (RESTORE) -> recovery")
    pb = config.N_BATCHES  # next batch id
    X, y = data_stream.make_batch(pb, poison=True)
    repo.insert_batch(mo, pb, X, y)
    poisoned_snap = g4d.snapshot(mo, snap_name(pb, "_poison"), config.DB,
                                 config.SAMPLES_TABLE)
    running.update(X, y)
    bad = running.evaluate(*holdout)
    n_after = g4d.count_at(mo, config.DB, config.SAMPLES_TABLE)
    log(f"ingested POISONED batch {pb} ({len(y)} rows, "
        f"{int(config.POISON_FRACTION*100)}% labels flipped)")
    log(f"  rows now={n_after}  model acc dropped to {bad['accuracy']} "
        f"(f1={bad['f1']})  <-- regression")

    good_snap = snap_name(config.N_BATCHES - 1)
    g4d.restore_table(mo, config.DB, config.SAMPLES_TABLE, good_snap)
    n_restored = g4d.count_at(mo, config.DB, config.SAMPLES_TABLE)
    log(f"RESTORE table to '{good_snap}'  ->  rows now={n_restored}")

    batches = repo.load_batches(mo)  # live = restored clean state
    recovered = ml.train_from_scratch(batches, seed=config.GLOBAL_SEED)
    rec = recovered.evaluate(*holdout)
    repo.register_model(mo, "v6_recovered", good_snap, n_restored, rec,
                        note="retrained after rollback")
    log(f"retrain on restored data -> acc={rec['accuracy']} f1={rec['f1']} "
        f"(recovered)")
    return recovered


def _relabel_to_ground_truth(mo, table):
    """Cleaning pipeline on a branch table: fix labels that disagree with
    re-derived ground truth. Returns the list of corrected ids.

    In a real project this is a human relabelling pass or a better labelling
    model; here we know the synthetic boundary, so we compute corrections
    exactly.
    """
    rows = mo.query(
        f"SELECT id, {', '.join(repo.FEATS)}, label FROM {config.DB}.{table}"
    )
    arr = np.array(rows, dtype=np.float64)
    ids = arr[:, 0].astype(np.int64)
    X = arr[:, 1:-1]
    y = arr[:, -1].astype(np.int64)
    true = (X @ data_stream.TRUE_W + data_stream.TRUE_B > 0).astype(np.int64)
    bad = np.nonzero(true != y)[0]
    mo.executemany(
        f"UPDATE {config.DB}.{table} SET label = %s WHERE id = %s",
        [(int(true[i]), int(ids[i])) for i in bad],
    )
    return [int(ids[i]) for i in bad]


def act5_branch_merge(mo, holdout):
    hr("ACT 5  Branch (DATA BRANCH CREATE) -> clean -> diff -> cherry-pick -> merge")
    # Branch the table with tracked lineage; this is what lets DATA BRANCH
    # DIFF/MERGE auto-detect the common ancestor for a row-level 3-way merge.
    g4d.branch_create(mo, config.DB, config.SAMPLES_TABLE, EXP_TABLE)
    log(f"branched table: {EXP_TABLE} = DATA BRANCH CREATE FROM {config.SAMPLES_TABLE}")

    # main baseline (carries ~5% label noise)
    main_model = ml.train_from_scratch(repo.load_batches(mo), seed=config.GLOBAL_SEED)
    main_m = main_model.evaluate(*holdout)
    n_main = g4d.count_at(mo, config.DB, config.SAMPLES_TABLE)

    # cleaning experiment lives only on the branch — main is untouched
    fixed_ids = _relabel_to_ground_truth(mo, EXP_TABLE)
    exp_model = ml.train_from_scratch(repo.load_batches(mo, table=EXP_TABLE),
                                      seed=config.GLOBAL_SEED)
    exp_m = exp_model.evaluate(*holdout)

    diff = g4d.branch_diff_summary(mo, config.DB, EXP_TABLE, config.SAMPLES_TABLE)
    log(f"main   ({n_main} rows, noisy): acc={main_m['accuracy']} f1={main_m['f1']}")
    log(f"branch (relabelled {len(fixed_ids)} rows): "
        f"acc={exp_m['accuracy']} f1={exp_m['f1']}")
    log(f"native DATA BRANCH DIFF branch AGAINST main: "
        f"UPDATED={diff.get('UPDATED', 0)} (row-level, vs lakeFS's whole-file diff)")
    if exp_m["accuracy"] <= main_m["accuracy"]:
        log("experiment did not beat main -> drop branch (main untouched)")
        mo.execute(f"DROP TABLE IF EXISTS {config.DB}.{EXP_TABLE}")
        return
    log(f"experiment WINS (+{round(exp_m['accuracy'] - main_m['accuracy'], 4)} acc)")

    # Staged rollout: cherry-pick only the reviewed batch-0 corrections first.
    batch0 = [i for i in fixed_ids if i < config.BATCH_SIZE]
    g4d.branch_pick(mo, config.DB, EXP_TABLE, config.SAMPLES_TABLE, batch0)
    rem = g4d.branch_diff_summary(mo, config.DB, EXP_TABLE, config.SAMPLES_TABLE)
    log(f"DATA BRANCH PICK {len(batch0)} reviewed rows (batch 0) into main  "
        f"-> remaining diff UPDATED={rem.get('UPDATED', 0)}")

    # Merge the remaining corrections natively (row-level 3-way, ACCEPT source).
    g4d.branch_merge(mo, config.DB, EXP_TABLE, config.SAMPLES_TABLE, conflict="ACCEPT")
    merged_snap = g4d.snapshot(mo, "mld_merged", config.DB, config.SAMPLES_TABLE)
    n_merged = g4d.count_at(mo, config.DB, config.SAMPLES_TABLE)
    merged_model = ml.train_from_scratch(repo.load_batches(mo), seed=config.GLOBAL_SEED)
    mm = merged_model.evaluate(*holdout)
    repo.register_model(mo, "v_merged", merged_snap, n_merged, mm,
                        note="native DATA BRANCH MERGE of cleaning branch")
    log(f"DATA BRANCH MERGE branch INTO main (WHEN CONFLICT ACCEPT)  "
        f"snapshot={merged_snap}  acc={mm['accuracy']} f1={mm['f1']}")
    mo.execute(f"DROP TABLE IF EXISTS {config.DB}.{EXP_TABLE}")


def print_registry(mo):
    hr("Model registry (data lineage: model_version -> data_snapshot)")
    print(f"  {'version':<14}{'snapshot':<12}{'rows':>7}  {'acc':>6} {'f1':>6}  note")
    for v, s, n, acc, f1, note in repo.show_registry(mo):
        print(f"  {v:<14}{s:<12}{n:>7}  {acc:>6} {f1:>6}  {note}")


def main():
    holdout = data_stream.make_holdout()
    with MO() as mo:
        cleanup(mo)
        repo.reset_db(mo)
        running = act1_stream(mo, holdout)
        act2_diff(mo)
        act3_reproduce(mo, holdout)
        recovered = act4_poison_rollback(mo, holdout, running)
        act5_branch_merge(mo, holdout)
        print_registry(mo)
        hr("Done. MatrixOne git4data exercised: snapshot / time-travel / "
           "restore / branch / diff / cherry-pick / merge")


if __name__ == "__main__":
    sys.exit(main())
