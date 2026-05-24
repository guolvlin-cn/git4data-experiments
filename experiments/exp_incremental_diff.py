"""Experiment: continuous learning that trains ONLY on the per-cycle delta.

A real continuous-learning pipeline gets, each cycle, some new rows plus late
corrections to old rows. Reprocessing the entire (ever-growing) dataset every
cycle is O(K*N) — quadratic. With a versioned store you can ask "what changed
since the snapshot I last trained on?" and touch only that delta.

Here `DATA BRANCH DIFF live AGAINST <last-trained snapshot>` returns exactly the
INSERTED + UPDATED rows; we `partial_fit` on just those. We log how many rows the
incremental pipeline touches vs. a naive full-retrain-every-cycle baseline.

Run:  python3 -m experiments.exp_incremental_diff
"""
import numpy as np

import config
from common import data_stream, model as ml
from matrixone.mo_client import MO
from matrixone import repo

DB = "mld_ml_exp"
TABLE = "samples"
CYCLES = 6
CORRECTIONS = 50  # late label fixes to already-ingested rows, per cycle


def reset(mo):
    mo.execute(f"DROP DATABASE IF EXISTS {DB}")
    mo.execute(f"CREATE DATABASE {DB}")
    cols = ", ".join(f"{f} DOUBLE" for f in repo.FEATS)
    mo.execute(
        f"CREATE TABLE {DB}.{TABLE} (id BIGINT PRIMARY KEY, batch INT, {cols}, label INT)"
    )


def correct_old_rows(mo, upto_batch):
    """Relabel a few already-ingested rows toward ground truth (creates UPDATEs)."""
    rows = mo.query(
        f"SELECT id, {', '.join(repo.FEATS)}, label FROM {DB}.{TABLE} "
        f"WHERE batch < {upto_batch} ORDER BY rand() LIMIT {CORRECTIONS}"
    )
    if not rows:
        return
    arr = np.array(rows, dtype=np.float64)
    ids, X, y = arr[:, 0].astype(np.int64), arr[:, 1:-1], arr[:, -1].astype(np.int64)
    true = (X @ data_stream.TRUE_W + data_stream.TRUE_B > 0).astype(np.int64)
    fixes = [(int(true[i]), int(ids[i])) for i in range(len(ids)) if true[i] != y[i]]
    if fixes:
        mo.executemany(f"UPDATE {DB}.{TABLE} SET label = %s WHERE id = %s", fixes)


def delta_since(mo, snap):
    """Return (X, y) for rows INSERTed/UPDATEd since `snap` (live AGAINST snap)."""
    out = mo.query(
        f"DATA BRANCH DIFF {DB}.{TABLE} AGAINST {DB}.{TABLE} {{snapshot='{snap}'}}"
    )
    keep = [r for r in out if r[1] in ("INSERT", "UPDATE")]
    if not keep:
        return np.empty((0, config.FEATURE_DIM)), np.empty((0,), dtype=np.int64)
    # row layout: (side, flag, id, batch, f0..f{d-1}, label)
    arr = np.array([[float(c) for c in r[2:]] for r in keep], dtype=np.float64)
    X = arr[:, 2:2 + config.FEATURE_DIM]
    y = arr[:, -1].astype(np.int64)
    return X, y


def main():
    holdout = data_stream.make_holdout()
    with MO() as mo:
        reset(mo)
        model = ml.IncrementalModel(seed=config.GLOBAL_SEED)
        prev_snap = None
        inc_total = 0      # rows the incremental pipeline touched (cumulative)
        full_total = 0     # rows a full-retrain-every-cycle baseline would touch

        print(f"{'cycle':>5} | {'rows in DB':>10} | {'delta(this cycle)':>17} | "
              f"{'incr cumⁱ':>10} | {'full-retrain cum':>16} | {'acc':>6}")
        print("-" * 82)
        for k in range(CYCLES):
            X, y = data_stream.make_batch(k)
            repo.insert_batch(mo, k, X, y, db=DB, table=TABLE)
            if k > 0:
                correct_old_rows(mo, upto_batch=k)
            snap = f"mldinc_s{k}"
            mo.execute(f"DROP SNAPSHOT IF EXISTS {snap}")
            mo.execute(f"CREATE SNAPSHOT {snap} FOR TABLE {DB} {TABLE}")

            if prev_snap is None:
                dX, dy = repo.load_xy(mo, db=DB, table=TABLE)  # cold start
            else:
                dX, dy = delta_since(mo, prev_snap)
            model.update(dX, dy)
            m = model.evaluate(*holdout)

            total = int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.{TABLE}"))
            inc_total += len(dy)
            full_total += total
            print(f"{k:>5} | {total:>10} | {len(dy):>17} | {inc_total:>10} | "
                  f"{full_total:>16} | {m['accuracy']:>6}")
            prev_snap = snap

        for k in range(CYCLES):
            mo.execute(f"DROP SNAPSHOT IF EXISTS mldinc_s{k}")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        print(f"\nIncremental touched {inc_total} rows total; "
              f"full-retrain-every-cycle would touch {full_total} "
              f"({full_total / max(inc_total,1):.1f}x more). "
              f"DATA BRANCH DIFF makes per-cycle cost ∝ delta, not dataset size.")


if __name__ == "__main__":
    main()
