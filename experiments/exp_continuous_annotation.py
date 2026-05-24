"""Scenario 1: multi-annotator / continuous labelling vs weekly training runs.

Annotators add & correct labels every day; a training job runs ~weekly and must
pin "the dataset as of run time". The recurring questions:
  - exactly how much did the data differ between two training runs, and where?
  - can we reconstruct the state at an arbitrary past instant we forgot to tag?

MatrixOne answers both:
  - snapshot per training run  -> DATA BRANCH DIFF run_w2 AGAINST run_w1 gives the
    exact INSERTED / UPDATED / DELETED rows between the two runs.
  - PITR -> RESTORE ... FROM PITR "<timestamp>" reconstructs ANY past instant,
    even one with no explicit snapshot (lakeFS only has discrete commits).

Run:  python3 -m experiments.exp_continuous_annotation
"""
import time

import config
from matrixone.mo_client import MO

DB = "mld_annot"
T = "labels"


def n(mo):
    return int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.{T}"))


def add_labels(mo, start_id, count, annotator):
    base = start_id
    rows = [(base + i, base + i, (base + i) % 2, annotator) for i in range(count)]
    mo.executemany(f"INSERT INTO {DB}.{T} (id,item_id,label,annotator) VALUES (%s,%s,%s,%s)", rows)


def main():
    acct = config.mo_account_name()
    with MO() as mo:
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")
        mo.execute(
            f"CREATE TABLE {DB}.{T} (id BIGINT PRIMARY KEY, item_id INT, label INT, "
            f"annotator VARCHAR(16))"
        )
        for s in ("mld_run_w1", "mld_run_w2", "mld_cur"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS {s}")
        mo.execute(f"DROP PITR IF EXISTS mld_annot_pitr")
        mo.execute(f"CREATE PITR mld_annot_pitr FOR DATABASE {DB} RANGE 1 'h'")

        # ---- week 1: 500 labels accumulate, training run 1 pins a snapshot ----
        add_labels(mo, 0, 500, "alice")
        mo.execute(f"CREATE SNAPSHOT mld_run_w1 FOR TABLE {DB} {T}")
        print(f"training run W1 -> snapshot mld_run_w1  ({n(mo)} labels)")

        # ---- mid-week: more labels arrive; capture an instant we do NOT snapshot ----
        time.sleep(2)
        add_labels(mo, 500, 150, "bob")
        time.sleep(1)
        t_unsnapshotted = mo.scalar(f"SELECT now()")
        print(f"  ...Wed 3pm (no snapshot taken): {n(mo)} labels  @ {t_unsnapshotted}")
        time.sleep(2)
        add_labels(mo, 650, 150, "bob")
        # corrections: relabel 80 earlier items
        mo.execute(f"UPDATE {DB}.{T} SET label = 1 - label WHERE id < 80")

        # ---- week 2: training run 2 pins a snapshot ----
        mo.execute(f"CREATE SNAPSHOT mld_run_w2 FOR TABLE {DB} {T}")
        print(f"training run W2 -> snapshot mld_run_w2  ({n(mo)} labels)")

        # ---- Q1: exactly how do the two training sets differ? ----
        d = {r[0]: int(r[1]) for r in mo.query(
            f"DATA BRANCH DIFF {DB}.{T} {{snapshot='mld_run_w2'}} "
            f"AGAINST {DB}.{T} {{snapshot='mld_run_w1'}} OUTPUT SUMMARY")}
        print(f"\nW1 -> W2 delta (DATA BRANCH DIFF): "
              f"INSERTED={d.get('INSERTED',0)}  UPDATED={d.get('UPDATED',0)}  "
              f"DELETED={d.get('DELETED',0)}")
        print("  => 'training set grew by 300 new labels + 80 relabels' — precisely.")

        # ---- Q2: reconstruct the un-snapshotted Wed-3pm state via PITR ----
        mo.execute(f"CREATE SNAPSHOT mld_cur FOR TABLE {DB} {T}")          # safety
        mo.execute(f'RESTORE DATABASE {DB} TABLE {T} FROM PITR mld_annot_pitr "{t_unsnapshotted}"')
        recovered = n(mo)
        mo.execute(f"RESTORE ACCOUNT {acct} DATABASE {DB} TABLE {T} FROM SNAPSHOT mld_cur")
        print(f"\nPITR restore to Wed-3pm '{t_unsnapshotted}' -> {recovered} labels "
              f"(no snapshot existed there); then rolled forward to current {n(mo)}.")
        print("  => arbitrary point-in-time recovery; lakeFS could only return a "
              "discrete commit, not an un-tagged instant.")

        for s in ("mld_run_w1", "mld_run_w2", "mld_cur"):
            mo.execute(f"DROP SNAPSHOT IF EXISTS {s}")
        mo.execute(f"DROP PITR IF EXISTS mld_annot_pitr")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")


if __name__ == "__main__":
    main()
