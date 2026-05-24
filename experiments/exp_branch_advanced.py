"""Experiment: two boundaries of MatrixOne's branch model that matter for ML.

1. SCHEMA EVOLUTION — DATA BRANCH DIFF/MERGE require IDENTICAL schema. If a
   branch adds a feature column, diff/merge refuse ("schema is not equivalent").
   So you can't diff/merge across a feature-schema change; align schemas first.

2. WHOLE-DATABASE ATOMIC VERSIONING — DATA BRANCH DIFF/MERGE are per-table, but
   `CREATE SNAPSHOT ... FOR DATABASE` + `RESTORE ... DATABASE` version and revert
   MANY tables (features + labels + metadata) together, atomically. That covers
   the "one consistent version across the whole training set" need.

Run:  python3 -m experiments.exp_branch_advanced
"""
import config
from matrixone.mo_client import MO

DB = "mld_adv"


def main():
    with MO() as mo:
        acct = config.mo_account_name()
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")

        print("== 1. Schema evolution across a branch ==")
        mo.execute(f"CREATE TABLE {DB}.feat (id INT PRIMARY KEY, f0 DOUBLE, label INT)")
        mo.execute(f"INSERT INTO {DB}.feat VALUES (1,0.1,0),(2,0.2,1),(3,0.3,0)")
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.feat_b FROM {DB}.feat")
        mo.execute(f"ALTER TABLE {DB}.feat_b ADD COLUMN f1 DOUBLE DEFAULT 0.0")
        mo.execute(f"UPDATE {DB}.feat_b SET f1 = 0.9 WHERE id = 2")
        try:
            mo.query(f"DATA BRANCH DIFF {DB}.feat_b AGAINST {DB}.feat OUTPUT SUMMARY")
            print("  DIFF across added column -> unexpectedly succeeded")
        except Exception as e:
            print(f"  DIFF across added column -> refused: {str(e).split(':')[-1].strip()[:60]}")
        print("  => feature-schema changes break diff/merge; must align schema first.")

        print("\n== 2. Whole-database atomic snapshot/restore (features + labels) ==")
        mo.execute(f"CREATE TABLE {DB}.features (id INT PRIMARY KEY, f0 DOUBLE)")
        mo.execute(f"CREATE TABLE {DB}.labels (id INT PRIMARY KEY, y INT)")
        mo.execute(f"INSERT INTO {DB}.features VALUES (1,0.1),(2,0.2)")
        mo.execute(f"INSERT INTO {DB}.labels VALUES (1,0),(2,1)")
        mo.execute(f"DROP SNAPSHOT IF EXISTS mld_adv_dbsnap")
        mo.execute(f"CREATE SNAPSHOT mld_adv_dbsnap FOR DATABASE {DB}")
        # diverge BOTH tables after the snapshot
        mo.execute(f"INSERT INTO {DB}.features VALUES (3,0.3)")
        mo.execute(f"INSERT INTO {DB}.labels VALUES (3,0)")

        def counts(snap=""):
            s = f" {{snapshot='{snap}'}}" if snap else ""
            f = mo.scalar(f"SELECT COUNT(*) FROM {DB}.features{s}")
            l = mo.scalar(f"SELECT COUNT(*) FROM {DB}.labels{s}")
            return f, l

        print(f"  live now:           features={counts()[0]} labels={counts()[1]}")
        print(f"  at db snapshot:     features={counts('mld_adv_dbsnap')[0]} "
              f"labels={counts('mld_adv_dbsnap')[1]}  (consistent point-in-time)")
        mo.execute(
            f"RESTORE ACCOUNT {acct} DATABASE {DB} FROM SNAPSHOT mld_adv_dbsnap"
        )
        print(f"  after db RESTORE:   features={counts()[0]} labels={counts()[1]}  "
              f"(both reverted atomically)")

        mo.execute(f"DROP SNAPSHOT IF EXISTS mld_adv_dbsnap")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")


if __name__ == "__main__":
    main()
