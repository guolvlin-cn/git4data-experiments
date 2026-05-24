"""Experiment: two annotation teams relabel in parallel, then merge — conflicts.

A common ML-data workflow: two people (or pipelines) branch the label set and
relabel overlapping rows differently. A real version-control system must DETECT
the collision (not silently lose one side) and let you choose a resolution.

We branch twice from the same base, make A and B disagree on one shared row,
merge A in, then merge B with each WHEN CONFLICT policy:
  FAIL   -> errors on the conflicting row (default; forces a human decision)
  SKIP   -> keeps the destination's value, drops B's conflicting change
  ACCEPT -> overwrites with B's value

Run:  python3 -m experiments.exp_concurrent_merge
"""
from matrixone.mo_client import MO

DB = "mld_merge_exp"


def setup(mo):
    mo.execute(f"CREATE DATABASE IF NOT EXISTS {DB}")
    for t in ("labels", "team_a", "team_b"):
        mo.execute(f"DROP TABLE IF EXISTS {DB}.{t}")
    mo.execute(f"CREATE TABLE {DB}.labels (id INT PRIMARY KEY, label INT)")
    mo.execute(f"INSERT INTO {DB}.labels VALUES (1,0),(2,0),(3,0),(4,0),(5,0)")


def show(mo, msg):
    rows = dict(mo.query(f"SELECT id,label FROM {DB}.labels ORDER BY id"))
    print(f"  {msg:<46} labels={rows}")


def main():
    with MO() as mo:
        setup(mo)
        show(mo, "base (all 0)")

        # Two parallel relabel branches off the same base.
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.team_a FROM {DB}.labels")
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.team_b FROM {DB}.labels")
        mo.execute(f"UPDATE {DB}.team_a SET label=1 WHERE id IN (1,2,3)")  # A: 1,2,3 -> 1
        mo.execute(f"UPDATE {DB}.team_b SET label=2 WHERE id IN (3,4,5)")  # B: 3,4,5 -> 2
        print("  team_a relabels {1,2,3}->1   team_b relabels {3,4,5}->2   "
              "(row 3 disagrees)")

        # Merge A cleanly (no conflicts vs base).
        mo.execute(f"DATA BRANCH MERGE {DB}.team_a INTO {DB}.labels WHEN CONFLICT FAIL")
        show(mo, "after MERGE team_a (FAIL)")

        # Merge B: row 3 now conflicts (both B and base changed it from original).
        try:
            mo.execute(
                f"DATA BRANCH MERGE {DB}.team_b INTO {DB}.labels WHEN CONFLICT FAIL"
            )
            print("  MERGE team_b WHEN CONFLICT FAIL -> unexpectedly succeeded")
        except Exception as e:
            print(f"  MERGE team_b WHEN CONFLICT FAIL -> detected conflict: "
                  f"{str(e).split('conflict')[-1].strip()[:60]}")

        mo.execute(f"DATA BRANCH MERGE {DB}.team_b INTO {DB}.labels WHEN CONFLICT SKIP")
        show(mo, "after MERGE team_b (SKIP: keep A's row 3)")

        # Reset and show ACCEPT taking B's value instead.
        setup(mo)
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.team_a FROM {DB}.labels")
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.team_b FROM {DB}.labels")
        mo.execute(f"UPDATE {DB}.team_a SET label=1 WHERE id IN (1,2,3)")
        mo.execute(f"UPDATE {DB}.team_b SET label=2 WHERE id IN (3,4,5)")
        mo.execute(f"DATA BRANCH MERGE {DB}.team_a INTO {DB}.labels WHEN CONFLICT FAIL")
        mo.execute(f"DATA BRANCH MERGE {DB}.team_b INTO {DB}.labels WHEN CONFLICT ACCEPT")
        show(mo, "after MERGE team_b (ACCEPT: take B's row 3)")

        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        print("\nTakeaway: row-level 3-way merge DETECTS the row-3 collision and lets "
              "you choose FAIL / SKIP / ACCEPT — lakeFS would see whole-file conflicts.")


if __name__ == "__main__":
    main()
