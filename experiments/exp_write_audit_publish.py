"""Scenario 3: Write-Audit-Publish (WAP) — the standard MLOps data pattern.

WRITE new data to an isolated staging branch; AUDIT it with quality gates;
PUBLISH (atomically promote to prod) only if every gate passes — prod is never
exposed to un-audited data.

MatrixOne maps WAP directly:
  WRITE   = DATA BRANCH CREATE staging FROM prod; insert the new batch
  AUDIT   = SQL quality gates run on staging (prod untouched throughout)
  PUBLISH = DATA BRANCH MERGE staging INTO prod   (atomic, row-level)

(lakeFS does the same shape: branch -> checks -> merge; it audits via an external
engine and merges whole files, MatrixOne audits in SQL and merges rows.)

Run:  python3 -m experiments.exp_write_audit_publish
"""
from matrixone.mo_client import MO

DB = "mld_wap"
PROD = "prod_labels"
STG = "staging"
NEW_BASE = 1_000_000      # ids of the incoming batch
EVAL_LO, EVAL_HI = 5000, 5049


def prod_count(mo):
    return int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.{PROD}"))


def audit(mo, table, where_new):
    """Run quality gates on the new rows of `table`; return {gate: violations}."""
    g = {}
    g["invalid_label"] = int(mo.scalar(
        f"SELECT COUNT(*) FROM {DB}.{table} WHERE {where_new} AND label NOT IN (0,1)"))
    g["dup_item"] = int(mo.scalar(
        f"SELECT COUNT(*) FROM (SELECT item_id FROM {DB}.{table} WHERE {where_new} "
        f"GROUP BY item_id HAVING COUNT(*) > 1) d"))
    g["eval_contam"] = int(mo.scalar(
        f"SELECT COUNT(*) FROM {DB}.{table} WHERE {where_new} "
        f"AND item_id BETWEEN {EVAL_LO} AND {EVAL_HI}"))
    pos = int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.{table} WHERE {where_new} AND label=1"))
    tot = int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.{table} WHERE {where_new}"))
    ratio = pos / tot if tot else 0
    g["class_imbalance"] = 0 if 0.4 <= ratio <= 0.6 else 1
    g["_ratio"] = round(ratio, 3)
    return g


def gates_pass(g):
    return all(v == 0 for k, v in g.items() if not k.startswith("_"))


def main():
    with MO() as mo:
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")
        mo.execute(
            f"CREATE TABLE {DB}.{PROD} (id BIGINT PRIMARY KEY, item_id INT, label INT)")
        # clean, balanced production data
        mo.execute(
            f"INSERT INTO {DB}.{PROD} SELECT result, result, result%2 "
            f"FROM generate_series(0,999) g")
        print(f"prod starts with {prod_count(mo)} clean labels\n")

        # ---- WRITE: stage a new annotation batch (with planted problems) ----
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.{STG} FROM {DB}.{PROD}")
        where_new = f"id >= {NEW_BASE}"
        # 300 good rows
        mo.execute(
            f"INSERT INTO {DB}.{STG} SELECT {NEW_BASE}+result, 2000+result, result%2 "
            f"FROM generate_series(0,299) g")
        # planted issues: 15 invalid labels, 15 duplicate item_ids, 10 eval-contaminated
        mo.execute(
            f"INSERT INTO {DB}.{STG} SELECT {NEW_BASE}+1000+result, 9000+result, 9 "
            f"FROM generate_series(0,14) g")
        mo.execute(
            f"INSERT INTO {DB}.{STG} SELECT {NEW_BASE}+2000+result, 2000+result, result%2 "
            f"FROM generate_series(0,14) g")   # item_id 2000.. collide -> duplicates
        mo.execute(
            f"INSERT INTO {DB}.{STG} SELECT {NEW_BASE}+3000+result, {EVAL_LO}+result, result%2 "
            f"FROM generate_series(0,9) g")
        print(f"WRITE: staged new batch on branch '{STG}' "
              f"({int(mo.scalar(f'SELECT COUNT(*) FROM {DB}.{STG} WHERE {where_new}'))} new rows)")

        # ---- AUDIT #1 (prod untouched) ----
        g1 = audit(mo, STG, where_new)
        print(f"AUDIT #1 on staging: {g1}")
        print(f"  prod still {prod_count(mo)} rows (untouched), gates_pass={gates_pass(g1)}")

        # ---- REMEDIATE on staging ----
        mo.execute(f"DELETE FROM {DB}.{STG} WHERE {where_new} AND label NOT IN (0,1)")
        mo.execute(f"DELETE FROM {DB}.{STG} WHERE {where_new} AND item_id BETWEEN {EVAL_LO} AND {EVAL_HI}")
        mo.execute(
            f"DELETE FROM {DB}.{STG} WHERE id IN (SELECT id FROM (SELECT id, "
            f"ROW_NUMBER() OVER (PARTITION BY item_id ORDER BY id) rn FROM {DB}.{STG} "
            f"WHERE {where_new}) z WHERE rn > 1)")
        g2 = audit(mo, STG, where_new)
        print(f"\nremediated; AUDIT #2: {g2}  gates_pass={gates_pass(g2)}")

        # ---- PUBLISH only if audit passes ----
        if gates_pass(g2):
            mo.execute(f"DATA BRANCH MERGE {DB}.{STG} INTO {DB}.{PROD} WHEN CONFLICT ACCEPT")
            print(f"PUBLISH: DATA BRANCH MERGE staging -> prod (atomic).  "
                  f"prod now {prod_count(mo)} rows.")
        else:
            print("audit failed again -> NOT published; prod untouched.")
        mo.execute(f"DROP TABLE IF EXISTS {DB}.{STG}")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        print("\nWAP held: prod never saw un-audited rows; bad data was caught & fixed on "
              "the branch; publish was a single atomic row-level merge.")


if __name__ == "__main__":
    main()
