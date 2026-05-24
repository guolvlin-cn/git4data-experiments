"""Experiment: RLHF / DPO preference-data curation on MatrixOne.

Preference data = many annotators labelling "is response A or B better" for each
prompt pair. The reward-model team needs: inter-annotator AGREEMENT analytics, a
CONSENSUS label per pair, exclusion of CONTESTED pairs, senior adjudication of a
few of them, and a reproducible, versioned preference set per reward-model run.

All of that is SQL aggregation + row-level versioning — a strong fit for
MatrixOne. lakeFS would store label files and need an external engine for the
agreement math, with only file-level history.

Run:  python3 -m experiments.exp_rlhf_preference
"""
import random

from matrixone.mo_client import MO

DB = "mld_rlhf"
M = 3000        # preference pairs
ANNOTATORS = ["ann_a", "ann_b", "ann_c"]


def gen(seed=11):
    rng = random.Random(seed)
    rows = []
    for pid in range(M):
        true = rng.random() < 0.5            # latent "better" response
        for ann in ANNOTATORS:
            agree = rng.random() < 0.90       # each annotator is 90% reliable
            rows.append((pid, ann, int(true if agree else (not true))))
    return rows


def main():
    rows = gen()
    with MO() as mo:
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")
        mo.execute(
            f"CREATE TABLE {DB}.votes (pair_id INT, annotator VARCHAR(8), label INT, "
            f"PRIMARY KEY (pair_id, annotator))"
        )
        mo.executemany(f"INSERT INTO {DB}.votes VALUES (%s,%s,%s)", rows)
        print(f"{M} pairs x {len(ANNOTATORS)} annotators = {len(rows)} votes\n")

        # ---- inter-annotator agreement + consensus, all in SQL ----
        # explicit PK (pair_id) so the DATA BRANCH ops below have a key to use.
        mo.execute(
            f"CREATE TABLE {DB}.consensus (pair_id INT PRIMARY KEY, label INT, "
            f"agree_votes INT, n_votes INT)"
        )
        mo.execute(
            f"INSERT INTO {DB}.consensus "
            f"SELECT pair_id, "
            f"  CASE WHEN SUM(label) * 2 > COUNT(*) THEN 1 ELSE 0 END AS label, "
            f"  CASE WHEN SUM(label) > COUNT(*) - SUM(label) "
            f"       THEN SUM(label) ELSE COUNT(*) - SUM(label) END AS agree_votes, "
            f"  COUNT(*) AS n_votes "
            f"FROM {DB}.votes GROUP BY pair_id"
        )
        unanimous = int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.consensus WHERE agree_votes = n_votes"))
        contested = int(mo.scalar(f"SELECT COUNT(*) FROM {DB}.consensus WHERE agree_votes < n_votes"))
        print("consensus built via SQL aggregation on the versioned votes:")
        print(f"  unanimous (3-0)  : {unanimous}")
        print(f"  contested (2-1)  : {contested}")

        mo.execute("DROP SNAPSHOT IF EXISTS mld_rlhf_v1")
        mo.execute(f"CREATE SNAPSHOT mld_rlhf_v1 FOR TABLE {DB} consensus")

        # ---- senior adjudicates 20 contested pairs on a REVIEW BRANCH ----
        contested_ids = [r[0] for r in mo.query(
            f"SELECT pair_id FROM {DB}.consensus WHERE agree_votes < n_votes "
            f"ORDER BY pair_id LIMIT 20")]
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.senior FROM {DB}.consensus")
        # senior overrides the shaky 2-1 majority on these pairs (flip to their call)
        mo.execute(
            f"UPDATE {DB}.senior SET label = 1 - label "
            f"WHERE pair_id IN ({', '.join(map(str, contested_ids))})"
        )
        d = {r[0]: int(r[1]) for r in mo.query(
            f"DATA BRANCH DIFF {DB}.senior AGAINST {DB}.consensus OUTPUT SUMMARY")}
        print(f"\nsenior review branch: re-labeled {len(contested_ids)} contested pairs "
              f"-> DATA BRANCH DIFF = UPDATED {d.get('UPDATED',0)} (row-level)")
        # cherry-pick ONLY the senior's adjudications back into consensus
        mo.execute(
            f"DATA BRANCH PICK {DB}.senior INTO {DB}.consensus "
            f"KEYS({', '.join(map(str, contested_ids))}) WHEN CONFLICT ACCEPT"
        )
        print(f"DATA BRANCH PICK applied senior's 20 adjudications into consensus")

        # ---- reward-model train_set = unanimous OR senior-adjudicated (SQL filter) ----
        rev = ", ".join(map(str, contested_ids))
        n_train = int(mo.scalar(
            f"SELECT COUNT(*) FROM {DB}.consensus "
            f"WHERE agree_votes = n_votes OR pair_id IN ({rev})"))
        print(f"\nreward-model train_set = unanimous + senior-adjudicated = {n_train} pairs "
              f"({unanimous} + {len(contested_ids)}); the other "
              f"{contested - len(contested_ids)} contested pairs stay excluded")
        dist = mo.query(
            f"SELECT label, COUNT(*) FROM {DB}.consensus "
            f"WHERE agree_votes = n_votes OR pair_id IN ({rev}) GROUP BY label ORDER BY label")
        print(f"train_set label balance: {{{', '.join(f'{l}:{c}' for l,c in dist)}}}")
        print("\nreproducible: consensus pinned at snapshot mld_rlhf_v1 for this "
              "reward-model run; a new annotator's votes -> recompute consensus -> "
              "DATA BRANCH DIFF gives the exact pairs whose label flipped.")

        mo.execute("DROP SNAPSHOT IF EXISTS mld_rlhf_v1")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        print("\nWhy MatrixOne fits RLHF preference data: agreement/consensus is SQL "
              "aggregation on versioned votes; contested pairs are a WHERE clause;\n"
              "adjudication is cherry-pick; each reward-model run pins a snapshot. "
              "lakeFS would need an external engine for all the math.")


if __name__ == "__main__":
    main()
