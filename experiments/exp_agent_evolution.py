"""Non-ML scenario: store an agent's trace in git4data and EVOLVE it by branching.

Idea: model the agent's evolvable "brain" as structured tables (a `memory` of
learned skills) plus an execution `trace` log — both versioned by git4data. Then
agent evolution becomes a git workflow:

  baseline -> branch the brain -> learn from the failed traces -> run the variant
  -> compare -> MERGE the winner (or discard). Each generation is a snapshot
  (= an immutable agent version); DATA BRANCH DIFF shows exactly what it learned;
  a bad mutation is undone with RESTORE; two evolution branches that touch the
  same skill collide -> conflict resolution; only validated skills are cherry-picked.

The "agent" is a deterministic simulator so the whole thing is reproducible:
a problem of type T is solved iff memory holds a rule for T with quality >= 0.5.

Run:  python3 -m experiments.exp_agent_evolution
"""
import config
from matrixone.mo_client import MO

DB = "mld_agent"
TYPES, PER = 10, 30
TASKSET = [(pid, pid % TYPES) for pid in range(TYPES * PER)]   # 300 problems, balanced
SOLVE_Q = 0.5


def run_agent(mo, run_id, mem="memory"):
    """Deterministic agent run against `mem`; logs traces; returns success rate."""
    rules = {int(t): float(q) for t, q in mo.query(
        f"SELECT problem_type, rule_quality FROM {DB}.{mem}")}
    rows, solved = [], 0
    for pid, t in TASKSET:
        q = rules.get(t, 0.0)
        ok = 1 if q >= SOLVE_Q else 0
        solved += ok
        rows.append((run_id * 1_000_000 + pid, run_id, pid, t, q, ok))
    mo.execute(f"DELETE FROM {DB}.trace WHERE run_id = %s", (run_id,))
    mo.executemany(
        f"INSERT INTO {DB}.trace (id,run_id,problem_id,problem_type,used_quality,success) "
        f"VALUES (%s,%s,%s,%s,%s,%s)", rows)
    return solved / len(TASKSET)


def learn_from_failures(mo, mem_branch, run_id, k, gen):
    """Inspect the failed traces of `run_id`; learn rules for the k worst types."""
    fails = [int(r[0]) for r in mo.query(
        f"SELECT problem_type FROM {DB}.trace WHERE run_id={run_id} AND success=0 "
        f"GROUP BY problem_type ORDER BY COUNT(*) DESC, problem_type LIMIT {k}")]
    for t in fails:
        mo.execute(f"INSERT INTO {DB}.{mem_branch} VALUES (%s, %s, %s)", (t, 0.9, f"gen{gen}"))
    return fails


def diff_summary(mo, target, base):
    return {r[0]: int(r[1]) for r in mo.query(
        f"DATA BRANCH DIFF {DB}.{target} AGAINST {DB}.{base} OUTPUT SUMMARY")}


def hr(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


def main():
    acct = config.mo_account_name()
    with MO() as mo:
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")
        mo.execute(f"CREATE DATABASE {DB}")
        mo.execute(
            f"CREATE TABLE {DB}.memory (problem_type INT PRIMARY KEY, rule_quality DOUBLE, "
            f"learned_in VARCHAR(16))")
        mo.execute(
            f"CREATE TABLE {DB}.trace (id BIGINT PRIMARY KEY, run_id INT, problem_id INT, "
            f"problem_type INT, used_quality DOUBLE, success INT)")
        for g in range(6):
            mo.execute(f"DROP SNAPSHOT IF EXISTS mld_agent_gen{g}")

        # ---- ACT 1: baseline agent (knows types 0-4), trace stored & snapshotted ----
        hr("ACT 1  Baseline agent + trace stored & versioned (generation 0)")
        for t in range(5):
            mo.execute(f"INSERT INTO {DB}.memory VALUES (%s,0.9,'seed')", (t,))
        sr = run_agent(mo, run_id=0)
        mo.execute(f"CREATE SNAPSHOT mld_agent_gen0 FOR TABLE {DB} memory")
        last_run, cur_sr, gen = 0, sr, 0
        print(f"gen0: memory has {mo.scalar(f'SELECT COUNT(*) FROM {DB}.memory')} skills; "
              f"success={cur_sr:.0%}; traces logged (run_id=0); snapshot mld_agent_gen0")

        # ---- ACT 2: evolution loop — branch, learn from failures, compare, merge ----
        hr("ACT 2  Evolve by branching: learn from failed traces -> compare -> MERGE")
        while cur_sr < 1.0 and gen < 5:
            gen += 1
            mo.execute(f"DROP TABLE IF EXISTS {DB}.memory_exp")
            mo.execute(f"DATA BRANCH CREATE TABLE {DB}.memory_exp FROM {DB}.memory")
            learned = learn_from_failures(mo, "memory_exp", last_run, k=2, gen=gen)
            sr_branch = run_agent(mo, run_id=900 + gen, mem="memory_exp")
            verdict = "MERGE" if sr_branch > cur_sr else "discard"
            print(f"  gen{gen}: branch learned rules for types {learned} -> "
                  f"branch success={sr_branch:.0%} vs main {cur_sr:.0%}  [{verdict}]")
            if sr_branch > cur_sr:
                mo.execute(f"DATA BRANCH MERGE {DB}.memory_exp INTO {DB}.memory WHEN CONFLICT ACCEPT")
                mo.execute(f"CREATE SNAPSHOT mld_agent_gen{gen} FOR TABLE {DB} memory")
                d = diff_summary_snap(mo, f"mld_agent_gen{gen}", f"mld_agent_gen{gen-1}")
                cur_sr = run_agent(mo, run_id=gen)
                last_run = gen
                print(f"         merged -> gen{gen} snapshot; DATA BRANCH DIFF vs gen{gen-1} "
                      f"INSERTED={d.get('INSERTED',0)} (what it learned); success now {cur_sr:.0%}")
            mo.execute(f"DROP TABLE IF EXISTS {DB}.memory_exp")
        print(f"\n  evolution converged: success={cur_sr:.0%} after gen{gen} "
              f"(each generation = an immutable agent snapshot, fully reproducible)")

        # ---- ACT 3: a bad mutation regresses the agent -> RESTORE rollback ----
        hr("ACT 3  Bad mutation -> regression -> RESTORE rollback")
        good_snap = f"mld_agent_gen{gen}"
        mo.execute(f"UPDATE {DB}.memory SET rule_quality=0.2 WHERE problem_type=3")  # forget a skill
        bad_sr = run_agent(mo, run_id=777)
        print(f"  applied a faulty self-edit (skill #3 degraded) -> success drops to {bad_sr:.0%}")
        mo.execute(f"RESTORE ACCOUNT {acct} DATABASE {DB} TABLE memory FROM SNAPSHOT {good_snap}")
        rec_sr = run_agent(mo, run_id=778)
        print(f"  RESTORE memory FROM SNAPSHOT {good_snap} -> success recovered to {rec_sr:.0%}")

        # ---- ACT 4: two evolution branches collide on the same skill + cherry-pick ----
        hr("ACT 4  Parallel evolution branches conflict on one skill + cherry-pick")
        for b in ("explorer_a", "explorer_b"):
            mo.execute(f"DROP TABLE IF EXISTS {DB}.{b}")
            mo.execute(f"DATA BRANCH CREATE TABLE {DB}.{b} FROM {DB}.memory")
        mo.execute(f"UPDATE {DB}.explorer_a SET rule_quality=0.6  WHERE problem_type=0")
        mo.execute(f"UPDATE {DB}.explorer_b SET rule_quality=0.99 WHERE problem_type=0")
        mo.execute(f"DATA BRANCH MERGE {DB}.explorer_a INTO {DB}.memory WHEN CONFLICT FAIL")
        try:
            mo.execute(f"DATA BRANCH MERGE {DB}.explorer_b INTO {DB}.memory WHEN CONFLICT FAIL")
            print("  explorer_b merged with FAIL -> unexpected (no conflict?)")
        except Exception as e:
            print(f"  explorer_b vs explorer_a on skill#0 -> conflict detected: "
                  f"{str(e).split('conflict')[-1].strip()[:55]}")
        mo.execute(f"DATA BRANCH MERGE {DB}.explorer_b INTO {DB}.memory WHEN CONFLICT ACCEPT")
        print(f"  resolved with ACCEPT -> skill#0 quality = "
              f"{mo.scalar(f'SELECT rule_quality FROM {DB}.memory WHERE problem_type=0')}")

        # cherry-pick: a branch refines 4 skills, only 2 are validated -> PICK those 2
        mo.execute(f"DROP TABLE IF EXISTS {DB}.explorer_c")
        mo.execute(f"DATA BRANCH CREATE TABLE {DB}.explorer_c FROM {DB}.memory")
        mo.execute(f"UPDATE {DB}.explorer_c SET rule_quality=0.95 WHERE problem_type IN (1,2,3,4)")
        mo.execute(f"DATA BRANCH PICK {DB}.explorer_c INTO {DB}.memory KEYS(1,2) WHEN CONFLICT ACCEPT")
        picked = mo.query(f"SELECT problem_type, rule_quality FROM {DB}.memory "
                          f"WHERE problem_type IN (1,2,3,4) ORDER BY problem_type")
        print(f"  cherry-pick validated skills {{1,2}} only -> {dict((int(t),q) for t,q in picked)} "
              f"(3,4 untouched)")

        for t in ("explorer_a", "explorer_b", "explorer_c"):
            mo.execute(f"DROP TABLE IF EXISTS {DB}.{t}")
        for g in range(gen + 1):
            mo.execute(f"DROP SNAPSHOT IF EXISTS mld_agent_gen{g}")
        mo.execute(f"DROP DATABASE IF EXISTS {DB}")

        hr("Verdict")
        print("  Works. git4data maps cleanly onto agent evolution:")
        print("   trace+brain stored as versioned tables; branch = isolated variant;")
        print("   learn-from-failed-traces + compare + MERGE = a generation; snapshot =")
        print("   immutable agent version; DATA BRANCH DIFF = 'what it learned'; RESTORE =")
        print("   undo a bad self-edit; conflict policies = reconcile parallel explorers;")
        print("   PICK = promote only validated skills. All reproducible & auditable.")


def diff_summary_snap(mo, target_snap, base_snap):
    """DIFF same table across two snapshots (memory @ two generations)."""
    return {r[0]: int(r[1]) for r in mo.query(
        f"DATA BRANCH DIFF {DB}.memory {{snapshot='{target_snap}'}} "
        f"AGAINST {DB}.memory {{snapshot='{base_snap}'}} OUTPUT SUMMARY")}


if __name__ == "__main__":
    main()
