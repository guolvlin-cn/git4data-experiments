"""ClickHouse vs MatrixOne as an agent/OTel TRACE backend — key-point comparison.

ClickHouse is the de-facto store for OTel traces (OTel Collector has a first-class
ClickHouse exporter; SigNoz/Uptrace are built on it). So for the agent-trace
scenario it's the right baseline. We put the SAME OTel-shaped spans into both and
compare on the points that matter for this scenario:

  ingest | analytical queries | VERSIONING | row-level MUTABILITY | JOINs

ClickHouse (in-process via chdb) vs MatrixOne (remote cloud — note the network
caveat: this is NOT a fair raw-speed test, ClickHouse is a specialized OLAP
engine and will win on ingest/scan regardless; the point is the CAPABILITY axes).

Run:  python3 -m experiments.exp_clickhouse_vs_matrixone   (needs `chdb`)
"""
import random
import time

from chdb import session as chs

import config
from matrixone.mo_client import MO

DB = "mld_chcmp"
N_TRACES = 250
COLS = ("trace_id", "span_id", "parent_span_id", "name", "kind", "start_ns",
        "duration_ms", "status", "gen_ai_operation", "gen_ai_request_model",
        "input_tokens", "output_tokens", "ingest_run", "eval_label")


def gen_spans(n_traces, model, run_tag, base, seed):
    rng = random.Random(seed)
    rows = []
    t = base
    sid = run_tag * 10_000_000           # disjoint span-id space per run -> globally unique
    for i in range(n_traces):
        tid = f"{run_tag:02x}{i:030x}"
        root = f"{sid:016x}"; sid += 1
        fail = (i % 7 == 0)
        rows.append((tid, root, "", "invoke_agent research-bot", "INTERNAL", t,
                     round(rng.uniform(20, 80), 3), "ERROR" if fail else "OK",
                     "invoke_agent", model, 0, 0, run_tag, ""))
        for j, (nm, op) in enumerate([(f"chat {model}", "chat"),
                                      ("execute_tool web_search", "execute_tool"),
                                      (f"chat {model}", "chat")]):
            cid = f"{sid:016x}"; sid += 1
            it = rng.randint(200, 600) if op == "chat" else 0
            ot = rng.randint(50, 200) if op == "chat" else 0
            st = "ERROR" if (fail and op == "execute_tool") else "OK"
            rows.append((tid, cid, root, nm, "CLIENT" if op == "chat" else "INTERNAL",
                         t + j + 1, round(rng.uniform(5, 60), 3), st, op, model, it, ot, run_tag, ""))
        t += 100
    return rows


def hr(s):
    print("\n" + "=" * 74 + f"\n  {s}\n" + "=" * 74)


# ---------------- ClickHouse (chdb, in-process) ----------------
def ch_lit(v):
    return str(v) if isinstance(v, (int, float)) else "'" + str(v).replace("'", "''") + "'"


def run_clickhouse(spans):
    sess = chs.Session()
    sess.query("CREATE DATABASE IF NOT EXISTS t", "CSV")
    sess.query(
        "CREATE TABLE t.spans (trace_id String, span_id String, parent_span_id String, "
        "name String, kind String, start_ns Int64, duration_ms Float64, status String, "
        "gen_ai_operation String, gen_ai_request_model String, input_tokens Int32, "
        "output_tokens Int32, ingest_run Int32, eval_label String) "
        "ENGINE = MergeTree ORDER BY (trace_id, span_id)", "CSV")
    values = ",".join("(" + ",".join(ch_lit(v) for v in r) + ")" for r in spans)
    t0 = time.perf_counter()
    sess.query(f"INSERT INTO t.spans VALUES {values}", "CSV")
    ingest_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    q_model = str(sess.query(
        "SELECT gen_ai_request_model, count(), sum(input_tokens+output_tokens), "
        "round(avg(duration_ms),2) FROM t.spans WHERE gen_ai_operation='chat' "
        "GROUP BY gen_ai_request_model", "CSV")).strip()
    q_err = str(sess.query("SELECT count() FROM t.spans WHERE status='ERROR'", "CSV")).strip()
    q_p95 = str(sess.query("SELECT round(quantile(0.95)(duration_ms),2) FROM t.spans", "CSV")).strip()
    query_ms = (time.perf_counter() - t0) * 1000

    # mutability: attaching an eval label = a MUTATION (ALTER ... UPDATE), async/heavy
    mut_note = "OK but it is a background MUTATION (rewrites parts), not a row update"
    try:
        sess.query("ALTER TABLE t.spans UPDATE eval_label='bad' WHERE status='ERROR'", "CSV")
    except Exception as e:
        mut_note = f"mutation error: {str(e)[:60]}"
    sess.close()
    return dict(ingest=ingest_ms, query=query_ms, by_model=q_model, errors=q_err,
                p95=q_p95, mut=mut_note)


# ---------------- MatrixOne (remote) ----------------
def run_matrixone(mo, spans_v1, spans_v2):
    acct = config.mo_account_name()
    mo.execute(f"DROP DATABASE IF EXISTS {DB}")
    mo.execute(f"CREATE DATABASE {DB}")
    mo.execute(
        f"CREATE TABLE {DB}.spans (trace_id VARCHAR(64), span_id VARCHAR(32), "
        f"parent_span_id VARCHAR(32), name VARCHAR(96), kind VARCHAR(16), start_ns BIGINT, "
        f"duration_ms DOUBLE, status VARCHAR(8), gen_ai_operation VARCHAR(24), "
        f"gen_ai_request_model VARCHAR(48), input_tokens INT, output_tokens INT, "
        f"ingest_run INT, eval_label VARCHAR(16), PRIMARY KEY (trace_id, span_id))")
    ins = f"INSERT INTO {DB}.spans ({','.join(COLS)}) VALUES ({','.join(['%s']*len(COLS))})"
    t0 = time.perf_counter()
    mo.executemany(ins, spans_v1)
    ingest_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    by_model = mo.query(
        f"SELECT gen_ai_request_model, COUNT(*), SUM(input_tokens+output_tokens), "
        f"ROUND(AVG(duration_ms),2) FROM {DB}.spans WHERE gen_ai_operation='chat' "
        f"GROUP BY gen_ai_request_model")
    errs = mo.scalar(f"SELECT COUNT(*) FROM {DB}.spans WHERE status='ERROR'")
    mx = mo.scalar(f"SELECT ROUND(MAX(duration_ms),2) FROM {DB}.spans")
    query_ms = (time.perf_counter() - t0) * 1000

    # --- capability: VERSIONING (git4data) — ClickHouse has no equivalent ---
    mo.execute("DROP SNAPSHOT IF EXISTS mld_ch_v1")
    mo.execute(f"CREATE SNAPSHOT mld_ch_v1 FOR TABLE {DB} spans")
    mo.executemany(ins, spans_v2)                       # agent v2 traces arrive
    mo.execute("DROP SNAPSHOT IF EXISTS mld_ch_v2")
    mo.execute(f"CREATE SNAPSHOT mld_ch_v2 FOR TABLE {DB} spans")
    d = {r[0]: int(r[1]) for r in mo.query(
        f"DATA BRANCH DIFF {DB}.spans {{snapshot='mld_ch_v2'}} "
        f"AGAINST {DB}.spans {{snapshot='mld_ch_v1'}} OUTPUT SUMMARY")}

    # --- capability: row-level MUTABILITY — attach a human-eval label, transactional ---
    t0 = time.perf_counter()
    mo.execute(f"UPDATE {DB}.spans SET eval_label='bad' WHERE status='ERROR'")
    mut_ms = (time.perf_counter() - t0) * 1000
    labeled = mo.scalar(f"SELECT COUNT(*) FROM {DB}.spans WHERE eval_label='bad'")

    # --- capability: JOIN to a model_registry (unified SQL store) ---
    mo.execute(f"CREATE TABLE {DB}.model_registry (model VARCHAR(48) PRIMARY KEY, cost_per_1k DOUBLE)")
    mo.execute(f"INSERT INTO {DB}.model_registry VALUES ('gpt-4o',0.005),('gpt-4o-mini',0.0006)")
    joined = mo.query(
        f"SELECT s.gen_ai_request_model, ROUND(SUM((s.input_tokens+s.output_tokens)/1000.0*m.cost_per_1k),4) "
        f"FROM {DB}.spans s JOIN {DB}.model_registry m ON s.gen_ai_request_model=m.model "
        f"WHERE s.gen_ai_operation='chat' GROUP BY s.gen_ai_request_model")

    mo.execute("DROP SNAPSHOT IF EXISTS mld_ch_v1")
    mo.execute("DROP SNAPSHOT IF EXISTS mld_ch_v2")
    mo.execute(f"DROP DATABASE IF EXISTS {DB}")
    return dict(ingest=ingest_ms, query=query_ms, by_model=by_model, errors=errs,
                maxdur=mx, diff_inserted=d.get("INSERTED", 0), mut_ms=mut_ms,
                labeled=labeled, joined=joined)


def main():
    spans_v1 = gen_spans(N_TRACES, "gpt-4o", 1, 0, seed=1)
    spans_v2 = gen_spans(N_TRACES, "gpt-4o-mini", 2, N_TRACES, seed=2)
    n = len(spans_v1)

    hr(f"Same {n} OTel spans into BOTH (ClickHouse in-process via chdb, MatrixOne remote)")
    ch = run_clickhouse(spans_v1)
    with MO() as mo:
        mo_ = run_matrixone(mo, spans_v1, spans_v2)

    print(f"  ingest {n} spans:   ClickHouse {ch['ingest']:.0f} ms   |   MatrixOne {mo_['ingest']:.0f} ms (remote)")
    print(f"  observability queries: ClickHouse {ch['query']:.0f} ms   |   MatrixOne {mo_['query']:.0f} ms (remote)")
    print(f"  error spans:        ClickHouse {ch['errors']}   |   MatrixOne {mo_['errors']}  (same data, same answer)")
    print(f"  ClickHouse has rich OLAP fns: p95 duration = {ch['p95']} ms (quantile()); "
          f"MatrixOne used MAX={mo_['maxdur']} (no native quantile)")

    hr("Capability points for THIS scenario (agent traces + iteration)")
    print("  VERSIONING (snapshot/branch/diff/PITR):")
    print(f"    ClickHouse: none (no git-for-data; you'd keep version columns or separate tables)")
    print(f"    MatrixOne:  CREATE SNAPSHOT per agent version + DATA BRANCH DIFF v2 vs v1 = "
          f"INSERTED {mo_['diff_inserted']} spans (row-level)")
    print("  ROW-LEVEL MUTABILITY (attach human-eval labels / corrections to spans):")
    print(f"    ClickHouse: {ch['mut']}")
    print(f"    MatrixOne:  plain UPDATE … WHERE status='ERROR' -> labeled {mo_['labeled']} spans "
          f"in {mo_['mut_ms']:.0f} ms (transactional, instant)")
    print("  JOIN to other structured data (unified SQL store):")
    print(f"    MatrixOne:  spans ⋈ model_registry -> cost per model: "
          f"{[(m, c) for m, c in mo_['joined']]}")
    print("    ClickHouse: joins supported but limited/awkward; typically a dedicated OLAP store")

    hr("Verdict")
    print("  ClickHouse WINS the pure trace-store job: ingest throughput, OLAP scan speed +")
    print("    rich functions (quantile/uniq), native TTL retention, and the mature OTel/SigNoz")
    print("    /Grafana ecosystem. For high-volume real-time agent monitoring, use ClickHouse.")
    print("  MatrixOne's edge for the AGENT-ITERATION angle: the SAME trace store is also")
    print("    git4data-VERSIONED (snapshot per eval/agent version, row-level DIFF, PITR),")
    print("    row-level MUTABLE (annotate/score/correct spans transactionally), and JOINable")
    print("    to versioned datasets/feature/model tables in one SQL engine. So: observability")
    print("    backend + versioned, annotatable, unified agent-iteration substrate.")
    print("  Pragmatic combo: ClickHouse for the firehose of monitoring traces; MatrixOne for")
    print("    the curated subset you version/annotate/evaluate and tie to training data.")
    print("\n  (caveat: MatrixOne here is REMOTE cloud — network latency inflates its ms; a")
    print("   co-located MO narrows ingest/query gaps. ClickHouse still wins raw OLAP by design.)")


if __name__ == "__main__":
    main()
