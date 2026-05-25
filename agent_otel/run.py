"""Run a REAL agent and land its OpenTelemetry traces in MatrixOne, then query
them back with SQL and snapshot the trace store with git4data.

LLM backend auto-selects: real Claude/OpenAI if a key is set, else an offline
deterministic planner (so this runs with no keys). Either way the agent loop,
tools, OTel instrumentation and MatrixOne export are real.

Run:  python3 -m agent_otel.run
      ANTHROPIC_API_KEY=... python3 -m agent_otel.run     # use real Claude
"""
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import config
from matrixone.mo_client import MO
from .agent import Agent
from .exporter import MatrixOneSpanExporter, SPANS_TABLE, create_spans_table
from .llm import make_llm

DB = "agent_otel_demo"
TASKS = [
    "What is 47 * 19?",
    "What is the population of France, doubled?",
    "Who wrote Hamlet?",
    "What is 10% of the speed of light (km/s)?",
    "What is the population of Atlantis?",   # unknown -> tool error path
]


def hr(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


def main():
    acct = config.mo_account_name()
    mo = MO()
    create_spans_table(mo, DB)

    # wire a real OTel TracerProvider to the MatrixOne exporter
    exporter = MatrixOneSpanExporter(mo, DB, run_tag=1)
    provider = TracerProvider(resource=Resource.create({"service.name": "research-assistant"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("agent_otel")

    llm = make_llm()
    agent = Agent(llm, tracer)

    hr(f"Running a REAL agent (LLM backend: {llm.system} / {llm.model})")
    for task in TASKS:
        answer = agent.run(task)
        print(f"  Q: {task}\n     A: {answer}")

    # ---- query the traces back from MatrixOne (SQL observability) ----
    hr("Traces landed in MatrixOne — query them with SQL")
    n_tr = mo.scalar(f"SELECT COUNT(DISTINCT trace_id) FROM {DB}.{SPANS_TABLE}")
    n_sp = mo.scalar(f"SELECT COUNT(*) FROM {DB}.{SPANS_TABLE}")
    print(f"  {n_tr} agent traces, {n_sp} spans ingested via the OTel SpanExporter\n")

    # reconstruct the multi-step trace (the 'population of France, doubled' one)
    tid = mo.scalar(
        f"SELECT s.trace_id FROM {DB}.{SPANS_TABLE} s WHERE s.name LIKE 'execute_tool calculator' "
        f"AND EXISTS (SELECT 1 FROM {DB}.{SPANS_TABLE} k WHERE k.trace_id=s.trace_id "
        f"AND k.name='execute_tool kb_lookup') LIMIT 1")
    if tid:
        rows = mo.query(
            f"SELECT span_id, parent_span_id, name, status, input_tokens, output_tokens, "
            f"json_extract(attributes,'$.\"tool.result\"') "
            f"FROM {DB}.{SPANS_TABLE} WHERE trace_id=%s ORDER BY start_ns", (tid,))
        kids = {}
        for sid, pid, name, st, it, ot, res in rows:
            kids.setdefault(pid, []).append((sid, name, st, it, ot, res))

        def walk(pid, d):
            for sid, name, st, it, ot, res in kids.get(pid, []):
                extra = f"  tok={it}+{ot}" if (it or ot) else ""
                if res not in (None, "null"):
                    extra += f"  ->{str(res).strip(chr(34))}"
                err = "  [ERROR]" if st == "ERROR" else ""
                print(f"    {'  ' * d}└─ {name}{extra}{err}")
                walk(sid, d + 1)
        print("  trace tree of a multi-step run:")
        walk("", 0)

    print("\n  per-trace rollup (SQL):")
    for tr, ns, tok, errs in mo.query(
        f"SELECT trace_id, COUNT(*), SUM(input_tokens+output_tokens), "
        f"SUM(CASE WHEN status='ERROR' THEN 1 ELSE 0 END) "
        f"FROM {DB}.{SPANS_TABLE} GROUP BY trace_id ORDER BY trace_id"):
        print(f"    {tr[:14]}…  spans={ns}  tokens={tok}  errors={errs}")

    print("\n  tool-call frequency + error span search (SQL):")
    for name, c in mo.query(
        f"SELECT name, COUNT(*) FROM {DB}.{SPANS_TABLE} WHERE gen_ai_operation='execute_tool' "
        f"GROUP BY name ORDER BY 2 DESC"):
        print(f"    {name}: {c}")
    for tr, name, msg in mo.query(
        f"SELECT trace_id, name, status_msg FROM {DB}.{SPANS_TABLE} WHERE status='ERROR' "
        f"AND gen_ai_operation='execute_tool'"):
        print(f"    ERROR span: {name} in {tr[:14]}…  ({msg})")

    # ---- git4data: the trace store is also versionable ----
    hr("git4data on the trace store")
    mo.execute("DROP SNAPSHOT IF EXISTS agent_otel_run1")
    mo.execute(f"CREATE SNAPSHOT agent_otel_run1 FOR TABLE {DB} {SPANS_TABLE}")
    print("  CREATE SNAPSHOT agent_otel_run1 — pin this batch of agent traces (a run/version).")
    print("  => same store does OTel observability AND versioned agent iteration")
    print("     (cross-version DATA BRANCH DIFF + SQL A/B shown in experiments/exp_otel_agent_trace.py).")

    mo.execute("DROP SNAPSHOT IF EXISTS agent_otel_run1")
    mo.execute(f"DROP DATABASE IF EXISTS {DB}")
    provider.shutdown()
    mo.close()
    hr("Done — a real agent ran, its OTel spans flowed into MatrixOne, queried via SQL.")


if __name__ == "__main__":
    main()
