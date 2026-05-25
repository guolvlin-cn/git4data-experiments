"""A real OpenTelemetry SpanExporter that writes spans into MatrixOne.

This is the integration point: in production the same mapping lives inside an
OpenTelemetry Collector exporter (agent SDK -> OTLP -> Collector -> MatrixOne).
Here we plug it straight into the agent's TracerProvider so a real agent run
lands its `gen_ai.*` spans in a MatrixOne table.
"""
import json

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

SPANS_TABLE = "spans"


def create_spans_table(mo, db):
    mo.execute(f"DROP DATABASE IF EXISTS {db}")
    mo.execute(f"CREATE DATABASE {db}")
    mo.execute(
        f"CREATE TABLE {db}.{SPANS_TABLE} (trace_id VARCHAR(32), span_id VARCHAR(16), "
        f"parent_span_id VARCHAR(16), name VARCHAR(160), kind VARCHAR(16), "
        f"start_ns BIGINT, end_ns BIGINT, duration_ms DOUBLE, status VARCHAR(16), "
        f"status_msg VARCHAR(256), service_name VARCHAR(64), gen_ai_operation VARCHAR(32), "
        f"gen_ai_system VARCHAR(32), gen_ai_request_model VARCHAR(64), input_tokens INT, "
        f"output_tokens INT, attributes JSON, ingest_run INT, PRIMARY KEY (trace_id, span_id))")


class MatrixOneSpanExporter(SpanExporter):
    def __init__(self, mo, db, run_tag=1):
        self.mo = mo
        self.db = db
        self.run_tag = run_tag

    def export(self, spans):
        rows = []
        for s in spans:
            a = dict(s.attributes or {})
            rows.append((
                format(s.context.trace_id, "032x"),
                format(s.context.span_id, "016x"),
                format(s.parent.span_id, "016x") if s.parent else "",
                s.name[:160],
                s.kind.name,
                int(s.start_time), int(s.end_time),
                round((s.end_time - s.start_time) / 1e6, 3),
                s.status.status_code.name,
                (s.status.description or "")[:256],
                s.resource.attributes.get("service.name") if s.resource else None,
                a.get("gen_ai.operation.name"),
                a.get("gen_ai.system"),
                a.get("gen_ai.request.model"),
                int(a.get("gen_ai.usage.input_tokens", 0)),
                int(a.get("gen_ai.usage.output_tokens", 0)),
                json.dumps(a, ensure_ascii=False, default=str),
                self.run_tag,
            ))
        with self.mo.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.db}.{SPANS_TABLE} (trace_id,span_id,parent_span_id,"
                f"name,kind,start_ns,end_ns,duration_ms,status,status_msg,service_name,"
                f"gen_ai_operation,gen_ai_system,gen_ai_request_model,input_tokens,"
                f"output_tokens,attributes,ingest_run) VALUES "
                f"({','.join(['%s']*18)})", rows)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass
