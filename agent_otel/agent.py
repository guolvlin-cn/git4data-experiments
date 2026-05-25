"""A real agent loop, instrumented with OpenTelemetry (GenAI semantic conventions).

Each run emits a real span tree through the OTel SDK:
    invoke_agent {name}            (root, gen_ai.operation.name=invoke_agent)
      ├─ chat {model}              (each LLM decision; gen_ai.usage.*_tokens)
      ├─ execute_tool {tool}       (each tool call; ERROR status on failure)
      └─ ...
The spans are exported by whatever processor/exporter the TracerProvider is wired
with — here, MatrixOneSpanExporter.
"""
import json

from opentelemetry.trace import SpanKind, Status, StatusCode

from .tools import TOOLS

MAX_STEPS = 6


class Agent:
    def __init__(self, llm, tracer, name="research-assistant"):
        self.llm = llm
        self.tracer = tracer
        self.name = name

    def run(self, task):
        with self.tracer.start_as_current_span(
            f"invoke_agent {self.name}", kind=SpanKind.INTERNAL,
            attributes={"gen_ai.operation.name": "invoke_agent",
                        "gen_ai.agent.name": self.name, "input.value": task}) as root:
            self.llm.begin(task)
            prev = None
            for _ in range(MAX_STEPS):
                with self.tracer.start_as_current_span(
                    f"chat {self.llm.model}", kind=SpanKind.CLIENT,
                    attributes={"gen_ai.operation.name": "chat",
                                "gen_ai.system": self.llm.system,
                                "gen_ai.request.model": self.llm.model}) as cs:
                    decision, usage = self.llm.step(prev)
                    cs.set_attribute("gen_ai.usage.input_tokens", usage["in"])
                    cs.set_attribute("gen_ai.usage.output_tokens", usage["out"])
                    cs.set_attribute("gen_ai.response.finish_reasons",
                                     "tool_calls" if decision["type"] == "tool_call" else "stop")

                if decision["type"] == "final":
                    root.set_attribute("output.value", str(decision["text"])[:300])
                    return decision["text"]

                tool, args = decision["tool"], decision["args"]
                with self.tracer.start_as_current_span(
                    f"execute_tool {tool}", kind=SpanKind.INTERNAL,
                    attributes={"gen_ai.operation.name": "execute_tool", "tool.name": tool,
                                "tool.arguments": json.dumps(args, default=str)}) as ts:
                    try:
                        result = TOOLS[tool](**args)
                        ts.set_attribute("tool.result", str(result)[:200])
                        ok = True
                    except Exception as e:
                        result = f"{type(e).__name__}: {e}"
                        ts.set_status(Status(StatusCode.ERROR, str(e)))
                        ok = False
                prev = {"tool": tool, "args": args, "result": result, "ok": ok}

            root.set_status(Status(StatusCode.ERROR, "max steps exceeded"))
            return "(no answer: step limit)"
