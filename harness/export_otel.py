"""Exports saved calls to an OpenTelemetry backend (Langfuse), with the judge's verdicts attached as scores.

Usage:
    python -m harness.export_otel runs/<folder>

Each call becomes one trace shaped like Pipecat's built-in tracing: a "conversation" span, one "turn" span per
caller-and-agent exchange, and inside each turn a "llm" span for the agent's model call plus one span per tool
call. Times come from the saved trace, so the timeline matches the call. Trace ids are derived from the run and
call name (and span ids from their place in the call), so they are stable across exports. Langfuse v4 still stores
a second export as extra events, so each run is exported once (see MARKER below).

Needs LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env (written by infra/langfuse/make_env.py).
"""
import argparse
import base64
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from harness import config
from harness.trace import existing_run


class FixedIds(IdGenerator):
    """Trace and span ids derived from the call and the span's place in it, so re-exporting a run updates the
    same traces and spans in Langfuse instead of adding duplicates. Set `trace_id` and `next_span_key` first."""
    trace_id = 0
    next_span_key = ""

    def generate_trace_id(self) -> int:
        return self.trace_id

    def generate_span_id(self) -> int:
        digest = hashlib.sha256(f"{self.trace_id:032x}/{self.next_span_key}".encode()).digest()
        return int.from_bytes(digest[:8], "big") or 1


def call_trace_id(run_name: str, call_name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{run_name}/{call_name}".encode()).digest()[:16], "big")


def auth_header() -> str:
    lf = config.langfuse()
    token = f"{lf['public_key']}:{lf['secret_key']}"
    return "Basic " + base64.b64encode(token.encode()).decode()


def make_tracer(ids: FixedIds):
    exporter = OTLPSpanExporter(endpoint=f"{config.langfuse()['host']}/api/public/otel/v1/traces",
                                headers={"Authorization": auth_header()})
    provider = TracerProvider(resource=Resource.create({"service.name": "voice-eval"}), id_generator=ids)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, provider.get_tracer("voice-eval")


def to_ns(start: datetime, seconds: float) -> int:
    return int((start.timestamp() + seconds) * 1e9)


def export_call(tracer, ids: FixedIds, trace: dict, score: dict | None, run_name: str, call_name: str) -> None:
    start = datetime.fromisoformat(trace["started_at"])
    turns = trace["turns"]
    end_s = turns[-1]["t_end_s"] if turns else trace.get("duration_s", 0)
    persona = trace["persona"]
    failed = [k for k, v in (score or {}).get("verdicts", {}).items() if v["result"] == "fail"]
    # Langfuse v4 reads trace-level fields from each observation, so every span carries them.
    common = {
        "langfuse.session.id": run_name,  # groups all calls of one run
        "langfuse.trace.name": f"{persona['name']} ({call_name})",
        "langfuse.trace.tags": [persona.get("category", ""), f"agent:{trace['models']['agent']}",
                                f"script:{trace.get('agent_script', {}).get('version', '?')}"]
                               + [f"FAIL:{k}" for k in failed],
    }

    def span(key: str, name: str, context=None, start_s: float = 0.0, **attributes):
        ids.next_span_key = key
        return tracer.start_span(name, context=context, start_time=to_ns(start, start_s),
                                 attributes={**common, **attributes})

    root = span("conversation", "conversation", **{
        "conversation.id": f"{run_name}/{call_name}",
        "langfuse.trace.metadata.persona_id": trace["persona_id"],
        "langfuse.trace.metadata.persona_tests": persona.get("tests", ""),
        "langfuse.trace.metadata.caller_model": trace["models"]["caller"],
        "langfuse.trace.metadata.end_reason": str(trace["end_reason"]),
        "langfuse.trace.metadata.checklist_version": (score or {}).get("checklist_version", ""),
        "langfuse.trace.input": turns[0]["text"] if turns else "",
        "langfuse.trace.output": turns[-1]["text"] if turns else "",
    })
    root_ctx = set_span_in_context(root)
    model = trace["models"]["agent"]

    # Pair each caller turn with the agent turn that answers it; the agent's opening line is turn 0 on its own.
    pending_caller = None
    for t in turns:
        if t["speaker"] == "caller":
            pending_caller = t
            continue
        turn_start = pending_caller["t_start_s"] if pending_caller else t["t_start_s"]
        heard = pending_caller["text"] if pending_caller else "[call connected]"
        turn_span = span(f"turn-{t['index']}", "turn", root_ctx, turn_start, **{
            "turn.number": t["index"],
            "langfuse.observation.input": heard,
            "langfuse.observation.output": t["text"],
        })
        turn_ctx = set_span_in_context(turn_span)
        waited = sum(r["waited_s"] for r in t.get("retries", []))
        llm_span = span(f"llm-{t['index']}", "llm", turn_ctx, t["t_start_s"], **{
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": model,
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": t.get("tokens", {}).get("prompt", 0),
            "gen_ai.usage.output_tokens": t.get("tokens", {}).get("completion", 0),
            "langfuse.observation.input": heard,
            "langfuse.observation.output": t["text"],
            "langfuse.observation.metadata.rate_limit_wait_s": waited,
            "langfuse.observation.metadata.retries": len(t.get("retries", [])),
        })
        llm_span.end(end_time=to_ns(start, t["t_end_s"]))
        for n, tc in enumerate(t.get("tool_calls", [])):
            tool_span = span(f"tool-{t['index']}-{n}", f"tool: {tc['name']}", turn_ctx, t["t_end_s"], **{
                "langfuse.observation.type": "tool",
                "langfuse.observation.input": json.dumps(tc["args"], ensure_ascii=False),
                "langfuse.observation.output": tc["result"],
            })
            tool_span.end(end_time=to_ns(start, t["t_end_s"]))
        turn_span.end(end_time=to_ns(start, t["t_end_s"]))
        pending_caller = None

    for err in trace.get("errors", []):
        root.set_status(Status(StatusCode.ERROR, f"{err['type']}: {err['message'][:200]}"))
    root.end(end_time=to_ns(start, end_s))


def post_scores(trace_id_hex: str, score: dict) -> None:
    """One BOOLEAN score per check (pass=1, fail=0; n/a is skipped), with the reason as the comment."""
    items = [{"id": f"{trace_id_hex}-{name}", "traceId": trace_id_hex, "name": name, "dataType": "BOOLEAN",
              "value": 1 if v["result"] == "pass" else 0, "comment": v["why"][:500] or None}
             for name, v in score.get("verdicts", {}).items() if v["result"] in ("pass", "fail")]
    with httpx.Client(base_url=config.langfuse()["host"], headers={"Authorization": auth_header()}) as http:
        for item in items:
            http.post("/api/public/scores", json=item).raise_for_status()


def export_run(run_dir: Path) -> int:
    ids = FixedIds()
    provider, tracer = make_tracer(ids)
    count = 0
    for trace_path in sorted((run_dir / "traces").glob("*.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        score_path = run_dir / "scores" / trace_path.name
        score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else None
        ids.trace_id = call_trace_id(run_dir.name, trace_path.stem)
        export_call(tracer, ids, trace, score, run_dir.name, trace_path.stem)
        if score:
            post_scores(f"{ids.trace_id:032x}", score)
        count += 1
    provider.force_flush()
    provider.shutdown()
    return count


MARKER = "exported_otel.json"


if __name__ == "__main__":
    # Ids are stable, but Langfuse v4 (events-only mode) stores every upload as new events, so a second export
    # of the same run shows each span twice. Export a run once; --force re-sends anyway.
    parser = argparse.ArgumentParser(prog="python -m harness export")
    parser.add_argument("run_dirs", nargs="+", type=existing_run)
    parser.add_argument("--force", action="store_true", help="send again even if already exported")
    args = parser.parse_args()
    if not config.langfuse():
        sys.exit("Langfuse is not configured (LANGFUSE_* in .env); see README.")
    for run_dir in args.run_dirs:
        marker = run_dir / MARKER
        if marker.exists() and not args.force:
            print(f"{run_dir.name}: already exported ({marker.name}); use --force to send again")
            continue
        count = export_run(run_dir)
        marker.write_text(json.dumps({"exported_at": datetime.now().isoformat(timespec="seconds"),
                                      "host": config.langfuse()["host"], "calls": count}), encoding="utf-8")
        print(f"{run_dir.name}: exported {count} calls to {config.langfuse()['host']}")
