"""The trace format: one JSON file per call, with every turn, its timing, tools used and errors.

Shared by the text runner, the talk mode and (in its own way) the voice server, so the judge, report, labelling
page and Langfuse export can read any of them.
"""

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from harness.config import (
    ROOT as ROOT,  # re-exported: other modules import ROOT from here
)


def new_trace(persona: dict, models: dict, agent_script: dict, **extra) -> dict:
    return {
        "persona_id": persona["id"],
        "persona": persona,
        "models": models,
        "agent_script": agent_script,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "turns": [],
        "errors": [],
        "caller_issues": [],  # problems caused by the simulated caller, kept apart from agent failures
        "end_reason": None,
        **extra,
    }


class TurnRecorder:
    """Appends turns to a trace with times relative to the start of the call."""

    def __init__(self, trace: dict):
        self.trace = trace
        self.t0 = time.perf_counter()

    def record(self, speaker: str, text: str, started: float, **fields) -> None:
        now = time.perf_counter()
        self.trace["turns"].append(
            {
                "index": len(self.trace["turns"]),
                "speaker": speaker,
                "text": text,
                "t_start_s": round(started - self.t0, 3),
                "t_end_s": round(now - self.t0, 3),
                "response_ms": round((now - started) * 1000),
                **fields,
            }
        )

    def record_agent(self, turn, started: float) -> None:
        """`turn` is an AgentTurn (see agent_interface)."""
        self.record(
            "agent",
            turn.text,
            started,
            tool_calls=[asdict(t) for t in turn.tool_calls],
            tokens=turn.tokens,
            llm_ms=turn.llm_ms,
            retries=turn.retries,
            guardrails=getattr(turn, "guardrails", []),
        )

    def finish(self, end_reason: str | None = None) -> dict:
        if end_reason:
            self.trace["end_reason"] = end_reason
        self.trace["duration_s"] = round(time.perf_counter() - self.t0, 3)
        return self.trace


def existing_run(value: str) -> Path:
    path = Path(value)
    if not (path / "traces").is_dir():
        raise argparse.ArgumentTypeError(f"{value} is not a run folder (no traces/ inside)")
    return path


def save_trace(trace: dict, run_dir: Path, name: str | None = None) -> Path:
    path = Path(run_dir) / "traces" / f"{name or trace['persona_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_trace(trace: dict) -> None:
    p = trace["persona"]
    print(f"\nPersona: {p['name']} ({p['id']}): {p['summary']}")
    print(f"Agent model: {trace['models']['agent']} | Caller model: {trace['models']['caller']}\n")
    for t in trace["turns"]:
        who = "AGENT " if t["speaker"] == "agent" else "CALLER"
        print(f"[{t['t_start_s']:6.1f}s] {who}: {t['text']}")
        for tc in t.get("tool_calls", []):
            if tc["name"] != "lookup_drug_info":
                print(f"           -> TOOL {tc['name']} {json.dumps(tc['args'], ensure_ascii=False)}")
            else:
                print(f"           -> TOOL lookup_drug_info({tc['args'].get('question', '')!r})")
        if t["speaker"] == "agent":
            print(f"           (agent took {t['response_ms']} ms)")
        for r in t.get("retries", []):
            print(f"           !! API retry: {r['error']}, waited {r['waited_s']} s")
    for issue in trace.get("caller_issues", []):
        print(f"CALLER ISSUE at turn {issue['turn']}: {issue['issue']}")
    for e in trace.get("errors", []):
        print(f"ERROR: {e['type']}: {e['message']}")
    print(f"\nCall ended: {trace['end_reason']} after {trace['duration_s']} s, {len(trace['turns'])} turns")
