"""Runs many simulated calls, judges each one, and builds the report.

Usage:
    python -m harness run                          # every persona once, new run folder
    python -m harness run --repeats 6              # every persona 6 times
    python -m harness run --personas linda_easy raj_dose_advice
    python -m harness run --resume runs/<folder>   # finish an interrupted run

Calls that already have a trace and a score in the run folder are skipped, so an
interrupted run can be resumed without paying for the same calls twice.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from harness import config, llm
from harness.agents.text_agent import AGENT_SCRIPT_VERSION, GUARDRAILS_ON
from harness.index import build_index
from harness.judge import CHECKLIST_VERSION, print_score, score_and_save
from harness.personas import all_persona_ids, load_persona
from harness.report import build_report
from harness.runner import run_call
from harness.trace import ROOT, load_json, save_trace


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m harness run")
    parser.add_argument("--personas", nargs="*", help="persona ids (default: all)")
    parser.add_argument("--repeats", type=int, default=1, help="calls per persona")
    parser.add_argument("--resume", help="existing run folder to continue")
    parser.add_argument("--name", help="run folder name (default: date-time)")
    parser.add_argument("--title", help="one line shown on the study index, e.g. 'Before: Ava v2, no guardrails'")
    parser.add_argument("--agent-model", default=llm.AGENT_MODEL, help="agent under test (a Groq model id)")
    args = parser.parse_args()

    run_dir = (
        Path(args.resume) if args.resume else ROOT / "runs" / (args.name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    # Persona-number order puts the side-effect personas first, in case a quota cuts the run short.
    persona_ids = args.personas or all_persona_ids()

    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "personas": persona_ids,
                    "repeats": args.repeats,
                    "models": {"agent": args.agent_model, "caller": llm.CALLER_MODEL, "judge": llm.JUDGE_MODEL},
                    "agent_script": AGENT_SCRIPT_VERSION,
                    "guardrails": GUARDRAILS_ON,
                    "checklist": CHECKLIST_VERSION,
                    "title": args.title,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    # A resumed run keeps the agent it started with, so one run never mixes two agents.
    agent_model = load_json(meta_path)["models"]["agent"]

    jobs = [(pid, r) for r in range(1, args.repeats + 1) for pid in persona_ids]
    t0 = time.perf_counter()
    try:
        run_jobs(jobs, run_dir, t0, agent_model)
    except llm.QuotaExhausted as exc:
        print(f"\nSTOPPED: provider daily limit reached. Resume later with --resume {run_dir}\n{exc}")
    report_path = build_report(run_dir)
    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Report: {report_path}")
    export_to_langfuse(run_dir)
    print(f"Study index: {build_index()}")


def export_to_langfuse(run_dir: Path) -> None:
    """Send the scored run to Langfuse once, if configured. An unreachable Langfuse doesn't fail the run."""
    if not config.langfuse():
        return
    from harness.export_otel import export_once

    try:
        count = export_once(run_dir)
    except Exception as exc:
        print(f"Langfuse export skipped ({type(exc).__name__}); later: python -m harness export {run_dir}")
        return
    if count is not None:
        print(f"Sent {count} calls to Langfuse: {config.langfuse()['host']}")


def run_jobs(jobs: list, run_dir: Path, t0: float, agent_model: str) -> None:
    for n, (pid, r) in enumerate(jobs, 1):
        name = f"{pid}__r{r}"
        trace_path, score_path = run_dir / "traces" / f"{name}.json", run_dir / "scores" / f"{name}.json"
        if trace_path.exists() and score_path.exists():
            continue
        print(f"[{n}/{len(jobs)}] {name}  ({time.perf_counter() - t0:.0f} s elapsed)", flush=True)
        if trace_path.exists():
            trace = load_json(trace_path)
        else:
            trace = run_call(load_persona(pid), agent_model)
            trace["call_name"] = name
            save_trace(trace, run_dir, name)
        print_score(score_and_save(trace, run_dir, name))


if __name__ == "__main__":
    main()
