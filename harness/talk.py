"""Talk to the agent yourself: you type as the patient, and the call is saved and judged like any other.

Usage:
    python -m harness talk --name Sean

Play a made-up patient. Do not type real health information. Type /bye to end the call.
"""

import argparse
import time
from datetime import datetime

from harness import llm
from harness.agents.text_agent import TextAgent
from harness.judge import print_score, score_and_save
from harness.personas import live_persona
from harness.report import build_report
from harness.trace import ROOT, TurnRecorder, new_trace, print_trace, save_trace


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m harness talk")
    parser.add_argument("--agent-model", default=llm.AGENT_MODEL)
    parser.add_argument("--name", default="Alex", help="first name the agent will greet (use a made-up one)")
    args = parser.parse_args()

    agent = TextAgent(patient_first_name=args.name, model=args.agent_model)
    persona = live_persona(args.name, "typed")
    trace = new_trace(
        persona,
        {"agent": agent.model, "caller": "human"},
        {"version": agent.script_version, "sha": agent.script_sha, "guardrails": agent.guardrails_on},
    )
    rec = TurnRecorder(trace)

    def agent_turn(fn, *a):
        started = time.perf_counter()
        turn = fn(*a)
        rec.record_agent(turn, started)
        print(f"\nAVA: {turn.text}")
        for tc in turn.tool_calls:
            if tc.name != "lookup_drug_info":
                print(f"     (tool: {tc.name} {tc.args})")
        return turn

    print("Type as the patient (made-up details only). /bye to hang up.\n")
    turn = agent_turn(agent.start)
    end_reason = "agent_ended"
    while not turn.ended:
        started = time.perf_counter()
        text = input("\nYOU: ").strip()
        if text == "/bye":
            end_reason = "caller_hung_up"
            break
        rec.record("caller", text, started)
        turn = agent_turn(agent.reply, text)
    rec.finish(end_reason)

    run_dir = ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-live")
    name = "live__r1"
    trace["call_name"] = name
    save_trace(trace, run_dir, name)
    print("\nScoring the call...")
    score = score_and_save(trace, run_dir, name)
    print_trace(trace)
    print_score(score)
    print(f"\nReport: {build_report(run_dir)}")


if __name__ == "__main__":
    main()
