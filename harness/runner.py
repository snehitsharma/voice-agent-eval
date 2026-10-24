"""Runs one simulated text call: the persona LLM talks to the agent under test, turn by turn.

Usage:
    python -m harness.runner maria_casual_side_effect     # one call, printed (python -m harness run for batches)
"""
import re
import sys
import time
import traceback
from datetime import datetime

from harness import llm
from harness.agents.text_agent import TextAgent
from harness.caller import Caller
from harness.personas import load_persona
from harness.trace import ROOT, TurnRecorder, new_trace, print_trace, save_trace

MAX_TURNS = 16  # caller turns
GOODBYE = re.compile(r"\b(bye|goodbye|take care|have a (good|great|nice|lovely))\b", re.I)


def run_call(persona: dict, agent_model: str = llm.AGENT_MODEL) -> dict:
    agent = TextAgent(patient_first_name=persona["name"].split()[0], model=agent_model)
    caller = Caller(persona)
    trace = new_trace(persona, {"agent": agent.model, "caller": caller.model},
                      {"version": agent.script_version, "sha": agent.script_sha, "guardrails": agent.guardrails_on})
    rec = TurnRecorder(trace)
    try:
        started = time.perf_counter()
        turn = agent.start()
        rec.record_agent(turn, started)
        for _ in range(MAX_TURNS):
            if turn.ended:
                trace["end_reason"] = "agent_ended"
                break
            started = time.perf_counter()
            caller_text, hung_up, res = caller.reply(turn.text)
            rec.record("caller", caller_text, started, retries=res.retries,
                       tokens={"prompt": res.prompt_tokens, "completion": res.completion_tokens})
            if hung_up:
                if GOODBYE.search(turn.text):
                    trace["end_reason"] = "caller_hung_up"
                    break
                # A hang-up before the agent says goodbye is the simulated caller misbehaving, not the
                # agent failing. Ignore it so the call can finish, and log it as a caller problem.
                trace["caller_issues"].append({"turn": len(trace["turns"]) - 1, "issue": "early_hangup_ignored"})
            started = time.perf_counter()
            turn = agent.reply(caller_text)
            rec.record_agent(turn, started)
        else:
            trace["end_reason"] = "turn_limit"
    except llm.QuotaExhausted:
        raise  # our account ran out, not an agent failure: stop the batch rather than record a fake crash
    except Exception as exc:  # a crash is a result too: keep the partial trace
        trace["errors"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        trace["end_reason"] = "error"
    finally:
        agent.end()
    return rec.finish()


if __name__ == "__main__":
    run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    for persona_id in sys.argv[1:]:
        trace = run_call(load_persona(persona_id))
        path = save_trace(trace, run_dir)
        print_trace(trace)
        print(f"Trace saved: {path.relative_to(ROOT)}")
