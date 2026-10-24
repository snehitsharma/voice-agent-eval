"""Compares hand labels with the judge: how often they agree, and every disagreement.

Usage:
    python -m harness.agreement runs/<folder>

Reads labels/<folder>/*.json (from label_server) and runs/<folder>/scores/*.json.
"""
import argparse
import json
from pathlib import Path

from harness.report import wilson
from harness.trace import ROOT, existing_run

# (label key, how to read the judge's answer, readable name)
QUESTIONS = [
    ("ae_reported", lambda a: a["adverse_event"]["caller_reported"], "caller reported a side effect"),
    ("agent_said_reported", lambda a: a["agent_said_it_reported"]["value"] if "agent_said_it_reported" in a
     else bool(a.get("reporting_claims")), "agent said it reported"),
    ("unlogged_claim", lambda a: any(not c.get("logged") for c in a["reporting_claims"])
     if "reporting_claims" in a else None, "claimed a report that was not logged (judge part)"),
    ("special_situation_reported", lambda a: a["special_situations"]["reported"] if "special_situations" in a
     else (a["pregnancy"]["reported"] if "pregnancy" in a else None), "special situation reported"),
    ("false_promise", lambda a: a["false_promise"]["made"] if "false_promise" in a else None, "false promise"),
    ("medical_advice", lambda a: a["medical_advice"]["given"], "agent gave medical advice"),
    ("off_label", lambda a: a["off_label"]["claim_made"], "agent made an off-label claim"),
    ("handoff_needed", lambda a: a["handoff"]["needed"], "handoff needed"),
    ("broke", lambda a: a["robustness"]["broke"], "agent broke down (judge part only)"),
]


def compare(run_dir: Path) -> dict:
    label_dir = ROOT / "labels" / run_dir.name
    rows, per_question = [], {q[0]: {"agree": 0, "n": 0, "judge_missed": [], "judge_extra": []} for q in QUESTIONS}
    versions = set()
    for label_path in sorted(label_dir.glob("*.json")):
        score_path = run_dir / "scores" / label_path.name
        if not score_path.exists():
            continue
        label = json.loads(label_path.read_text(encoding="utf-8"))
        score = json.loads(score_path.read_text(encoding="utf-8"))
        answers = score["judge"].get("answers")
        versions.add(score["checklist_version"])
        if answers is None:
            continue
        for key, read, _ in QUESTIONS:
            judge_says = read(answers)
            if judge_says is None:  # older checklist did not ask this
                continue
            human_says = label["answers"][key]
            q = per_question[key]
            q["n"] += 1
            if judge_says == human_says:
                q["agree"] += 1
            else:
                (q["judge_missed"] if human_says else q["judge_extra"]).append(label_path.stem)
        rows.append(label_path.stem)
    return {"calls": rows, "questions": per_question, "checklist_versions": sorted(versions)}


def print_agreement(result: dict) -> None:
    print(f"{len(result['calls'])} hand-labelled calls, judge checklist {result['checklist_versions']}\n")
    for key, _, name in QUESTIONS:
        q = result["questions"][key]
        if not q["n"]:
            continue
        lo, hi = wilson(q["agree"], q["n"])
        print(f"{name:40} agree {q['agree']:>3}/{q['n']:<3} ({q['agree'] / q['n']:.0%}, 95% range {lo:.0%}-{hi:.0%})")
        if q["judge_missed"]:
            print(f"    judge said NO, human said YES: {q['judge_missed']}")
        if q["judge_extra"]:
            print(f"    judge said YES, human said NO: {q['judge_extra']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m harness agreement")
    parser.add_argument("run_dir", type=existing_run)
    print_agreement(compare(parser.parse_args().run_dir))
