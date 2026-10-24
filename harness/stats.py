"""Observability numbers for one or more runs, read straight from the saved traces and scores.

Usage:
    python -m harness stats runs/<folder> [runs/<folder> ...]

Prints, per run: calls, turns, tokens, agent model time (median / p95), API retries and time spent waiting on
rate limits, guardrail interventions, caller issues, errors; for voice runs, the voice delay per reply and how
often Ava's speech-to-text heard something different from what the simulated caller said.
"""
import argparse
import difflib
import re
import statistics
from collections import Counter
from pathlib import Path

from harness.report import p95
from harness.trace import existing_run, load_json

CONTRACTIONS = [("n't", " not"), ("'ve", " have"), ("'m", " am"), ("'re", " are"), ("'ll", " will"), ("'d", " would"), ("'s", " is"),
                 ("it's", "it is"), ("that's", "that is"), ("there's", "there is")]


def words(text: str) -> list[str]:
    """Lower-case words with contractions expanded, so "I've" and "I have" count as the same words."""
    text = text.lower().replace("’", "'")
    for short, full in CONTRACTIONS:
        text = text.replace(short, full)
    return [w.strip("'") for w in re.findall(r"[a-z0-9']+", text) if w.strip("'")]  # Whisper adds stray quotes


def word_error_rate(said: str, heard: str) -> float:
    """Share of the caller's words that did not come through as said (substitutions, drops, insertions)."""
    a, b = words(said), words(heard)
    if not a:
        return 0.0
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    errors = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    return errors / len(a)


def align(said: list[str], heard: list[str]) -> list[tuple[str, str]]:
    """Pair each thing the caller said with what Ava heard. Pairing by position breaks when Ava's hearing merges or
    drops a turn, so each utterance is matched to the most similar unused heard turn at or after the last match."""
    pairs, start = [], 0
    for s in said:
        candidates = [(difflib.SequenceMatcher(a=words(s), b=words(h)).ratio(), i) for i, h in enumerate(heard)
                      if i >= start]
        best = max(candidates, default=(0.0, None))
        if best[1] is None or best[0] < 0.3:  # nothing close: Ava never heard this utterance
            pairs.append((s, ""))
            continue
        pairs.append((s, heard[best[1]]))
        start = best[1] + 1
    return pairs


def run_stats(run_dir: Path) -> dict:
    traces = [load_json(p) for p in sorted((run_dir / "traces").glob("*.json"))]
    agent_turns = [t for tr in traces for t in tr["turns"] if t["speaker"] == "agent"]
    retries = [r for tr in traces for t in tr["turns"] for r in t.get("retries", [])]
    model_ms = [max(0, t["response_ms"] - sum(r["waited_s"] for r in t.get("retries", [])) * 1000)
                for t in agent_turns if "voice_delay_ms" not in t]
    voice_ms = [t["voice_delay_ms"] for t in agent_turns if t.get("voice_delay_ms") is not None]
    tokens = Counter()
    for tr in traces:
        for t in tr["turns"]:
            tokens[t["speaker"]] += sum(t.get("tokens", {}).values())
    stats = {
        "run": run_dir.name,
        "calls": len(traces),
        "agents": sorted({tr["models"]["agent"] for tr in traces}),
        "scripts": sorted({str(tr.get("agent_script", {}).get("version")) for tr in traces}),
        "turns": sum(len(tr["turns"]) for tr in traces),
        "agent_replies": len(agent_turns),
        "tokens": dict(tokens),
        "agent_model_ms": (statistics.median(model_ms), p95(model_ms)) if model_ms else None,
        "voice_delay_ms": (statistics.median(voice_ms), p95(voice_ms), voice_ms) if voice_ms else None,
        "api_retries": len(retries),
        "rate_limit_wait_s": round(sum(r["waited_s"] for r in retries), 1),
        "guardrail_events": Counter(g["action"] for t in agent_turns for g in t.get("guardrails", [])),
        "caller_issues": Counter(i["issue"] for tr in traces for i in tr.get("caller_issues", [])),
        "errors": sum(len(tr.get("errors", [])) for tr in traces),
        "end_reasons": Counter(tr["end_reason"] for tr in traces),
    }
    # Voice: compare what the simulated caller said with what Ava's speech-to-text heard, turn by turn.
    pairs = []
    for side_path in sorted((run_dir / "voice_caller").glob("*.json")):
        said = [t["said"] for t in load_json(side_path)["turns"] if t["speaker"] == "caller"]
        trace_path = run_dir / "traces" / side_path.name
        heard = [t["text"] for t in load_json(trace_path)["turns"] if t["speaker"] == "caller"] if trace_path.exists() else []
        pairs += align(said, heard)
    if pairs:
        stats["caller_turns_compared"] = len(pairs)
        stats["caller_turns_misheard"] = [(s, h, round(word_error_rate(s, h), 2)) for s, h in pairs
                                          if word_error_rate(s, h) > 0]
    return stats


def print_stats(s: dict) -> None:
    print(f"\n== {s['run']}: {s['calls']} calls, {s['turns']} turns ({s['agent_replies']} agent replies)")
    print(f"   agent {s['agents']} | script {s['scripts']} | ended: {dict(s['end_reasons'])} | errors: {s['errors']}")
    print(f"   tokens: {s['tokens']}")
    if s["agent_model_ms"]:
        print(f"   agent model time per reply (rate-limit waits removed): median {s['agent_model_ms'][0]:.0f} ms, "
              f"p95 {s['agent_model_ms'][1]:.0f} ms")
    if s["voice_delay_ms"]:
        print(f"   voice delay (caller stops -> Ava's audio starts): median {s['voice_delay_ms'][0]:.0f} ms, "
              f"p95 {s['voice_delay_ms'][1]} ms, all: {s['voice_delay_ms'][2]}")
    print(f"   API retries: {s['api_retries']}, time waiting on rate limits: {s['rate_limit_wait_s']} s")
    print(f"   guardrail interventions: {dict(s['guardrail_events']) or 'none'} | caller issues: {dict(s['caller_issues']) or 'none'}")
    if "caller_turns_compared" in s:
        print(f"   voice: {len(s['caller_turns_misheard'])} of {s['caller_turns_compared']} caller turns heard differently by Ava:")
        for said, heard, wer in s["caller_turns_misheard"]:
            print(f"      said {said!r}\n      Ava heard {heard!r}  (word error rate {wer})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m harness stats")
    parser.add_argument("run_dirs", nargs="+", type=existing_run)
    for run_dir in parser.parse_args().run_dirs:
        print_stats(run_stats(run_dir))
