"""Builds the HTML report for a run: a summary page plus one readable page per call.

Usage:
    python -m harness.report runs/<folder>
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

from harness.trace import existing_run

CHECKS = ["adverse_event", "special_situation", "medical_advice", "off_label", "false_promise", "handoff",
          "task_completion", "robustness"]
NA = {"result": "n/a", "why": "", "turn": None}  # for checks an older checklist version did not have
CHECK_NAMES = {
    "adverse_event": "Adverse event handling", "special_situation": "Special situations reported (v3+)",
    "false_promise": "No false promises (v5+)",
    "medical_advice": "No medical advice", "off_label": "No off-label claims",
    "handoff": "Right handoff", "task_completion": "Task completion", "robustness": "Robustness",
}

CSS = """
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e3e3e6; --card:#f6f6f8;
        --fail:#b3261e; --fail-bg:#fdecea; --pass:#1e7a3c; --pass-bg:#e8f5ec; --warn:#8a5a00; --warn-bg:#fff4dc; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
        --bg:#151517; --fg:#ececef; --muted:#9a9aa2; --line:#2c2c31; --card:#1e1e22;
        --fail:#ff8a80; --fail-bg:#3a1e1c; --pass:#7ad69a; --pass-bg:#16301f; --warn:#ffcc66; --warn-bg:#33290f; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0; padding:24px 16px; }
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; }
.muted { color: var(--muted); } .note { background:var(--warn-bg); color:var(--fg); padding:10px 14px; border-radius:8px; }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-top:16px; }
.card { background:var(--card); border-radius:10px; padding:14px; } .card b { font-size: 26px; display:block; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align: top; }
.fail { color:var(--fail); font-weight:600; } .pass { color:var(--pass); } .na { color:var(--muted); }
.turn { padding:8px 12px; border-radius:8px; margin:6px 0; background:var(--card); }
.turn.agent { border-left: 3px solid #4a7bd0; } .turn.caller { border-left: 3px solid #999; }
.turn.flagged { background: var(--fail-bg); } .tool { font-family: ui-monospace, monospace; font-size: 13px; color: var(--muted); }
a { color: #3b6fd6; } .scroll { overflow-x: auto; }
"""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes out of n (honest error bars for small samples)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{escape(title)}</title>"
            f"<style>{CSS}</style></head><body><main>{body}</main></body></html>")


def load_run(run_dir: Path) -> list[dict]:
    calls = []
    for score_path in sorted((run_dir / "scores").glob("*.json")):
        trace_path = run_dir / "traces" / score_path.name
        if trace_path.exists():
            calls.append({"name": score_path.stem,
                          "score": json.loads(score_path.read_text(encoding="utf-8")),
                          "trace": json.loads(trace_path.read_text(encoding="utf-8"))})
    return calls


def call_page(call: dict) -> str:
    trace, score = call["trace"], call["score"]
    p = trace["persona"]
    bad_turns = {v.get("turn") for v in score["verdicts"].values() if v["result"] == "fail"}
    rows = "".join(
        f"<tr><td>{CHECK_NAMES[c]}</td><td class='{_cls(v['result'])}'>{v['result'].upper()}</td>"
        f"<td>{escape(v['why'])}{' (turn ' + str(v['turn']) + ')' if v.get('turn') is not None else ''}</td></tr>"
        for c in CHECKS if (v := score["verdicts"].get(c)))
    turns = []
    for t in trace["turns"]:
        tools = "".join(f"<div class='tool'>→ tool {escape(tc['name'])} {escape(json.dumps(tc['args'], ensure_ascii=False))}</div>"
                        for tc in t.get("tool_calls", []))
        retries = "".join(f"<div class='tool'>API retry: {escape(r['error'])}, waited {r['waited_s']} s</div>"
                          for r in t.get("retries", []))
        timing = f" · reply took {t['response_ms']} ms" if t["speaker"] == "agent" else ""
        turns.append(f"<div class='turn {t['speaker']}{' flagged' if t['index'] in bad_turns else ''}'>"
                     f"<div class='muted'>turn {t['index']} · {t['speaker'].upper()} · {t['t_start_s']} s{timing}</div>"
                     f"{escape(t['text'])}{tools}{retries}</div>")
    review = "".join(f"<li>{escape(n)}</li>" for n in score.get("needs_review", []))
    body = (f"<p><a href='../report.html'>← back to report</a></p>"
            f"<h1>{escape(p['name'])}: {escape(call['name'])}</h1>"
            f"<p class='muted'>Persona #{p.get('number', '?')} ({escape(p.get('category', ''))}): {escape(p['summary'])}<br>"
            f"Tests: {escape(p.get('tests', ''))}<br>Call ended: {escape(str(trace['end_reason']))}, "
            f"{len(trace['turns'])} turns, {trace['duration_s']} s</p>"
            f"<p class='note'>Simulated caller, not a real person. Fictional drug.</p>"
            f"<h2>Verdicts</h2><div class='scroll'><table><tr><th>Check</th><th>Result</th><th>Why</th></tr>{rows}</table></div>"
            + (f"<h2>Marked for human review</h2><ul>{review}</ul>" if review else "")
            + f"<h2>Transcript</h2><p class='muted'>Turns behind a failed verdict are highlighted.</p>{''.join(turns)}")
    return page(f"{p['name']} call", body)


def _cls(result: str) -> str:
    return {"pass": "pass", "fail": "fail"}.get(result, "na")


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    calls = load_run(run_dir)
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    (run_dir / "calls").mkdir(exist_ok=True)
    for call in calls:
        (run_dir / "calls" / f"{call['name']}.html").write_text(call_page(call), encoding="utf-8")

    judged = [c for c in calls if c["score"]["verdicts"]]
    judge_failed = len(calls) - len(judged)

    # Adverse events: the headline number.
    reported = [c for c in judged if c["score"]["judge"]["answers"]["adverse_event"]["caller_reported"]]
    caught = [c for c in reported if c["score"]["code"]["flag_tool_turns"]]
    missed = [c for c in reported if not c["score"]["code"]["flag_tool_turns"]]
    claimed = [c for c in judged if c["score"]["verdicts"]["adverse_event"]["why"].startswith("claimed but not logged")]
    false_alarms = [c for c in judged if c["score"]["verdicts"]["adverse_event"].get("false_alarm")]
    lo, hi = wilson(len(caught), len(reported))
    persona_ae_not_said = [c for c in judged if c["trace"]["persona"]["expected"].get("reports_adverse_event")
                           and not c["score"]["judge"]["answers"]["adverse_event"]["caller_reported"]]

    check_rows = ""
    for check in CHECKS:
        counts = Counter(c["score"]["verdicts"].get(check, NA)["result"] for c in judged)
        applicable = counts["pass"] + counts["fail"]
        check_rows += (f"<tr><td>{CHECK_NAMES[check]}</td><td class='pass'>{counts['pass']}</td>"
                       f"<td class='fail'>{counts['fail']}</td><td class='na'>{counts['n/a']}</td>"
                       f"<td>{f'{counts['pass'] / applicable:.0%}' if applicable else '-'}</td></tr>")

    groups = defaultdict(lambda: [0, 0])
    for c in judged:
        g = groups[c["trace"]["persona"].get("category", "?")]
        g[0] += 1
        g[1] += any(v["result"] == "fail" for v in c["score"]["verdicts"].values())
    group_rows = "".join(f"<tr><td>{escape(k)}</td><td>{n}</td><td class='fail'>{f}</td></tr>"
                         for k, (n, f) in sorted(groups.items()))

    delays = [d for c in calls for d in c["score"]["code"]["agent_delays_ms"]]
    tokens = Counter()
    for c in calls:
        for t in c["trace"]["turns"]:
            tokens[t["speaker"]] += sum(t.get("tokens", {}).values())
        tokens["judge"] += sum(c["score"]["judge"].get("tokens", {}).values())

    failures = ""
    for c in judged:
        for check in CHECKS:
            v = c["score"]["verdicts"].get(check, NA)
            if v["result"] == "fail":
                failures += (f"<tr><td><a href='calls/{c['name']}.html'>{escape(c['name'])}</a></td>"
                             f"<td>{CHECK_NAMES[check]}</td><td>{escape(v['why'][:200])}</td>"
                             f"<td>{'' if v.get('turn') is None else v['turn']}</td></tr>")
    reviews = "".join(f"<li><a href='calls/{c['name']}.html'>{escape(c['name'])}</a>: {escape(n)}</li>"
                      for c in calls for n in c["score"].get("needs_review", []))

    models = meta.get("models", {})
    graders = Counter((c["score"]["judge"].get("model", "?"), c["score"]["checklist_version"]) for c in judged)
    mixed = "" if len(graders) <= 1 else (
        "<p class='note'><b>Mixed grading.</b> Calls in this run were scored by different judges or rule versions, so "
        "totals are not comparable until it is rescored: "
        + ", ".join(f"{n} by {escape(m)} / {escape(v)}" for (m, v), n in graders.most_common()) + ".</p>")
    body = f"""
<h1>Voice agent eval: run {escape(run_dir.name)}</h1>
<p class='muted'>{len(calls)} calls from {len({c['trace']['persona_id'] for c in calls})} personas ·
agent {escape(models.get('agent', '?'))} · caller {escape(models.get('caller', '?'))} · judge {escape(models.get('judge', '?'))} ·
checklist {escape(judged[0]['score']['checklist_version'] if judged else '?')}</p>
<p class='note'><b>Read with care.</b> Simulated callers (an LLM playing invented patients), not real people. Fictional drug.
The judge is an LLM and has not yet been checked against hand labels for this run, so its verdicts may be wrong.
Small samples: the ranges shown are 95% intervals.</p>
{mixed}

<div class='cards'>
  <div class='card'><b>{len(caught)} of {len(reported)}</b>reported side effects were flagged
     <div class='muted'>95% range {lo:.0%}–{hi:.0%}</div></div>
  <div class='card'><b class='fail'>{len(missed)}</b>side effects missed</div>
  <div class='card'><b class='fail'>{len(claimed)}</b>told the caller it reported, but did not</div>
  <div class='card'><b>{len(false_alarms)}</b>false alarms (flagged with nothing reported)</div>
</div>

<h2>Checks</h2>
<div class='scroll'><table><tr><th>Check</th><th>Pass</th><th>Fail</th><th>n/a</th><th>Pass rate</th></tr>{check_rows}</table></div>
<p class='muted'>Judge could not score {judge_failed} call(s). {len(persona_ae_not_said)} call(s) where the persona was meant to
report a side effect but the judge says the caller never did (a caller or judge problem, not counted as an agent miss).</p>

<h2>By persona group</h2>
<div class='scroll'><table><tr><th>Group</th><th>Calls</th><th>Calls with any failure</th></tr>{group_rows}</table></div>

<h2>Delay</h2>
<p>{f"Agent reply time over {len(delays)} replies: median <b>{statistics.median(delays):.0f} ms</b>, p95 <b>{p95(delays)} ms</b>." if delays else "No replies."}
<span class='muted'>Text only: model time with rate-limit waits removed. Real voice delay (speech in and out) is not measured yet.</span></p>

<h2>Tokens used</h2>
<p>Agent {tokens['agent']:,} · caller {tokens['caller']:,} · judge {tokens['judge']:,}</p>

<h2>Failures</h2>
<div class='scroll'><table><tr><th>Call</th><th>Check</th><th>Why</th><th>Turn</th></tr>{failures or "<tr><td colspan=4>None</td></tr>"}</table></div>

<h2>Marked for human review</h2>
<ul>{reviews or "<li>None</li>"}</ul>
"""
    report_path = run_dir / "report.html"
    report_path.write_text(page(f"Eval run {run_dir.name}", body), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m harness report")
    parser.add_argument("run_dir", type=existing_run)
    print(build_report(parser.parse_args().run_dir))
