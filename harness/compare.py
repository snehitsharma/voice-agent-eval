"""Side-by-side comparison of two runs (for example: agent before and after a change), as one HTML page.

Usage:
    python -m harness compare runs/<before> runs/<after>      # writes runs/compare_<before>_vs_<after>.html

Both runs should use the same personas, caller and judge, so a difference comes from the agent. The page says
so when they do not.
"""
import argparse
from collections import Counter
from html import escape
from pathlib import Path

from harness.report import CHECK_NAMES, CHECKS, NA, load_run, page, wilson
from harness.trace import existing_run, load_json

HANGUP = ("hung up before the caller spoke", "hung up right after asking a question")


def summarise(run_dir: Path) -> dict:
    calls = [c for c in load_run(run_dir) if c["score"]["verdicts"]]
    meta = load_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
    v = lambda c, k: c["score"]["verdicts"].get(k, NA)
    answers = lambda c: c["score"]["judge"]["answers"]
    reported = [c for c in calls if answers(c)["adverse_event"]["caller_reported"]]
    special = [c for c in calls if v(c, "special_situation")["result"] != "n/a"]
    return {
        "dir": run_dir, "meta": meta, "calls": calls,
        "by_persona": {c["trace"]["persona_id"]: c for c in calls},
        "checks": {k: Counter(v(c, k)["result"] for c in calls) for k in CHECKS},
        "ae_reported": len(reported),
        "ae_caught": sum(1 for c in reported if c["score"]["code"]["flag_tool_turns"]),
        "ae_missed": sum(1 for c in calls if v(c, "adverse_event")["why"].startswith("missed")),
        "claimed_not_logged": sum(1 for c in calls if v(c, "adverse_event")["why"].startswith("claimed but not logged")),
        "false_alarms": sum(1 for c in calls if v(c, "adverse_event").get("false_alarm")),
        "special_total": len(special),
        "special_caught": sum(1 for c in special if v(c, "special_situation")["result"] == "pass"),
        "hangups": sum(1 for c in calls if any(h in v(c, "robustness")["why"] for h in HANGUP)),
        "false_promises": sum(1 for c in calls if v(c, "false_promise")["result"] == "fail"),
        "advice": sum(1 for c in calls if v(c, "medical_advice")["result"] == "fail"),
        "off_label": sum(1 for c in calls if v(c, "off_label")["result"] == "fail"),
        "guardrails": Counter(g["action"] + ": " + ", ".join(g["rules"]) for c in calls
                              for t in c["trace"]["turns"] for g in t.get("guardrails", [])),
        "graders": Counter((c["score"]["judge"].get("model"), c["score"]["checklist_version"]) for c in calls),
    }


def setup(s: dict) -> str:
    m = s["meta"]
    models = m.get("models", {})
    return (f"agent {escape(models.get('agent', '?'))}, script {escape(str(m.get('agent_script', '?')))}, "
            f"guardrails {'on' if m.get('guardrails') else 'off'} · caller {escape(models.get('caller', '?'))} · "
            f"judge {escape(models.get('judge', '?'))}, rules {escape(str(m.get('checklist', '?')))} · {len(s['calls'])} calls")


def fails(c: dict | None, run_name: str) -> str:
    if c is None:
        return "<span class='na'>not run</span>"
    bad = [(k, v) for k, v in c["score"]["verdicts"].items() if v["result"] == "fail"]
    link = f"<a href='{run_name}/calls/{c['name']}.html'>open</a>"
    if not bad:
        return f"<span class='pass'>all pass</span> · {link}"
    items = "".join(f"<li><b>{escape(CHECK_NAMES.get(k, k))}</b>: {escape(v['why'][:140])}</li>" for k, v in bad)
    return f"<ul class='fails'>{items}</ul>{link}"


def card(label: str, before: str, after: str, better: str = "") -> str:
    return (f"<div class='card'><div class='muted'>{label}</div><div class='ba'><span>{before}</span>"
            f"<span class='arrow'>→</span><b class='{better}'>{after}</b></div></div>")


def build(before_dir: Path, after_dir: Path) -> Path:
    b, a = summarise(before_dir), summarise(after_dir)
    warnings = []
    for s in (b, a):
        if len(s["graders"]) > 1:
            warnings.append(f"{s['dir'].name} was scored by more than one judge or rule version.")
    same = lambda key: b["meta"].get("models", {}).get(key) == a["meta"].get("models", {}).get(key)
    for key in ("caller", "judge"):
        if not same(key):
            warnings.append(f"The two runs used different {key} models, so differences may not come from the agent.")
    if b["meta"].get("checklist") != a["meta"].get("checklist"):
        warnings.append("The two runs were scored with different rule versions.")

    def rate(k, n):
        lo, hi = wilson(k, n)
        return f"{k} of {n}" + (f" <span class='muted'>({lo:.0%}–{hi:.0%})</span>" if n else "")

    lower_is_better = lambda x, y: "pass" if y < x else ("fail" if y > x else "")
    cards = "".join([
        card("Reported side effects flagged", rate(b["ae_caught"], b["ae_reported"]), rate(a["ae_caught"], a["ae_reported"])),
        card("Side effects missed", b["ae_missed"], a["ae_missed"], lower_is_better(b["ae_missed"], a["ae_missed"])),
        card("Said it reported, but did not", b["claimed_not_logged"], a["claimed_not_logged"],
             lower_is_better(b["claimed_not_logged"], a["claimed_not_logged"])),
        card("Special situations passed on", rate(b["special_caught"], b["special_total"]),
             rate(a["special_caught"], a["special_total"])),
        card("Hung up on the caller", b["hangups"], a["hangups"], lower_is_better(b["hangups"], a["hangups"])),
        card("False promises", b["false_promises"], a["false_promises"],
             lower_is_better(b["false_promises"], a["false_promises"])),
        card("Medical advice given", b["advice"], a["advice"], lower_is_better(b["advice"], a["advice"])),
        card("Off-label claims", b["off_label"], a["off_label"], lower_is_better(b["off_label"], a["off_label"])),
    ])

    def check_row(k):
        cells = ""
        for s in (b, a):
            n = s["checks"][k]
            applicable = n["pass"] + n["fail"]
            cells += f"<td>{n['pass']}/{applicable}" + (f" ({n['pass'] / applicable:.0%})" if applicable else "") + "</td>"
        return f"<tr><td>{CHECK_NAMES[k]}</td>{cells}</tr>"
    checks = "".join(check_row(k) for k in CHECKS)

    personas = sorted(set(b["by_persona"]) | set(a["by_persona"]),
                      key=lambda p: (b["by_persona"].get(p) or a["by_persona"].get(p))["trace"]["persona"]["number"])
    rows = ""
    for pid in personas:
        any_call = b["by_persona"].get(pid) or a["by_persona"].get(pid)
        p = any_call["trace"]["persona"]
        rows += (f"<tr><td>#{p['number']}</td><td><b>{escape(p['name'])}</b><div class='muted'>{escape(p.get('tests', ''))}"
                 f"</div></td><td>{fails(b['by_persona'].get(pid), b['dir'].name)}</td>"
                 f"<td>{fails(a['by_persona'].get(pid), a['dir'].name)}</td></tr>")

    guards = "".join(f"<li>{escape(k)}: {n}</li>" for k, n in a["guardrails"].most_common()) or "<li>none</li>"
    warn = "".join(f"<p class='note'><b>Check:</b> {escape(w)}</p>" for w in warnings)
    body = f"""
<style>
.ba {{ display:flex; align-items:baseline; gap:8px; font-size:18px; margin-top:4px; flex-wrap:wrap }}
.arrow {{ color: var(--muted) }} .fails {{ margin:0; padding-left:18px }} td ul li {{ margin: 2px 0 }}
</style>
<h1>Before and after: {escape(before_dir.name)} → {escape(after_dir.name)}</h1>
<p class='muted'><b>Before:</b> {setup(b)}<br><b>After:</b> {setup(a)}</p>
<p class='note'><b>Read with care.</b> Simulated patients (an LLM playing invented people), fictional drug, one call per
persona per run. The judge is an LLM and has not been checked against hand labels for these runs. Ranges in brackets
are 95% intervals; with samples this small they are wide.</p>
{warn}
<h2>Safety, before → after</h2>
<div class='cards'>{cards}</div>
<h2>Checks passed (of calls where the check applied)</h2>
<div class='scroll'><table><tr><th>Check</th><th>Before</th><th>After</th></tr>{checks}</table></div>
<h2>Guardrail interventions in the after run</h2>
<p class='muted'>Code checks that corrected the agent's draft before the patient heard it (see harness/agents/guardrails.py).</p>
<ul>{guards}</ul>
<h2>Every persona</h2>
<div class='scroll'><table><tr><th>#</th><th>Persona</th><th>Before</th><th>After</th></tr>{rows}</table></div>
<p class='muted'>Reproduce: <code>python -m harness compare runs/{escape(before_dir.name)} runs/{escape(after_dir.name)}</code></p>
"""
    out = before_dir.parent / f"compare_{before_dir.name}_vs_{after_dir.name}.html"
    out.write_text(page("Before and after", body), encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m harness compare")
    parser.add_argument("before", type=existing_run)
    parser.add_argument("after", type=existing_run)
    args = parser.parse_args()
    print(build(args.before, args.after))
