"""The study index: one page that lists every result in reading order (runs/index.html).

Usage:
    python -m harness index          # also rebuilt automatically at the end of every run

Order: before/after comparisons, main study runs (runs/study-*), live conversations, voice calls, write-up, then
earlier development runs. Each entry shows its setup (agent, script, guardrails, judge, rules) and call count, so
nothing on the page can be mistaken for a result from a different setup.
"""

import argparse
from html import escape
from pathlib import Path

from harness.report import page
from harness.trace import ROOT, load_json

RUNS = ROOT / "runs"


def describe(run_dir: Path) -> dict:
    traces = sorted((run_dir / "traces").glob("*.json"))
    first = load_json(traces[0]) if traces else {}
    meta = load_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
    models = meta.get("models") or first.get("models", {})
    script = meta.get("agent_script") or first.get("agent_script", {}).get("version", "?")
    guard = meta.get("guardrails", first.get("agent_script", {}).get("guardrails"))
    scores = sorted((run_dir / "scores").glob("*.json"))
    rules = sorted({load_json(s)["checklist_version"] for s in scores}) if scores else []
    voice = (run_dir / "voice_caller").exists() or "voice" in first
    kind = (
        "study"
        if run_dir.name.startswith("study-")
        else "voice"
        if voice
        else "live"
        if run_dir.name.endswith("-live")
        else "dev"
    )
    title = meta.get("title") or {
        "voice": "Simulated voice call (phone audio, Pipecat)",
        "live": "Live chat: a person typing as a made-up patient",
    }.get(kind, "Development run")
    setup = (
        f"agent {models.get('agent', '?')} · script {script} · guardrails "
        f"{'on' if guard else 'off' if guard is not None else '?'} · caller {models.get('caller', '?')} · "
        f"judge rules {', '.join(rules) or 'not scored'} · {len(traces)} call(s)"
    )
    return {
        "dir": run_dir,
        "kind": kind,
        "title": title,
        "setup": setup,
        "report": (run_dir / "report.html").exists(),
        "langfuse": (run_dir / "exported_otel.json").exists(),
    }


def entry(d: dict) -> str:
    name = d["dir"].name
    link = f"<a href='{name}/report.html'>{escape(name)}</a>" if d["report"] else escape(name)
    tag = " · <span class='pass'>in Langfuse</span>" if d["langfuse"] else ""
    return f"<li><b>{escape(d['title'])}</b><br>{link}{tag}<div class='muted'>{escape(d['setup'])}</div></li>"


def build_index() -> Path:
    runs = [
        describe(p) for p in sorted(RUNS.iterdir()) if p.is_dir() and p.name != "archive" and (p / "traces").exists()
    ]
    compares = sorted(RUNS.glob("compare_*.html"))
    section = lambda kind: "".join(entry(d) for d in runs if d["kind"] == kind) or "<li class='muted'>none yet</li>"
    comp = (
        "".join(
            f"<li><a href='{c.name}'><b>{escape(c.stem.removeprefix('compare_').replace('_vs_', '  →  '))}</b></a></li>"
            for c in compares
        )
        or "<li class='muted'>none yet</li>"
    )
    body = f"""
<h1>Voice Agent Eval: study index</h1>
<p class='note'>Simulated patients (an LLM playing invented people), a fictional drug (Zelvora) and a test agent (Ava).
No real people or health data. The judge is an LLM; its verdicts are unverified until checked against hand labels.</p>
<h2>1. Start here: before and after</h2><ul>{comp}</ul>
<h2>2. Study runs</h2><ul>{section("study")}</ul>
<h2>3. Live conversations</h2><ul>{section("live")}</ul>
<h2>4. Voice</h2><ul>{section("voice")}</ul>
<h2>5. Write-up</h2>
<ul><li><a href='../docs/observability.md'>Observability: seeing what a voice agent actually did</a></li>
<li><a href='../evals/README.md'>Grading rules: version history (v1 → v5)</a></li>
<li>Langfuse dashboard (local): <a href='http://localhost:3000'>http://localhost:3000</a></li></ul>
<h2>6. Earlier development runs</h2>
<details><summary class='muted'>Show ({sum(1 for d in runs if d["kind"] == "dev")})</summary><ul>{section("dev")}</ul></details>
"""
    out = RUNS / "index.html"
    out.write_text(page("Study index", body), encoding="utf-8")
    return out


if __name__ == "__main__":
    argparse.ArgumentParser(prog="python -m harness index", description="Rebuild runs/index.html").parse_args()
    print(build_index())
