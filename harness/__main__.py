"""One entry point for the harness.

    python -m harness run [--personas ...] [--repeats N] [--agent-model M] [--resume runs/<folder>]
    python -m harness talk [--agent-model M] [--name Sean]     you type as the patient
    python -m harness judge runs/<folder>                      score (or rescore) every call in a run
    python -m harness report runs/<folder>                     rebuild the HTML report
    python -m harness label runs/<folder> [port]               hand-label calls in the browser
    python -m harness agreement runs/<folder>                  judge vs your labels
    python -m harness export runs/<folder> [--force]           send calls and scores to Langfuse
    python -m harness stats runs/<folder> [...]                observability numbers from the traces
    python -m harness compare runs/<before> runs/<after>       side-by-side HTML comparison of two runs
    python -m harness index                                    rebuild runs/index.html (everything, in reading order)

Voice commands use the separate voice environment; see README.md.
"""

import runpy
import sys

COMMANDS = {
    "run": "harness.batch",
    "talk": "harness.talk",
    "judge": "harness.judge",
    "report": "harness.report",
    "label": "harness.label_server",
    "agreement": "harness.agreement",
    "export": "harness.export_otel",
    "stats": "harness.stats",
    "compare": "harness.compare",
    "index": "harness.index",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 2)
    command = sys.argv[1]
    sys.argv = [f"harness {command}", *sys.argv[2:]]  # each module reads its own arguments
    runpy.run_module(COMMANDS[command], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
