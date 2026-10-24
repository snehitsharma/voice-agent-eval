"""Loads the simulated callers in data/personas/ (made-up people; see data/personas/*.yaml)."""
import yaml

from harness.trace import ROOT

PERSONA_DIR = ROOT / "data" / "personas"


def load_persona(persona_id: str) -> dict:
    return yaml.safe_load((PERSONA_DIR / f"{persona_id}.yaml").read_text(encoding="utf-8"))


def live_persona(first_name: str, how: str) -> dict:
    """Stand-in persona for a real person playing a made-up patient (typed chat or phone call). It has no answer
    key, so nothing is marked for review against expectations."""
    return {"id": "live" if how == "typed" else f"live_{how}", "number": 0, "category": "live",
            "name": f"{first_name} (live, {how})", "summary": f"A real person ({how}) playing a made-up patient.",
            "tests": f"live conversation ({how})", "character": "", "expected": {}}


def all_persona_ids() -> list[str]:
    """Ordered by persona number, so side-effect personas (#1-11) come first."""
    personas = [load_persona(p.stem) for p in PERSONA_DIR.glob("*.yaml")]
    return [p["id"] for p in sorted(personas, key=lambda p: p["number"])]
