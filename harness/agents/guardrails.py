"""Checks on each agent reply before the patient hears it.

The script asks the model to behave; these make the most serious failures hard to commit anyway. A reply that breaks
a rule gets one retry with an explanation, and anything still wrong is fixed in code. Interventions are recorded on
the agent turn ("guardrails"), so reports show how often the model needed correcting.
"""

import re
from dataclasses import dataclass

SAFETY_TEAM = re.compile(r"safety team", re.I)
CLAIM_VERB = re.compile(
    r"\b(pass(ed|ing)?|not(e|ed|ing)|shar(e|ed|ing)|record(ed|ing)?|log(ged|ging)?|report(ed|ing)?|"
    r"let (our|the) safety team know|flag(ged|ging)?)\b",
    re.I,
)
# Emoji: pictographs, symbols and dingbats, and the variation selector that turns a symbol into an emoji.
EMOJI_CHARS = f"{chr(0x1F300)}-{chr(0x1FAFF)}{chr(0x2600)}-{chr(0x27BF)}{chr(0xFE0F)}"
EMOJI = re.compile(f"[{EMOJI_CHARS}]")
FORMATTING = re.compile(r"\*\*|__|^\s*#+\s|^\s*[-*•]\s", re.M)


@dataclass
class Problem:
    rule: str
    tell_model: str  # what the model is told when it gets another try


def words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def claims_report(text: str) -> bool:
    return bool(SAFETY_TEAM.search(text) and CLAIM_VERB.search(text))


def check(
    text: str, tools: list[str], *, flagged_this_turn: bool, caller_has_spoken: bool, earlier_replies: list[str]
) -> list[Problem]:
    problems = []
    if claims_report(text) and not (flagged_this_turn or "flag_adverse_event" in tools):
        problems.append(
            Problem(
                "unlogged_report_claim",
                "You told the patient you passed something to the safety team, but you did not call "
                "flag_adverse_event. Call flag_adverse_event now for what the patient reported, then reply.",
            )
        )
    if "end_call" in tools and not caller_has_spoken:
        problems.append(
            Problem(
                "end_before_caller_spoke",
                "Do not end the call: the patient has not said anything yet. Greet them and wait.",
            )
        )
    if "end_call" in tools and "?" in text:
        problems.append(
            Problem(
                "end_after_question",
                "Do not end the call in a reply that asks the patient a question. Ask the question and "
                "wait for the answer; end the call only after the goodbye.",
            )
        )
    if len(words(text)) >= 15 and words(text) in {words(r) for r in earlier_replies}:
        problems.append(
            Problem(
                "repeated_reply", "You already said exactly that. Respond to what the patient just said, in new words."
            )
        )
    return problems


def clean_for_speech(text: str) -> str:
    """Removes emoji and formatting, which text-to-speech would read out or garble."""
    return re.sub(r"[ \t]{2,}", " ", FORMATTING.sub("", EMOJI.sub("", text))).strip()
