"""Plain-code checks on each agent reply before the patient gets it.

The prompt asks the model to behave; these checks make the worst failures (found by the tester) hard to commit
even when the model slips. Each rule says what it found and what to tell the model, the agent gets one chance to
answer again, and anything still wrong is fixed in code. Every intervention is written to the trace
("guardrails" on the agent turn), so the report and Langfuse show how often the model needed correcting.
"""
import re
from dataclasses import dataclass

SAFETY_TEAM = re.compile(r"safety team", re.I)
CLAIM_VERB = re.compile(r"\b(pass(ed|ing)?|not(e|ed|ing)|shar(e|ed|ing)|record(ed|ing)?|log(ged|ging)?|report(ed|ing)?|"
                        r"let (our|the) safety team know|flag(ged|ging)?)\b", re.I)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿️]")
FORMATTING = re.compile(r"\*\*|__|^\s*#+\s|^\s*[-*•]\s", re.M)


@dataclass
class Problem:
    rule: str
    tell_model: str  # what the model is told when it gets another try


def words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def claims_report(text: str) -> bool:
    return bool(SAFETY_TEAM.search(text) and CLAIM_VERB.search(text))


def check(text: str, tools: list[str], *, flagged_this_turn: bool, caller_has_spoken: bool,
          earlier_replies: list[str]) -> list[Problem]:
    problems = []
    if claims_report(text) and not (flagged_this_turn or "flag_adverse_event" in tools):
        problems.append(Problem("unlogged_report_claim",
                                "You told the patient you passed something to the safety team, but you did not call "
                                "flag_adverse_event. Call flag_adverse_event now for what the patient reported, then reply."))
    if "end_call" in tools and not caller_has_spoken:
        problems.append(Problem("end_before_caller_spoke",
                                "Do not end the call: the patient has not said anything yet. Greet them and wait."))
    if "end_call" in tools and "?" in text:
        problems.append(Problem("end_after_question",
                                "Do not end the call in a reply that asks the patient a question. Ask the question and "
                                "wait for the answer; end the call only after the goodbye."))
    if len(words(text)) >= 15 and words(text) in {words(r) for r in earlier_replies}:
        problems.append(Problem("repeated_reply",
                                "You already said exactly that. Respond to what the patient just said, in new words."))
    return problems


def clean_for_speech(text: str) -> str:
    """Emoji and text formatting would be read out or garbled by text-to-speech, so they are removed."""
    return re.sub(r"[ \t]{2,}", " ", FORMATTING.sub("", EMOJI.sub("", text))).strip()
