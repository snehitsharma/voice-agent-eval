"""Guardrails on the agent: a fake model makes each known mistake, and the agent must catch it."""
import json
from types import SimpleNamespace as NS

import pytest

from harness import llm
from harness.agents import guardrails, text_agent
from harness.agents.text_agent import TextAgent


def tool_call(name, args=None, n=0):
    return NS(id=f"call_{name}_{n}", function=NS(name=name, arguments=json.dumps(args or {})))


def reply(text="", *calls):
    return llm.LLMResult(NS(content=text, tool_calls=list(calls) or None), 10, 100, 10, [])


@pytest.fixture
def scripted(monkeypatch):
    """Make llm.chat return these replies in order, and keep the messages the agent sent each time."""
    sent = []

    def use(*replies):
        queue = list(replies)

        def fake_chat(model, messages, tools=None, **extra):
            sent.append([dict(m) for m in messages])
            return queue.pop(0)
        monkeypatch.setattr(text_agent.llm, "chat", fake_chat)
        return sent
    return use


def test_check_finds_each_known_mistake():
    found = lambda text, tools, **kw: {p.rule for p in guardrails.check(
        text, tools, **{"flagged_this_turn": False, "caller_has_spoken": True, "earlier_replies": [], **kw})}
    assert found("I've passed that to our safety team.", []) == {"unlogged_report_claim"}
    assert found("I've passed that to our safety team.", ["flag_adverse_event"]) == set()
    assert found("Hi Derek, this is Ava.", ["end_call"], caller_has_spoken=False) == {"end_before_caller_spoke"}
    assert found("Do you have any questions?", ["end_call"]) == {"end_after_question"}
    line = "Have a great day! If you need anything else, feel free to contact our support line."
    assert found(line, [], earlier_replies=[line.upper()]) == {"repeated_reply"}
    assert guardrails.clean_for_speech("Have a great day! 😊") == "Have a great day!"


def test_end_call_with_question_gets_a_second_try(scripted):
    sent = scripted(reply("Any questions?", tool_call("end_call")), reply("Do you have any questions about Zelvora?"))
    agent = TextAgent("Helen", model="fake")
    agent.caller_has_spoken = True
    turn = agent.reply("Yes, I take it every day.")
    assert not turn.ended and turn.text == "Do you have any questions about Zelvora?"
    assert turn.guardrails[0]["action"] == "asked_model_again"
    assert "Before this reply reaches the patient" in sent[1][-1]["content"]  # the model was told why


def test_end_call_blocked_if_the_retry_is_still_wrong(scripted):
    scripted(reply("Any questions?", tool_call("end_call")), reply("Any other questions?", tool_call("end_call", n=1)))
    agent = TextAgent("Helen", model="fake")
    agent.caller_has_spoken = True
    turn = agent.reply("No.")
    assert not turn.ended and [g["action"] for g in turn.guardrails] == ["asked_model_again", "blocked_end_call"]


def test_unlogged_claim_is_made_true_by_flagging(scripted):
    scripted(reply("I'll note that for the safety team."), reply("I'll note that too for the safety team."))
    agent = TextAgent("Sean", model="fake")
    turn = agent.reply("I've been a bit itchy, probably my allergies.")
    flags = [t for t in turn.tool_calls if t.name == "flag_adverse_event"]
    assert flags and flags[0].args["source"] == "guardrail"
    assert flags[0].args["description"] == "I've been a bit itchy, probably my allergies."


def test_guardrails_can_be_switched_off(scripted):
    scripted(reply("Hi Derek.", tool_call("end_call")))
    turn = TextAgent("Derek", model="fake", guardrails_on=False).start()
    assert turn.ended and turn.guardrails == []
