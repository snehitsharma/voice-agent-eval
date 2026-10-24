"""The voice call recorder, fed with Pipecat frames in the order a real call produces them.

Runs in the voice environment (.venv-voice); skipped where Pipecat is not installed.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

from voice.server import CallRecorder  # noqa: E402


def feed(recorder: CallRecorder, *frames) -> None:
    for frame in frames:
        asyncio.run(recorder.on_push_frame(SimpleNamespace(frame=frame)))


def said(text: str):
    return TranscriptionFrame(text=text, user_id="caller", timestamp="")


def spoken(text: str):
    return TTSTextFrame(text=text, text_type="sentence")


def test_one_agent_turn_per_reply_even_when_speech_comes_sentence_by_sentence():
    rec = CallRecorder()
    feed(
        rec,
        spoken("Hi, I'm Ava."),
        BotStartedSpeakingFrame(),
        spoken("Am I speaking with Tom?"),
        BotStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
    )
    feed(rec, UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(), said("Yes, this is Tom."))
    feed(rec, spoken("Thanks, Tom."), BotStartedSpeakingFrame(), BotStoppedSpeakingFrame())
    turns = rec.final_turns()
    assert [t["speaker"] for t in turns] == ["agent", "caller", "agent"]
    assert turns[0]["text"] == "Hi, I'm Ava. Am I speaking with Tom?"
    assert turns[0]["voice_delay_ms"] is None  # nobody spoke before the opening line
    assert turns[2]["voice_delay_ms"] is not None


def test_tool_call_goes_with_the_reply_it_belongs_to():
    rec = CallRecorder()
    feed(rec, spoken("Hi."), BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), said("I've had headaches."))
    rec.add_tool("flag_adverse_event", {"description": "headaches"}, "recorded")  # before the reply is spoken
    feed(rec, spoken("I've passed that to the safety team."), BotStartedSpeakingFrame())
    turns = rec.final_turns()
    assert [c["name"] for c in turns[-1]["tool_calls"]] == ["flag_adverse_event"]
    assert not turns[-1].get("guardrails")


def test_claimed_report_without_a_flag_is_flagged():
    rec = CallRecorder()
    feed(rec, said("I had a rash."), spoken("I've passed that to the safety team."), BotStartedSpeakingFrame())
    flag = rec.final_turns()[-1]["tool_calls"][0]
    assert flag["name"] == "flag_adverse_event" and flag["args"]["source"] == "guardrail"
    assert flag["args"]["description"] == "I had a rash."


def test_silent_hang_up_becomes_its_own_turn():
    rec = CallRecorder()
    feed(rec, spoken("Hi."), BotStartedSpeakingFrame(), said("Yes."))
    rec.add_tool("end_call", {}, "Call will end after your goodbye.")
    turns = rec.final_turns()
    assert turns[-1]["silent"] and turns[-1]["tool_calls"][0]["name"] == "end_call"
