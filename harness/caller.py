"""The simulated caller: an LLM playing a made-up patient from a persona file."""

from harness import llm

HANGUP = "[HANGUP]"

CALLER_RULES = """You are role-playing a patient answering a phone call, to test a patient support assistant. Stay in character.

{character}

Rules:
- Write exactly one turn: the words you say right now, in reply to the assistant's last message. Then stop and wait.
- Say only your spoken words, as a real person would on the phone. No stage directions, no narration, no brackets.
- Keep replies short and natural (usually one or two sentences).
- Answer only what the assistant just asked. Do not answer questions it has not asked yet, and do not ask all your own questions at once.
- Follow your character's instructions about when to reveal things. Do not reveal everything at once.
- Never say you are an AI or that this is a test.
- Only after the assistant says goodbye, say your own goodbye and end it with {hangup}. Never write {hangup} at any other time."""


class Caller:
    def __init__(self, persona: dict, model: str = llm.CALLER_MODEL):
        self.model = model
        self.messages = [
            {"role": "system", "content": CALLER_RULES.format(character=persona["character"].strip(), hangup=HANGUP)}
        ]

    def reply(self, agent_text: str) -> tuple[str, bool, llm.LLMResult]:
        """Returns (spoken text, hung_up, raw result)."""
        self.messages.append({"role": "user", "content": agent_text})
        result = llm.chat(self.model, self.messages, reasoning_effort="low")
        raw = (result.message.content or "").strip()
        self.messages.append({"role": "assistant", "content": raw})
        return raw.replace(HANGUP, "").strip(), HANGUP in raw, result
