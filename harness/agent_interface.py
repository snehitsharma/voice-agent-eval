"""The contract every agent under test follows, so the harness can test any agent."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    name: str
    args: dict
    result: str


@dataclass
class AgentTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    ended: bool = False  # the agent hung up
    llm_ms: int = 0  # time spent waiting on the model (0 if the agent does not report it)
    tokens: dict = field(default_factory=dict)
    retries: list = field(default_factory=list)  # failed API attempts behind this turn, if any
    guardrails: list = field(default_factory=list)  # code checks that corrected this reply (see agents/guardrails.py)


class Agent(Protocol):
    def start(self) -> AgentTurn:
        """The agent's opening line when the call connects."""

    def reply(self, caller_text: str) -> AgentTurn:
        """What the agent says (and does) after the caller speaks."""

    def end(self) -> None:
        """Clean up when the call is over."""
