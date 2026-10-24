"""The agent under test, text version: an LLM following the support-call script, with tools and guardrails."""

import hashlib
import json
import re

from harness import config, llm
from harness.agent_interface import AgentTurn, ToolCall
from harness.agents import guardrails

DATA = config.ROOT / "data"
SCRIPT_FILES = {"v1": "agent_script_v1.md", "v2": "agent_script_v2.md", "v3": "agent_script_v3.md"}
AGENT_SCRIPT_VERSION, GUARDRAILS_ON = config.AGENT_SCRIPT_VERSION, config.GUARDRAILS_ON

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_drug_info",
            "description": "Look up approved patient information about a drug. The only allowed source of drug facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug": {"type": "string", "description": "Drug name, e.g. Zelvora"},
                    "question": {"type": "string", "description": "What you need to know"},
                },
                "required": ["drug", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_adverse_event",
            "description": "Report to the safety team: a side effect, symptom or health problem the patient mentioned, "
            "or a pregnancy or breastfeeding while taking the drug.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What the patient said, in their words where possible",
                    },
                    "seriousness": {"type": "string", "enum": ["non_serious", "serious", "unsure"]},
                    "kind": {
                        "type": "string",
                        "enum": ["adverse_event", "pregnancy_exposure"],
                        "description": "adverse_event unless this is a pregnancy or breastfeeding report",
                    },
                },
                "required": ["description", "seriousness"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Hand the call to a human team member.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call after saying goodbye.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

STOPWORDS = set(
    "the a an and or of to is it in for on with can i my you your do does what how if be are this that".split()
)


def load_label_sections(drug: str) -> list[tuple[str, str]] | None:
    path = DATA / "drugs" / f"{drug.strip().lower()}.md"
    if not path.exists():
        return None
    sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]
    return [(s.split("\n", 1)[0].strip(), s.split("\n", 1)[1].strip()) for s in sections]


def lookup_drug_info(drug: str, question: str) -> str:
    """Keyword retrieval over the label: the two sections that share the most words with the question.
    The label is one page, so this is enough; a real catalogue of labels would want embeddings."""
    sections = load_label_sections(drug)
    if sections is None:
        return f"No approved information found for '{drug}'."
    words = {w for w in re.findall(r"[a-z]+", question.lower()) if w not in STOPWORDS}
    overlap = lambda section: len(words & set(re.findall(r"[a-z]+", " ".join(section).lower())))
    best = sorted(sections, key=overlap, reverse=True)[:2]
    return "\n\n".join(f"{title}: {body}" for title, body in best)


def run_tool(name: str, args: dict) -> str:
    """What each tool returns to the model. Shared by the text and voice agents."""
    if name == "lookup_drug_info":
        return lookup_drug_info(args.get("drug", ""), args.get("question", ""))
    return {
        "flag_adverse_event": "Adverse event recorded and sent to the safety team.",
        "transfer_to_human": "Transfer requested. A team member will join the call.",
        "end_call": "Call will end after your goodbye.",
    }.get(name, f"Unknown tool '{name}'.")


def strip_reasoning(text: str) -> str:
    """Qwen sometimes emits its reasoning even with reasoning off, and a voice agent would read it out."""
    text = re.sub(r"(?s)^.*</think>", "", text)
    return text.replace("<think>", "").strip()


def parse_args(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_unparsed": raw}


class TextAgent:
    MAX_TOOL_ROUNDS = 4

    def __init__(
        self,
        patient_first_name: str,
        model: str = llm.AGENT_MODEL,
        script_version: str = AGENT_SCRIPT_VERSION,
        guardrails_on: bool = GUARDRAILS_ON,
    ):
        self.model = model
        self.script_version = script_version
        self.guardrails_on = guardrails_on
        script = (DATA / SCRIPT_FILES[script_version]).read_text(encoding="utf-8")
        self.script_sha = hashlib.sha256(script.encode()).hexdigest()[:10]
        self.messages = [{"role": "system", "content": script}]
        self.patient_first_name = patient_first_name
        self.caller_has_spoken = False
        self.last_caller_text = ""
        self.replies: list[str] = []

    def start(self) -> AgentTurn:
        return self._respond(f"[The call has connected. The patient, {self.patient_first_name}, has picked up.]")

    def reply(self, caller_text: str) -> AgentTurn:
        self.caller_has_spoken, self.last_caller_text = True, caller_text
        return self._respond(caller_text)

    def end(self) -> None:
        pass

    def _respond(self, user_text: str) -> AgentTurn:
        self.messages.append({"role": "user", "content": user_text})
        turn = AgentTurn(text="", tokens={"prompt": 0, "completion": 0})
        texts, flagged, retry_left = [], False, self.guardrails_on
        for _ in range(self.MAX_TOOL_ROUNDS + 1):  # one extra round for a guardrail retry
            message = self._ask_model(turn)
            text = strip_reasoning(message.content or "")
            calls = list(message.tool_calls or [])
            problems = []
            if self.guardrails_on:
                text = guardrails.clean_for_speech(text)
                problems = guardrails.check(
                    " ".join(texts + [text]),
                    [c.function.name for c in calls],
                    flagged_this_turn=flagged,
                    caller_has_spoken=self.caller_has_spoken,
                    earlier_replies=self.replies,
                )
            if problems and retry_left:
                retry_left = False
                self._ask_again(turn, problems, text, calls)
                continue
            calls = self._block_bad_hangup(turn, problems, calls)
            if text:
                texts.append(text)
            flagged |= self._run_calls(turn, text, calls)
            if not calls or turn.ended:
                break
        turn.text = " ".join(texts)
        if self.guardrails_on and guardrails.claims_report(turn.text) and not flagged:
            self._flag_what_was_claimed(turn)
        self.replies.append(turn.text)
        return turn

    def _ask_model(self, turn: AgentTurn):
        result = llm.chat(self.model, self.messages, tools=TOOLS, reasoning_effort="none")
        turn.llm_ms += result.latency_ms
        turn.tokens["prompt"] += result.prompt_tokens
        turn.tokens["completion"] += result.completion_tokens
        turn.retries += result.retries
        return result.message

    def _ask_again(self, turn: AgentTurn, problems: list, draft: str, calls: list) -> None:
        turn.guardrails.append(
            {
                "rules": [p.rule for p in problems],
                "action": "asked_model_again",
                "draft": draft,
                "draft_tools": [c.function.name for c in calls],
            }
        )
        self.messages.append(
            {
                "role": "system",
                "content": "Before this reply reaches the patient: " + " ".join(p.tell_model for p in problems),
            }
        )

    @staticmethod
    def _block_bad_hangup(turn: AgentTurn, problems: list, calls: list) -> list:
        """If the retry still hangs up on the caller, drop the end_call."""
        rules = {p.rule for p in problems}
        if not rules & {"end_before_caller_spoke", "end_after_question"}:
            return calls
        turn.guardrails.append({"rules": sorted(rules), "action": "blocked_end_call"})
        return [c for c in calls if c.function.name != "end_call"]

    def _run_calls(self, turn: AgentTurn, text: str, calls: list) -> bool:
        """Record the assistant message, run its tool calls and return whether one of them was a flag."""
        message = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in calls
            ]
        self.messages.append(message)
        flagged = False
        for call in calls:
            args = parse_args(call.function.arguments)
            output = run_tool(call.function.name, args)
            turn.tool_calls.append(ToolCall(call.function.name, args, output))
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
            flagged |= call.function.name == "flag_adverse_event"
            turn.ended |= call.function.name == "end_call"
        return flagged

    def _flag_what_was_claimed(self, turn: AgentTurn) -> None:
        """The patient was told it was reported, so report it. Over-reporting is the safe side."""
        args = {
            "description": self.last_caller_text,
            "seriousness": "unsure",
            "kind": "adverse_event",
            "source": "guardrail",
        }
        turn.tool_calls.append(ToolCall("flag_adverse_event", args, run_tool("flag_adverse_event", args)))
        turn.guardrails.append({"rules": ["unlogged_report_claim"], "action": "auto_flagged"})
