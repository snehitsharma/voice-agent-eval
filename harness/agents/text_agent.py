"""The agent under test, text version: an LLM following the support-call script, with tools."""
import hashlib
import json
import re
from pathlib import Path

from harness import config, llm
from harness.agent_interface import AgentTurn, ToolCall
from harness.agents import guardrails

DATA = Path(__file__).resolve().parents[2] / "data"
# v2 adds pregnancy reporting; v3 targets every failure the tester found in v2 (see data/agent_script_v3.md)
SCRIPT_FILES = {"v1": "agent_script_v1.md", "v2": "agent_script_v2.md", "v3": "agent_script_v3.md"}
AGENT_SCRIPT_VERSION, GUARDRAILS_ON = config.AGENT_SCRIPT_VERSION, config.GUARDRAILS_ON

TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_drug_info",
        "description": "Look up approved patient information about a drug. The only allowed source of drug facts.",
        "parameters": {"type": "object", "properties": {
            "drug": {"type": "string", "description": "Drug name, e.g. Zelvora"},
            "question": {"type": "string", "description": "What you need to know"},
        }, "required": ["drug", "question"]}}},
    {"type": "function", "function": {
        "name": "flag_adverse_event",
        "description": "Report to the safety team: a side effect, symptom or health problem the patient mentioned, "
                       "or a pregnancy or breastfeeding while taking the drug.",
        "parameters": {"type": "object", "properties": {
            "description": {"type": "string", "description": "What the patient said, in their words where possible"},
            "seriousness": {"type": "string", "enum": ["non_serious", "serious", "unsure"]},
            "kind": {"type": "string", "enum": ["adverse_event", "pregnancy_exposure"],
                     "description": "adverse_event unless this is a pregnancy or breastfeeding report"},
        }, "required": ["description", "seriousness"]}}},
    {"type": "function", "function": {
        "name": "transfer_to_human",
        "description": "Hand the call to a human team member.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "end_call",
        "description": "End the call after saying goodbye.",
        "parameters": {"type": "object", "properties": {}}}},
]

STOPWORDS = set("the a an and or of to is it in for on with can i my you your do does what how if be are this that".split())


def load_label_sections(drug: str) -> list[tuple[str, str]] | None:
    path = DATA / "drugs" / f"{drug.strip().lower()}.md"
    if not path.exists():
        return None
    sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]
    return [(s.split("\n", 1)[0].strip(), s.split("\n", 1)[1].strip()) for s in sections]


def lookup_drug_info(drug: str, question: str) -> str:
    """Simple keyword retrieval: return the two label sections that best match the question."""
    sections = load_label_sections(drug)
    if sections is None:
        return f"No approved information found for '{drug}'."
    words = {w for w in re.findall(r"[a-z]+", question.lower()) if w not in STOPWORDS}
    scored = sorted(sections, key=lambda s: -len(words & set(re.findall(r"[a-z]+", (s[0] + " " + s[1]).lower()))))
    return "\n\n".join(f"{title}: {body}" for title, body in scored[:2])


class TextAgent:
    MAX_TOOL_ROUNDS = 4

    def __init__(self, patient_first_name: str, model: str = llm.AGENT_MODEL, script_version: str = AGENT_SCRIPT_VERSION,
                 guardrails_on: bool = GUARDRAILS_ON):
        self.model = model
        self.script_version = script_version
        self.guardrails_on = guardrails_on
        script = (DATA / SCRIPT_FILES[script_version]).read_text(encoding="utf-8")
        self.script_sha = hashlib.sha256(script.encode()).hexdigest()[:10]  # proves which exact text was used
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
        texts, flagged, second_try_left = [], False, self.guardrails_on
        for _ in range(self.MAX_TOOL_ROUNDS + 1):  # +1 leaves room for one guardrail retry
            result = llm.chat(self.model, self.messages, tools=TOOLS, reasoning_effort="none")
            turn.llm_ms += result.latency_ms
            turn.tokens["prompt"] += result.prompt_tokens
            turn.tokens["completion"] += result.completion_tokens
            turn.retries += result.retries
            msg = result.message
            text = strip_reasoning(msg.content or "")
            calls = list(msg.tool_calls or [])
            problems = []
            if self.guardrails_on:
                text = guardrails.clean_for_speech(text)
                problems = guardrails.check(" ".join(texts + [text]), [tc.function.name for tc in calls],
                                            flagged_this_turn=flagged, caller_has_spoken=self.caller_has_spoken,
                                            earlier_replies=self.replies)
            if problems and second_try_left:  # discard this draft and let the model answer again
                second_try_left = False
                turn.guardrails.append({"rules": [p.rule for p in problems], "action": "asked_model_again",
                                        "draft": text, "draft_tools": [tc.function.name for tc in calls]})
                self.messages.append({"role": "system", "content": "Before this reply reaches the patient: "
                                      + " ".join(p.tell_model for p in problems)})
                continue
            rules = {p.rule for p in problems}
            if rules & {"end_before_caller_spoke", "end_after_question"}:  # still wrong on the retry: block it
                calls = [tc for tc in calls if tc.function.name != "end_call"]
                turn.guardrails.append({"rules": sorted(rules), "action": "blocked_end_call"})
            if text:
                texts.append(text)
            self.messages.append({"role": "assistant", "content": text, **({"tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in calls]} if calls else {})})
            for tc in calls:
                output = self._run_tool(tc.function.name, tc.function.arguments)
                turn.tool_calls.append(ToolCall(tc.function.name, _parse_args(tc.function.arguments), output))
                flagged |= tc.function.name == "flag_adverse_event"
                if tc.function.name == "end_call":
                    turn.ended = True
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            if not calls or turn.ended:  # no tools: the reply is complete; end_call: the goodbye was already said
                break
        turn.text = " ".join(t for t in texts if t)
        if self.guardrails_on and guardrails.claims_report(turn.text) and not flagged:
            # The patient was told it was reported, so make it true: over-reporting is the safe side in pharma.
            args = {"description": self.last_caller_text, "seriousness": "unsure", "kind": "adverse_event",
                    "source": "guardrail"}
            turn.tool_calls.append(ToolCall("flag_adverse_event", args, self._run_tool("flag_adverse_event", "{}")))
            turn.guardrails.append({"rules": ["unlogged_report_claim"], "action": "auto_flagged"})
        self.replies.append(turn.text)
        return turn

    def _run_tool(self, name: str, raw_args: str) -> str:
        args = _parse_args(raw_args)
        if name == "lookup_drug_info":
            return lookup_drug_info(args.get("drug", ""), args.get("question", ""))
        if name == "flag_adverse_event":
            return "Adverse event recorded and sent to the safety team."
        if name == "transfer_to_human":
            return "Transfer requested. A team member will join the call."
        if name == "end_call":
            return "Call will end after your goodbye."
        return f"Unknown tool '{name}'."


def strip_reasoning(text: str) -> str:
    """Qwen sometimes emits its private reasoning even with reasoning off. A voice agent would read it
    aloud, so drop everything up to a closing </think> and any stray tags. The harness still checks
    every agent reply for leaked reasoning, because another agent may not filter it."""
    text = re.sub(r"(?s)^.*</think>", "", text)
    return text.replace("<think>", "").strip()


def _parse_args(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_unparsed": raw}
