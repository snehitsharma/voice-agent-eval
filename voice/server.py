"""Voice Ava: the same agent as the text harness (script, tools, drug lookup), on a Pipecat voice pipeline.

    Twilio (or the simulated caller) --websocket--> Sarvam STT --> Groq LLM + tools --> Sarvam TTS --> back

Run (uses the separate voice environment):
    .venv-voice\\Scripts\\python -m uvicorn voice.server:app --port 8000

Endpoints:
    WS   /ws                Twilio Media Streams protocol; the simulated voice caller connects here

Every call is saved as a trace in the harness format (runs/<run>/traces/<call>.json), so the same judge scores
it. Agent turns record the voice delay: from the moment the caller stopped speaking to the moment Ava's audio
started. Pipecat's own OpenTelemetry spans (STT, LLM, TTS with time to first byte) go to Langfuse when
LANGFUSE_* is set.
"""
import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    FunctionCallResultProperties,
    LLMRunFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the harness package lives next to voice/

from harness import config  # noqa: E402  (also loads .env)
from harness.agents.guardrails import claims_report  # noqa: E402
from harness.agents.text_agent import SCRIPT_FILES, TOOLS, lookup_drug_info  # noqa: E402
from harness.personas import live_persona  # noqa: E402

ROOT = config.ROOT
SAFETY_TOOLS = {"flag_adverse_event", "transfer_to_human"}  # never cancelled by an interruption
AGENT_MODEL, AGENT_SCRIPT_VERSION, GUARDRAILS_ON = config.AGENT_MODEL, config.AGENT_SCRIPT_VERSION, config.GUARDRAILS_ON

SCRIPT_SHA = hashlib.sha256((ROOT / "data" / SCRIPT_FILES[AGENT_SCRIPT_VERSION]).read_bytes()).hexdigest()[:10]

app = FastAPI()


def setup_langfuse_tracing() -> bool:
    lf = config.langfuse()
    if not lf:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from pipecat.utils.tracing.setup import setup_tracing
    token = base64.b64encode(f"{lf['public_key']}:{lf['secret_key']}".encode()).decode()
    exporter = OTLPSpanExporter(endpoint=f"{lf['host']}/api/public/otel/v1/traces",
                                headers={"Authorization": f"Basic {token}"})
    return setup_tracing(service_name="voice-ava-pipecat", exporter=exporter)


TRACING = setup_langfuse_tracing()


class CallRecorder(BaseObserver):
    """Builds the harness-format transcript and timings from the frames flowing through the pipeline."""

    def __init__(self, t0: float):
        super().__init__()
        self.t0 = t0
        self.turns: list[dict] = []
        self.seen: set[int] = set()
        self.caller_text: list[str] = []
        self.user_started = self.user_stopped = None
        self.agent_turn: dict | None = None
        self.pending_tools: list[dict] = []
        self.ended_by_agent = False

    def now(self) -> float:
        return round(time.perf_counter() - self.t0, 3)

    def add_tool(self, name: str, args: dict, result: str) -> None:
        # Tools are called before Ava speaks the reply they belong to, so they go to the open (or next) agent turn.
        call = {"name": name, "args": args, "result": result}
        (self.agent_turn["tool_calls"] if self.agent_turn and not self.caller_text else self.pending_tools).append(call)

    def close_caller_turn(self) -> None:
        if self.caller_text:
            start = self.user_started if self.user_started is not None else self.now()
            end = self.user_stopped if self.user_stopped is not None else self.now()
            self.turns.append({"index": len(self.turns), "speaker": "caller", "text": " ".join(self.caller_text),
                               "t_start_s": start, "t_end_s": end, "response_ms": round((end - start) * 1000)})
            self.caller_text = []

    def caller_has_spoken(self) -> bool:
        return bool(self.caller_text) or any(t["speaker"] == "caller" for t in self.turns)

    def last_caller_words(self) -> str:
        if self.caller_text:
            return " ".join(self.caller_text)
        return next((t["text"] for t in reversed(self.turns) if t["speaker"] == "caller"), "")

    def finish_agent_turn(self) -> None:
        """Guardrail: spoken words cannot be taken back, so a "passed to the safety team" claim without a flag is
        made true by flagging what the caller last said (over-reporting is the safe side)."""
        turn = self.agent_turn
        if turn and claims_report(turn["text"]) and not any(c["name"] == "flag_adverse_event" for c in turn["tool_calls"]):
            turn["tool_calls"].append({"name": "flag_adverse_event", "result": "Adverse event recorded and sent to the safety team.",
                                       "args": {"description": self.last_caller_words(), "seriousness": "unsure",
                                                "kind": "adverse_event", "source": "guardrail"}})
            turn.setdefault("guardrails", []).append({"rules": ["unlogged_report_claim"], "action": "auto_flagged"})
        self.agent_turn = None

    def open_agent_turn(self) -> dict:
        """Everything Ava says between two caller turns is one agent turn (Pipecat reports speech per sentence,
        and sends a sentence's text slightly before its audio starts)."""
        if self.agent_turn is None:
            self.close_caller_turn()
            now = self.now()
            self.agent_turn = {"index": len(self.turns), "speaker": "agent", "text": "", "t_start_s": now,
                               "t_end_s": now, "response_ms": 0, "voice_delay_ms": None, "spoke": False,
                               "tool_calls": self.pending_tools, "retries": []}
            self.pending_tools = []
            self.turns.append(self.agent_turn)
        return self.agent_turn

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if frame.id in self.seen:  # observers see a frame at every hop; count it once
            return
        self.seen.add(frame.id)
        if isinstance(frame, UserStartedSpeakingFrame):
            self.user_started = self.now()
            if self.agent_turn and self.agent_turn["spoke"] and self.agent_turn["t_end_s"] >= self.user_started:
                self.agent_turn["interrupted"] = True  # caller talked over Ava
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_stopped = self.now()
        elif isinstance(frame, TranscriptionFrame):
            self.finish_agent_turn()  # the caller has spoken, so Ava's next words start a new turn
            self.caller_text.append(frame.text.strip())
        elif isinstance(frame, TTSTextFrame):
            turn = self.open_agent_turn()
            turn["text"] = (turn["text"] + " " + frame.text).strip()
        elif isinstance(frame, BotStartedSpeakingFrame):
            turn = self.open_agent_turn()
            if not turn["spoke"]:
                turn["spoke"] = True
                turn["t_start_s"] = self.now()
                # Voice delay: caller stopped speaking -> Ava's first audio. None for the opening line.
                if turn["index"] > 0 and self.user_stopped is not None:
                    turn["voice_delay_ms"] = turn["response_ms"] = round((turn["t_start_s"] - self.user_stopped) * 1000)
        elif isinstance(frame, BotStoppedSpeakingFrame) and self.agent_turn is not None:
            self.agent_turn["t_end_s"] = self.now()


def speech_services():
    """Ava's ears and voice for the configured VOICE_STACK, plus a description for the trace."""
    if config.VOICE_STACK == "elevenlabs":
        stt = GroqSTTService(api_key=config.groq_api_key(),
                             settings=GroqSTTService.Settings(model=config.WHISPER_MODEL, language=Language.EN))
        tts = ElevenLabsTTSService(api_key=os.environ["ELEVENLABS_API_KEY"].strip(), settings=ElevenLabsTTSService.Settings(
            voice=config.ELEVENLABS_AGENT_VOICE, model=config.ELEVENLABS_MODEL))
        return stt, tts, {"stt": f"groq {config.WHISPER_MODEL}",
                          "tts": f"elevenlabs {config.ELEVENLABS_MODEL} {config.ELEVENLABS_AGENT_VOICE}"}
    stt = SarvamSTTService(api_key=os.environ["SARVAM_API_KEY"], settings=SarvamSTTService.Settings(language=config.LANGUAGE))
    tts = SarvamTTSService(api_key=os.environ["SARVAM_API_KEY"],
                           settings=SarvamTTSService.Settings(voice=config.AGENT_VOICE, language=config.LANGUAGE))
    return stt, tts, {"stt": f"sarvam ({config.LANGUAGE})", "tts": f"sarvam {config.AGENT_VOICE}"}


def tools_schema() -> ToolsSchema:
    return ToolsSchema(standard_tools=[
        FunctionSchema(name=t["function"]["name"], description=t["function"]["description"],
                       properties=t["function"]["parameters"].get("properties", {}),
                       required=t["function"]["parameters"].get("required", []))
        for t in TOOLS])


@app.websocket("/ws")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    # Twilio sends "connected" then "start" (with the stream id and our custom parameters) before any audio.
    start = None
    try:
        while start is None:
            message = json.loads(await websocket.receive_text())
            if message.get("event") == "start":
                start = message["start"]
    except WebSocketDisconnect:  # the other side hung up before the call started; nothing to record
        logger.info("websocket closed before the call started")
        return
    params = start.get("customParameters", {})
    mode = params.get("mode", "phone")  # "phone" (real Twilio call) or "sim" (simulated caller)
    first_name = params.get("first_name", "there")
    run_name = params.get("run") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-voice"
    call_name = params.get("call") or "voice__r1"
    persona = json.loads(params["persona"]) if params.get("persona") else live_persona(first_name, "phone")

    serializer = TwilioFrameSerializer(
        stream_sid=start["streamSid"], call_sid=start.get("callSid"),
        account_sid=os.getenv("TWILIO_ACCOUNT_SID") if mode == "phone" else None,
        auth_token=os.getenv("TWILIO_AUTH_TOKEN") if mode == "phone" else None,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=(mode == "phone")))
    transport = FastAPIWebsocketTransport(websocket=websocket, params=FastAPIWebsocketParams(
        audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer))

    script = (ROOT / "data" / SCRIPT_FILES[AGENT_SCRIPT_VERSION]).read_text(encoding="utf-8")
    script += ("\n\nThis is a live voice call: your words are spoken aloud by text-to-speech. Never use emoji, "
               "lists or formatting.")
    stt, tts, speech = speech_services()
    llm = GroqLLMService(api_key=config.groq_api_key(), settings=GroqLLMService.Settings(
        model=AGENT_MODEL, reasoning_effort="none", system_instruction=script))
    context = LLMContext(tools=tools_schema())
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=config.VAD_STOP_SECS))))

    t0 = time.perf_counter()
    recorder = CallRecorder(t0)

    async def on_tool(fn_params):
        name, args = fn_params.function_name, dict(fn_params.arguments or {})
        if name == "lookup_drug_info":
            result = lookup_drug_info(args.get("drug", ""), args.get("question", ""))
        elif name == "flag_adverse_event":
            result = "Adverse event recorded and sent to the safety team."
        elif name == "transfer_to_human":
            result = "Transfer requested. A team member will join the call."  # no real transfer in this test tool
        else:  # end_call
            result = "Call ending."
        if name == "end_call" and GUARDRAILS_ON:
            spoken = recorder.agent_turn["text"] if recorder.agent_turn else ""
            rule = ("end_before_caller_spoke" if not recorder.caller_has_spoken()
                    else "end_after_question" if "?" in spoken else None)
            if rule:  # Guardrail: do not hang up on the caller; keep listening instead
                recorder.add_tool(name, args, "Blocked: wait for the patient to answer before ending the call.")
                turn = recorder.agent_turn or recorder.open_agent_turn()
                turn.setdefault("guardrails", []).append({"rules": [rule], "action": "blocked_end_call"})
                await fn_params.result_callback("Do not end the call yet. Wait for the patient to answer.",
                                                properties=FunctionCallResultProperties(run_llm=False))
                return
        recorder.add_tool(name, args, result)
        if name == "end_call":
            recorder.ended_by_agent = True
            await fn_params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
            await fn_params.llm.push_frame(EndWorkerFrame())  # after the goodbye already queued for speech
        else:
            await fn_params.result_callback(result)

    for tool in TOOLS:
        name = tool["function"]["name"]
        # A safety report or handoff must survive the caller talking over Ava: Pipecat cancels in-flight tool calls
        # on interruption by default, and in testing a patient's "goodbye" cancelled a side-effect report.
        llm.register_function(name, on_tool, cancel_on_interruption=name not in SAFETY_TOOLS)

    pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000,
                              enable_metrics=True, enable_usage_metrics=True),
        observers=[recorder], enable_tracing=TRACING, conversation_id=f"{run_name}/{call_name}",
        additional_span_attributes={"langfuse.session.id": run_name})
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_connected(_transport, _client):
        context.add_message({"role": "user",
                             "content": f"[The call has connected. The patient, {first_name}, has picked up.]"})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client):
        await runner.cancel()

    started_at = datetime.now().isoformat(timespec="seconds")
    errors = []
    try:
        await runner.run()
    except Exception as exc:  # a crash is a result too: keep the partial trace
        logger.exception("voice call failed")
        errors.append({"type": type(exc).__name__, "message": str(exc), "traceback": ""})
    recorder.finish_agent_turn()
    recorder.close_caller_turn()
    if recorder.pending_tools:
        # Tools called after the caller's last words with nothing spoken (e.g. end_call with no goodbye) are a
        # silent agent turn of their own, so the trace shows Ava hanging up without replying.
        now = recorder.now()
        recorder.turns.append({"index": len(recorder.turns), "speaker": "agent", "text": "", "t_start_s": now,
                               "t_end_s": now, "response_ms": 0, "voice_delay_ms": None,
                               "tool_calls": recorder.pending_tools, "retries": [], "silent": True})
    for t in recorder.turns:
        t.setdefault("tool_calls", [])
        t.pop("spoke", None)
    trace = {
        "persona_id": persona["id"], "persona": persona, "call_name": call_name,
        "models": {"agent": AGENT_MODEL, "caller": "voice-sim" if mode == "sim" else "human-phone"},
        "agent_script": {"version": AGENT_SCRIPT_VERSION, "sha": SCRIPT_SHA, "voice_suffix": True, "guardrails": GUARDRAILS_ON},
        "voice": {**speech, "stack": config.VOICE_STACK, "transport": "twilio-media-streams",
                  "vad_stop_secs": config.VAD_STOP_SECS,
                  "mode": mode},
        "started_at": started_at, "turns": recorder.turns, "errors": errors, "caller_issues": [],
        "end_reason": "error" if errors else ("agent_ended" if recorder.ended_by_agent else "caller_hung_up"),
        "duration_s": round(time.perf_counter() - t0, 3),
    }
    out = ROOT / "runs" / run_name / "traces" / f"{call_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"trace saved: {out}")
