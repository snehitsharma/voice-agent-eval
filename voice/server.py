"""Voice Ava: the same agent as the text harness (script, tools, guardrails) on a Pipecat voice pipeline.

    caller --Twilio Media Streams websocket--> speech-to-text --> Groq LLM + tools --> text-to-speech --> caller

The simulated voice caller (voice/sim_caller.py) connects to /ws the way Twilio does. Each call is saved as a trace
in the harness format, so the same judge scores it, and each agent turn records the voice delay: caller stops
speaking -> Ava's audio starts. Pipecat's own spans (STT, LLM, TTS) go to Langfuse when it is configured.

    .venv-voice\\Scripts\\python -m uvicorn voice.server:app --port 8000
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
from harness.agents.text_agent import SCRIPT_FILES, TOOLS, run_tool  # noqa: E402
from harness.personas import live_persona  # noqa: E402

AGENT_MODEL, SCRIPT_VERSION, GUARDRAILS_ON = config.AGENT_MODEL, config.AGENT_SCRIPT_VERSION, config.GUARDRAILS_ON
SCRIPT = (config.ROOT / "data" / SCRIPT_FILES[SCRIPT_VERSION]).read_text(encoding="utf-8")
SCRIPT_SHA = hashlib.sha256(SCRIPT.encode()).hexdigest()[:10]
VOICE_NOTE = "\n\nThis is a live voice call: your words are spoken aloud. Never use emoji, lists or formatting."
# Pipecat cancels in-flight tool calls when the caller starts talking. A side-effect report or a handoff must not
# be lost that way, for example when the caller says goodbye while Ava pauses to log a report.
SAFETY_TOOLS = {"flag_adverse_event", "transfer_to_human"}

app = FastAPI()


def setup_langfuse_tracing() -> bool:
    lf = config.langfuse()
    if not lf:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from pipecat.utils.tracing.setup import setup_tracing

    token = base64.b64encode(f"{lf['public_key']}:{lf['secret_key']}".encode()).decode()
    exporter = OTLPSpanExporter(
        endpoint=f"{lf['host']}/api/public/otel/v1/traces", headers={"Authorization": f"Basic {token}"}
    )
    return setup_tracing(service_name="voice-ava-pipecat", exporter=exporter)


TRACING = setup_langfuse_tracing()


class CallRecorder(BaseObserver):
    """Builds the transcript and timings from the frames flowing through the pipeline.

    Pipecat reports Ava's speech per sentence and sends each sentence's text slightly before its audio, so
    everything she says between two caller turns is collected into one agent turn."""

    def __init__(self):
        super().__init__()
        self.t0 = time.perf_counter()
        self.turns: list[dict] = []
        self.seen: set[int] = set()
        self.caller_text: list[str] = []
        self.user_started = self.user_stopped = None
        self.agent_turn: dict | None = None
        self.pending_tools: list[dict] = []
        self.ended_by_agent = False

    def now(self) -> float:
        return round(time.perf_counter() - self.t0, 3)

    def caller_has_spoken(self) -> bool:
        return bool(self.caller_text) or any(t["speaker"] == "caller" for t in self.turns)

    def last_caller_words(self) -> str:
        if self.caller_text:
            return " ".join(self.caller_text)
        return next((t["text"] for t in reversed(self.turns) if t["speaker"] == "caller"), "")

    def add_tool(self, name: str, args: dict, result: str) -> None:
        # Tool calls happen before the reply they belong to is spoken, so they wait for the next agent turn.
        call = {"name": name, "args": args, "result": result}
        if self.agent_turn and not self.caller_text:
            self.agent_turn["tool_calls"].append(call)
        else:
            self.pending_tools.append(call)

    def close_caller_turn(self) -> None:
        if not self.caller_text:
            return
        start = self.user_started if self.user_started is not None else self.now()
        end = self.user_stopped if self.user_stopped is not None else self.now()
        self.turns.append(
            {
                "index": len(self.turns),
                "speaker": "caller",
                "text": " ".join(self.caller_text),
                "t_start_s": start,
                "t_end_s": end,
                "response_ms": round((end - start) * 1000),
            }
        )
        self.caller_text = []

    def open_agent_turn(self) -> dict:
        if self.agent_turn is None:
            self.close_caller_turn()
            now = self.now()
            self.agent_turn = {
                "index": len(self.turns),
                "speaker": "agent",
                "text": "",
                "t_start_s": now,
                "t_end_s": now,
                "response_ms": 0,
                "voice_delay_ms": None,
                "spoke": False,
                "tool_calls": self.pending_tools,
                "retries": [],
            }
            self.pending_tools = []
            self.turns.append(self.agent_turn)
        return self.agent_turn

    def finish_agent_turn(self) -> None:
        """Spoken words can't be taken back, so a "passed to the safety team" claim with no flag gets flagged."""
        turn = self.agent_turn
        if (
            turn
            and claims_report(turn["text"])
            and not any(c["name"] == "flag_adverse_event" for c in turn["tool_calls"])
        ):
            args = {
                "description": self.last_caller_words(),
                "seriousness": "unsure",
                "kind": "adverse_event",
                "source": "guardrail",
            }
            turn["tool_calls"].append(
                {"name": "flag_adverse_event", "args": args, "result": run_tool("flag_adverse_event", args)}
            )
            turn.setdefault("guardrails", []).append({"rules": ["unlogged_report_claim"], "action": "auto_flagged"})
        self.agent_turn = None

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if frame.id in self.seen:  # observers see a frame at every hop
            return
        self.seen.add(frame.id)
        if isinstance(frame, UserStartedSpeakingFrame):
            self.user_started = self.now()
            turn = self.agent_turn
            if turn and turn["spoke"] and turn["t_end_s"] >= self.user_started:
                turn["interrupted"] = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_stopped = self.now()
        elif isinstance(frame, TranscriptionFrame):
            self.finish_agent_turn()
            self.caller_text.append(frame.text.strip())
        elif isinstance(frame, TTSTextFrame):
            turn = self.open_agent_turn()
            turn["text"] = (turn["text"] + " " + frame.text).strip()
        elif isinstance(frame, BotStartedSpeakingFrame):
            turn = self.open_agent_turn()
            if not turn["spoke"]:
                turn["spoke"] = True
                turn["t_start_s"] = self.now()
                if turn["index"] > 0 and self.user_stopped is not None:  # the opening line has no delay
                    turn["voice_delay_ms"] = turn["response_ms"] = round((turn["t_start_s"] - self.user_stopped) * 1000)
        elif isinstance(frame, BotStoppedSpeakingFrame) and self.agent_turn is not None:
            self.agent_turn["t_end_s"] = self.now()

    def final_turns(self) -> list[dict]:
        self.finish_agent_turn()
        self.close_caller_turn()
        if self.pending_tools:  # tools called after the last words with nothing spoken, e.g. a silent end_call
            now = self.now()
            self.turns.append(
                {
                    "index": len(self.turns),
                    "speaker": "agent",
                    "text": "",
                    "t_start_s": now,
                    "t_end_s": now,
                    "response_ms": 0,
                    "voice_delay_ms": None,
                    "tool_calls": self.pending_tools,
                    "retries": [],
                    "silent": True,
                }
            )
            self.pending_tools = []
        for t in self.turns:
            t.setdefault("tool_calls", [])
            t.pop("spoke", None)
        return self.turns


def speech_services():
    """Ava's speech-to-text and text-to-speech for the configured VOICE_STACK, and a description for the trace."""
    if config.VOICE_STACK == "elevenlabs":
        stt = GroqSTTService(
            api_key=config.groq_api_key(),
            settings=GroqSTTService.Settings(model=config.WHISPER_MODEL, language=Language.EN),
        )
        tts = ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"].strip(),
            settings=ElevenLabsTTSService.Settings(voice=config.ELEVENLABS_AGENT_VOICE, model=config.ELEVENLABS_MODEL),
        )
        return (
            stt,
            tts,
            {
                "stt": f"groq {config.WHISPER_MODEL}",
                "tts": f"elevenlabs {config.ELEVENLABS_MODEL} {config.ELEVENLABS_AGENT_VOICE}",
            },
        )
    sarvam_key = os.environ["SARVAM_API_KEY"]
    stt = SarvamSTTService(api_key=sarvam_key, settings=SarvamSTTService.Settings(language=config.LANGUAGE))
    tts = SarvamTTSService(
        api_key=sarvam_key, settings=SarvamTTSService.Settings(voice=config.AGENT_VOICE, language=config.LANGUAGE)
    )
    return stt, tts, {"stt": f"sarvam ({config.LANGUAGE})", "tts": f"sarvam {config.AGENT_VOICE}"}


def tools_schema() -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=t["function"]["name"],
                description=t["function"]["description"],
                properties=t["function"]["parameters"].get("properties", {}),
                required=t["function"]["parameters"].get("required", []),
            )
            for t in TOOLS
        ]
    )


def tool_handler(recorder: CallRecorder):
    async def on_tool(call):
        name, args = call.function_name, dict(call.arguments or {})
        if name == "end_call" and GUARDRAILS_ON:
            spoken = recorder.agent_turn["text"] if recorder.agent_turn else ""
            rule = (
                "end_before_caller_spoke"
                if not recorder.caller_has_spoken()
                else "end_after_question"
                if "?" in spoken
                else None
            )
            if rule:  # don't hang up on the caller; keep listening instead
                recorder.add_tool(name, args, "Blocked: wait for the patient to answer before ending the call.")
                turn = recorder.agent_turn or recorder.open_agent_turn()
                turn.setdefault("guardrails", []).append({"rules": [rule], "action": "blocked_end_call"})
                await call.result_callback(
                    "Do not end the call yet. Wait for the patient to answer.",
                    properties=FunctionCallResultProperties(run_llm=False),
                )
                return
        result = run_tool(name, args)
        recorder.add_tool(name, args, result)
        if name == "end_call":
            recorder.ended_by_agent = True
            await call.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
            await call.llm.push_frame(EndWorkerFrame())  # ends after the goodbye that is already queued
        else:
            await call.result_callback(result)

    return on_tool


async def wait_for_start(websocket: WebSocket) -> dict | None:
    """Twilio sends "connected" and then "start" (stream id and our custom parameters) before any audio."""
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if message.get("event") == "start":
                return message["start"]
    except WebSocketDisconnect:
        return None


@app.websocket("/ws")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    start = await wait_for_start(websocket)
    if start is None:
        logger.info("websocket closed before the call started")
        return
    params = start.get("customParameters", {})
    mode = params.get("mode", "phone")  # "sim" for the simulated caller
    first_name = params.get("first_name", "there")
    run_name = params.get("run") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-voice"
    call_name = params.get("call") or "voice__r1"
    persona = json.loads(params["persona"]) if params.get("persona") else live_persona(first_name, "phone")

    phone = mode == "phone"
    serializer = TwilioFrameSerializer(
        stream_sid=start["streamSid"],
        call_sid=start.get("callSid"),
        account_sid=os.getenv("TWILIO_ACCOUNT_SID") if phone else None,
        auth_token=os.getenv("TWILIO_AUTH_TOKEN") if phone else None,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=phone),
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer
        ),
    )

    stt, tts, speech = speech_services()
    llm = GroqLLMService(
        api_key=config.groq_api_key(),
        settings=GroqLLMService.Settings(
            model=AGENT_MODEL, reasoning_effort="none", system_instruction=SCRIPT + VOICE_NOTE
        ),
    )
    context = LLMContext(tools=tools_schema())
    vad = SileroVADAnalyzer(params=VADParams(stop_secs=config.VAD_STOP_SECS))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=vad)
    )

    recorder = CallRecorder()
    on_tool = tool_handler(recorder)
    for tool in TOOLS:
        name = tool["function"]["name"]
        llm.register_function(name, on_tool, cancel_on_interruption=name not in SAFETY_TOOLS)

    pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000, audio_out_sample_rate=8000, enable_metrics=True, enable_usage_metrics=True
        ),
        observers=[recorder],
        enable_tracing=TRACING,
        conversation_id=f"{run_name}/{call_name}",
        additional_span_attributes={"langfuse.session.id": run_name},
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_connected(_transport, _client):
        context.add_message(
            {"role": "user", "content": f"[The call has connected. The patient, {first_name}, has picked up.]"}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client):
        await runner.cancel()

    started_at = datetime.now().isoformat(timespec="seconds")
    errors = []
    try:
        await runner.run()
    except Exception as exc:  # keep the partial trace; a crash is a result too
        logger.exception("voice call failed")
        errors.append({"type": type(exc).__name__, "message": str(exc), "traceback": ""})

    trace = {
        "persona_id": persona["id"],
        "persona": persona,
        "call_name": call_name,
        "models": {"agent": AGENT_MODEL, "caller": "voice-sim" if mode == "sim" else "human-phone"},
        "agent_script": {
            "version": SCRIPT_VERSION,
            "sha": SCRIPT_SHA,
            "voice_suffix": True,
            "guardrails": GUARDRAILS_ON,
        },
        "voice": {
            **speech,
            "stack": config.VOICE_STACK,
            "transport": "twilio-media-streams",
            "vad_stop_secs": config.VAD_STOP_SECS,
            "mode": mode,
        },
        "started_at": started_at,
        "turns": recorder.final_turns(),
        "errors": errors,
        "caller_issues": [],
        "end_reason": "error" if errors else ("agent_ended" if recorder.ended_by_agent else "caller_hung_up"),
        "duration_s": recorder.now(),
    }
    out = config.ROOT / "runs" / run_name / "traces" / f"{call_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"trace saved: {out}")
