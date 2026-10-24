"""Phone calls to Ava over Twilio webhooks, turn by turn (no streaming).

Twilio's <Gather input="speech"> listens, decides when the caller has stopped, transcribes what they said and posts
the text here. Ava (the same TextAgent as the text harness) replies, and her reply is played in Sarvam's voice from
/gather/audio/. This works on Twilio trial accounts, which cannot stream call audio. Turns are request/response, so
replies are slower than with the streaming server in voice/server.py; the trace records which transport was used.

    .venv-voice\\Scripts\\python -m uvicorn voice.gather:app --port 8000      # then voice/make_call.py
"""

import base64
import hmac
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from fastapi import FastAPI, Request
from fastapi.responses import Response
from loguru import logger
from sarvamai import SarvamAI
from twilio.request_validator import RequestValidator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the harness package lives next to voice/

from harness import config  # noqa: E402  (also loads .env)
from harness.agents.text_agent import TextAgent  # noqa: E402
from harness.batch import export_to_langfuse  # noqa: E402
from harness.index import build_index  # noqa: E402
from harness.judge import score_and_save  # noqa: E402
from harness.personas import live_persona  # noqa: E402
from harness.report import build_report  # noqa: E402
from harness.trace import TurnRecorder, new_trace, save_trace  # noqa: E402
from voice.gather_key import call_key  # noqa: E402

app = FastAPI()
calls: dict[str, dict] = {}  # CallSid -> {"agent", "rec", "trace"}
audio: dict[str, bytes] = {}  # clip id -> WAV bytes, fetched once by Twilio
audio_lock = threading.Lock()


@lru_cache(maxsize=1)
def sarvam() -> SarvamAI:
    return SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])


@lru_cache(maxsize=1)
def validator() -> RequestValidator:
    return RequestValidator(os.environ["TWILIO_AUTH_TOKEN"])


def public_host(request: Request) -> str:
    """The host Twilio used to reach us through the tunnel."""
    return request.headers.get("x-forwarded-host") or request.headers.get("host")


def public_url(request: Request) -> str:
    query = f"?{request.url.query}" if request.url.query else ""
    return f"https://{public_host(request)}{request.url.path}{query}"


async def twilio_form(request: Request) -> dict | None:
    """The posted form, or None if the request doesn't carry our call key.

    Twilio's signature is checked too, but only logged: through the tunnel it matched on status callbacks and not
    on the first webhook of a call, so the call key is what actually authorises a request."""
    form = {k: str(v) for k, v in (await request.form()).items()}
    if not hmac.compare_digest(request.query_params.get("k", ""), call_key()):
        logger.warning("rejected a webhook request without the call key")
        return None
    if not validator().validate(public_url(request), form, request.headers.get("x-twilio-signature", "")):
        logger.info(f"Twilio signature did not match for {request.url.path}")
    return form


def speak(request: Request, text: str) -> str:
    """TwiML that plays `text` in Ava's Sarvam voice."""
    if not text:
        return ""
    result = sarvam().text_to_speech.convert(
        text=text, target_language_code=config.LANGUAGE, speaker=config.AGENT_VOICE, speech_sample_rate=8000
    )
    clip_id = uuid.uuid4().hex
    with audio_lock:
        audio[clip_id] = base64.b64decode("".join(result.audios))
    return f"<Play>{escape(f'https://{public_host(request)}/gather/audio/{clip_id}.wav')}</Play>"


def listen(request: Request, say: str) -> str:
    """Speak, then listen. Twilio decides when the caller has finished; silence also comes back to us."""
    action = escape(f"https://{public_host(request)}/gather/turn?k={call_key()}")  # Twilio needs an absolute URL
    return (
        f'<Gather input="speech" action="{action}" method="POST" speechTimeout="auto" '
        f'language={quoteattr(config.LANGUAGE)} actionOnEmptyResult="true" bargeIn="true">{say}</Gather>'
    )


def twiml(body: str) -> Response:
    return Response(
        content=f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>', media_type="application/xml"
    )


def record_reply(call: dict, reply, started: float) -> None:
    call["rec"].record_agent(reply, started)
    last = call["trace"]["turns"][-1]
    last["server_ms"] = last["response_ms"]  # our side only: model plus speech synthesis


@app.post("/gather/start")
async def start(request: Request):
    form = await twilio_form(request)
    if form is None:
        return Response(status_code=403)
    name = request.query_params.get("name", "there")
    agent = TextAgent(patient_first_name=name)
    trace = new_trace(
        live_persona(name, "phone"),
        {"agent": agent.model, "caller": "human-phone"},
        {"version": agent.script_version, "sha": agent.script_sha, "guardrails": agent.guardrails_on},
        voice={
            "stt": "twilio <Gather> speech",
            "tts": f"sarvam {config.AGENT_VOICE} via <Play>",
            "transport": "twilio-webhooks",
            "mode": "phone",
        },
    )
    call = calls[form["CallSid"]] = {"agent": agent, "rec": TurnRecorder(trace), "trace": trace}
    started = time.perf_counter()
    greeting = agent.start()
    say = speak(request, greeting.text)
    record_reply(call, greeting, started)
    return twiml(listen(request, say))


@app.post("/gather/turn")
async def turn(request: Request):
    form = await twilio_form(request)
    if form is None:
        return Response(status_code=403)
    call = calls.get(form.get("CallSid", ""))
    if call is None:
        return twiml("<Say>Sorry, this call has expired.</Say><Hangup/>")
    heard = form.get("SpeechResult", "").strip()
    started = time.perf_counter()
    call["rec"].record(
        "caller",
        heard or "(silence)",
        started,
        stt_confidence=float(form["Confidence"]) if form.get("Confidence") else None,
    )
    reply = call["agent"].reply(heard or "[The patient said nothing.]")
    say = speak(request, reply.text)
    record_reply(call, reply, started)
    if reply.ended:
        call["trace"]["end_reason"] = "agent_ended"
        return twiml(f"{say}<Hangup/>")
    return twiml(listen(request, say))


@app.get("/gather/audio/{clip_id}.wav")
async def audio_file(clip_id: str):
    with audio_lock:
        data = audio.pop(clip_id, None)
    return Response(content=data, media_type="audio/wav") if data else Response(status_code=404)


@app.post("/gather/status")
async def status(request: Request):
    """Twilio says the call ended: save the trace, then score it, report it and export it in the background."""
    form = await twilio_form(request)
    if form is None:
        return Response(status_code=403)
    call = calls.pop(form.get("CallSid", ""), None)
    if call is None or form.get("CallStatus") != "completed":
        return Response(status_code=204)
    trace = call["trace"]
    call["rec"].finish(trace["end_reason"] or "caller_hung_up")
    run_dir = config.ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-phone")
    trace["call_name"] = "phone__r1"
    save_trace(trace, run_dir, "phone__r1")
    threading.Thread(target=finish_call, args=(trace, run_dir), daemon=True).start()
    return Response(status_code=204)


def finish_call(trace: dict, run_dir: Path) -> None:
    try:
        score_and_save(trace, run_dir, "phone__r1")
        logger.info(f"phone call scored: {build_report(run_dir)}")
        export_to_langfuse(run_dir)
        build_index()
    except Exception:
        logger.exception(f"scoring the phone call failed; later: python -m harness judge {run_dir}")
