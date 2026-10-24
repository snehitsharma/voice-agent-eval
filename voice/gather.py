"""Phone calls to Ava over Twilio webhooks only (no websocket audio stream), turn by turn.

Twilio's <Gather input="speech"> listens, decides when the caller has stopped speaking, transcribes it and POSTs
the text here. Ava (the same TextAgent as the text harness: script, tools, guardrails) replies; her words are
spoken with Sarvam's voice (played by Twilio from /gather/audio/...).
Works on Twilio trial accounts, which cannot stream call audio (Media Streams).

Compared with voice/server.py (Pipecat, streaming, used by the simulated voice calls): Twilio does the speech-to-text instead of Sarvam, and turns
are request/response, so replies are slower. The trace records the transport so results are never mixed up.

Run (voice environment), then start a call with voice/make_call.py:
    .venv-voice\\Scripts\\python -m uvicorn voice.gather:app --port 8000
"""
import base64
import hmac
import os
import sys
import threading
import time
import uuid
from datetime import datetime
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
from harness.personas import live_persona  # noqa: E402
from harness.trace import TurnRecorder, new_trace, save_trace  # noqa: E402
from voice.gather_key import call_key  # noqa: E402

app = FastAPI()
sarvam = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
validator = RequestValidator(os.environ["TWILIO_AUTH_TOKEN"])

calls: dict[str, dict] = {}    # CallSid -> {agent, rec, trace}
audio: dict[str, bytes] = {}   # audio id -> WAV bytes, fetched once by Twilio
audio_lock = threading.Lock()


def public_url(request: Request) -> str:
    """The URL Twilio used (through the tunnel), needed to check its signature and to build absolute links."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    query = f"?{request.url.query}" if request.url.query else ""
    return f"https://{host}{request.url.path}{query}"


async def twilio_form(request: Request) -> dict | None:
    """The POSTed form, or None if the request does not carry our key. Twilio's signature is also checked and
    logged: in testing it matched Twilio's status callbacks but not the first call webhook, so it is not used to
    reject requests until that difference is understood."""
    form = {k: str(v) for k, v in (await request.form()).items()}
    if not hmac.compare_digest(request.query_params.get("k", ""), call_key()):
        logger.warning("rejected a webhook request without the call key")
        return None
    if not validator.validate(public_url(request), form, request.headers.get("x-twilio-signature", "")):
        logger.info(f"Twilio signature did not match for {request.url.path} (request accepted by call key)")
    return form


def speak(request: Request, text: str) -> str:
    """TwiML that plays `text` in Ava's Sarvam voice (Twilio fetches the clip from /gather/audio/...)."""
    if not text:
        return ""
    result = sarvam.text_to_speech.convert(text=text, target_language_code=config.LANGUAGE,
                                           speaker=config.AGENT_VOICE, speech_sample_rate=8000)
    audio_id = uuid.uuid4().hex
    with audio_lock:
        audio[audio_id] = base64.b64decode("".join(result.audios))
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    return f"<Play>{escape(f'https://{host}/gather/audio/{audio_id}.wav')}</Play>"


def listen(request: Request, say: str) -> str:
    """Speak, then listen for the caller. Twilio decides when they stopped speaking (speechTimeout=auto);
    actionOnEmptyResult means silence also comes back to us, so Ava can respond to it."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    action = escape(f"https://{host}/gather/turn?k={call_key()}")  # absolute: Twilio could not resolve a relative one
    return (f'<Gather input="speech" action="{action}" method="POST" speechTimeout="auto" '
            f'language={quoteattr(config.LANGUAGE)} actionOnEmptyResult="true" bargeIn="true">{say}</Gather>')


def twiml(body: str) -> Response:
    return Response(content=f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>',
                    media_type="application/xml")


@app.post("/gather/start")
async def start(request: Request):
    form = await twilio_form(request)
    if form is None:
        return Response(status_code=403)
    name = request.query_params.get("name", "there")
    agent = TextAgent(patient_first_name=name)
    trace = new_trace(live_persona(name, "phone"), {"agent": agent.model, "caller": "human-phone"},
                      {"version": agent.script_version, "sha": agent.script_sha, "guardrails": agent.guardrails_on},
                      voice={"stt": "twilio <Gather> speech", "tts": f"sarvam {config.AGENT_VOICE} via <Play>",
                             "transport": "twilio-webhooks", "mode": "phone"})
    rec = TurnRecorder(trace)
    calls[form["CallSid"]] = {"agent": agent, "rec": rec, "trace": trace}
    started = time.perf_counter()
    turn = agent.start()
    say = speak(request, turn.text)
    rec.record_agent(turn, started)
    trace["turns"][-1]["server_ms"] = trace["turns"][-1]["response_ms"]  # model + voice synthesis on our side
    return twiml(listen(request, say))


@app.post("/gather/turn")
async def turn(request: Request):
    form = await twilio_form(request)
    if form is None:
        return Response(status_code=403)
    call = calls.get(form.get("CallSid", ""))
    if call is None:
        return twiml("<Say>Sorry, this call has expired.</Say><Hangup/>")
    agent, rec, trace = call["agent"], call["rec"], call["trace"]
    heard = form.get("SpeechResult", "").strip()
    started = time.perf_counter()
    rec.record("caller", heard or "(silence)", started,
               stt_confidence=float(form["Confidence"]) if form.get("Confidence") else None)
    reply = agent.reply(heard or "[The patient said nothing.]")
    say = speak(request, reply.text)
    rec.record_agent(reply, started)
    trace["turns"][-1]["server_ms"] = trace["turns"][-1]["response_ms"]
    if reply.ended:
        trace["end_reason"] = "agent_ended"
        return twiml(f"{say}<Hangup/>")
    return twiml(listen(request, say))


@app.get("/gather/audio/{audio_id}.wav")
async def audio_file(audio_id: str):
    with audio_lock:
        data = audio.pop(audio_id, None)  # each clip is played once
    return Response(content=data, media_type="audio/wav") if data else Response(status_code=404)


@app.post("/gather/status")
async def status(request: Request):
    """Twilio reports the call ended: save the trace, then score it, report it and send it to Langfuse."""
    form = await twilio_form(request)
    if form is None:
        return Response(status_code=403)
    call = calls.pop(form.get("CallSid", ""), None)
    if call is None or form.get("CallStatus") != "completed":
        return Response(status_code=204)
    rec, trace = call["rec"], call["trace"]
    rec.finish(trace["end_reason"] or "caller_hung_up")
    run_dir = config.ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-phone")
    trace["call_name"] = "phone__r1"
    save_trace(trace, run_dir, "phone__r1")
    threading.Thread(target=finish_call, args=(trace, run_dir), daemon=True).start()
    return Response(status_code=204)


def finish_call(trace: dict, run_dir: Path) -> None:
    from harness.batch import export_to_langfuse
    from harness.index import build_index
    from harness.judge import score_and_save
    from harness.report import build_report
    try:
        score_and_save(trace, run_dir, "phone__r1")
        logger.info(f"phone call scored: {build_report(run_dir)}")
        export_to_langfuse(run_dir)
        build_index()
    except Exception:
        logger.exception(f"scoring the phone call failed; later: python -m harness judge {run_dir}")
