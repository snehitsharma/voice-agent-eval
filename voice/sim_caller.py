"""Simulated voice caller: phones voice Ava the way Twilio does, speaking and listening as phone audio.

    .venv-voice\\Scripts\\python -m voice.sim_caller maria_casual_side_effect

Each turn it listens (collects Ava's audio until she stops and transcribes it), thinks (the same persona LLM as the
text harness picks a reply) and speaks (synthesises the reply and streams it as 8 kHz audio in real time). In between
it sends silence, like an open phone line. The server saves the trace; this script saves what the caller said and
heard (runs/<run>/voice_caller/<call>.json), so mishearing can be measured on both sides.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

import audioop
import httpx
import websockets
from openai import OpenAI
from sarvamai import SarvamAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the harness package lives next to voice/

from harness import config  # noqa: E402  (also loads .env)
from harness.caller import Caller  # noqa: E402
from harness.personas import load_persona  # noqa: E402
from harness.runner import GOODBYE, MAX_TURNS  # noqa: E402

RATE = 8000
FRAME = 160  # 20 ms of 8 kHz mu-law
SILENCE = b"\xff" * FRAME  # mu-law silence
ROOT = config.ROOT
LANGUAGE, CALLER_VOICE = config.LANGUAGE, config.SIM_CALLER_VOICE
# Ava's turn is over after this much silence. Shorter values cut her off in the pause between two sentences.
AGENT_DONE_AFTER_S = 2.5
PAUSE_RECORD_S = 0.5  # silences inside one of Ava's replies longer than this are recorded
FIRST_WORDS_TIMEOUT_S = 25


class Line:
    """One open phone line: sends caller audio (or silence) every 20 ms and collects Ava's audio."""

    def __init__(self, ws, stream_sid: str):
        self.ws, self.stream_sid = ws, stream_sid
        self.outgoing = bytearray()
        self.incoming = bytearray()
        self.first_rx = self.last_rx = None
        self.pauses: list[float] = []  # silences inside Ava's current reply
        self.last_pauses: list[float] = []
        self.closed = False

    async def send_loop(self):
        next_tick = time.perf_counter()
        while not self.closed:
            chunk = bytes(self.outgoing[:FRAME]) if self.outgoing else SILENCE
            del self.outgoing[:FRAME]
            chunk = chunk.ljust(FRAME, b"\xff")
            try:
                await self.ws.send(
                    json.dumps(
                        {
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": base64.b64encode(chunk).decode()},
                        }
                    )
                )
            except websockets.ConnectionClosed:
                self.closed = True
                return
            next_tick += FRAME / RATE
            await asyncio.sleep(max(0, next_tick - time.perf_counter()))

    async def receive_loop(self):
        try:
            async for raw in self.ws:
                message = json.loads(raw)
                if message.get("event") == "media":
                    now = time.perf_counter()
                    if self.last_rx is not None:
                        # Audio arrives in bursts; a gap longer than the audio already queued is real silence.
                        played_until = self.first_rx + len(self.incoming) / RATE
                        if now - max(self.last_rx, played_until) > PAUSE_RECORD_S:
                            self.pauses.append(round(now - max(self.last_rx, played_until), 2))
                    self.first_rx = self.first_rx or now
                    self.last_rx = now
                    self.incoming += base64.b64decode(message["media"]["payload"])
        except websockets.ConnectionClosed:
            pass
        self.closed = True

    async def wait_for_agent_turn(self) -> bytes:
        """Ava's turn is over when no audio has arrived for a while AND her audio has had time to play."""
        waited_from = time.perf_counter()
        while not self.closed:
            now = time.perf_counter()
            if self.first_rx is None:
                if now - waited_from > FIRST_WORDS_TIMEOUT_S:
                    return b""  # she said nothing
            else:
                played_until = self.first_rx + len(self.incoming) / RATE
                if now - self.last_rx > AGENT_DONE_AFTER_S and now > played_until + 0.3:
                    break
            await asyncio.sleep(0.05)
        audio = bytes(self.incoming)
        self.incoming.clear()
        self.first_rx = self.last_rx = None
        self.last_pauses, self.pauses = self.pauses, []
        return audio

    async def say(self, ulaw: bytes):
        self.outgoing += ulaw
        while self.outgoing and not self.closed:  # wait until it has all been "spoken"
            await asyncio.sleep(0.02)


def ulaw_to_wav(ulaw: bytes) -> bytes:
    return pcm16_to_wav(audioop.ulaw2lin(ulaw, 2), RATE)


def wav_to_ulaw(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        pcm, rate, width, channels = (
            wf.readframes(wf.getnframes()),
            wf.getframerate(),
            wf.getsampwidth(),
            wf.getnchannels(),
        )
    if channels == 2:
        pcm = audioop.tomono(pcm, width, 0.5, 0.5)
    if width != 2:
        pcm = audioop.lin2lin(pcm, width, 2)
    if rate != RATE:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, RATE, None)
    return audioop.lin2ulaw(pcm, 2)


def pcm16_to_wav(pcm: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buffer.getvalue()


def caller_speech():
    """The simulated patient's ears (WAV -> text) and voice (text -> 8 kHz mu-law) for the configured VOICE_STACK."""
    if config.VOICE_STACK == "sarvam":
        sarvam = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

        def hear(wav: bytes) -> str:
            return sarvam.speech_to_text.transcribe(
                file=("agent.wav", wav, "audio/wav"), model="saarika:v2.5", language_code=LANGUAGE
            ).transcript

        def speak(text: str) -> bytes:
            tts = sarvam.text_to_speech.convert(
                text=text, target_language_code=LANGUAGE, speaker=CALLER_VOICE, speech_sample_rate=RATE
            )
            return wav_to_ulaw(base64.b64decode("".join(tts.audios)))

        return hear, speak, f"sarvam {CALLER_VOICE}"

    groq = OpenAI(api_key=config.groq_api_key(), base_url=config.GROQ_BASE_URL)

    def hear(wav: bytes) -> str:
        return groq.audio.transcriptions.create(
            model=config.WHISPER_MODEL, file=("agent.wav", wav, "audio/wav"), language="en"
        ).text

    if config.SIM_CALLER_TTS == "elevenlabs":

        def speak(text: str) -> bytes:
            r = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_CALLER_VOICE}",
                params={"output_format": "pcm_16000"},
                timeout=60,
                headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"].strip()},
                json={"text": text, "model_id": config.ELEVENLABS_MODEL},
            )
            r.raise_for_status()
            return wav_to_ulaw(pcm16_to_wav(r.content, 16000))

        return hear, speak, f"elevenlabs {config.ELEVENLABS_CALLER_VOICE}"

    def speak(text: str) -> bytes:
        speech = groq.audio.speech.create(
            model=config.GROQ_TTS_MODEL, voice=config.GROQ_TTS_VOICE, input=text, response_format="wav"
        )
        return wav_to_ulaw(speech.read())

    return hear, speak, f"groq {config.GROQ_TTS_MODEL} {config.GROQ_TTS_VOICE}"


def start_message(stream_sid: str, persona: dict, run_name: str, call_name: str) -> dict:
    """Twilio's "start" event, with the call details the server needs passed as custom parameters."""
    return {
        "event": "start",
        "streamSid": stream_sid,
        "start": {
            "streamSid": stream_sid,
            "callSid": "CA" + uuid.uuid4().hex,
            "accountSid": "ACsimulated",
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": RATE, "channels": 1},
            "customParameters": {
                "mode": "sim",
                "first_name": persona["name"].split()[0],
                "run": run_name,
                "call": call_name,
                "persona": json.dumps(persona),
            },
        },
    }


async def run_call(persona_id: str, run_name: str, call_name: str, url: str, caller_model: str | None) -> dict:
    persona = load_persona(persona_id)
    caller = Caller(persona, model=caller_model)
    hear, speak, voice_desc = caller_speech()
    stream_sid = "MZ" + uuid.uuid4().hex
    side = {
        "persona_id": persona_id,
        "call_name": call_name,
        "caller_model": caller.model,
        "voice_stack": config.VOICE_STACK,
        "caller_voice": voice_desc,
        "turns": [],
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    t0 = time.perf_counter()

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps(start_message(stream_sid, persona, run_name, call_name)))
        line = Line(ws, stream_sid)
        tasks = [asyncio.create_task(line.send_loop()), asyncio.create_task(line.receive_loop())]

        for _ in range(MAX_TURNS):
            agent_audio = await line.wait_for_agent_turn()
            if line.closed and not agent_audio:
                side["end_reason"] = "agent_ended"
                break
            heard = ""
            if agent_audio:
                heard = (await asyncio.to_thread(hear, ulaw_to_wav(agent_audio))).strip()
            side["turns"].append(
                {
                    "speaker": "agent",
                    "heard_by_caller": heard,
                    "pauses_s": line.last_pauses,
                    "audio_s": round(len(agent_audio) / RATE, 2),
                    "t_s": round(time.perf_counter() - t0, 2),
                }
            )
            if line.closed:
                side["end_reason"] = "agent_ended"
                break
            text, hung_up, _ = await asyncio.to_thread(caller.reply, heard or "(silence)")
            if text:
                ulaw = await asyncio.to_thread(speak, text)
                spoke_at = round(time.perf_counter() - t0, 2)
                await line.say(ulaw)
                side["turns"].append(
                    {"speaker": "caller", "said": text, "audio_s": round(len(ulaw) / RATE, 2), "t_s": spoke_at}
                )
            if hung_up and not GOODBYE.search(heard):
                # As in the text harness, hanging up before Ava's goodbye is the simulated caller's mistake.
                side.setdefault("caller_issues", []).append(
                    {"turn": len(side["turns"]) - 1, "issue": "early_hangup_ignored"}
                )
                hung_up = False
            if hung_up:
                side["end_reason"] = "caller_hung_up"
                await asyncio.sleep(1.0)
                await ws.send(json.dumps({"event": "stop", "streamSid": stream_sid}))
                break
        else:
            side["end_reason"] = "turn_limit"
        line.closed = True
        for task in tasks:
            task.cancel()
    side["duration_s"] = round(time.perf_counter() - t0, 2)
    out = ROOT / "runs" / run_name / "voice_caller" / f"{call_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(side, indent=2, ensure_ascii=False), encoding="utf-8")
    return side


def main():
    parser = argparse.ArgumentParser(prog="python -m voice.sim_caller")
    parser.add_argument("personas", nargs="+")
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--caller-model", default=config.CALLER_MODEL, help="Groq model playing the patient")
    parser.add_argument("--run", default=datetime.now().strftime("%Y%m%d-%H%M%S") + "-voice")
    args = parser.parse_args()
    for pid in args.personas:
        side = asyncio.run(run_call(pid, args.run, f"{pid}__r1", args.url, args.caller_model))
        print(f"\n{pid}: {side['end_reason']} after {side['duration_s']} s")
        for t in side["turns"]:
            print(
                f"  {'AVA (as heard)' if t['speaker'] == 'agent' else 'CALLER (said)'}: "
                f"{t.get('heard_by_caller', t.get('said'))}"
            )
    print(f"\nRun folder: runs/{args.run}  (score it with: python -m harness judge runs/{args.run})")


if __name__ == "__main__":
    main()
