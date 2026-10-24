"""Every setting in one place. Each one can be overridden with an environment variable (or a line in .env).

Models run on Groq (GROQ_API_KEY). Voice uses Sarvam (SARVAM_API_KEY). Langfuse is optional (LANGFUSE_*).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Models (Groq)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen/qwen3.8-27b")      # the agent under test (same model as voice-v1)
CALLER_MODEL = os.getenv("CALLER_MODEL", "openai/gpt-oss-120b")  # plays the simulated patients
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")    # a different model family from the agent

# Agent under test
AGENT_SCRIPT_VERSION = os.getenv("AGENT_SCRIPT_VERSION", "v3")    # data/agent_script_<version>.md
GUARDRAILS_ON = os.getenv("AGENT_GUARDRAILS", "1") != "0"         # 0 measures the model without code checks

# Judge
CHECKLIST_VERSION = os.getenv("JUDGE_VERSION", "v5")              # evals/checklist_<version>.md

# Voice (Sarvam speech, Pipecat)
LANGUAGE = os.getenv("LANGUAGE", "en-IN")
AGENT_VOICE = os.getenv("TTS_SPEAKER", "priya")                   # Ava, same voice as voice-v1
SIM_CALLER_VOICE = os.getenv("SIM_CALLER_VOICE", "rahul")         # simulated patient, a different voice
# Speech providers. "sarvam": Sarvam hears and speaks on both sides. "elevenlabs": Groq Whisper hears on both sides,
# Ava speaks with ElevenLabs, the simulated patient speaks with SIM_CALLER_TTS ("groq" = Orpheus, or "elevenlabs").
VOICE_STACK = os.getenv("VOICE_STACK", "sarvam")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")              # Groq
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
ELEVENLABS_AGENT_VOICE = os.getenv("ELEVENLABS_AGENT_VOICE", "EXAVITQu4vr4xnSDxMaL")  # "Sarah"
ELEVENLABS_CALLER_VOICE = os.getenv("ELEVENLABS_CALLER_VOICE", "CwhRBWXzGAHq8TQ4Fs17")  # "Roger"
SIM_CALLER_TTS = os.getenv("SIM_CALLER_TTS", "groq")
GROQ_TTS_MODEL, GROQ_TTS_VOICE = "canopylabs/orpheus-v1-english", os.getenv("GROQ_TTS_VOICE", "troy")
# Silence that counts as "the caller stopped speaking". Pipecat's default (0.2 s) split "I'm doing okay, thanks"
# into two turns at the comma; 0.8 s waits through normal pauses at the cost of ~0.6 s more delay per reply.
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.8"))


def groq_api_key() -> str:
    return os.environ["GROQ_API_KEY"].strip()


def langfuse() -> dict | None:
    """Langfuse connection settings, or None when it is not configured."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return None
    return {"host": os.environ["LANGFUSE_HOST"], "public_key": os.environ["LANGFUSE_PUBLIC_KEY"],
            "secret_key": os.environ["LANGFUSE_SECRET_KEY"]}
