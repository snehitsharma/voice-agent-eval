"""The one place that talks to the LLM provider (Groq, OpenAI-compatible API)."""
import time
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from harness import config

# Kept as names here because callers and traces refer to them as llm.AGENT_MODEL etc.
AGENT_MODEL, CALLER_MODEL, JUDGE_MODEL = config.AGENT_MODEL, config.CALLER_MODEL, config.JUDGE_MODEL

MAX_ATTEMPTS = 8  # per-minute rate limits can need several waits in a row
MAX_WAIT_S = 30   # cap on one wait, so a single bad request cannot stall a run

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.groq_api_key(), base_url=config.GROQ_BASE_URL, max_retries=0, timeout=30)
    return _client


class QuotaExhausted(Exception):
    """The account's daily token or request limit is used up. Waiting a few seconds will not help."""


@dataclass
class LLMResult:
    message: object  # the openai ChatCompletionMessage
    latency_ms: int  # total, including any retries
    prompt_tokens: int
    completion_tokens: int
    retries: list  # one entry per failed attempt, so slow turns can be explained


def chat(model: str, messages: list, tools: list | None = None, **extra) -> LLMResult:
    """One chat completion. `extra` goes to the provider as-is (e.g. reasoning_effort, response_format).

    Retries are done here rather than hidden inside the SDK, so each one is recorded on the turn."""
    kwargs = {"model": model, "messages": messages, "extra_body": extra or None}
    if tools:
        kwargs["tools"] = tools
    start = time.perf_counter()
    retries = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client().chat.completions.create(**kwargs)
            break
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
            if isinstance(exc, RateLimitError) and "per day" in str(exc):
                raise QuotaExhausted(str(exc)) from exc
            wait_s = min(_retry_after(exc) or 2 ** attempt, MAX_WAIT_S)
            retries.append({"attempt": attempt, "error": type(exc).__name__, "waited_s": wait_s})
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(wait_s)
    latency_ms = round((time.perf_counter() - start) * 1000)
    usage = response.usage
    return LLMResult(response.choices[0].message, latency_ms, usage.prompt_tokens, usage.completion_tokens, retries)


def _retry_after(exc) -> float | None:
    response = getattr(exc, "response", None)
    try:
        return float(response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None
