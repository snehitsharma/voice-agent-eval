# Voice Agent Eval

A test harness for voice agents in life sciences. Simulated patients call the agent under test, every call is
saved as a full trace, and a judge scores each call against a written checklist: did the agent finish its task,
catch every reported side effect, avoid medical advice and off-label claims, and hand off when it should.

**Everything is made up:** a fictional drug (Zelvora), fictional patients, fictional label. Simulated callers are
LLMs playing invented people, not real patients. This is a test tool on invented data; it is not HIPAA compliant or
medically validated.

## How it works

```
 persona (YAML) ──> simulated caller (LLM) ──┐
                                             │ text, or phone audio over a Twilio-style websocket
 agent under test ("Ava": script + tools) <──┘
        │
        └─> trace (every turn, timings, tool calls, errors) ──> judge = code checks + LLM judge
                                                                   │
                     HTML report · Langfuse (OpenTelemetry) <──────┴──> hand labels ──> judge agreement
```

- **Agent under test:** "Ava", a patient-support agent with a script (`data/agent_script_v2.md`) and four tools:
  look up the approved label, flag an adverse event, transfer to a human, end the call. The same agent runs as text
  (`harness/agents/text_agent.py`) and as voice (`voice/server.py`, Pipecat + Sarvam speech + Groq LLM).
  Code guardrails (`harness/agents/guardrails.py`) check each reply before the patient gets it, and every
  intervention is recorded in the trace.
  Any agent can be plugged in through `harness/agent_interface.py` (start / reply / end).
- **Simulated callers:** 31 personas in `data/personas/`, each aimed at a risk: side effects mentioned casually,
  explained away or hidden in a question; serious reactions; pushes for dose advice; off-label questions; callers
  who need a human; silence, rambling, hostility and prompt injection. Each persona carries an answer key that is
  hidden from the judge and used only to flag calls for review.
- **Judge:** plain-code checks on the trace (was the flag tool actually called, did the agent hang up on the
  caller, leaked reasoning, repeated replies) combined with an LLM judge that must quote the transcript. Rules are
  versioned: `evals/checklist_v5.md`, history in `evals/README.md`.
- **Trust in the judge:** a local labelling page shows calls without the judge's answers; `agreement` reports how
  often the judge matches the human, with every disagreement listed.
- **Observability:** traces are JSON files plus an HTML report per run, exported to a local Langfuse as
  OpenTelemetry (conversation → turn → LLM and tool spans, shaped like Pipecat's tracing). Voice calls also send
  Pipecat's own STT / LLM / TTS spans.

## Write-up

- **[Report: testing a pharma voice agent before it calls real patients](docs/report.md)** (start here)
- [Observability: seeing what a voice agent actually did](docs/observability.md)
- [Labelling guide](docs/labelling-guide.md)

## Setup

Python 3.13. Text harness:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`.env` needs `GROQ_API_KEY`. Every setting (models, agent script version, guardrails on/off, rule version, voices)
is in `harness/config.py`, each overridable by an environment variable:

| Setting | Default | Variable |
|---|---|---|
| Agent under test | `qwen/qwen3.8-27b` | `AGENT_MODEL` |
| Simulated caller | `openai/gpt-oss-120b` | `CALLER_MODEL` |
| Judge | `openai/gpt-oss-120b` | `JUDGE_MODEL` |
| Agent script | `v3` | `AGENT_SCRIPT_VERSION` |
| Guardrails | on | `AGENT_GUARDRAILS=0` turns them off |
| Grading rules | `v5` | `JUDGE_VERSION` |

## Docker

```
docker build -t voice-eval .                          # harness (308 MB)
docker run --rm voice-eval                            # lists the commands
docker run --rm voice-eval compare runs/study-before-ava-v2 runs/study-after-ava-v3    # no API keys needed
docker run --rm --env-file .env -v "%cd%/runs:/app/runs" voice-eval run --personas maria_casual_side_effect

docker build -t voice-eval:voice --target voice .     # + Pipecat voice agent and phone calls (1.1 GB)
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 voice-eval:voice
```

Secrets are never copied into an image: pass them with `--env-file .env`. To reach Langfuse on the same machine
from a container, add `-e LANGFUSE_HOST=http://host.docker.internal:3000`. CI (`.github/workflows/ci.yml`) lints,
runs the tests, checks that the saved results still reproduce, and builds the image.

## Commands

```
python -m harness run                      # every persona once; --repeats N, --personas ..., --resume runs/<folder>
python -m harness talk --name Sean         # you type as the patient; the call is scored like any other
python -m harness judge runs/<folder>      # score or rescore a run
python -m harness report runs/<folder>     # HTML report: runs/<folder>/report.html
python -m harness label runs/<folder>      # hand-label calls at http://localhost:8765
python -m harness agreement runs/<folder>  # judge vs hand labels
python -m harness export runs/<folder>     # send to Langfuse
python -m harness stats runs/<folder>      # timings, retries, guardrails, voice delay and mishearing
python -m harness compare runs/<a> runs/<b>  # before/after page
python -m harness index                    # runs/index.html: every result in reading order
.venv\Scripts\python -m pytest             # tests
```

(Run with `.venv\Scripts\python`.) Runs stop cleanly when a provider's daily quota is used up and resume with
`--resume`.

## Voice

Separate environment, because Pipecat brings many packages:

```
python -m venv .venv-voice
.venv-voice\Scripts\python -m pip install -r requirements-voice.txt
.venv-voice\Scripts\python -m uvicorn voice.server:app --port 8000      # voice Ava (Pipecat, streaming)
.venv-voice\Scripts\python -m voice.sim_caller maria_casual_side_effect  # simulated phone call
```

The simulated caller connects the way Twilio Media Streams does, speaks the persona's lines as 8 kHz phone audio and
transcribes Ava's audio to decide what to say. Each agent turn records the voice delay: caller stops speaking →
Ava's audio starts. Speech providers are chosen with `VOICE_STACK` (`sarvam`, or `elevenlabs` = ElevenLabs voice +
Groq Whisper hearing; see `harness/config.py`).

Real phone calls run turn by turn over Twilio webhooks (`voice/gather.py`), which works on Twilio trial accounts:

```
.venv-voice\Scripts\python -m uvicorn voice.gather:app --port 8000
cloudflared tunnel --url http://localhost:8000                          # public https URL for Twilio
.venv-voice\Scripts\python -m voice.make_call --public-url https://<tunnel-url> --name Sean
```

## Langfuse

```
.venv\Scripts\python infra\langfuse\make_env.py     # random secrets, login and API keys (not committed)
cd infra\langfuse && docker compose up -d            # http://localhost:3000, user admin@voice-eval.local
```

`docker-compose.yml` is Langfuse's official file, unmodified; `docker-compose.override.yml` keeps every port local
to the machine.

## Layout

```
data/        drug label, agent scripts, 31 personas
evals/       checklists and judge prompts, versioned
harness/     runner, caller, text agent, judge, report, labelling, agreement, export, talk mode
voice/       Pipecat voice server, simulated voice caller, Twilio phone calls
infra/       local Langfuse
runs/        one folder per run: traces/, scores/, report.html, calls/*.html; runs/index.html lists them
labels/      hand labels used to check the judge
docs/        report, observability chapter, labelling guide
tests/       pytest
```

## Status and limits

- Results, sample sizes and the commands that reproduce them: [docs/report.md](docs/report.md). Small samples are
  reported with 95% ranges.
- The judge was checked against 12 hand-labelled calls by one labeller.
- Simulated callers are LLMs and sometimes break character (answering several questions at once, hanging up
  early). Those are logged as caller issues, never counted against the agent.
- Simulated voice listening uses speech-to-text, which mishears some words a person would not ("ankles" heard as
  "anus"); such errors on the caller's side are not counted against the agent.
- Cost on Groq's paid tier, measured from saved runs: about 1 cent per simulated text call (agent, caller and judge).
