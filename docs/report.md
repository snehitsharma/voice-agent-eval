# Testing a pharma voice agent before it calls real patients

*Snehit Sharma · Python, Pipecat, Groq, Twilio, Sarvam, ElevenLabs, Langfuse/OpenTelemetry*

A pharma company's voice agent has to finish its task, never give medical advice, and pass on every reported side
effect. I built a test harness that calls such an agent with simulated patients, records every call, scores it
against written rules, and shows where it fails. Everything is invented: a fictional drug (Zelvora), simulated
patients played by an LLM, and a test agent ("Ava") on Groq's `qwen3.8-27b`.

## How it works

31 simulated patients, each built around one risk: side effects mentioned casually, brushed off, hidden in a
question or reported about someone else; serious reactions; pushes for dose advice; off-label questions; callers who
need a human. Each call is saved as a trace (every turn, timing, tool call, error), then scored by plain-code checks
(was the side-effect tool actually called, did the agent hang up on the caller) plus an LLM judge that has to quote
the transcript. The rules went through five versions as testing exposed gaps (`evals/README.md`).

## 1. The harness found real failures

In the first round, the agent:

- **hung up on a patient describing a serious allergic reaction**, during its own greeting, before he could speak;
- **told a patient "I've passed that to the safety team" without logging anything**;
- missed a swollen calf described calmly ("probably pulled something"), and an accidental double dose.

## 2. Fixing the agent, measured on the same 31 scenarios

New instructions plus code guardrails that check each reply before the patient hears it:

| | Before | After |
|---|---|---|
| Hung up on the patient | 4 | **0** |
| Said "I reported it" without logging it | 2 | **0** |
| Skipped confirming the patient's identity | 18 | **2** |
| Reported side effects flagged | 11 of 12 | 11 of 12 |
| Medical advice / false promises | 0 / 0 | 0 / 0 |

The guardrails stepped in 5 times in the 31 "after" calls, and each intervention is recorded in the trace.

## 3. Checking the judge against my own labels

I labelled 12 calls by hand, without seeing the judge's answers. On the key question, *did the patient report a side
effect?*, the judge agreed with me on **11 of 12**. It missed one rash introduced as "No, none. Well, I've had this
rash…". It agreed on 12 of 12 for medical advice and off-label claims. Its weakest area was matching each "I reported
it" to a logged report (8 of 12), and it once counted a patient's question ("could my daughter take it?") as an event.

## 4. Voice: what breaks only because it is voice

A Pipecat streaming agent (Sarvam, or ElevenLabs plus Whisper speech) took simulated phone calls on the 12 side-effect
scenarios:

- **Side effects caught in 10 of 11 complete calls; median reply delay about 0.6 s** (slowest 1.3–1.7 s).
- **An interruption cancelled a safety report.** The agent said "I", paused to log a rash, the patient took the pause
  as the end and said goodbye, and Pipecat cancelled the in-flight report. Fix: safety actions can no longer be
  cancelled by an interruption. The re-run logged the rash.
- **Speech detection cut patients off** at commas with the default 0.2 s silence threshold; it now waits 0.8 s.
- **Real phone calls** over Twilio: the phone line's speech recognition heard "double dose" as **"double doors"**.
  The agent asked to clarify, then reported it. One call dropped after a 5.4 s reply, the slowest of that call.

## 5. Observability

Every call is exported to Langfuse via OpenTelemetry (conversation → turn → LLM and tool spans), with the judge's
verdicts attached as scores. Separating the provider's rate-limit waits from model time showed that the long pauses
in early runs were rate limits (680 s across 13 calls) while the model answered in a median of 390 ms per reply.
Details: [docs/observability.md](observability.md).

## Limits

Simulated patients and an invented drug; one call per scenario; the judge was checked on 12 calls by one labeller;
the voice tests used clean synthetic speech. This is a test tool on invented data, not a compliant or medically
validated system.

## Reproduce

```
python -m harness compare runs/study-before-ava-v2 runs/study-after-ava-v3     # section 2
python -m harness agreement runs/study-after-ava-v3                           # section 3
python -m harness stats runs/study-voice-ava-v3 runs/study-voice-elevenlabs   # section 4
python -m harness stats runs/20261022-002637                                  # section 5
```
