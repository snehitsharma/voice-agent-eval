# Two images from one file:
#   docker build -t voice-eval .                       # harness: simulated text calls, judge, reports, labelling
#   docker build -t voice-eval:voice --target voice .   # + Pipecat voice agent, simulated voice calls, phone calls
# Secrets are never copied in; pass them at run time: docker run --env-file .env ...

FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --create-home --uid 1000 app && chown app:app /app

FROM base AS voice
COPY requirements.txt requirements-voice.txt ./
RUN pip install -r requirements.txt -r requirements-voice.txt
COPY --chown=app:app . .
USER app
EXPOSE 8000
# Voice Ava (Pipecat, streaming). For webhook phone calls instead: uvicorn voice.gather:app ...
CMD ["uvicorn", "voice.server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS harness
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=app:app . .
USER app
EXPOSE 8765
ENTRYPOINT ["python", "-m", "harness"]
# With no arguments it prints the list of commands.
CMD []
