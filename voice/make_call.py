"""Phone a real number and connect the call to Ava over Twilio webhooks.

Twilio rings TARGET_PHONE_NUMBER from TWILIO_PHONE_NUMBER; when it is answered, Twilio asks voice/gather.py what to
do, turn by turn (Twilio listens and transcribes, Ava replies, Sarvam speaks). When the call ends it is saved,
scored and sent to Langfuse automatically (runs/<date-time>-phone/).

Steps (voice environment):
    1. .venv-voice\\Scripts\\python -m uvicorn voice.gather:app --port 8000
    2. cloudflared tunnel --url http://localhost:8000        (or: ngrok http 8000) -> prints a public https URL
    3. .venv-voice\\Scripts\\python -m voice.make_call --public-url https://<that-url> --name Sean

Play a made-up patient on the call. Do not share real health information.
"""

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import quote

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the harness package lives next to voice/

from harness import config  # noqa: E402,F401  (loads .env)
from voice.gather_key import call_key  # noqa: E402

REQUIRED = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "TARGET_PHONE_NUMBER")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m voice.make_call")
    parser.add_argument(
        "--public-url",
        default=os.getenv("PUBLIC_BASE_URL"),
        help="public https URL that reaches the voice server (from cloudflared or ngrok)",
    )
    parser.add_argument("--name", default="there", help="first name Ava will greet (use a made-up one)")
    args = parser.parse_args()

    env = {k: os.getenv(k) for k in REQUIRED}
    missing = [k for k, v in env.items() if not v] + ([] if args.public_url else ["--public-url (or PUBLIC_BASE_URL)"])
    if missing:
        sys.exit(f"Missing: {', '.join(missing)}")
    base = args.public_url.rstrip("/")
    if not base.startswith("https://"):
        sys.exit("The public URL must start with https:// (Twilio needs a secure address).")

    client = Client(env["TWILIO_ACCOUNT_SID"], env["TWILIO_AUTH_TOKEN"])
    try:
        call = client.calls.create(
            to=env["TARGET_PHONE_NUMBER"],
            from_=env["TWILIO_PHONE_NUMBER"],
            url=f"{base}/gather/start?name={quote(args.name)}&k={call_key()}",
            status_callback=f"{base}/gather/status?k={call_key()}",
            status_callback_event=["completed"],
        )
    except TwilioRestException as exc:
        sys.exit(f"Twilio refused the call: {exc.msg}")
    print(f"Calling {env['TARGET_PHONE_NUMBER'][:-4]}****  (call {call.sid}). Answer it and talk to Ava.")
    print("When you hang up, the call is saved, scored and sent to Langfuse automatically (runs/<date-time>-phone).")


if __name__ == "__main__":
    main()
