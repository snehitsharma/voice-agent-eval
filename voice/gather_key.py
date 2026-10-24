"""The secret placed in every webhook address of a webhook-mode call (see voice/gather.py and voice/make_call.py).

Derived from the Twilio auth token, so the call script and the server compute the same value without storing it;
anyone without the token cannot guess it. A quick tunnel's address is random, but it is public while it is open.
"""

import hashlib
import hmac
import os


def call_key() -> str:
    return hmac.new(os.environ["TWILIO_AUTH_TOKEN"].encode(), b"voice-eval/gather", hashlib.sha256).hexdigest()[:24]
