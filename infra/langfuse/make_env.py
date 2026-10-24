"""Writes infra/langfuse/.env with random secrets and a ready-made login, project and API keys.

Usage (once, before `docker compose up -d`):
    python infra/langfuse/make_env.py

Also adds LANGFUSE_* connection settings to the project's main .env so the harness can send traces.
Refuses to overwrite an existing infra/langfuse/.env, because the databases keep the old secrets.
"""

import secrets
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

env_path = HERE / ".env"
if env_path.exists():
    raise SystemExit(f"{env_path} already exists; delete it (and the Docker volumes) to start over.")

public_key = "pk-lf-" + secrets.token_hex(16)
secret_key = "sk-lf-" + secrets.token_hex(16)
password = secrets.token_urlsafe(12)
db_password = secrets.token_hex(16)
minio_password = secrets.token_hex(16)

values = {
    "TELEMETRY_ENABLED": "false",
    "NEXTAUTH_SECRET": secrets.token_hex(32),
    "SALT": secrets.token_hex(16),
    "ENCRYPTION_KEY": secrets.token_hex(32),
    "POSTGRES_PASSWORD": db_password,
    "DATABASE_URL": f"postgresql://postgres:{db_password}@postgres:5432/postgres",
    "CLICKHOUSE_PASSWORD": secrets.token_hex(16),
    "REDIS_AUTH": secrets.token_hex(16),
    "MINIO_ROOT_PASSWORD": minio_password,
    "LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY": minio_password,
    "LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY": minio_password,
    "LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY": minio_password,
    "LANGFUSE_INIT_ORG_ID": "voice-eval",
    "LANGFUSE_INIT_ORG_NAME": "Voice Eval",
    "LANGFUSE_INIT_PROJECT_ID": "voice-eval",
    "LANGFUSE_INIT_PROJECT_NAME": "Voice Eval",
    "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": public_key,
    "LANGFUSE_INIT_PROJECT_SECRET_KEY": secret_key,
    "LANGFUSE_INIT_USER_EMAIL": "admin@voice-eval.local",
    "LANGFUSE_INIT_USER_NAME": "Admin",
    "LANGFUSE_INIT_USER_PASSWORD": password,
}
env_path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")

main_env = ROOT / ".env"
existing = main_env.read_text(encoding="utf-8") if main_env.exists() else ""
if "LANGFUSE_PUBLIC_KEY" not in existing:
    with main_env.open("a", encoding="utf-8") as f:
        f.write(
            f"\nLANGFUSE_HOST=http://localhost:3000\nLANGFUSE_PUBLIC_KEY={public_key}\n"
            f"LANGFUSE_SECRET_KEY={secret_key}\n"
        )

print(f"Wrote {env_path}")
print("Langfuse login: http://localhost:3000  user admin@voice-eval.local")
print(f"Password is LANGFUSE_INIT_USER_PASSWORD in {env_path}")
