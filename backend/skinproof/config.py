from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def _legacy_gemini_key() -> str | None:
    """Read the existing local key file without executing or logging it.

    This is a migration bridge for the current workspace only. Production
    deployments should use GEMINI_API_KEY or SKINPROOF_GEMINI_API_KEY and can
    disable this bridge with SKINPROOF_DISABLE_LEGACY_KEY_FILE=1.
    """

    if os.getenv("SKINPROOF_DISABLE_LEGACY_KEY_FILE", "").strip() == "1":
        return None
    # Fail secure: the bridge is only active when a deployment explicitly
    # opts into development mode. An unset/misconfigured SKINPROOF_ENV must
    # never re-enable a local dev convenience in a real deployment.
    if os.getenv("SKINPROOF_ENV", "production").strip().casefold() != "development":
        return None
    path = Path(os.getenv("SKINPROOF_LEGACY_KEY_FILE", "first.py"))
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    # Parse a quoted value only; never eval/import the file.
    match = re.search(r"['\"]([^'\"]{20,})['\"]", source)
    return match.group(1).strip() if match else None


@dataclass(frozen=True)
class Settings:
    db_path: Path
    photo_dir: Path | None
    database_url: str | None = None
    database_pool_min_size: int = 1
    database_pool_max_size: int = 10
    database_connect_timeout: int = 10
    raw_photo_retention_days: int = 730
    model_version: str = "deterministic-3.0"
    policy_version: str = "2026-01"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_enabled: bool = True
    firebase_project_id: str | None = None
    auth_required: bool = False
    admin_token: str | None = None
    cors_allowed_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        db_path = Path(os.getenv("SKINPROOF_DB_PATH", ".data/skinproof.sqlite3"))
        database_url = (
            os.getenv("SKINPROOF_DATABASE_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
            or os.getenv("POSTGRES_URL", "").strip()
            or None
        )
        photo_dir_value = os.getenv("SKINPROOF_PHOTO_DIR", "").strip()
        api_key = (
            os.getenv("SKINPROOF_GEMINI_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
            or _legacy_gemini_key()
        )
        enabled_value = os.getenv("SKINPROOF_GEMINI_ENABLED", "1").strip().casefold()
        auth_required_value = os.getenv("SKINPROOF_AUTH_REQUIRED", "0").strip().casefold()
        return cls(
            db_path=db_path,
            photo_dir=Path(photo_dir_value) if photo_dir_value else None,
            database_url=database_url,
            database_pool_min_size=max(1, int(os.getenv("SKINPROOF_DB_POOL_MIN_SIZE", "1"))),
            database_pool_max_size=max(1, int(os.getenv("SKINPROOF_DB_POOL_MAX_SIZE", "10"))),
            database_connect_timeout=max(1, int(os.getenv("SKINPROOF_DB_CONNECT_TIMEOUT", "10"))),
            raw_photo_retention_days=int(os.getenv("SKINPROOF_RAW_RETENTION_DAYS", "730")),
            model_version=os.getenv("SKINPROOF_MODEL_VERSION", "deterministic-3.0"),
            policy_version=os.getenv("SKINPROOF_POLICY_VERSION", "2026-01"),
            gemini_api_key=api_key or None,
            gemini_model=os.getenv("SKINPROOF_GEMINI_MODEL", "gemini-3.5-flash-lite"),
            gemini_enabled=enabled_value not in {"0", "false", "no", "off"},
            firebase_project_id=os.getenv("SKINPROOF_FIREBASE_PROJECT_ID", "").strip() or None,
            # Auth defaults OFF: the existing test suite and the unauthenticated
            # web client carry no bearer tokens and must keep passing/working.
            auth_required=auth_required_value in {"1", "true", "yes", "on"},
            admin_token=os.getenv("SKINPROOF_ADMIN_TOKEN", "").strip() or None,
            cors_allowed_origins=tuple(
                origin.strip()
                for origin in os.getenv("SKINPROOF_CORS_ALLOWED_ORIGINS", "").split(",")
                if origin.strip()
            ),
        )

    def prepare(self) -> None:
        if not self.database_url:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.photo_dir:
            self.photo_dir.mkdir(parents=True, exist_ok=True)
