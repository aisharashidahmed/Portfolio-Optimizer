"""Application configuration, loaded from the environment.

Nothing secret is hardcoded. Every value has a development-safe default except
the JWT signing key, which must be set explicitly outside of development.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_SECRET_KEY = "dev-only-insecure-key-do-not-use-in-production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: Literal["development", "test", "production"] = "development"

    # --- Auth ---
    secret_key: str = DEV_SECRET_KEY
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # --- Persistence ---
    database_url: str = "sqlite:///./portfolio.db"

    # --- Job queue ---
    redis_url: str = "redis://localhost:6379/0"
    result_ttl_seconds: int = 3600

    # --- API ---
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Finance defaults ---
    risk_free_rate: float = 0.045

    @field_validator("secret_key")
    @classmethod
    def reject_default_secret_in_production(cls, value: str, info) -> str:
        if value == DEV_SECRET_KEY and info.data.get("environment") == "production":
            raise ValueError(
                "SECRET_KEY must be set to a real value when ENVIRONMENT=production"
            )
        return value

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached so the environment is read once per process."""
    return Settings()
