
"""Environment variables and settings."""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os
from dotenv import load_dotenv


def _sanitize_database_url(url: str) -> str:
    """Strip Prisma-style '?schema=' params that asyncpg doesn't support."""
    if not url:
        return url
    if "?schema=" in url:
        return url.split("?")[0]
    return url

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    database_url: str = _sanitize_database_url(os.getenv(
        "DATABASE_URL",
        "postgresql://persona_i_admin:persona_i_admin_11_17!!@localhost:5432/persona_i"
    ))

    server_host: str = "0.0.0.0"
    server_port: int = int(os.getenv("PORT", 8002))
    cors_origins: List[str] = ["*"]

    default_model: str = "gpt-3.5-turbo"
    default_temperature: float = 0.7

    load_db_history: bool = os.getenv("ENABLE_DB_HISTORY", "true").lower() == "true"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
