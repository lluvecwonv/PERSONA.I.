
"""
환경 변수 및 설정 관리
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os
from dotenv import load_dotenv


def _sanitize_database_url(url: str) -> str:
    """
    asyncpg는 Prisma 스타일의 '?schema=' 파라미터를 지원하지 않으므로 제거한다.
    필요한 경우 .env에서는 기본 커넥션 문자열만 입력하도록 한다.
    """
    if not url:
        return url
    if "?schema=" in url:
        return url.split("?")[0]
    return url

# .env 파일 로드
load_dotenv()


class Settings(BaseSettings):
    """
    애플리케이션 설정
    """
    # OpenAI API 키
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Gemini API 키
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # 데이터베이스 설정 (PostgreSQL - READ ONLY)
    # 주의: asyncpg는 ?schema=public 지원 안 함 (Prisma 전용 파라미터)
    # Docker에서 PostgreSQL은 5433 포트로 매핑됨
    database_url: str = _sanitize_database_url(os.getenv(
        "DATABASE_URL",
        "postgresql://persona_i_admin:persona_i_admin_11_17!!@localhost:5432/persona_i"
    ))

    # 서버 설정
    server_host: str = "0.0.0.0"
    server_port: int = int(os.getenv("PORT", 8002))

    # CORS 허용 도메인
    cors_origins: List[str] = ["*"]

    # LangChain 기본 설정
    default_model: str = "gpt-3.5-turbo"
    default_temperature: float = 0.7

    # 게임 서버에서 DB 히스토리 로드 여부 (기본값 True)
    load_db_history: bool = os.getenv("ENABLE_DB_HISTORY", "true").lower() == "true"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # .env의 추가 필드 무시
    )


settings = Settings()
