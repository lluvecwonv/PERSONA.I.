
"""
환경 변수 및 설정 관리
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Settings(BaseSettings):
    """
    애플리케이션 설정
    """
    # OpenAI API 키
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # 데이터베이스 설정 (PostgreSQL - READ ONLY)
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:1234@localhost:5432/person_ai"
    )

    # 서버 설정
    server_host: str = "0.0.0.0"
    server_port: int = int(os.getenv("PORT", 8002))

    # CORS 허용 도메인
    cors_origins: List[str] = ["*"]

    # LangChain 기본 설정
    default_model: str = "gpt-3.5-turbo"
    default_temperature: float = 0.7

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # .env의 추가 필드 무시
    )


settings = Settings()
