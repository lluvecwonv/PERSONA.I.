FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY backend/openwebui_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 캐시 무효화 (변경 시 재빌드 트리거)
ARG CACHE_BUST=v5

# OpenWebUI 서버 코드 복사
COPY backend/openwebui_server/ .

# agents 폴더 복사 (공유 에이전트 사용)
COPY backend/agents /app/agents

# 데이터 디렉토리 생성
RUN mkdir -p /app/data

# 환경 변수 설정 (Railway에서 PORT를 자동 제공)
ENV PORT=8001

# OpenWebUI Server 실행 (Railway PORT 사용)
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
