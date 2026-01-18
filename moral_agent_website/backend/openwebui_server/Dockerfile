# Python 3.12 slim 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (SQLite 등)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 먼저 복사 (캐싱 최적화)
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 복사
COPY . .

# 데이터 디렉토리 생성 (SQLite DB 저장용)
RUN mkdir -p /app/data

# Railway에서 제공하는 PORT 환경변수 사용 (기본값: 8001)
ENV PORT=8001

# 포트 노출
EXPOSE ${PORT}

# Uvicorn으로 FastAPI 실행
# --host 0.0.0.0: 모든 인터페이스에서 접근 허용
# --port $PORT: Railway의 동적 포트 사용
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
