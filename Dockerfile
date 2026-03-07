FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/game_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG CACHE_BUST=v5

COPY backend/game_server/ .
COPY backend/agents /app/agents

RUN mkdir -p /app/data

ENV PORT=8002

CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
