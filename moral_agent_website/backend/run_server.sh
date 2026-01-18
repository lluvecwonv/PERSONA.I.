#!/bin/bash

# FastAPI + Cloudflare Tunnel 실행 스크립트
# 컴퓨터 켤 때마다 이 스크립트 실행하면 됩니다!

echo "================================================"
echo "🚀 AI Agent Server 시작"
echo "================================================"
echo ""

# 현재 디렉토리 확인
cd "/Users/yoonnchaewon/Library/Mobile Documents/com~apple~CloudDocs/llm_lab/moral_agnet/backend"

# 1. 기존 프로세스 종료 (포트 충돌 방지)
echo "🔍 기존 서버 확인 중..."
if lsof -Pi :8002 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  포트 8002에서 실행 중인 프로세스 종료 중..."
    lsof -ti:8002 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# 2. FastAPI 서버 시작 (백그라운드)
echo ""
echo "📦 FastAPI 서버 시작 중..."
nohup /usr/bin/python3 server_game.py > logs/server.log 2>&1 &
SERVER_PID=$!
echo "✅ FastAPI 서버 시작됨 (PID: $SERVER_PID)"

# 3. 서버가 준비될 때까지 대기
echo ""
echo "⏳ 서버 준비 중..."
for i in {1..10}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "✅ FastAPI 서버 준비 완료!"
        break
    fi
    sleep 1
    echo "   대기 중... ($i/10)"
done

# 4. Cloudflare Tunnel 시작 (고정 도메인 사용)
echo ""
echo "🌐 Cloudflare Tunnel 시작 중..."
nohup /opt/homebrew/bin/cloudflared tunnel run 189b4783-3ec4-4a68-8bb4-dbb9b6d9a393 > logs/tunnel.log 2>&1 &
TUNNEL_PID=$!
echo "✅ Cloudflare Tunnel 시작됨 (PID: $TUNNEL_PID)"

# 5. Tunnel 준비 대기
echo ""
echo "⏳ Tunnel 준비 중..."
sleep 3

# 고정 URL 사용
TUNNEL_URL="https://api.persona-ai-agent.com"

echo ""
echo "================================================"
echo "✅ 서버 실행 완료!"
echo "================================================"
echo ""
echo "📍 FastAPI 서버 (로컬):"
echo "   http://localhost:8002"
echo ""
echo "📍 외부 접속 URL (고정 도메인!):"
echo "   $TUNNEL_URL"
echo ""
echo "📍 Swagger 문서:"
echo "   $TUNNEL_URL/docs"
echo ""
echo "================================================"
echo ""
echo "💡 서버 종료 방법:"
echo "   kill -9 $SERVER_PID $TUNNEL_PID"
echo ""
echo "💡 로그 확인:"
echo "   tail -f logs/server.log"
echo "   tail -f logs/tunnel.log"
echo ""

echo "🎉 서버가 백그라운드에서 실행 중입니다!"
echo "   컴퓨터를 닫으면 서버가 멈춥니다."
echo ""
