# Game Server

AI ethics dialogue game FastAPI server. Connects with the NestJS backend to handle agent conversations.

---

## Project Structure

```
game_server/
├── server.py                 # FastAPI main server
├── langchain_service.py      # Agent orchestration service
├── config.py                 # Settings management
├── exceptions.py             # Custom exception classes
├── profile_messages.json     # Profile initial/final messages
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
cd backend
pip install -r game_server/requirements.txt
```

### 2. Environment variables

Create a `.env` file or export variables:

```bash
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key    # optional, for Gemini reflection
PORT=8002
DB_HOST=localhost
DB_PORT=5432
DB_USER=persona_i_admin
DB_PASSWORD=your-db-password
DB_NAME=persona_i
```

---

## Running the Server

### Start

```bash
cd backend
nohup python3 -m game_server.server > /tmp/game_server.log 2>&1 &
```

### Check logs

```bash
tail -f /tmp/game_server.log
```

### Stop

```bash
# Kill all game_server processes
pkill -f "game_server.server"
pkill -f "game_server"
pkill -f "server.py"

# Force kill anything on port 8002
lsof -ti:8002 | xargs kill -9
```

### Restart (full)

```bash
pkill -f "game_server.server"
pkill -f "game_server"
pkill -f "server.py"
lsof -ti:8002 | xargs kill -9
sleep 2

# Verify port is free
lsof -i :8002

# Start
cd backend
nohup python3 -m game_server.server > /tmp/game_server.log 2>&1 &
tail -f /tmp/game_server.log
```

Server runs at `http://localhost:8002`.

---

## API Endpoints

### Health Check

```
GET /health
```

Returns `{"status": "ok"}`.

### Game Chat

```
POST /api/game/context/{context_id}/chat
```

**Request:**
```json
{
  "agent_id": "agent-1",
  "message": "user message",
  "user_id": "user-uuid",
  "stream": false,
  "include_audio": false
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "response": "AI response",
    "metadata": {
      "stage": "stage1",
      "message_count": 1,
      "is_end": false
    }
  }
}
```

Set `stream: true` for SSE streaming mode.

### TTS

```
POST /api/tts
```

**Request:**
```json
{
  "text": "text to convert",
  "voice": "alloy",
  "model": "tts-1"
}
```

Available voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`.

---

## Agents

| Agent ID | Name | Model Key | Scenario | Voice |
|----------|------|-----------|----------|-------|
| `agent-1` | Artist Apprentice | `artist-apprentice` | AI art exhibition ethics (facilitator) | alloy |
| `agent-2` | Friend Agent | `friend-agent` | AI resurrection ethics (facilitator) | nova |
| `agent-3` | Colleague 1 | `colleague1` | AI art - deontological opposition | shimmer |
| `agent-4` | Colleague 2 | `colleague2` | AI art - utilitarian support | echo |
| `agent-5` | Jangmo | `jangmo` | AI resurrection - deontological opposition | fable |
| `agent-6` | Son | `son` | AI resurrection - utilitarian support | onyx |

### Adding a New Agent

1. Register in `server.py` `AGENTS` dict
2. Import and instantiate in `langchain_service.py.__init__()`
3. Route by model key in `_get_agent()`

---

## Testing

```bash
# Non-streaming
curl -X POST "http://localhost:8002/api/game/context/test-123/chat" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"agent-1","message":"안녕하세요","user_id":"user-123"}'

# TTS
curl -X POST http://localhost:8002/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"테스트입니다.","voice":"alloy"}'
```
