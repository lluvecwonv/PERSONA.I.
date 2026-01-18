# 게임 서버 개발 가이드

AI 윤리 대화 게임의 FastAPI 기반 게임 서버입니다. NestJS 백엔드와 연동하여 에이전트 대화를 처리합니다.

---

## 프로젝트 구조

```
game_server/
├── server.py                 # FastAPI 서버 메인 파일
├── langchain_service.py      # LangChain 서비스 (에이전트 통합)
├── config.py                 # 설정 관리
├── exceptions.py             # 커스텀 예외 클래스
├── utils.py                  # 유틸리티 함수
├── requirements.txt          # Python 의존성
├── Dockerfile               # Docker 이미지 빌드
├── README.md                # 이 문서
└── BACKEND_INTEGRATION.md   # 백엔드 연동 가이드
```

---

## 개발 환경 설정

### 1. Python 가상환경 생성

```bash
cd backend/game_server
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env` 파일을 생성하거나 환경 변수를 설정합니다:

```bash
OPENAI_API_KEY=your-openai-api-key
PORT=8002
HOST=0.0.0.0
```

---

## 서버 실행

### 개발 모드

```bash
python server.py
```

또는 uvicorn으로 직접 실행:

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8002
```

### 프로덕션 모드

```bash
uvicorn server:app --host 0.0.0.0 --port 8002
```

서버는 `http://localhost:8002`에서 실행됩니다.

---

## API 엔드포인트

### 헬스 체크

```
GET /health
```

서버 상태 확인용 엔드포인트입니다.

**응답:**
```json
{
  "status": "ok"
}
```

### 게임 대화

```
POST /api/game/context/{context_id}/chat
```

사용자 메시지를 받아 AI 에이전트의 응답을 반환합니다.

**Path Parameters:**
- `context_id`: 대화 컨텍스트 ID (UUID)

**Request Body:**
```json
{
  "agent_id": "agent-1",
  "message": "사용자 메시지",
  "user_id": "user-uuid"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "response": "AI 응답 메시지",
    "metadata": {
      "stage": "stage1",
      "message_count": 1,
      "turn_count": 1,
      "is_end": false,
      "is_first": false
    }
  }
}
```

---

## 에이전트 목록

게임 서버에서 사용 가능한 에이전트 목록입니다.

| Agent ID | 이름 | 모델명 | 설명 | 상태 | Voice |
|----------|------|--------|------|------|-------|
| `agent-1` | Artist Apprentice | `artist-apprentice` | AI 화가 윤리 대화 게임 - 화가 지망생 에이전트 | Active | alloy |
| `agent-2` | Friend Agent | `friend-agent` | AI 윤리 대화 게임 - 친구 에이전트 (죽은 사람을 재현하는 AI) | Active | nova |
| `agent-3` | Colleague 1 (동료 화가) | `colleague1` | AI 화가 윤리 대화 게임 - 의무론적 관점에서 AI 예술을 반대하는 동료 화가 | Active | shimmer |
| `agent-4` | Colleague 2 (동료 화가) | `colleague2` | AI 화가 윤리 대화 게임 - 공리주의 관점에서 AI 예술을 찬성하는 동료 화가 | Active | echo |
| `agent-5` | Jangmo (장모) | `jangmo` | AI 복원 윤리 대화 게임 - 의무론적 관점에서 AI 복원을 반대하는 장모 | Active | fable |
| `agent-6` | Son (아들) | `son` | AI 복원 윤리 대화 게임 - 공리주의 관점에서 AI 복원을 찬성하는 아들 | Active | onyx |

### 에이전트 상세 정보

#### agent-1: Artist Apprentice (화가 지망생)
- **시나리오**: AI 화가 윤리 대화
- **대화 단계**: 
  - Stage 1: 슬럼프 공감
  - Stage 2: AI 도구 제안
  - Stage 3: 선택 탐색
- **특징**: 자연스러운 한국어 대화, 감정 공감 중심 응답

#### agent-2: Friend Agent (친구)
- **시나리오**: 죽은 사람을 재현하는 AI 윤리 고민
- **대화 단계**:
  - Stage 1: 상황 공유
  - Stage 2: 의견 수렴
  - Stage 3: 깊은 탐구
- **특징**: 사용자 의도 파악 기반 응답 (긍정/부정/중립에 맞춰 일치하는 반응)

#### agent-3: Colleague 1 (동료 화가 - 의무론)
- **윤리 관점**: 의무론 (Deontology)
- **입장**: AI 예술 반대
- **핵심 원칙**: 예술의 본질, 인간성, 창작 과정의 가치 강조
- **모델**: Fine-tuned GPT-4.1-mini

#### agent-4: Colleague 2 (동료 화가 - 공리주의)
- **윤리 관점**: 공리주의 (Utilitarianism)
- **입장**: AI 예술 찬성
- **핵심 원칙**: 최대 다수의 최대 행복, 접근성, 효율성 강조
- **모델**: Fine-tuned GPT-4.1-mini

#### agent-5: Jangmo (장모 - 의무론)
- **시나리오**: 죽은 사람을 AI로 재현하는 기술 윤리 대화
- **윤리 관점**: 의무론 (Deontology)
- **입장**: AI 복원 기술 반대
- **캐릭터**: 일 년 전 딸을 잃은 여성 노인, 사위(플레이어)의 장모
- **핵심 원칙**: 죽은 사람의 동의 없이 복원하는 것의 윤리적 문제 강조
- **특징**: 노인 여성이 사위에게 쓰는 반말, 2-3문장 간결한 응답
- **모델**: Fine-tuned GPT-5-mini

#### agent-6: Son (아들 - 공리주의)
- **시나리오**: 죽은 사람을 AI로 재현하는 기술 윤리 대화
- **윤리 관점**: 공리주의 (Utilitarianism)
- **입장**: AI 복원 기술 찬성
- **캐릭터**: 일 년 전 어머니를 잃은 20대 청년, 아버지(플레이어)의 아들
- **핵심 원칙**: 가족의 행복과 이익 극대화 강조
- **특징**: 20대 청년이 아버지에게 쓰는 반말, 2-3문장 간결한 응답
- **모델**: Fine-tuned GPT-5-mini

---

## 에이전트 통합

### 에이전트 등록

`server.py`의 `AGENTS` 딕셔너리에 에이전트를 등록합니다:

```python
AGENTS = {
    "agent-1": {
        "id": "agent-1",
        "name": "Artist Apprentice",
        "model": "artist-apprentice",
        "description": "AI 화가 윤리 대화 게임 - 화가 지망생 에이전트",
        "voice": "alloy",
        "status": "active"
    },
    # ... 추가 에이전트
}
```

### LangChain 서비스에 에이전트 추가

`langchain_service.py`에서 에이전트를 초기화하고 `chat()` 메서드에 통합합니다:

1. **에이전트 클래스 import**
```python
from agents.your_agent.conversation_agent import ConversationAgent as YourAgent
```

2. **에이전트 인스턴스 생성**
```python
self.your_agent = YourAgent(api_key=api_key, model="gpt-5-mini-2025-08-07")
```

3. **chat() 메서드에서 모델명으로 라우팅**
```python
if model == "your-agent":
    agent = self.your_agent
```

---

## 음성(TTS) API

텍스트를 음성으로 변환하는 별도 API입니다. 채팅 응답 후 TTS 버튼을 눌렀을 때 호출합니다.

### 요청

```bash
POST /api/tts
Content-Type: application/json

{
  "text": "변환할 텍스트",
  "voice": "alloy",
  "model": "tts-1"
}
```

### 파라미터

| 파라미터 | 필수 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | O | - | 변환할 텍스트 (최대 4096자) |
| `voice` | X | `alloy` | 음성 종류: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `model` | X | `tts-1` | TTS 모델: `tts-1` (빠름), `tts-1-hd` (고품질) |

### 응답

```json
{
  "status": "success",
  "audio": "base64_encoded_mp3_data...",
  "format": "mp3",
  "voice": "alloy",
  "model": "tts-1"
}
```

### 빠른 테스트

```bash
uvicorn backend.game_server.server:app --reload --port 8002

# TTS API 테스트
curl -X POST http://localhost:8002/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"안녕하세요, 테스트입니다.","voice":"alloy"}'
```


### 채팅 응답에 음성 포함 (레거시)

채팅 API에서 `include_audio` 플래그를 사용하면 응답에 음성이 포함됩니다.
단, 응답 크기가 커질 수 있으므로 별도 TTS API 사용을 권장합니다.

```bash
curl -X POST http://localhost:8002/api/game/context/test/chat \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agent-1","message":"테스트","user_id":"u1","include_audio":true}'
```

SSE 스트리밍 모드에서는 마지막 이벤트에 `{"done":true,"audio":"..."}`가 포함됩니다.

---

## 세션 관리

게임 서버는 Open WebUI와 완전히 분리된 세션 저장소를 사용합니다.

### 세션 저장소 구조

- **Open WebUI 세션**: `self.sessions_openwebui` (Open WebUI 전용)
- **게임 서버 세션**: `self.sessions_game` (게임 서버 전용)

### Context ID 기반 세션

NestJS에서 전달하는 `context_id`를 세션 ID로 사용합니다:
- 각 대화 컨텍스트마다 고유한 UUID 사용
- `context_id`를 그대로 세션 ID로 사용하여 대화 상태 관리
- 여러 사용자가 동시에 여러 대화를 진행해도 각각 격리되어 관리

### 세션 초기화

새로운 `context_id`가 들어오면 자동으로 새 세션을 생성합니다:

```python
if session_id not in session_store:
    session_store[session_id] = self._create_new_session()
```

---

## 에러 처리

### 커스텀 예외

`exceptions.py`에 정의된 예외 클래스를 사용합니다:

- `APIKeyNotFoundError`: OpenAI API 키가 없을 때
- `ChainExecutionError`: LangChain 실행 중 오류 발생 시

### HTTP 예외

FastAPI의 `HTTPException`을 사용하여 적절한 HTTP 상태 코드를 반환합니다:

```python
if request.agent_id not in AGENTS:
    raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
```

---

## 로깅

로깅은 Python의 `logging` 모듈을 사용합니다:

```python
logger.info(f"Chat: user={request.user_id}, context={context_id}, agent={request.agent_id}")
logger.error(f"Error in chat message: {str(e)}", exc_info=True)
```

로그 레벨은 `INFO`로 설정되어 있으며, 에러 발생 시 전체 스택 트레이스를 기록합니다.

---

## 개발 팁

### 1. 에이전트 테스트

개발 중에는 직접 API를 호출하여 테스트할 수 있습니다:

```bash
curl -X POST "http://localhost:8002/api/game/context/test-123/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent-1",
    "message": "안녕하세요",
    "user_id": "user-123"
  }'
```

### 2. 디버깅

- `--reload` 옵션으로 코드 변경 시 자동 재시작
- 로그를 통해 요청/응답 흐름 추적
- `langchain_service.py`의 `chat()` 메서드에 브레이크포인트 설정

### 3. 에이전트 모델명 매핑

에이전트의 `model` 필드와 `langchain_service.py`의 모델명이 일치해야 합니다:

- `AGENTS["agent-1"]["model"]` = `"artist-apprentice"`
- `langchain_service.py`에서 `if model == "artist-apprentice"`로 라우팅

---

## Docker 배포

### Dockerfile 빌드

```bash
docker build -t game-server .
```

### Docker 실행

```bash
docker run -p 8002:8002 \
  -e OPENAI_API_KEY=your-key \
  -e PORT=8002 \
  game-server
```
