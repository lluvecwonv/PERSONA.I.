# Open WebUI 서버

AI 윤리 대화 게임의 Open WebUI 백엔드 서버입니다.

## 개요

- **포트**: 8001
- **용도**: Open WebUI 프론트엔드와 연동하여 대화형 AI 에이전트 제공
- **지원 모델**:
  - `moral-agent-spt`: DPO Fine-tuned 도덕적 에이전트
  - `moral-agent-gpt`: GPT Fine-tuned 도덕적 대화 에이전트
  - `artist-apprentice`: 화가 지망생 에이전트
  - `friend-agent`: 친구 에이전트 (죽은 사람을 재현하는 AI)
  - `colleague1`: 동료 화가 (의무론적 관점, AI 예술 반대)
  - `colleague2`: 동료 화가 (공리주의 관점, AI 예술 찬성)
  - `jangmo`: 장모 (의무론적 관점, AI 복원 반대)
  - `son`: 아들 (공리주의 관점, AI 복원 찬성)

## 실행 방법

### 1. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 입력하세요:

```bash
OPENAI_API_KEY=your-openai-api-key
DATABASE_URL=postgresql://person_ai_admin:1234@postgres:5432/person_ai
```

### 2. 로컬 실행

```bash
# Python 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python server.py
```

서버는 `http://localhost:8001`에서 실행됩니다.

### 3. Docker 실행

```bash
# Docker 이미지 빌드
docker build -t moral-agent-openwebui .

# Docker 컨테이너 실행
docker run -d \
  --name openai-backend \
  -p 8001:8001 \
  -e OPENAI_API_KEY=your-openai-api-key \
  -e DATABASE_URL=postgresql://person_ai_admin:1234@postgres:5432/person_ai \
  moral-agent-openwebui
```

## API 엔드포인트

### OpenAI 호환 API

- `GET /v1/models` - 사용 가능한 모델 목록 조회
- `POST /v1/chat/completions` - 채팅 완료 (OpenAI 호환)

### 헬스 체크

- `GET /health` - 서버 상태 확인

### 대화 히스토리

- `GET /api/history/{session_id}` - 세션 대화 기록 조회

## 프로젝트 구조

```
openwebui_server/
├── server.py                    # FastAPI 서버 메인 파일
├── langchain_service.py         # LangChain 서비스 로직
├── config.py                    # 설정 파일
├── db/                          # 데이터베이스 모델 및 서비스
├── artist_apprentice_agent/     # 화가 지망생 에이전트
├── friend_agent/                # 친구 에이전트
├── spt_agent/                   # SPT 에이전트
├── colleague1_agent/            # 동료 화가 1 (의무론)
├── colleague2_agent/            # 동료 화가 2 (공리주의)
├── requirements.txt             # Python 의존성
├── Dockerfile                   # Docker 이미지 빌드 파일
└── README.md                    # 이 문서
```

## 설정

### 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | - |
| `DATABASE_URL` | PostgreSQL 데이터베이스 URL | `postgresql://...` |
| `HOST` | 서버 호스트 | `0.0.0.0` |
| `PORT` | 서버 포트 | `8001` |

## Open WebUI 연동

Open WebUI 프론트엔드와 연동하려면 다음 환경 변수를 설정하세요:

```bash
# Open WebUI 컨테이너
OPENAI_API_BASE_URL=http://openai-backend:8001/v1
OPENAI_API_KEY=sk-dummy
```

## 주의사항

- Open WebUI 전용 세션 저장소를 사용합니다 (게임 서버와 격리)
- 대화 기록은 PostgreSQL 데이터베이스에 저장됩니다
- Fine-tuned 모델 사용 시 `ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs` 모델이 필요합니다

## 문제 해결

### 모델이 Open WebUI에서 보이지 않을 때

1. 서버가 정상적으로 실행되고 있는지 확인:
   ```bash
   curl http://localhost:8001/health
   ```

2. 모델 목록 확인:
   ```bash
   curl http://localhost:8001/v1/models
   ```

3. Open WebUI가 올바른 백엔드 URL을 사용하는지 확인

### 데이터베이스 연결 오류

- `DATABASE_URL`이 올바른지 확인
- PostgreSQL 서버가 실행 중인지 확인
- 방화벽 설정 확인

## 라이센스

이 프로젝트는 연구 및 교육 목적으로 사용됩니다.
