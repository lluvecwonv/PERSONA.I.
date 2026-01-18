# 게임 서버 백엔드 연동 가이드

백엔드 개발자를 위한 게임 서버 API 연동 문서입니다.

## 에피소드별 에이전트 매핑

| 에피소드 | agent_id | 에이전트 이름 | 설명 | 상태 |
|---------|----------|--------------|------|------|
| 1 | `agent-1` | 화가 지망생 (Artist Apprentice) | AI 화가 윤리 대화 - 화가 지망생 에이전트 | Active |
| 2 | `agent-2` | 친구 (Friend Agent) | 죽은 사람을 재현하는 AI에 대한 윤리적 고민 | Active |
| 3 | `agent-3` | 동료 화가 1 (Colleague 1) | 의무론적 관점에서 AI 예술을 반대하는 동료 화가 | Active |
| 4 | `agent-4` | 동료 화가 2 (Colleague 2) | 공리주의 관점에서 AI 예술을 찬성하는 동료 화가 | Active |
| 5 | `agent-5` | TBD | 추후 구현 예정 | Inactive |
| 6 | `agent-6` | TBD | 추후 구현 예정 | Inactive |

---

## API 엔드포인트

### 게임 대화 API
```
POST /api/game/context/{context_id}/chat
```

**서버 주소:**
- 개발: `http://localhost:8002`
- 프로덕션: `http://game-server:8002` (Docker 내부) 또는 `http://16.184.63.106:8002`

---

## 슬롯-에이전트 매핑 구현

NestJS에서 다음과 같이 슬롯별 에이전트를 매핑하세요:

```typescript
// constants/agents.ts
export const EPISODE_AGENT_MAP = {
  1: 'agent-1',  // 화가 지망생
  2: 'agent-2',  // 친구
  3: 'agent-3',  // 동료 화가 (의무론)
  4: 'agent-4',  // 동료 화가 (공리주의)
} as const;

// 타입 정의
export type EpisodeNumber = keyof typeof EPISODE_AGENT_MAP;
export type AgentId = typeof EPISODE_AGENT_MAP[EpisodeNumber];
```

---

## API 호출 예시

### TypeScript/NestJS

```typescript
import axios from 'axios';

interface GameChatRequest {
  agent_id: string;
  message: string;
  user_id: string;
}

interface GameChatResponse {
  status: 'success';
  data: {
    response: string;
    metadata: {
      stage: string;
      message_count: number;
      turn_count: number;
      is_end: boolean;
      is_first: boolean;
    };
  };
}

// 게임 대화 요청
async function sendGameMessage(
  episodeNumber: number,
  contextId: string,
  userId: string,
  message: string
): Promise<GameChatResponse> {
  const agentId = EPISODE_AGENT_MAP[episodeNumber];

  const response = await axios.post<GameChatResponse>(
    `http://game-server:8002/api/game/context/${contextId}/chat`,
    {
      agent_id: agentId,
      message: message,
      user_id: userId
    }
  );

  return response.data;
}
```

### 에피소드별 사용 예시

```typescript
// 에피소드 1: 화가 지망생
const response1 = await axios.post(
  'http://game-server:8002/api/game/context/uuid-here/chat',
  {
    agent_id: 'agent-1',
    message: '안녕하세요',
    user_id: 'user-uuid'
  }
);

// 에피소드 2: 친구
const response2 = await axios.post(
  'http://game-server:8002/api/game/context/uuid-here/chat',
  {
    agent_id: 'agent-2',
    message: '안녕하세요',
    user_id: 'user-uuid'
  }
);

// 에피소드 3: 동료 화가 1 (의무론)
const response3 = await axios.post(
  'http://game-server:8002/api/game/context/uuid-here/chat',
  {
    agent_id: 'agent-3',
    message: '안녕하세요',
    user_id: 'user-uuid'
  }
);

// 에피소드 4: 동료 화가 2 (공리주의)
const response4 = await axios.post(
  'http://game-server:8002/api/game/context/uuid-here/chat',
  {
    agent_id: 'agent-4',
    message: '안녕하세요',
    user_id: 'user-uuid'
  }
);
```

---

## 요청 스키마

```json
{
  "agent_id": "string",     // 필수: "agent-1" ~ "agent-4"
  "message": "string",      // 필수: 사용자 메시지
  "user_id": "string"       // 필수: 사용자 UUID (토큰에서 추출)
}
```

---

## 응답 스키마

### 성공 응답 (200)

```json
{
  "status": "success",
  "data": {
    "response": "AI 에이전트의 응답 메시지",
    "metadata": {
      "stage": "stage1",
      "message_count": 5,
      "turn_count": 3,
      "is_end": false,
      "is_first": false
    }
  }
}
```

