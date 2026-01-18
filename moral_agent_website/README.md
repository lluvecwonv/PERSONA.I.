# AI 윤리 대화 에이전트 개발 가이드

LangGraph 기반 상태 머신으로 구현된 대화형 AI 에이전트 개발 문서입니다.

---

## 에이전트 아키텍처

### LangGraph 기반 상태 머신

모든 에이전트는 LangGraph의 StateGraph를 사용하여 대화 흐름을 관리합니다:
- **State**: 대화 상태를 담는 딕셔너리 (messages, current_stage, turn_count, is_end 등)
- **Node**: 각 대화 단계를 처리하는 핸들러 함수
- **Edge**: 단계 간 전환 경로 (순차적 또는 조건부)
- **Conditional Edge**: 조건에 따라 다음 단계를 결정하는 함수

### 에이전트 구조

```
agent/
├── conversation_agent.py    # LangGraph 기반 대화 엔진
├── prompts/
│   ├── system_prompt.txt    # 시스템 프롬프트
│   ├── stage1_*.txt         # 단계별 프롬프트
│   ├── stage2_*.txt
│   └── stage3_*.txt
├── utils.py                 # 유틸리티 함수 (응답 필터링 등)
└── README.md
```

---

## 구현된 에이전트

### 1. Artist Apprentice (화가 지망생) - agent-1

**모델**: GPT-4o-mini (기본 모델)
**경로**: `backend/artist_apprentice_agent/`
**시나리오**: AI 화가 윤리 대화

#### 대화 단계

1. **Stage 1 (슬럼프 공감)**: 화가 지망생의 슬럼프 상황 공유
2. **Stage 2 (AI 도구 제안)**: 친구가 AI 그림 도구 제안
3. **Stage 3 (선택 탐색)**: 사용자의 선택과 이유 탐구

#### 주요 특징

- 자연스러운 한국어 대화
- 감정 공감 중심 응답
- 선택에 대한 이유 탐구 (질문만, 조언 금지)

---

### 2. Friend Agent (친구) - agent-2

**모델**: GPT-4o-mini (기본 모델)
**경로**: `backend/friend_agent/`
**시나리오**: 죽은 사람을 재현하는 AI 윤리 고민

#### 대화 단계

1. **Stage 1 (상황 공유)**: 죽은 할아버지를 AI로 재현하는 서비스 소개
2. **Stage 2 (의견 수렴)**: 사용자 의견에 대한 공감적 응답
3. **Stage 3 (깊은 탐구)**: 선택의 근거 탐색

#### 핵심 구현: 의도 파악 기반 응답

`stage2_acknowledgment.txt` 프롬프트에서 사용자의 의도를 먼저 파악한 후 같은 방향으로 응답합니다:

```
1. 의도 파악 (긍정/부정/중립)
   - "도움이 될 것 같아" = 긍정
   - "걱정돼", "위험해" = 부정
   - "모르겠어" = 중립

2. 상황 인정
   - 사용자가 말한 내용을 있는 그대로 받아들이기

3. 일치하는 반응
   - 긍정 → 긍정 응답
   - 부정 → 부정 응답
   - 절대 반대 방향 응답 금지
```

**문제 사례**:
```
사용자: "도움이 될 것 같아"  (긍정)
잘못된 응답: "걱정이 될 수 있을 것 같아"  (부정) ← 의도 불일치!
올바른 응답: "그렇구나, 도움이 될 수 있겠다" (긍정) ← 의도 일치
```

---

### 3. Colleague 1 (동료 화가 - 의무론) - agent-3

**모델**: Fine-tuned GPT-4.1-mini
**윤리 관점**: 의무론 (Deontology)

#### Fine-tuning 전략

- **방법**: Direct Preference Optimization (DPO)
- **데이터**: 의무론적 관점의 선호/비선호 응답 쌍
- **목표**: AI 예술을 반대하는 의무론적 입장 강화
- **핵심 원칙**: 예술의 본질, 인간성, 창작 과정의 가치 강조


#### 시스템 프롬프트 특징

```
규칙:
- "현재" 단어 절대 사용 금지
- 의무론적 관점에서 AI 예술 반대
- 예술의 본질과 인간성 강조
- 자연스러운 일상 한국어 사용
```

---

### 4. Colleague 2 (동료 화가 - 공리주의) - agent-4

**모델**: Fine-tuned GPT-4.1-mini
**Fine-tuned Model ID**: `ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs`
**경로**: `backend/colleague2_agent/`
**윤리 관점**: 공리주의 (Utilitarianism)

#### Fine-tuning 전략

- **방법**: Direct Preference Optimization (DPO)
- **데이터**: 공리주의적 관점의 선호/비선호 응답 쌍
- **목표**: AI 예술을 찬성하는 공리주의적 입장 강화
- **핵심 원칙**: 최대 다수의 최대 행복, 접근성, 효율성 강조

#### 시스템 프롬프트 특징

```
규칙:
- 공리주의적 관점에서 AI 예술 찬성
- 최대 다수의 최대 행복 강조
- 예술의 접근성과 효율성 중시
- 자연스러운 일상 한국어 사용
```

---

## Fine-tuning 상세

### DPO (Direct Preference Optimization)

Fine-tuning은 OpenAI의 Fine-tuning API를 사용하여 DPO 방식으로 진행되었습니다.

#### 학습 데이터 구조

DPO 학습 데이터는 다음 형식으로 구성됩니다:
- **prompt**: 사용자 메시지
- **chosen**: 선호하는 응답 (해당 윤리적 관점에 부합하는 응답)
- **rejected**: 비선호 응답 (해당 윤리적 관점에 부합하지 않는 응답)

#### Fine-tuning 파라미터

- **Base Model**: `gpt-4.1-mini-2025-04-14`
- **Method**: DPO (Direct Preference Optimization)
- **Organization**: idl-lab
- **Model Version**: moral-agent-v1
- **Model ID**: `ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs`

### 알려진 이슈 및 해결

#### "현재" 단어 반복 문제

**원인**: Fine-tuned 모델의 버그
**증상**: "현재" 단어가 응답에 반복적으로 출력

**해결 방법**:
1. **시스템 프롬프트에 금지 명시**: 프롬프트에 "현재" 단어 사용 금지 규칙 추가
2. **utils.py에서 정규식 필터링**: `clean_gpt_response()` 함수로 응답 후처리
3. **응답 전송 전 실시간 제거**: `conversation_agent.py`의 `process()` 함수에서 필터링 적용

---

## 공통 구현 패턴

### 1. 프롬프트 외부화

모든 프롬프트는 텍스트 파일(`prompts/` 디렉토리)로 관리합니다. 이렇게 하면:
- 코드 수정 없이 프롬프트만 변경 가능
- 버전 관리가 용이함
- 각 에이전트별로 독립적인 프롬프트 관리 가능

각 에이전트 디렉토리 내 `prompts/` 폴더에 단계별 프롬프트 파일을 저장하고, `load_prompt()` 함수로 로드합니다.

### 2. 단계별 핸들러

각 대화 단계(stage)를 별도의 핸들러 함수로 분리합니다:
- **stage1_handler**: 첫 단계 (상황 소개, 인사 등)
- **stage2_handler**: 두 번째 단계 (의견 수렴, 질문 등)
- **stage3_handler**: 세 번째 단계 (심화 토론, 결론 등)

각 핸들러는 현재 상태(state)를 받아서 다음 상태를 반환하며, 턴 수와 현재 단계 정보를 업데이트합니다.

### 3. 조건부 전환

대화 단계 전환은 조건부 함수로 제어합니다:
- **턴 수 기반**: 일정 턴 수가 지나면 다음 단계로 자동 전환
- **키워드 기반**: 사용자가 특정 키워드("종료", "끝" 등)를 입력하면 대화 종료
- **의도 감지**: LLM이 사용자의 의도를 분석하여 단계 전환 여부 결정

LangGraph의 `add_conditional_edges`를 사용하여 각 노드에서 다음 노드로의 전환 조건을 정의합니다.

---
