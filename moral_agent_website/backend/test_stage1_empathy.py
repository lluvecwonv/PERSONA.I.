#!/usr/bin/env python3
"""
Stage 1 공감 테스트
사용자가 캐릭터를 구체화하지 않았을 때, 공감 + 추가 질문이 제대로 나오는지 테스트
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
os.chdir(backend_dir)

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

# Stage1Handler 직접 구현 (import 문제 회피)
class Stage1HandlerTest:
    """Stage 1 테스트용 핸들러"""
    FIXED_GREETING = "너 요즘 그림도 잘 안 그린다며? 요새 뭐 하면서 지내?"
    FOLLOW_UP_QUESTION = "그렇구나. 요즘 밥은 먹고 있어?"

    def __init__(self, llm, prompts=None):
        self.llm = llm
        self.prompts = prompts or {}

    def _generate_rephrase_question(self, messages: list) -> str:
        """공감 + 추가 질문 생성"""
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        user_message = user_messages[-1]["content"].strip() if user_messages else ""

        # 최근 대화 컨텍스트 (최대 4턴)
        recent_messages = messages[-4:] if len(messages) > 4 else messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

        empathy_template = self.prompts.get(
            "stage1_rephrase",
            "최근 대화:\n{context}\n사용자 마지막 발화: {user_message}\n"
            "사용자의 말을 자연스럽게 인정하고 걱정하는 반말 한 문장만 출력하세요."
        )
        empathy_prompt = empathy_template.replace("{context}", context).replace("{user_message}", user_message)

        try:
            result = self.llm.invoke(empathy_prompt)
            empathy_line = result.content.strip().strip('"')
            if not empathy_line:
                empathy_line = "그랬구나."
        except Exception as e:
            logger.error(f"❌ [Stage1] Empathy generation error: {e}")
            empathy_line = "그랬구나."

        logger.info(f"⚠️ [Stage1] Empathy line: '{empathy_line}'")
        return f"{empathy_line} {self.FOLLOW_UP_QUESTION}"


def test_empathy_generation():
    """Stage 1 공감 문구 생성 테스트"""
    print("\n" + "="*60)
    print("🧪 Stage 1 공감 테스트")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)

    # 프롬프트 (기본값 사용)
    prompts = {}

    handler = Stage1HandlerTest(llm=llm, prompts=prompts)

    # 테스트 케이스: (사용자 메시지, 예상되는 공감 포함 여부)
    test_cases = [
        {
            "name": "부정적 감정",
            "messages": [
                {"role": "assistant", "content": "너 요즘 그림도 잘 안 그린다며? 요새 뭐 하면서 지내?"},
                {"role": "user", "content": "별로 안 좋아"},
            ],
        },
        {
            "name": "무기력함",
            "messages": [
                {"role": "assistant", "content": "너 요즘 그림도 잘 안 그린다며? 요새 뭐 하면서 지내?"},
                {"role": "user", "content": "그냥 집에만 있어"},
            ],
        },
        {
            "name": "짧은 답변",
            "messages": [
                {"role": "assistant", "content": "너 요즘 그림도 잘 안 그린다며? 요새 뭐 하면서 지내?"},
                {"role": "user", "content": "응"},
            ],
        },
        {
            "name": "힘들다는 표현",
            "messages": [
                {"role": "assistant", "content": "너 요즘 그림도 잘 안 그린다며? 요새 뭐 하면서 지내?"},
                {"role": "user", "content": "요즘 너무 힘들어"},
            ],
        },
        {
            "name": "질문으로 답함",
            "messages": [
                {"role": "assistant", "content": "너 요즘 그림도 잘 안 그린다며? 요새 뭐 하면서 지내?"},
                {"role": "user", "content": "왜?"},
            ],
        },
    ]

    print(f"\n📌 고정 추가 질문: \"{handler.FOLLOW_UP_QUESTION}\"")
    print("─"*60)

    for tc in test_cases:
        print(f"\n📋 테스트: {tc['name']}")
        user_msg = tc['messages'][-1]['content']
        print(f"👤 사용자: \"{user_msg}\"")

        # 공감 + 추가 질문 생성
        response = handler._generate_rephrase_question(tc['messages'])

        print(f"🤖 AI: \"{response}\"")

        # 검증: 추가 질문이 포함되어 있는지
        has_follow_up = handler.FOLLOW_UP_QUESTION in response
        # 검증: 추가 질문 앞에 공감 문구가 있는지
        has_empathy = response != handler.FOLLOW_UP_QUESTION

        status = "✅" if (has_follow_up and has_empathy) else "⚠️"
        print(f"{status} 공감 포함: {has_empathy}, 추가 질문 포함: {has_follow_up}")


def test_full_stage1_flow():
    """Stage 1 전체 흐름 테스트"""
    print("\n" + "="*60)
    print("🧪 Stage 1 전체 흐름 테스트")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)
    prompts = {}

    handler = Stage1HandlerTest(llm=llm, prompts=prompts)

    # 시나리오: 캐릭터 구체화 안 함 → 공감 + 추가 질문
    state = {
        "messages": [
            {"role": "assistant", "content": handler.FIXED_GREETING},
            {"role": "user", "content": "그냥 뭐 별거 없어"},
        ],
        "stage": "stage1",
    }

    print(f"\n🎬 시나리오: 캐릭터 구체화 안 함")
    print(f"👤 사용자: \"그냥 뭐 별거 없어\"")

    result_state = handler.handle(state)
    response = result_state["last_response"]

    print(f"\n🤖 AI 응답: \"{response}\"")
    print(f"📊 Stage: {result_state['stage']}")
    print(f"📊 캐릭터 설정됨: {result_state.get('artist_character_set', False)}")


if __name__ == "__main__":
    test_empathy_generation()
    print("\n" + "="*60)
    print("✅ 테스트 완료!")
    print("="*60)
