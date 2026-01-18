#!/usr/bin/env python3
"""
off_topic_answer 기능 테스트
- 사용자가 질문을 무시하고 다른 얘기할 때 제대로 감지하는지 확인
"""
import os
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.path.join(backend_dir, "game_server"))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI


def format_prompt(prompt_template: str, **kwargs) -> str:
    """프롬프트 템플릿에 변수를 채웁니다."""
    result = prompt_template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def load_prompts(agent_name: str) -> dict:
    """프롬프트 파일들을 로드"""
    prompts = {}
    prompt_dir = os.path.join(backend_dir, "agents", agent_name, "prompts")

    for filename in os.listdir(prompt_dir):
        if filename.endswith(".txt"):
            key = filename.replace(".txt", "")
            with open(os.path.join(prompt_dir, filename), "r", encoding="utf-8") as f:
                prompts[key] = f.read()

    return prompts


def test_intent_detection_with_llm(analyzer: ChatOpenAI, prompt_template: str, question: str, user_msg: str) -> str:
    """LLM으로 intent 감지 테스트"""
    prompt = format_prompt(
        prompt_template,
        user_message=user_msg,
        current_question=question,
        context="",
        dont_know_count=0,
        next_topic_question=question
    )

    try:
        result = analyzer.invoke(prompt)
        intent = result.content.strip().lower()
        return intent
    except Exception as e:
        logger.error(f"Error: {e}")
        return "error"


def test_friend_agent_off_topic():
    """친구 에이전트 off_topic_answer 테스트"""
    print("\n" + "="*60)
    print("🧪 친구 에이전트 (friend_agent) off_topic_answer 테스트")
    print("="*60)

    prompts = load_prompts("friend_agent")
    analyzer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt_template = prompts.get("detect_intent", "")

    # 테스트 케이스: (질문, 사용자 응답, 예상 intent)
    test_cases = [
        # === off_topic_answer 케이스 (질문 무시) ===
        ("혹시 구매한다면 우려되는 점이 있어?", "돈 벌겠지", "off_topic"),
        ("혹시 구매한다면 우려되는 점이 있어?", "회사가 이익 보겠지뭐", "off_topic"),
        ("혹시 구매한다면 우려되는 점이 있어?", "편리하잖아", "off_topic"),
        ("혹시 구매한다면 우려되는 점이 있어?", "외로운 사람들한테 도움될듯", "off_topic"),
        ("이 서비스가 누군가에게 피해가 될 수 있을까?", "기술이 좋아진거지", "off_topic"),
        ("이 서비스가 누군가에게 피해가 될 수 있을까?", "그냥 편해보여", "off_topic"),
        ("고인의 동의 없이 만들어도 괜찮을까?", "돈만 되면 되지", "off_topic"),
        ("고인의 동의 없이 만들어도 괜찮을까?", "기술 발전이니까", "off_topic"),

        # === answer 케이스 (질문에 직접 답함) ===
        ("혹시 구매한다면 우려되는 점이 있어?", "개인정보 유출이 걱정돼", "answer"),
        ("혹시 구매한다면 우려되는 점이 있어?", "우려되는 점은 없어", "answer"),
        ("혹시 구매한다면 우려되는 점이 있어?", "데이터가 해킹당하면 어떡해", "answer"),
        ("혹시 구매한다면 우려되는 점이 있어?", "별로 걱정 안 돼", "answer"),
        ("이 서비스가 누군가에게 피해가 될 수 있을까?", "유족들이 힘들어할 수도", "answer"),
        ("이 서비스가 누군가에게 피해가 될 수 있을까?", "피해는 없을 것 같아", "answer"),
        ("고인의 동의 없이 만들어도 괜찮을까?", "동의가 필요하지 않을까", "answer"),
        ("고인의 동의 없이 만들어도 괜찮을까?", "괜찮을 것 같아", "answer"),
    ]

    passed = 0
    failed = 0

    for question, user_msg, expected in test_cases:
        print(f"\n📋 질문: \"{question}\"")
        print(f"👤 사용자: \"{user_msg}\"")
        print(f"🎯 예상: {expected}")

        result = test_intent_detection_with_llm(analyzer, prompt_template, question, user_msg)

        # off_topic_answer 또는 off_topic이면 매칭
        is_pass = (expected in result) or (expected == "off_topic" and "off_topic" in result)
        status = "✅ PASS" if is_pass else "❌ FAIL"
        print(f"📊 결과: {result} {status}")

        if is_pass:
            passed += 1
        else:
            failed += 1

    print(f"\n📈 친구 에이전트 결과: {passed}/{passed+failed} 통과")
    return passed, failed


def test_artist_agent_off_topic():
    """화가 지망생 에이전트 off_topic_answer 테스트"""
    print("\n" + "="*60)
    print("🧪 화가 지망생 에이전트 (artist_apprentice_agent) off_topic_answer 테스트")
    print("="*60)

    prompts = load_prompts("artist_apprentice_agent")
    analyzer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt_template = prompts.get("detect_intent", "")

    # 테스트 케이스: (질문, 사용자 응답, 예상 intent)
    test_cases = [
        # === off_topic_answer 케이스 (질문 무시) ===
        ("이 전시가 누군가에게 피해가 될 수 있을까요?", "돈 벌겠지", "off_topic"),
        ("이 전시가 누군가에게 피해가 될 수 있을까요?", "예술관이 이익 보겠지뭐", "off_topic"),
        ("이 전시가 누군가에게 피해가 될 수 있을까요?", "기술 발전이잖아요", "off_topic"),
        ("이 전시가 누군가에게 피해가 될 수 있을까요?", "관람객들이 좋아하겠죠", "off_topic"),
        ("AI 작품이 공정하다고 생각하세요?", "예술관이 돈 벌겠네요", "off_topic"),
        ("AI 작품이 공정하다고 생각하세요?", "신기한 기술이에요", "off_topic"),
        ("작가의 동의 없이 학습해도 괜찮을까요?", "AI가 그린 거 멋있어요", "off_topic"),
        ("작가의 동의 없이 학습해도 괜찮을까요?", "기술이 좋아졌네요", "off_topic"),

        # === answer 케이스 (질문에 직접 답함) ===
        ("이 전시가 누군가에게 피해가 될 수 있을까요?", "기존 화가들이 피해볼 것 같아요", "answer"),
        ("이 전시가 누군가에게 피해가 될 수 있을까요?", "피해는 없을 것 같아요", "answer"),
        ("이 전시가 누군가에게 피해가 될 수 있을까요?", "일자리를 잃는 사람이 있을 수도요", "answer"),
        ("이 전시가 누군가에게 피해가 될 수 있을까요?", "별로 피해볼 사람은 없을 듯해요", "answer"),
        ("AI 작품이 공정하다고 생각하세요?", "공정하지 않아요, 다른 작품 배꼈으니까", "answer"),
        ("AI 작품이 공정하다고 생각하세요?", "공정하다고 생각해요", "answer"),
        ("작가의 동의 없이 학습해도 괜찮을까요?", "동의가 필요하다고 생각해요", "answer"),
        ("작가의 동의 없이 학습해도 괜찮을까요?", "괜찮을 것 같아요", "answer"),
    ]

    passed = 0
    failed = 0

    for question, user_msg, expected in test_cases:
        print(f"\n📋 질문: \"{question}\"")
        print(f"👤 사용자: \"{user_msg}\"")
        print(f"🎯 예상: {expected}")

        result = test_intent_detection_with_llm(analyzer, prompt_template, question, user_msg)

        # off_topic_answer 또는 off_topic이면 매칭
        is_pass = (expected in result) or (expected == "off_topic" and "off_topic" in result)
        status = "✅ PASS" if is_pass else "❌ FAIL"
        print(f"📊 결과: {result} {status}")

        if is_pass:
            passed += 1
        else:
            failed += 1

    print(f"\n📈 화가 지망생 에이전트 결과: {passed}/{passed+failed} 통과")
    return passed, failed


if __name__ == "__main__":
    print("🚀 off_topic_answer 테스트 시작")
    print("="*60)

    friend_passed, friend_failed = test_friend_agent_off_topic()
    artist_passed, artist_failed = test_artist_agent_off_topic()

    total_passed = friend_passed + artist_passed
    total_failed = friend_failed + artist_failed

    print("\n" + "="*60)
    print("📊 최종 결과")
    print("="*60)
    print(f"친구 에이전트: {friend_passed}/{friend_passed+friend_failed}")
    print(f"화가 지망생 에이전트: {artist_passed}/{artist_passed+artist_failed}")
    print(f"총합: {total_passed}/{total_passed+total_failed}")

    if total_failed == 0:
        print("\n🎉 모든 테스트 통과!")
    else:
        print(f"\n⚠️ {total_failed}개 테스트 실패")
