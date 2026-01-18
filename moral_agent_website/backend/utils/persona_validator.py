"""
페르소나 검증 및 수정 유틸리티
colleague1, colleague2, jangmo, son 에이전트의 응답이 페르소나에 맞는지 검증하고 필요시 수정

평가 기준:
- 윤리적 입장 (의무론/공리주의)
- 말투 (존댓말/반말)
- SPT 질문 전략 사용
- 책임 상기
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 프롬프트 파일 경로
BACKEND_DIR = Path(__file__).parent.parent
PROMPT_PATHS = {
    "colleague1": BACKEND_DIR / "agents" / "colleague1_agent" / "prompts" / "ai_artist_deontological.txt",
    "colleague2": BACKEND_DIR / "agents" / "colleague2_agent" / "prompts" / "ai_artist_utilitarian.txt",
    "jangmo": BACKEND_DIR / "agents" / "jangmo_agent" / "prompts" / "jangmo_deontological.txt",
    "son": BACKEND_DIR / "agents" / "son_agent" / "prompts" / "son_utilitarian.txt",
}


def load_persona_prompt(agent_type: str) -> str:
    """에이전트별 페르소나 프롬프트 파일 로드"""
    prompt_path = PROMPT_PATHS.get(agent_type)
    if prompt_path and prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to load prompt for {agent_type}: {e}")
    return ""


# 에이전트별 페르소나 설정 (프롬프트 파일 기반)
PERSONA_RULES = {
    "colleague1": {
        "name": "동료1 (50대 여성 선배 화가)",
        "tone": "반말",  # 프롬프트: "친한 친구 사이의 반말", "~하나?", "~한가?" 스타일
        "tone_target": "후배 동료",
        "stance": "의무론 - AI 전시 반대",
        "stance_keywords": ["규칙", "원칙", "의무", "기준", "도덕", "자격", "책임"],
        "tone_markers": ["하나", "한가", "군", "네", "지", "야"],  # 반말 질문형
        "wrong_tone_markers": ["요", "습니다", "세요", "까요"],  # 존댓말은 잘못됨
        "question_style": "~하나?, ~한가?",  # 프롬프트에서 명시된 질문 스타일
        "rules": [
            "친한 친구 사이의 반말 사용 (해, 야, 지)",
            "질문할 때 '~하나?', '~한가?' 스타일 사용",
            "의무론적 관점 (규칙, 원칙, 도덕적 의무 강조)",
            "AI 작품 전시에 반대하는 입장",
            "'의무론', '공리주의' 단어 사용 금지",
        ]
    },
    "colleague2": {
        "name": "동료2 (30대 남성 후배 화가)",
        "tone": "존댓말",
        "tone_target": "선생님",
        "stance": "책임 중심 - AI 전시 찬성",
        "stance_keywords": ["이익", "결과", "행복", "전체", "많은 사람", "책임"],
        "tone_markers": ["요", "습니다", "세요", "까요", "선생님"],
        "wrong_tone_markers": ["야", "해", "냐", "하나", "한가"],
        "rules": [
            "선생님에게 존댓말 사용 (~요, ~습니다)",
            "'선생님' 호칭 사용",
            "책임 중심 관점 (전체 이익, 결과 중시)",
            "AI 작품 전시에 찬성하는 입장",
            "'의무론', '공리주의' 단어 사용 금지",
        ]
    },
    "jangmo": {
        "name": "장모 (노인 여성)",
        "tone": "반말",
        "tone_target": "사위",
        "stance": "의무론 - AI 부활 반대",
        "stance_keywords": ["존엄", "도리", "원칙", "지켜야", "선", "동의"],
        "tone_markers": ["야", "해", "지", "어", "아", "거든"],
        "wrong_tone_markers": ["요", "습니다", "세요", "니다", "시면", "께서"],
        "rules": [
            "사위에게 반말 사용 (해, 야, 지)",
            "의무론적 관점 (존엄성, 도리, 원칙 강조)",
            "AI 부활 서비스에 반대하는 입장",
            "'의무론', '공리주의' 단어 사용 금지",
        ]
    },
    "son": {
        "name": "아들 (20대 초반 남성)",
        "tone": "존댓말",
        "tone_target": "아버지",
        "stance": "책임 중심 - AI 부활 찬성",
        "stance_keywords": ["행복", "위안", "가족", "좋", "덜 슬프", "책임"],
        "tone_markers": ["요", "습니다", "아버지", "세요", "께서"],
        "wrong_tone_markers": ["해봐", "냐"],  # "야", "해", "지"는 존댓말에도 포함될 수 있음 (생각해요 등)
        "rules": [
            "아버지에게 존댓말 사용 (~요, ~습니다)",
            "'엄마' 또는 '우리 엄마' 호칭 사용",
            "책임 중심 관점 (가족의 행복, 위안 중시)",
            "AI 부활 서비스에 찬성하는 입장",
            "'의무론', '공리주의' 단어 사용 금지",
        ]
    },
}


class PersonaValidator:
    """페르소나 검증 및 수정 클래스"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.3)

    def check_tone(self, response: str, agent_type: str) -> Tuple[bool, str]:
        """
        말투가 페르소나에 맞는지 간단히 체크

        Returns:
            (맞음 여부, 이유)
        """
        persona = PERSONA_RULES.get(agent_type)
        if not persona:
            return True, "unknown_agent"

        response_end = response[-20:] if len(response) > 20 else response

        # 올바른 말투 마커 확인
        has_correct = any(marker in response for marker in persona["tone_markers"])

        # 잘못된 말투 마커 확인 (문장 끝에서)
        has_wrong = any(marker in response_end for marker in persona["wrong_tone_markers"])

        if persona["tone"] == "존댓말":
            if has_correct and not has_wrong:
                return True, "correct_formal"
            elif has_wrong:
                return False, "should_be_formal"
            else:
                return False, "missing_formal_markers"
        else:  # 반말
            if has_wrong:  # wrong_tone_markers가 실제로는 존댓말 마커
                return False, "should_be_informal"
            else:
                return True, "correct_informal"

    async def fix_response(self, response: str, agent_type: str) -> str:
        """
        페르소나에 맞지 않는 응답을 수정
        프롬프트 파일을 활용하여 더 정확한 수정

        Args:
            response: 원본 응답
            agent_type: 에이전트 유형

        Returns:
            수정된 응답
        """
        persona = PERSONA_RULES.get(agent_type)
        if not persona:
            return response

        # 프롬프트 파일에서 핵심 스타일 규칙 추출
        persona_prompt = load_persona_prompt(agent_type)
        style_excerpt = ""
        if persona_prompt:
            # 스타일 관련 부분만 추출 (최대 500자)
            if "style:" in persona_prompt.lower():
                style_start = persona_prompt.lower().find("style:")
                style_excerpt = persona_prompt[style_start:style_start+500]
            elif "**style" in persona_prompt.lower():
                style_start = persona_prompt.lower().find("**style")
                style_excerpt = persona_prompt[style_start:style_start+500]

        fix_prompt = f"""다음 응답을 페르소나에 맞게 수정해주세요.

## 원본 응답
{response}

## 페르소나
- 캐릭터: {persona['name']}
- 말투: {persona['tone']} ({persona['tone_target']}에게)
- 입장: {persona['stance']}
- 규칙:
{chr(10).join([f"  - {r}" for r in persona['rules']])}

{f"## 스타일 참고 (프롬프트에서 발췌){chr(10)}{style_excerpt}" if style_excerpt else ""}

## 수정 지시
1. 말투가 맞지 않으면 올바른 말투로 수정
   - {persona['tone']}을 사용해야 함
   - {persona['tone_target']}에게 말하는 상황
2. 입장이 맞지 않으면 해당 입장에 맞게 수정
3. 원래 의미와 길이는 최대한 유지
4. 자연스러운 한국어로 수정
5. '의무론', '공리주의', 'utilitarian' 단어 사용 금지
6. 수정된 응답만 출력 (설명 없이)
"""

        try:
            result = await self.llm.ainvoke(fix_prompt)
            fixed = result.content.strip()

            if fixed != response:
                logger.info(f"🔧 [PersonaValidator] 응답 수정됨 ({agent_type})")
                logger.info(f"   원본: {response[:50]}...")
                logger.info(f"   수정: {fixed[:50]}...")

            return fixed
        except Exception as e:
            logger.error(f"❌ [PersonaValidator] 수정 오류: {e}")
            return response

    async def validate_and_fix(self, response: str, agent_type: str) -> Tuple[str, bool]:
        """
        응답을 검증하고 필요시 수정

        Args:
            response: 원본 응답
            agent_type: 에이전트 유형

        Returns:
            (수정된 응답, 수정 여부)
        """
        # friend, artist_apprentice는 검증 제외
        if agent_type in ["friend", "artist_apprentice"]:
            return response, False

        # 말투 체크
        is_correct, reason = self.check_tone(response, agent_type)

        if is_correct:
            logger.info(f"✅ [PersonaValidator] 페르소나 OK ({agent_type}): {reason}")
            return response, False

        # 수정 필요
        logger.info(f"⚠️ [PersonaValidator] 페르소나 수정 필요 ({agent_type}): {reason}")
        fixed = await self.fix_response(response, agent_type)

        return fixed, fixed != response


async def validate_and_fix_persona(
    response: str,
    agent_type: str,
    api_key: str
) -> Tuple[str, bool]:
    """
    편의 함수: 응답을 검증하고 필요시 수정

    Args:
        response: 원본 응답
        agent_type: 에이전트 유형
        api_key: OpenAI API 키

    Returns:
        (수정된 응답, 수정 여부)
    """
    # friend, artist_apprentice는 검증 제외
    if agent_type in ["friend", "artist_apprentice"]:
        return response, False

    validator = PersonaValidator(api_key)
    return await validator.validate_and_fix(response, agent_type)
