"""Persona validation and correction for agent responses."""
import logging
from pathlib import Path
from typing import Tuple
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent.parent
PROMPT_PATHS = {
    "colleague1": BACKEND_DIR / "agents" / "colleague1_agent" / "prompts" / "response_prompt.txt",
    "colleague2": BACKEND_DIR / "agents" / "colleague2_agent" / "prompts" / "response_prompt.txt",
    "jangmo": BACKEND_DIR / "agents" / "jangmo_agent" / "prompts" / "response_prompt.txt",
    "son": BACKEND_DIR / "agents" / "son_agent" / "prompts" / "response_prompt.txt",
}


def load_persona_prompt(agent_type: str) -> str:
    prompt_path = PROMPT_PATHS.get(agent_type)
    if prompt_path and prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to load prompt for {agent_type}: {e}")
    return ""


PERSONA_RULES = {
    "colleague1": {
        "name": "동료1 (50대 여성 선배 화가)",
        "tone": "반말",
        "tone_target": "후배 동료",
        "stance": "의무론 - AI 전시 반대",
        "stance_keywords": ["규칙", "원칙", "의무", "기준", "도덕", "자격", "책임"],
        "tone_markers": ["하나", "한가", "군", "네", "지", "야"],
        "wrong_tone_markers": ["요", "습니다", "세요", "까요"],
        "question_style": "~하나?, ~한가?",
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
        # "야", "해", "지" can appear in formal speech too (e.g. 생각해요)
        "wrong_tone_markers": ["해봐", "냐"],
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

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.3)

    def check_tone(self, response: str, agent_type: str) -> Tuple[bool, str]:
        """Check if the response tone matches the persona."""
        persona = PERSONA_RULES.get(agent_type)
        if not persona:
            return True, "unknown_agent"

        response_end = response[-20:] if len(response) > 20 else response

        has_correct = any(marker in response for marker in persona["tone_markers"])
        has_wrong = any(marker in response_end for marker in persona["wrong_tone_markers"])

        if persona["tone"] == "존댓말":
            if has_correct and not has_wrong:
                return True, "correct_formal"
            elif has_wrong:
                return False, "should_be_formal"
            else:
                return False, "missing_formal_markers"
        else:
            if has_wrong:
                return False, "should_be_informal"
            else:
                return True, "correct_informal"

    async def fix_response(self, response: str, agent_type: str) -> str:
        """Fix a response that doesn't match the persona, using the prompt file for accuracy."""
        persona = PERSONA_RULES.get(agent_type)
        if not persona:
            return response

        persona_prompt = load_persona_prompt(agent_type)
        style_excerpt = ""
        if persona_prompt:
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
                logger.info(f"[PersonaValidator] Response fixed ({agent_type})")
                logger.info(f"  Original: {response[:50]}...")
                logger.info(f"  Fixed: {fixed[:50]}...")

            return fixed
        except Exception as e:
            logger.error(f"[PersonaValidator] Fix error: {e}")
            return response

    async def validate_and_fix(self, response: str, agent_type: str) -> Tuple[str, bool]:
        """Validate response and fix if needed. Returns (response, was_fixed)."""
        if agent_type in ["friend", "artist_apprentice"]:
            return response, False

        is_correct, reason = self.check_tone(response, agent_type)

        if is_correct:
            logger.info(f"[PersonaValidator] Persona OK ({agent_type}): {reason}")
            return response, False

        logger.info(f"[PersonaValidator] Persona fix needed ({agent_type}): {reason}")
        fixed = await self.fix_response(response, agent_type)

        return fixed, fixed != response


async def validate_and_fix_persona(
    response: str,
    agent_type: str,
    api_key: str
) -> Tuple[str, bool]:
    """Convenience function: validate and fix a response."""
    if agent_type in ["friend", "artist_apprentice"]:
        return response, False

    validator = PersonaValidator(api_key)
    return await validator.validate_and_fix(response, agent_type)
