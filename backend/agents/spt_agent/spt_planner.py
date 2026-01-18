"""
SPT Planner
질문 후보만 생성 (결정권 없음)

핵심 역할:
- SPT 전략에 따른 질문 후보 생성
- 현재 프레임에 맞는 질문 제안
- 결정은 Controller에게 위임
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .dst import SPTFrame, DialogueState
from .utils import clean_gpt_response

logger = logging.getLogger(__name__)

# 프롬프트 파일 로드
def _load_prompt(filename: str) -> str:
    """prompts 폴더에서 프롬프트 파일 로드"""
    prompt_path = Path(__file__).parent / "prompts" / filename
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load prompt {filename}: {e}")
        return ""


class SPTStrategy(Enum):
    """SPT 전략 유형"""
    SIMILAR_EXPERIENCE = "similar_experience"      # 유사 경험 떠올리기
    COMPARE_CONTRAST = "compare_contrast"          # 비교/대조
    CONTEXT_CONSIDERATION = "context_consideration"  # 상황 맥락 고려
    BACKGROUND_INFO = "background_info"            # 배경 정보 활용
    PERSPECTIVE_TAKING = "perspective_taking"      # 입장 바꿔 생각하기
    REFLECTION = "reflection"                      # 성찰
    EMOTION_REGULATION = "emotion_regulation"      # 감정 조절


@dataclass
class QuestionCandidate:
    """질문 후보"""
    question: str
    strategy: SPTStrategy
    target_frame: SPTFrame
    keywords: List[str]  # 핵심 키워드 (off_topic 감지용)
    priority: int = 1    # 우선순위 (1이 가장 높음)


class SPTPlanner:
    """
    SPT Planner: 질문 후보 생성기

    "어떤 질문을 할 수 있는가?" (결정 아님, 제안만)

    출력:
    - 질문 후보 1~3개
    - 각 후보의 SPT 전략
    - 핵심 키워드 (off_topic 감지용)
    """

    # 프레임별 시스템 프롬프트
    FRAME_PROMPTS = {
        SPTFrame.RAPPORT: """
당신은 대화 초반에 라포를 형성하는 단계입니다.
- 사용자의 관심사나 감정에 공감하세요
- 가벼운 질문으로 대화를 시작하세요
- 아직 깊은 윤리적 질문은 피하세요
""",
        SPTFrame.EXPLORE: """
당신은 사용자의 입장을 탐색하는 단계입니다.
- 사용자의 관점과 이유를 파악하세요
- "왜 그렇게 생각해?" 같은 개방형 질문을 사용하세요
- 사용자의 가치관과 우선순위를 이해하려 하세요
""",
        SPTFrame.CHALLENGE: """
당신은 새로운 관점을 제시하는 단계입니다.
- 다른 이해관계자의 관점을 고려하도록 유도하세요
- "~의 입장에서는 어떨까?" 형태의 질문을 사용하세요
- 사용자의 기존 관점에 부드럽게 도전하세요
""",
        SPTFrame.REFLECT: """
당신은 사용자가 성찰하도록 돕는 단계입니다.
- 대화에서 나온 다양한 관점을 연결하세요
- 사용자가 자신의 생각 변화를 인식하도록 하세요
- 깊은 성찰을 유도하는 질문을 사용하세요
""",
        SPTFrame.CLOSE: """
당신은 대화를 마무리하는 단계입니다.
- 대화의 핵심 내용을 요약하세요
- 사용자의 성장이나 통찰을 인정하세요
- 열린 결말로 마무리하세요
"""
    }

    # SPT 전략별 질문 템플릿
    STRATEGY_TEMPLATES = {
        SPTStrategy.SIMILAR_EXPERIENCE: [
            "혹시 비슷한 상황을 경험한 적 있어?",
            "이런 상황이 떠오르는 경험이 있어?",
        ],
        SPTStrategy.COMPARE_CONTRAST: [
            "만약 {stakeholder}라면 어떻게 느꼈을까?",
            "이 상황을 {another_context}에 비교하면 어떨까?",
        ],
        SPTStrategy.CONTEXT_CONSIDERATION: [
            "이 상황에서 특별히 고려해야 할 점이 뭘까?",
            "어떤 맥락에서 이런 일이 일어났을까?",
        ],
        SPTStrategy.PERSPECTIVE_TAKING: [
            "{stakeholder}의 입장에서 생각해보면 어떨까?",
            "만약 네가 {stakeholder}라면 어떤 마음일까?",
        ],
        SPTStrategy.REFLECTION: [
            "처음 생각과 지금 생각이 달라진 부분이 있어?",
            "이 대화에서 새롭게 알게 된 점이 있어?",
        ],
    }

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API key
            model: 사용할 모델 (빠른 응답을 위해 mini 권장)
        """
        self.api_key = api_key
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.7,
            max_tokens=300
        )

    async def generate_candidates(
        self,
        state: DialogueState,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        topic_context: str = ""
    ) -> List[QuestionCandidate]:
        """
        질문 후보 생성

        Args:
            state: 현재 대화 상태
            user_message: 사용자의 최근 메시지
            conversation_history: 대화 기록
            topic_context: 주제 컨텍스트 (예: "AI 예술 전시")

        Returns:
            질문 후보 리스트 (1~3개)
        """
        frame = state.current_frame
        frame_prompt = self.FRAME_PROMPTS.get(frame, self.FRAME_PROMPTS[SPTFrame.EXPLORE])

        # 대화 기록 포맷팅
        history_text = self._format_history(conversation_history[-6:])

        # 시스템 프롬프트 구성 (txt 파일에서 로드)
        prompt_template = _load_prompt("spt_planner_system.txt")
        system_prompt = prompt_template.format(frame_prompt=frame_prompt)

        user_prompt = f"""
주제: {topic_context if topic_context else "도덕적 딜레마"}

최근 대화:
{history_text}

사용자의 마지막 발언: "{user_message}"

위 대화 흐름에 자연스럽게 이어질 SPT 질문 후보를 생성하세요.
"""

        try:
            result = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            content = clean_gpt_response(result.content)
            candidates = self._parse_candidates(content, frame)

            logger.info(f"🎯 [Planner] Generated {len(candidates)} question candidates")
            return candidates

        except Exception as e:
            logger.error(f"❌ [Planner] Error generating candidates: {e}")
            return self._get_fallback_candidates(frame)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """대화 기록 포맷팅"""
        if not history:
            return "(대화 기록 없음)"

        lines = []
        for msg in history:
            role = "사용자" if msg.get("role") == "user" else "에이전트"
            content = msg.get("content", "")[:100]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _parse_candidates(
        self,
        content: str,
        frame: SPTFrame
    ) -> List[QuestionCandidate]:
        """LLM 출력을 파싱하여 QuestionCandidate 리스트 생성"""
        candidates = []

        # 간단한 파싱 (질문: / 키워드: 형식)
        blocks = content.split("---")

        for i, block in enumerate(blocks):
            lines = block.strip().split("\n")
            question = ""
            perspective = ""

            for line in lines:
                line = line.strip()
                if line.startswith("질문") and ":" in line:
                    question = line.split(":", 1)[1].strip()
                elif line.startswith("관점") and ":" in line:
                    perspective = line.split(":", 1)[1].strip()

            if question:
                candidates.append(QuestionCandidate(
                    question=question,
                    strategy=self._infer_strategy_from_perspective(perspective),
                    target_frame=frame,
                    keywords=self._extract_keywords(question),
                    priority=i + 1
                ))

        return candidates[:3]  # 최대 3개

    def _infer_strategy_from_perspective(self, perspective: str) -> SPTStrategy:
        """관점에서 SPT 전략 추론"""
        if "개인" in perspective or "가족" in perspective:
            return SPTStrategy.SIMILAR_EXPERIENCE
        elif "제3자" in perspective or "타인" in perspective:
            return SPTStrategy.PERSPECTIVE_TAKING
        elif "사회" in perspective or "규범" in perspective:
            return SPTStrategy.CONTEXT_CONSIDERATION
        elif "의무" in perspective or "책임" in perspective:
            return SPTStrategy.REFLECTION

        return SPTStrategy.PERSPECTIVE_TAKING  # 기본값

    def _infer_strategy(self, question: str) -> SPTStrategy:
        """질문에서 SPT 전략 추론"""
        question_lower = question.lower()

        if "입장" in question_lower or "관점" in question_lower:
            return SPTStrategy.PERSPECTIVE_TAKING
        elif "비슷" in question_lower or "경험" in question_lower:
            return SPTStrategy.SIMILAR_EXPERIENCE
        elif "비교" in question_lower or "다른" in question_lower:
            return SPTStrategy.COMPARE_CONTRAST
        elif "상황" in question_lower or "맥락" in question_lower:
            return SPTStrategy.CONTEXT_CONSIDERATION
        elif "달라" in question_lower or "변화" in question_lower:
            return SPTStrategy.REFLECTION

        return SPTStrategy.PERSPECTIVE_TAKING  # 기본값

    def _extract_keywords(self, question: str) -> List[str]:
        """질문에서 핵심 키워드 추출 (간단한 규칙 기반)"""
        # 불용어 제외하고 주요 단어 추출
        stopwords = {"어떻게", "왜", "뭐", "뭘", "이", "그", "저", "것", "수", "게", "를", "을", "가", "는", "은", "에", "의"}

        words = question.replace("?", "").split()
        keywords = [w for w in words if w not in stopwords and len(w) > 1]

        return keywords[:3]

    def _get_fallback_candidates(self, frame: SPTFrame) -> List[QuestionCandidate]:
        """폴백 질문 후보"""
        fallbacks = {
            SPTFrame.RAPPORT: QuestionCandidate(
                question="이 주제에 대해 평소에 어떻게 생각해왔어?",
                strategy=SPTStrategy.CONTEXT_CONSIDERATION,
                target_frame=frame,
                keywords=["생각", "평소"],
                priority=1
            ),
            SPTFrame.EXPLORE: QuestionCandidate(
                question="왜 그렇게 생각해?",
                strategy=SPTStrategy.SIMILAR_EXPERIENCE,
                target_frame=frame,
                keywords=["왜", "생각", "이유"],
                priority=1
            ),
            SPTFrame.CHALLENGE: QuestionCandidate(
                question="다른 사람의 입장에서는 어떻게 보일까?",
                strategy=SPTStrategy.PERSPECTIVE_TAKING,
                target_frame=frame,
                keywords=["다른", "입장", "사람"],
                priority=1
            ),
            SPTFrame.REFLECT: QuestionCandidate(
                question="이 대화에서 새롭게 생각하게 된 점이 있어?",
                strategy=SPTStrategy.REFLECTION,
                target_frame=frame,
                keywords=["새롭게", "생각", "변화"],
                priority=1
            ),
            SPTFrame.CLOSE: QuestionCandidate(
                question="오늘 대화에서 가장 기억에 남는 게 뭐야?",
                strategy=SPTStrategy.REFLECTION,
                target_frame=frame,
                keywords=["기억", "대화"],
                priority=1
            ),
        }

        return [fallbacks.get(frame, fallbacks[SPTFrame.EXPLORE])]
