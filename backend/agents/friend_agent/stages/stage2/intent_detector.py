"""
Intent detection for Stage 2
사용자가 AI 예술에 대한 의견을 냈는지 판단
"""
import logging
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# utils.py 임포트
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import format_prompt

logger = logging.getLogger(__name__)


class IntentDetector:
    """Stage 2 사용자 의도 감지 클래스"""

    OPINION_KEYWORDS = (
        "원하", "원해", "싶어", "하고싶", "만나고", "만날거", "만나고싶",
        "찬성", "반대", "괜찮", "필요", "해야", "하고싶다", "사용하",
        "쓰고싶", "믿어", "믿고싶", "좋아", "싫어", "할래", "할거야"
    )
    YES_RESPONSES = ("응", "웅", "그래", "맞아", "좋아", "당연")
    DONT_KNOW_KEYWORDS = ("모르겠", "모르겠어", "몰라", "글쎄", "잘모르", "잘 모르")
    # ✨ 에이전트에게 의견을 묻는 패턴
    ASK_OPINION_KEYWORDS = ("너는", "넌", "네생각", "네 생각", "너생각", "너 생각", "니생각", "넌어떻게", "너는어떻게", "넌 어떻게", "너는 어떻게")

    def __init__(self, analyzer: ChatOpenAI, prompts: dict):
        """
        Args:
            analyzer: Intent detection용 LLM
            prompts: 프롬프트 딕셔너리
        """
        self.analyzer = analyzer
        self.prompts = prompts

    def detect(self, user_message: str, messages: list = None) -> str:
        """
        사용자가 AI 예술에 대한 의견을 냈는지 판단

        Args:
            user_message: 사용자 메시지
            messages: 전체 대화 기록 (컨텍스트)

        Returns:
            "opinion" - 명확한 의견을 냄 (Stage 3로 전환)
            "unclear" - 모호하거나 무관한 답변 (다시 질문)
            "dont_know" - 모르겠다고 함 (설명 필요)
            "ask_opinion" - 에이전트에게 의견을 물음 (잘 모르겠다고 답변)
        """
        if not user_message:
            return "unclear"

        normalized = user_message.strip().lower().replace(" ", "")

        # ✨ 에이전트에게 의견을 묻는지 먼저 체크
        if any(keyword in normalized for keyword in self.ASK_OPINION_KEYWORDS):
            logger.info(f"✅ [Stage2Intent] Heuristic ask_opinion detected: {user_message}")
            return "ask_opinion"

        if any(keyword in normalized for keyword in self.DONT_KNOW_KEYWORDS):
            logger.info(f"✅ [Stage2Intent] Heuristic dont_know detected: {user_message}")
            return "dont_know"

        if normalized in self.YES_RESPONSES:
            logger.info(f"✅ [Stage2Intent] Heuristic opinion detected (yes-response): {user_message}")
            return "opinion"

        if any(keyword in normalized for keyword in self.OPINION_KEYWORDS):
            logger.info(f"✅ [Stage2Intent] Heuristic opinion keyword detected: {user_message}")
            return "opinion"

        # 최근 4턴(8개 메시지)의 대화 컨텍스트 생성
        conversation_context = ""
        if messages:
            recent_messages = messages[-8:] if len(messages) > 8 else messages
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_messages
            ])

        # ✨ txt 파일에서 프롬프트 로드 (conversation_context 추가)
        intent_prompt = format_prompt(
            self.prompts.get("stage2_detect_intent", ""),
            user_message=user_message,
            conversation_context=conversation_context
        )

        try:
            result = self.analyzer.invoke(intent_prompt)
            decision = result.content.strip().lower()

            if "opinion" in decision:
                logger.info(f"Detected opinion: {user_message}")
                return "opinion"
            elif "dont_know" in decision or "dont know" in decision:
                logger.info(f"Detected dont_know: {user_message}")
                return "dont_know"
            else:
                logger.info(f"Detected unclear: {user_message}")
                return "unclear"
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return "unclear"
