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
        """
        # 최근 2턴(4개 메시지)의 대화 컨텍스트 생성 (Stage2 질문 포함을 위해 확장)
        conversation_context = ""
        if messages:
            recent_messages = messages[-4:] if len(messages) > 4 else messages
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
            logger.info(f"🔍 [Stage2 IntentDetector] ========== INTENT DETECTION START ==========")
            logger.info(f"🔍 [Stage2 IntentDetector] User message: '{user_message}'")
            logger.info(f"🔍 [Stage2 IntentDetector] Conversation context:\n{conversation_context}")
            logger.info(f"🔍 [Stage2 IntentDetector] Context length: {len(conversation_context)} chars")

            result = self.analyzer.invoke(intent_prompt)
            decision = result.content.strip().lower()

            logger.info(f"🔍 [Stage2 IntentDetector] LLM raw response: '{decision}'")

            if "opinion" in decision:
                logger.info(f"✅ [Stage2 IntentDetector] Detected: OPINION → Stage 3 transition")
                return "opinion"
            elif "dont_know" in decision or "dont know" in decision:
                logger.info(f"⚠️ [Stage2 IntentDetector] Detected: DONT_KNOW → Need explanation")
                return "dont_know"
            else:
                logger.info(f"⚠️ [Stage2 IntentDetector] Detected: UNCLEAR → Rephrase question")
                return "unclear"
        except Exception as e:
            logger.error(f"❌ [Stage2 IntentDetector] Error: {e}")
            return "unclear"
