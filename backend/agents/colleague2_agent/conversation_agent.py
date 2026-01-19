"""
Colleague2 Agent (후배 화가 - 공리주의적 관점, AI 예술 찬성)
LangChain 기반 대화 에이전트
✨ SPT Reflection Framework 통합 (별도 SPT Agent 없음)
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

sys.path.append(str(Path(__file__).parent.parent))
from .utils import clean_gpt_response

logger = logging.getLogger(__name__)


def _load_spt_reflection_framework() -> str:
    """Load SPT reflection framework from file"""
    framework_path = Path(__file__).parent.parent / "spt_agent" / "prompts" / "spt_reflection_framework.txt"
    try:
        with open(framework_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load SPT reflection framework: {e}")
        return ""


class Colleague2Agent:
    """
    후배 화가 대화 에이전트 (AI 예술 찬성 - 공리주의적 관점)
    - LangChain ChatOpenAI 사용
    - 30대 초반 남성 후배 화가 캐릭터
    - ✨ SPT Reflection Framework 직접 통합 (별도 SPT Agent 없음)
    - 선배에게 존댓말 사용
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        spt_agent_v2: Optional[Any] = None  # 하위 호환성 유지, 사용하지 않음
    ):
        """
        Args:
            api_key: OpenAI API key
            model: 모델 ID
            spt_agent_v2: (deprecated) 하위 호환성을 위해 유지, 사용되지 않음
        """
        self.api_key = api_key
        self.model = model

        self.session_store: Dict[str, ChatMessageHistory] = {}

        # 페르소나 프롬프트 로드
        prompt_path = Path(__file__).parent / "prompts" / "ai_artist_utilitarian.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.persona_prompt = f.read().strip()
                logger.info("✅ Colleague2 persona prompt loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Colleague2 persona prompt: {e}")
            self.persona_prompt = "당신은 AI 예술에 찬성하는 30대 남성 화가입니다. 사회적 이익과 혜택을 근거로 찬성 의견을 말합니다."

        # SPT Reflection Framework 로드
        self.spt_framework = _load_spt_reflection_framework()
        if self.spt_framework:
            logger.info("✅ SPT Reflection Framework loaded successfully")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """세션별 히스토리 가져오기 (없으면 생성)"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
            logger.info(f"✅ [Colleague2] Created new session: {session_id}")
        return self.session_store[session_id]

    def clear_session(self, session_id: str) -> bool:
        """세션 히스토리 삭제"""
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info(f"✅ [Colleague2] Cleared session: {session_id}")
            return True
        return False

    def _build_messages(self, messages: List[Dict[str, str]]) -> List:
        """
        페르소나 + SPT Reflection Framework 결합 프롬프트 구성
        """
        # 페르소나 + SPT Framework 결합
        combined_prompt = f"""{self.persona_prompt}

=== SPT REFLECTION FRAMEWORK ===
{self.spt_framework}
=== END FRAMEWORK ===
"""

        lc_messages = [SystemMessage(content=combined_prompt)]

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

        return lc_messages

    def _create_llm(self, max_tokens: int, streaming: bool, temperature: float = 0.7) -> ChatOpenAI:
        """LLM 인스턴스 생성"""
        kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "streaming": streaming,
        }
        if "gpt-5" in self.model:
            kwargs["temperature"] = 1.0
            if max_tokens:
                kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,
        session_id: str = "default"
    ) -> str:
        """
        비스트리밍 대화 - SPT Reflection Framework 통합
        """
        try:
            last_user_msg = self._extract_last_user_message(messages)

            llm = self._create_llm(max_tokens, streaming=False, temperature=temperature)
            lc_messages = self._build_messages(messages)
            result = await llm.ainvoke(lc_messages)

            raw_content = result.content.strip()
            cleaned_content = clean_gpt_response(raw_content)

            history = self.get_session_history(session_id)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_content)

            logger.info(f"✅ [Colleague2] session_id={session_id}, response: '{cleaned_content}'")
            return cleaned_content

        except Exception as e:
            logger.error(f"❌ [Colleague2] Error: {e}", exc_info=True)
            return "선생님, 다시 한 번만 말씀해주시겠어요?"

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,
        session_id: str = "default"
    ):
        """
        스트리밍 대화 - SPT Reflection Framework 통합
        """
        try:
            last_user_msg = self._extract_last_user_message(messages)

            llm = self._create_llm(max_tokens, streaming=True, temperature=temperature)
            lc_messages = self._build_messages(messages)

            full_response = ""
            async for chunk in llm.astream(lc_messages):
                chunk_text = self._extract_chunk_text(chunk)
                if chunk_text:
                    full_response += chunk_text
                    yield chunk_text

            cleaned_response = clean_gpt_response(full_response)

            history = self.get_session_history(session_id)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_response)

            logger.info(f"✅ [Colleague2] session_id={session_id}, streamed: '{cleaned_response}'")

        except Exception as e:
            logger.error(f"❌ [Colleague2] Streaming error: {e}", exc_info=True)
            for char in "선생님, 다시 한 번만 말씀해주시겠어요?":
                yield char

    def _extract_chunk_text(self, chunk) -> str:
        """스트리밍 청크에서 텍스트 추출"""
        if hasattr(chunk, "content"):
            if isinstance(chunk.content, list):
                return "".join(part["text"] for part in chunk.content if isinstance(part, dict) and part.get("text"))
            else:
                return str(chunk.content)
        else:
            return str(chunk)

    @staticmethod
    def _extract_last_user_message(messages: List[Dict[str, str]]) -> str:
        """마지막 유저 메시지 추출"""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"].strip()
        return ""
