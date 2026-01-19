"""
Colleague1 Agent (동료 화가 - 의무론적 관점, AI 예술 반대)
LangChain 기반 대화 에이전트
✨ 간소화된 SPT 통합 (Reflection 제거)
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import logging
import sys
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

if TYPE_CHECKING:
    from agents.spt_agent.spt_agent_v2 import SPTAgentV2

sys.path.append(str(Path(__file__).parent.parent))
from .utils import clean_gpt_response

logger = logging.getLogger(__name__)


class Colleague1Agent:
    """
    동료 화가 대화 에이전트 (AI 예술 반대 - 의무론적 관점)
    - LangChain ChatOpenAI 사용
    - 50대 여성 선배 화가 캐릭터
    - ✨ 간소화된 SPT 통합 (항상 SPT 호출)
    - 후배에게 반말 사용
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        spt_agent_v2: Optional['SPTAgentV2'] = None
    ):
        """
        Args:
            api_key: OpenAI API key
            model: 모델 ID (fine-tuned moral agent model)
            spt_agent_v2: SPT Agent V2 인스턴스 (선택적)
        """
        self.api_key = api_key
        self.model = model
        self.spt_agent_v2 = spt_agent_v2

        self.session_store: Dict[str, ChatMessageHistory] = {}

        prompt_path = Path(__file__).parent / "prompts" / "ai_artist_deontological.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
                logger.info("✅ Colleague1 system prompt loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Colleague1 system prompt: {e}")
            self.system_prompt = "당신은 AI 예술에 반대하는 50대 여성 화가입니다. 의무와 책임을 근거로 반대 의견을 말합니다."

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """세션별 히스토리 가져오기 (없으면 생성)"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
            logger.info(f"✅ [Colleague1] Created new session: {session_id}")
        return self.session_store[session_id]

    def clear_session(self, session_id: str) -> bool:
        """세션 히스토리 삭제"""
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info(f"✅ [Colleague1] Cleared session: {session_id}")
            return True
        return False

    def _build_messages_with_spt(
        self,
        messages: List[Dict[str, str]],
        spt_instruction: Optional[str] = None
    ) -> List:
        """
        SPT instruction을 포함한 프롬프트 구성
        """
        system_prompt = self.system_prompt

        if spt_instruction:
            system_prompt += f"""

🧠 SPT Strategy Instructions:
{spt_instruction}

Follow these instructions while maintaining your character's stance and speech style.
"""

        lc_messages = [SystemMessage(content=system_prompt)]

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
        비스트리밍 대화 - 간소화된 SPT 통합
        1) 항상 SPT Agent V2 호출
        2) SPT instruction 기반으로 응답 생성
        """
        try:
            last_user_msg = self._extract_last_user_message(messages)

            # 항상 SPT Agent V2 호출
            spt_instruction = None
            if self.spt_agent_v2:
                try:
                    spt_result = await self.spt_agent_v2.process(
                        session_id=f"동료 화가_{session_id}",
                        user_message=last_user_msg,
                        conversation_history=messages,
                        topic_context="AI 예술"
                    )
                    spt_instruction = spt_result["instruction"]
                    logger.info(f"🧠 [Colleague1] SPT instruction: {spt_instruction}")
                except Exception as spt_error:
                    logger.error(f"⚠️ [Colleague1] SPT Agent V2 error (continuing without SPT): {spt_error}")

            llm = self._create_llm(max_tokens, streaming=False, temperature=temperature)
            lc_messages = self._build_messages_with_spt(messages, spt_instruction)
            result = await llm.ainvoke(lc_messages)

            raw_content = result.content.strip()
            cleaned_content = clean_gpt_response(raw_content)

            history = self.get_session_history(session_id)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_content)

            logger.info(f"✅ [Colleague1] session_id={session_id}, response: '{cleaned_content}'")
            return cleaned_content

        except Exception as e:
            logger.error(f"❌ [Colleague1] Error: {e}", exc_info=True)
            return "미안하네, 다시 한번 말해주겠나?"

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,
        session_id: str = "default"
    ):
        """
        스트리밍 대화 - 간소화된 SPT 통합 (streaming)
        """
        try:
            last_user_msg = self._extract_last_user_message(messages)

            # 항상 SPT Agent V2 호출
            spt_instruction = None
            if self.spt_agent_v2:
                try:
                    spt_result = await self.spt_agent_v2.process(
                        session_id=f"동료 화가_{session_id}",
                        user_message=last_user_msg,
                        conversation_history=messages,
                        topic_context="AI 예술"
                    )
                    spt_instruction = spt_result["instruction"]
                    logger.info(f"🧠 [Colleague1] SPT instruction: {spt_instruction}")
                except Exception as spt_error:
                    logger.error(f"⚠️ [Colleague1] SPT Agent V2 error (continuing without SPT): {spt_error}")

            llm = self._create_llm(max_tokens, streaming=True, temperature=temperature)
            lc_messages = self._build_messages_with_spt(messages, spt_instruction)

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

            logger.info(f"✅ [Colleague1] session_id={session_id}, streamed: '{cleaned_response}'")

        except Exception as e:
            logger.error(f"❌ [Colleague1] Streaming error: {e}", exc_info=True)
            for char in "미안하네, 다시 한번 말해주겠나?":
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
