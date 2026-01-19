"""
Jangmo Agent (장모 - 의무론적 관점, AI 복원 반대)
LangChain 기반 대화 에이전트
✨ Chain of Persona (CoP) + 조건부 SPT 방식
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import logging
import sys
import re
import random
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    ChatGoogleGenerativeAI = None

if TYPE_CHECKING:
    from agents.spt_agent.spt_agent_v2 import SPTAgentV2

sys.path.append(str(Path(__file__).parent.parent))
from .utils import clean_gpt_response

logger = logging.getLogger(__name__)


class JangmoAgent:
    """
    장모 대화 에이전트 (AI 복원 반대 - 책임 중심 관점)
    - LangChain ChatOpenAI 사용
    - 노인 여성 장모 캐릭터
    - ✨ Chain of Persona (CoP) + 조건부 SPT 방식
    - 사위에게 반말 사용
    """

    def __init__(
        self,
        api_key: str,
        model: str = "ft:gpt-4.1-mini-2025-04-14:idl-lab:moral-agent-v1:CfBlCpDs",
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

        self._session_keywords: Dict[str, List[str]] = {}

        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if HAS_GEMINI and gemini_api_key:
            self.analyzer = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                google_api_key=gemini_api_key
            )
            logger.info("✅ [Jangmo] Using Gemini for persona validation")
        else:
            self.analyzer = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=api_key
            )
            logger.info("⚠️ [Jangmo] Using gpt-4o-mini for persona validation (Gemini not available)")

        prompt_path = Path(__file__).parent / "prompts" / "jangmo_deontological.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
                logger.info("✅ Jangmo system prompt loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Jangmo system prompt: {e}")
            self.system_prompt = "당신은 사위에게 아내(당신의 딸)를 AI로 복원하는 것을 반대하는 장모입니다. 사위와 대화할 때는 '네 아내'라고 표현합니다. 책임과 예의, 약속을 근거로 반대 의견을 말합니다."

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """✨ 세션별 히스토리 가져오기 (없으면 생성)"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
            logger.info(f"✅ [Jangmo] Created new session: {session_id}")
        return self.session_store[session_id]

    def clear_session(self, session_id: str) -> bool:
        """세션 히스토리 삭제"""
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info(f"✅ [Jangmo] Cleared session: {session_id}")
            return True
        return False

    async def _generate_self_reflection(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3
    ) -> str:
        """
        1단계: 페르소나 자기 성찰 생성
        - ✨ Gemini 모델 사용 (반복 문제 해결)
        - 낮은 temperature (0.3) 사용 (일관성 중시)
        - 내부 사고 과정 생성
        - Chain of Persona + 조건부 SPT

        Args:
            messages: 대화 메시지 리스트
            temperature: LLM temperature (낮을수록 일관적)

        Returns:
            Self-reflection 텍스트
        """
        reflection_prompt_messages = self._build_reflection_prompt(messages)
        reflection_prompt_text = reflection_prompt_messages[0].content  # SystemMessage에서 텍스트 추출

        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.analyzer.invoke, reflection_prompt_text)

        reflection = result.content.strip()

        return reflection

    def _build_reflection_prompt(
        self,
        messages: List[Dict[str, str]]
    ) -> List:
        """
        Self-Reflection용 프롬프트 구성
        - reflection_prompt.txt 파일에서 로드
        """
        last_user_msg = self._extract_last_user_message(messages)
        recent_history = self._get_recent_history(messages, limit=3)
        recent_response = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                recent_response = msg["content"].strip()
                break

        prompt_path = Path(__file__).parent / "prompts" / "reflection_prompt.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        reflection_instruction = template.format(
            system_prompt=self.system_prompt,
            last_user_msg=last_user_msg,
            recent_history=recent_history,
            recent_response=recent_response
        )

        return [SystemMessage(content=reflection_instruction)]

    def _build_messages_with_reflection_and_spt(
        self,
        messages: List[Dict[str, str]],
        reflection: str,
        spt_instruction: Optional[str] = None
    ) -> List:
        """
        Self-reflection + SPT instruction을 포함한 프롬프트 구성
        """
        prompt_path = Path(__file__).parent / "prompts" / "response_prompt.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        spt_section = ""
        if spt_instruction:
            spt_section = f"""
🧠 SPT Strategy Instructions:
{spt_instruction}

Follow these instructions while maintaining your character's stance and speech style.
"""

        response_instruction = template.format(
            reflection=reflection,
            spt_section=spt_section
        )

        system_prompt = self.system_prompt + "\n\n" + response_instruction
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
        max_tokens: int = 150,  # 응답 길이 (1문장)
        session_id: str = "default"  # ✨ 세션 ID 추가
    ) -> str:
        """
        비스트리밍 대화 - ✨ 2-step CoP + SPT Agent V2 통합
        1) Self-reflection 생성
        2) SPT 필요 시 SPT Agent V2 호출
        3) Reflection + SPT instruction 기반으로 Final response 생성
        """
        try:
            reflection = await self._generate_self_reflection(messages, temperature=0.3)
            logger.info(f"✨ [Jangmo] Self-reflection: {reflection[:100]}...")

            spt_needed = self._parse_spt_necessity(reflection)

            last_user_msg = self._extract_last_user_message(messages)
            logger.info(f"🔍 [Jangmo] SPT needed: {spt_needed}")

            spt_instruction = None
            if spt_needed and self.spt_agent_v2:
                stored_keywords = self._get_stored_keywords(session_id)

                try:
                    spt_result = await self.spt_agent_v2.process(
                        session_id=f"장모_{session_id}",  # 에이전트별 고유 세션
                        user_message=last_user_msg,
                        conversation_history=messages,
                        topic_context="AI 복원",
                        question_keywords=stored_keywords
                    )

                    spt_instruction = spt_result["instruction"]
                    logger.info(f"🧠 [Jangmo] SPT instruction: {spt_instruction[:100]}...")

                    if spt_result.get("suggested_keywords"):
                        self._store_keywords(session_id, spt_result["suggested_keywords"])

                except Exception as spt_error:
                    logger.error(f"⚠️ [Jangmo] SPT Agent V2 error (continuing without SPT): {spt_error}")

            llm = self._create_llm(max_tokens, streaming=False, temperature=temperature)
            lc_messages = self._build_messages_with_reflection_and_spt(messages, reflection, spt_instruction)
            result = await llm.ainvoke(lc_messages)

            raw_content = result.content.strip()
            cleaned_content = clean_gpt_response(raw_content)

            if spt_needed and self.spt_agent_v2 and spt_instruction:
                try:
                    has_question = "?" in cleaned_content
                    self.spt_agent_v2.update_after_agent_response(
                        session_id=f"장모_{session_id}",
                        agent_response=cleaned_content,
                        asked_question=has_question
                    )
                except Exception as update_error:
                    logger.error(f"⚠️ [Jangmo] SPT state update error (non-critical): {update_error}")

            history = self.get_session_history(session_id)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_content)

            logger.info(f"✅ [Jangmo] session_id={session_id}, response: '{cleaned_content[:50]}...'")
            return cleaned_content

        except Exception as e:
            logger.error(f"❌ [Jangmo] Error: {e}", exc_info=True)
            return "미안해, 다시 한번 말해줄래?"

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,  # 응답 길이 (1문장)
        session_id: str = "default"  # ✨ 세션 ID 추가
    ):
        """
        스트리밍 대화 - ✨ 2-step CoP + SPT Agent V2 통합 (streaming)
        """
        try:
            reflection = await self._generate_self_reflection(messages, temperature=0.3)
            logger.info(f"✨ [Jangmo] Self-reflection: {reflection[:100]}...")

            spt_needed = self._parse_spt_necessity(reflection)

            last_user_msg = self._extract_last_user_message(messages)
            logger.info(f"🔍 [Jangmo] SPT needed: {spt_needed}")

            spt_instruction = None
            if spt_needed and self.spt_agent_v2:
                stored_keywords = self._get_stored_keywords(session_id)

                try:
                    spt_result = await self.spt_agent_v2.process(
                        session_id=f"장모_{session_id}",
                        user_message=last_user_msg,
                        conversation_history=messages,
                        topic_context="AI 복원",
                        question_keywords=stored_keywords
                    )

                    spt_instruction = spt_result["instruction"]
                    logger.info(f"🧠 [Jangmo] SPT instruction: {spt_instruction[:100]}...")

                    if spt_result.get("suggested_keywords"):
                        self._store_keywords(session_id, spt_result["suggested_keywords"])

                except Exception as spt_error:
                    logger.error(f"⚠️ [Jangmo] SPT Agent V2 error (continuing without SPT): {spt_error}")

            llm = self._create_llm(max_tokens, streaming=True, temperature=temperature)
            lc_messages = self._build_messages_with_reflection_and_spt(messages, reflection, spt_instruction)

            full_response = ""
            async for chunk in llm.astream(lc_messages):
                chunk_text = self._extract_chunk_text(chunk)
                if chunk_text:
                    full_response += chunk_text
                    yield chunk_text

            cleaned_response = clean_gpt_response(full_response)

            if spt_needed and self.spt_agent_v2 and spt_instruction:
                try:
                    has_question = "?" in cleaned_response
                    self.spt_agent_v2.update_after_agent_response(
                        session_id=f"장모_{session_id}",
                        agent_response=cleaned_response,
                        asked_question=has_question
                    )
                except Exception as update_error:
                    logger.error(f"⚠️ [Jangmo] SPT state update error (non-critical): {update_error}")

            history = self.get_session_history(session_id)
            last_user_msg = self._extract_last_user_message(messages)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_response)

            logger.info(f"✅ [Jangmo] session_id={session_id}, streamed: '{cleaned_response[:50]}...'")

        except Exception as e:
            logger.error(f"❌ [Jangmo] Streaming error: {e}", exc_info=True)
            for char in "미안해, 다시 한번 말해줄래?":
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

    def _get_recent_history(self, messages: List[Dict[str, str]], limit: int = 3) -> str:
        """
        최근 N턴의 대화 히스토리 요약 (SPT 판단에 사용)

        Args:
            messages: 전체 대화 메시지 리스트
            limit: 최근 몇 턴을 가져올지 (기본 3턴)

        Returns:
            요약된 대화 히스토리 문자열
        """
        recent = messages[-(limit*2):] if len(messages) > limit*2 else messages
        lines = []
        for msg in recent:
            role = "사용자" if msg.get("role") == "user" else "AI"
            content = msg.get("content", "")[:100]  # 100자 제한
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "[대화 시작]"

    def _parse_spt_necessity(self, reflection: str) -> bool:
        """
        Reflection 출력에서 SPT 필요성 판단 결과 추출

        Args:
            reflection: Self-reflection LLM 출력 텍스트

        Returns:
            True if SPT is needed, False otherwise
        """
        match = re.search(r'SPT\s*(?:필요성?|required)\s*[:：]\s*(YES|NO)', reflection, re.IGNORECASE)
        if match:
            return match.group(1).upper() == "YES"

        if ("질문 1:" in reflection and "답변 1:" in reflection) or (
            "Question 1:" in reflection and "Answer 1:" in reflection
        ):
            return True

        return False

    def _store_keywords(self, session_id: str, keywords: List[str]) -> None:
        """
        다음 턴의 off-topic 감지를 위한 키워드 저장

        Args:
            session_id: 세션 ID
            keywords: 저장할 키워드 리스트
        """
        self._session_keywords[session_id] = keywords
        logger.info(f"📌 [Jangmo] Stored keywords for {session_id}: {keywords}")

    def _get_stored_keywords(self, session_id: str) -> List[str]:
        """
        저장된 키워드 조회

        Args:
            session_id: 세션 ID

        Returns:
            저장된 키워드 리스트 (없으면 빈 리스트)
        """
        return self._session_keywords.get(session_id, [])
