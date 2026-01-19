"""
Colleague2 Agent (후배 화가 - 공리주의적 관점, AI 예술 찬성)
LangChain 기반 대화 에이전트
✨ Two-Phase SPT Reflection (자기 인식 → 응답 생성)
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


def _load_reflection_prompt() -> str:
    """Load reflection prompt from this agent's prompts folder"""
    prompt_path = Path(__file__).parent / "prompts" / "reflection_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load reflection prompt: {e}")
        return ""


class Colleague2Agent:
    """
    후배 화가 대화 에이전트 (AI 예술 찬성 - 공리주의적 관점)
    - LangChain ChatOpenAI 사용
    - 30대 초반 남성 후배 화가 캐릭터
    - ✨ Two-Phase SPT: 자기 인식 → 응답 생성
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

        # Reflection Prompt 로드
        self.reflection_prompt = _load_reflection_prompt()
        if self.reflection_prompt:
            logger.info("✅ Reflection prompt loaded successfully")

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

    def _format_history(self, messages: List[Dict[str, str]], limit: int = 12) -> str:
        """대화 히스토리를 텍스트로 포맷팅"""
        if not messages:
            return "(No previous conversation)"

        recent = messages[-limit:] if len(messages) > limit else messages
        lines = []
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if content:
                speaker = "User" if role == "user" else "Agent"
                lines.append(f"{speaker}: {content}")

        return "\n".join(lines) if lines else "(No previous conversation)"

    async def _perform_spt_reflection(
        self,
        messages: List[Dict[str, str]],
        last_user_msg: str,
        session_id: str
    ) -> Dict[str, str]:
        """
        Phase 1: SPT 자기 인식 단계 - txt 파일 기반 프롬프트
        디버그 로그에 전체 과정 출력
        """
        logger.info(f"🧠 [PHASE1_START] session_id={session_id}, Performing SPT reflection...")

        # 최근 대화 히스토리 구성
        history_text = self._format_history(messages)

        # reflection_prompt.txt에서 플레이스홀더 치환
        reflection_prompt = self.reflection_prompt.format(
            last_user_msg=last_user_msg,
            history=history_text
        )

        try:
            # 빠른 모델로 reflection 수행
            reflection_llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=self.api_key,
                temperature=0.3,
                max_tokens=500
            )

            result = await reflection_llm.ainvoke([
                SystemMessage(content=reflection_prompt)
            ])

            reflection_text = result.content.strip()

            # 디버그 로그 출력
            logger.info(f"🧠 [SPT_REFLECTION] session_id={session_id}")
            logger.info(f"🧠 [SPT_REFLECTION] User message: '{last_user_msg}'")
            logger.info(f"🧠 [SPT_REFLECTION] Full output:\n{reflection_text}")
            logger.info(f"🧠 [PHASE1_COMPLETE] Reflection done")

            return {"reflection": reflection_text}

        except Exception as e:
            logger.error(f"🧠 [SPT_REFLECTION] Error: {e}")
            return {"reflection": "Reflection failed - respond naturally based on persona."}

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
        Two-Phase 대화:
        Phase 1: SPT 자기 인식 (디버그 로그 출력)
        Phase 2: 응답 생성
        """
        try:
            last_user_msg = self._extract_last_user_message(messages)

            # Phase 1: SPT Reflection (디버그 로그 출력)
            reflection = await self._perform_spt_reflection(messages, last_user_msg, session_id)

            # Phase 2: Response Generation
            logger.info(f"🎭 [PHASE2_START] session_id={session_id}, Generating response...")

            llm = self._create_llm(max_tokens, streaming=False, temperature=temperature)

            # Phase 2 메시지 구성: Reflection 결과 + 대화 히스토리
            combined_prompt = f"""Based on your self-reflection analysis below, generate your response in character.

=== REFLECTION ANALYSIS ===
{reflection['reflection']}
=== END REFLECTION ===

IMPORTANT: Output ONLY your final in-character response in Korean. Do NOT output the reflection analysis again."""

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

            result = await llm.ainvoke(lc_messages)

            raw_content = result.content.strip()
            cleaned_content = clean_gpt_response(raw_content)

            logger.info(f"🎭 [PHASE2_COMPLETE] session_id={session_id}, response: '{cleaned_content}'")

            # 세션 히스토리 저장
            history = self.get_session_history(session_id)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_content)

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
        Two-Phase 스트리밍 대화:
        Phase 1: SPT 자기 인식 (디버그 로그 출력)
        Phase 2: 응답 스트리밍
        """
        try:
            last_user_msg = self._extract_last_user_message(messages)

            # Phase 1: SPT Reflection (디버그 로그 출력)
            reflection = await self._perform_spt_reflection(messages, last_user_msg, session_id)

            # Phase 2: Response Generation (Streaming)
            logger.info(f"🎭 [PHASE2_START] session_id={session_id}, Generating streamed response...")

            llm = self._create_llm(max_tokens, streaming=True, temperature=temperature)

            # Phase 2 메시지 구성: Reflection 결과 + 대화 히스토리
            combined_prompt = f"""Based on your self-reflection analysis below, generate your response in character.

=== REFLECTION ANALYSIS ===
{reflection['reflection']}
=== END REFLECTION ===

IMPORTANT: Output ONLY your final in-character response in Korean. Do NOT output the reflection analysis again."""

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

            full_response = ""
            async for chunk in llm.astream(lc_messages):
                chunk_text = self._extract_chunk_text(chunk)
                if chunk_text:
                    full_response += chunk_text
                    yield chunk_text

            cleaned_response = clean_gpt_response(full_response)
            logger.info(f"🎭 [PHASE2_COMPLETE] session_id={session_id}, streamed: '{cleaned_response}'")

            # 세션 히스토리 저장
            history = self.get_session_history(session_id)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_response)

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
