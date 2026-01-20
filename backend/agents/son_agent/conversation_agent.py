"""
Son Agent (아들 - 책임 중심 관점, AI 복원 찬성)
LangChain 기반 대화 에이전트
✨ Three-Phase Architecture: Basic Reflection → SPT Planner (조건부) → Response
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys
import os
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Gemini 지원 (선택적)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    ChatGoogleGenerativeAI = None

sys.path.append(str(Path(__file__).parent.parent))
from .utils import clean_gpt_response

logger = logging.getLogger(__name__)


def _load_reflection_prompt() -> str:
    """Load basic reflection prompt (Step 1, 2 only)"""
    prompt_path = Path(__file__).parent / "prompts" / "reflection_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load reflection prompt: {e}")
        return ""


def _load_spt_planner_prompt() -> str:
    """Load SPT planner prompt (Step 3 - 5 questions)"""
    prompt_path = Path(__file__).parent / "prompts" / "spt_planner_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load SPT planner prompt: {e}")
        return ""


def _load_response_prompt() -> str:
    """Load response generation prompt (Phase 2)"""
    prompt_path = Path(__file__).parent / "prompts" / "response_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load response prompt: {e}")
        return ""


class SonAgent:
    """
    아들 대화 에이전트 (AI 복원 찬성 - 책임 중심 관점)
    - LangChain ChatOpenAI 사용
    - 20대 초반 남성 아들 캐릭터
    - ✨ Three-Phase: Basic Reflection → SPT Planner (조건부) → Response
    - 아버지에게 존댓말 사용
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
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")

        self.session_store: Dict[str, ChatMessageHistory] = {}

        # 페르소나 프롬프트 로드
        prompt_path = Path(__file__).parent / "prompts" / "son_utilitarian.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.persona_prompt = f.read().strip()
                logger.info("✅ Son persona prompt loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Son persona prompt: {e}")
            self.persona_prompt = "당신은 아버지에게 어머니를 AI로 복원하는 것을 찬성하는 아들입니다. 가족의 행복과 이익을 근거로 찬성 의견을 말합니다."

        # Reflection Prompt 로드 (Step 1, 2)
        self.reflection_prompt = _load_reflection_prompt()
        if self.reflection_prompt:
            logger.info("✅ Reflection prompt loaded successfully")

        # SPT Planner Prompt 로드 (Step 3)
        self.spt_planner_prompt = _load_spt_planner_prompt()
        if self.spt_planner_prompt:
            logger.info("✅ SPT Planner prompt loaded successfully")

        # Response Prompt 로드 (Phase 2)
        self.response_prompt = _load_response_prompt()
        if self.response_prompt:
            logger.info("✅ Response prompt loaded successfully")

        # ✨ Phase 2 응답용 GPT-5-mini LLM 설정
        self.response_llm = ChatOpenAI(
            model="gpt-5-mini",
            api_key=self.api_key,
            temperature=1.0,
            max_completion_tokens=300
        )
        logger.info("✅ [Son] Using Gemini 2.5 Flash for Phase 1 & 1.5, GPT-5-mini for Phase 2")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """세션별 히스토리 가져오기 (없으면 생성)"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
            logger.info(f"✅ [Son] Created new session: {session_id}")
        return self.session_store[session_id]

    def clear_session(self, session_id: str) -> bool:
        """세션 히스토리 삭제"""
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info(f"✅ [Son] Cleared session: {session_id}")
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

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 파싱"""
        try:
            # ```json ... ``` 블록 제거
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"JSON parsing error: {e}, text: {text[:200]}")
            return {}

    async def _perform_basic_reflection(
        self,
        messages: List[Dict[str, str]],
        last_user_msg: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Phase 1: Basic Reflection - SPT 필요 여부만 판단
        Returns: { spt_required: bool, context_analysis: str, user_utterance_type: str }
        """
        logger.info(f"🧠 [PHASE1_START] session_id={session_id}, Performing basic reflection...")

        history_text = self._format_history(messages)

        reflection_prompt = self.reflection_prompt.format(
            last_user_msg=last_user_msg,
            history=history_text
        )

        try:
            # ✨ Phase 1: Gemini 2.5 Flash 사용
            reflection_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.google_api_key,
                temperature=0.3,
                max_output_tokens=500
            )

            result = await reflection_llm.ainvoke([
                HumanMessage(content=reflection_prompt)
            ])

            reflection_text = result.content.strip()
            logger.info(f"🧠 [PHASE1_OUTPUT] {reflection_text}")

            # JSON 파싱
            parsed = self._parse_json_response(reflection_text)

            spt_required = parsed.get("spt_required", False)
            context_analysis = parsed.get("context_analysis", "")
            user_utterance_type = parsed.get("user_utterance_type", "Question")

            logger.info(f"🧠 [SPT_REQUIRED] {spt_required}")
            logger.info(f"🧠 [PHASE1_COMPLETE] Basic reflection done")

            return {
                "spt_required": spt_required,
                "context_analysis": context_analysis,
                "user_utterance_type": user_utterance_type
            }

        except Exception as e:
            logger.error(f"🧠 [PHASE1_ERROR] {e}")
            return {
                "spt_required": False,
                "context_analysis": "Reflection failed",
                "user_utterance_type": "Question"
            }

    async def _perform_spt_planning(
        self,
        messages: List[Dict[str, str]],
        last_user_msg: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Phase 1.5: SPT Planner - 5개 질문 수행 (SPT=YES일 때만 호출)
        Returns: { stakeholders, empathy_analysis, stance_alignment, blind_spot, strategic_question }
        """
        logger.info(f"🎯 [SPT_PLANNER_START] session_id={session_id}, Performing SPT planning...")

        history_text = self._format_history(messages)

        spt_prompt = self.spt_planner_prompt.format(
            last_user_msg=last_user_msg,
            history=history_text
        )

        try:
            # ✨ Phase 1.5: Gemini 2.5 Flash 사용
            spt_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.google_api_key,
                temperature=0.3,
                max_output_tokens=500
            )

            result = await spt_llm.ainvoke([
                HumanMessage(content=spt_prompt)
            ])

            spt_text = result.content.strip()
            logger.info(f"🎯 [SPT_PLANNER_OUTPUT] {spt_text}")

            # JSON 파싱
            parsed = self._parse_json_response(spt_text)

            strategic_question = parsed.get("strategic_question", "")
            stakeholders = parsed.get("stakeholders", [])
            blind_spot = parsed.get("blind_spot", "")

            logger.info(f"🎯 [STRATEGIC_QUESTION] {strategic_question}")
            logger.info(f"🎯 [SPT_PLANNER_COMPLETE] SPT planning done")

            # Phase 2용 포맷팅된 섹션 생성
            formatted_section = ""
            if strategic_question:
                formatted_section = f"""=== SPT ANALYSIS (MUST USE) ===
Stakeholders: {', '.join(stakeholders)}
Blind spot user is missing: {blind_spot}
Strategic question to ask: {strategic_question}
=== END SPT ===

CRITICAL: You MUST include the strategic question in your response to guide the user's thinking."""

            return {
                "stakeholders": stakeholders,
                "empathy_analysis": parsed.get("empathy_analysis", ""),
                "stance_alignment": parsed.get("stance_alignment", ""),
                "blind_spot": blind_spot,
                "strategic_question": strategic_question,
                "formatted_section": formatted_section
            }

        except Exception as e:
            logger.error(f"🎯 [SPT_PLANNER_ERROR] {e}")
            return {
                "stakeholders": [],
                "empathy_analysis": "",
                "stance_alignment": "",
                "blind_spot": "",
                "strategic_question": "",
                "formatted_section": ""
            }

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

    def _build_response_prompt(
        self,
        reflection: Dict[str, Any],
        spt_result: Optional[Dict[str, Any]]
    ) -> str:
        """Phase 2용 응답 생성 프롬프트 구성"""

        # SPT 섹션은 SPT Planner에서 이미 포맷팅됨
        spt_section = spt_result.get("formatted_section", "") if spt_result else ""

        return self.response_prompt.format(
            user_utterance_type=reflection.get('user_utterance_type', 'Question'),
            context_analysis=reflection.get('context_analysis', ''),
            spt_section=spt_section
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,
        session_id: str = "default"
    ) -> str:
        """
        Three-Phase 대화:
        Phase 1: Basic Reflection (SPT 필요 여부 판단)
        Phase 1.5: SPT Planner (SPT=YES일 때만)
        Phase 2: 응답 생성
        """
        try:
            last_user_msg = self._extract_last_user_message(messages)

            # Phase 1: Basic Reflection
            reflection = await self._perform_basic_reflection(messages, last_user_msg, session_id)

            # Phase 1.5: SPT Planning (조건부)
            spt_result = None
            if reflection.get("spt_required", False):
                spt_result = await self._perform_spt_planning(messages, last_user_msg, session_id)

            # Phase 2: Response Generation (Gemini 우선)
            logger.info(f"🎭 [PHASE2_START] session_id={session_id}, Generating response...")

            # Gemini 사용 가능하면 Gemini, 아니면 OpenAI
            if self.response_llm:
                llm = self.response_llm
                logger.info(f"🎭 [PHASE2] Using Gemini for response")
            else:
                llm = self._create_llm(max_tokens, streaming=False, temperature=temperature)
                logger.info(f"🎭 [PHASE2] Using OpenAI for response")

            combined_prompt = self._build_response_prompt(reflection, spt_result)

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
            logger.error(f"❌ [Son] Error: {e}", exc_info=True)
            return "아버지, 다시 한 번만 말씀해주시겠어요?"

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,
        session_id: str = "default"
    ):
        """
        Three-Phase 스트리밍 대화:
        Phase 1: Basic Reflection (SPT 필요 여부 판단)
        Phase 1.5: SPT Planner (SPT=YES일 때만)
        Phase 2: 응답 스트리밍
        """
        try:
            last_user_msg = self._extract_last_user_message(messages)

            # Phase 1: Basic Reflection
            reflection = await self._perform_basic_reflection(messages, last_user_msg, session_id)

            # Phase 1.5: SPT Planning (조건부)
            spt_result = None
            if reflection.get("spt_required", False):
                spt_result = await self._perform_spt_planning(messages, last_user_msg, session_id)

            # Phase 2: Response Generation (Streaming)
            logger.info(f"🎭 [PHASE2_START] session_id={session_id}, Generating streamed response...")

            llm = self._create_llm(max_tokens, streaming=True, temperature=temperature)

            combined_prompt = self._build_response_prompt(reflection, spt_result)

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
            logger.error(f"❌ [Son] Streaming error: {e}", exc_info=True)
            for char in "아버지, 다시 한 번만 말씀해주시겠어요?":
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
