"""
SPT (Social Perspective Taking) Agent.
Response Type Classification + Dynamic System Instruction support.
"""
from typing import List, Dict, Optional
from pathlib import Path
import logging
import sys
import re
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

sys.path.append(str(Path(__file__).parent.parent))
from .utils import clean_gpt_response

logger = logging.getLogger(__name__)


class ResponseType:
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    UNCERTAIN = "UNCERTAIN"
    SHORT = "SHORT"
    QUESTION = "QUESTION"
    OTHER = "OTHER"


# Dynamic system instructions per response type (Korean, part of prompt logic - not comments)
RESPONSE_TYPE_INSTRUCTIONS = {
    ResponseType.AGREE: """
[사용자 상태: 동의함]
- 기존 주장을 단순 반복하지 말 것
- 새로운 이해관계자나 사회적 영향을 추가로 제시할 것
- "맞아", "그래" 같은 감정적 동조만 하지 말고, 반드시 새로운 논거를 포함할 것
- 질문은 최대 1개로 제한할 것
""",
    ResponseType.DISAGREE: """
[사용자 상태: 반대함]
- 사용자의 반대 논거를 먼저 인정(Mirror)할 것
- 그 논거의 타당성을 인정(Validate)한 후
- 그 기준 안에서 딜레마나 긴장을 제시(Tension Question)할 것
- 자신의 입장을 일방적으로 반복하지 말 것
""",
    ResponseType.UNCERTAIN: """
[사용자 상태: 모르겠음/불확실함]
- 질문만 던지지 말 것 (사용자가 이미 혼란스러운 상태)
- 먼저 구체적인 관점/사례/비유를 제시할 것
- 사용자의 망설임에 공감을 표현할 것
- "어떤 부분이 가장 걸리세요?" 같은 구체화 질문을 던질 것
""",
    ResponseType.SHORT: """
[사용자 상태: 단답/짧은 응답]
- 사용자가 생각을 정리하지 못한 상태임
- 새로운 질문을 던지기 전에 먼저 이해를 돕는 설명을 제공할 것
- 구체적인 예시나 상황을 들어 대화를 이끌 것
- "왜 그렇게 생각해?" 같은 근거 요청 질문을 자연스럽게 할 것
""",
    ResponseType.QUESTION: """
[사용자 상태: 질문함]
- 사용자의 질문에 직접적으로 답변할 것
- 답변 후 자신의 입장과 연결할 것
- 너무 긴 설명보다는 핵심만 간결하게 전달할 것
""",
    ResponseType.OTHER: """
[사용자 상태: 기타]
- 대화를 자연스럽게 이어갈 것
- SPT 전략을 활용해 상대의 관점을 이해하도록 유도할 것
"""
}


class SPTAgent:

    # Rule-based classification patterns
    AGREE_PATTERNS = [
        r'^(응|웅|ㅇㅇ|그래|맞아|맞어|그럴것같아|그럴거같아|그렇겠다|동의해|인정|ㅇㅋ|좋아|그치|그렇지|그런것같아|그런거같아|네|예|맞습니다|동의합니다|그렇네|그러네)$',
        r'(그럴것같|그럴거같|동의|맞는것같|맞는거같|인정해|좋은것같|좋은거같)',
    ]
    DISAGREE_PATTERNS = [
        r'^(아니|아냐|아뇨|노|ㄴㄴ|반대|그건아니|그건 아닌|싫어|별로)$',
        r'(아닌것같|아닌거같|반대야|동의못해|그건좀|그건 좀|모르겠어|싫은데|별로야)',
    ]
    UNCERTAIN_PATTERNS = [
        r'^(글쎄|몰라|모르겠|잘모르겠|음|흠|애매|모르겠어|모르겠네|잘 모르겠|글쎄요|모르겠어요)$',
        r'(모르겠|글쎄|애매해|확실하지|확실하지않|헷갈|고민|어렵다|어려워)',
    ]
    SHORT_PATTERNS = [
        r'^.{1,5}$',
        r'^(ㅎㅎ|ㅋㅋ|ㄱㄱ|ㅇㅇ|아|오|허|흠|음|네|예|응)$',
    ]
    QUESTION_PATTERNS = [
        r'\?$',
        r'(왜|어떻게|뭐|뭔|무슨|어디|언제|누가|뭘)',
    ]

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

        self.session_store: Dict[str, ChatMessageHistory] = {}

        # Classifier LLM: prefer Gemini, fall back to GPT
        gemini_api_key = os.getenv("GOOGLE_API_KEY", "")
        if HAS_GEMINI and gemini_api_key:
            self.classifier_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                google_api_key=gemini_api_key
            )
            logger.info("[SPT] Using Gemini for response type classification")
        else:
            self.classifier_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=api_key
            )
            logger.info("[SPT] Using gpt-4o-mini for response type classification (Gemini not available)")

        prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
                logger.info("SPT system prompt loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SPT system prompt: {e}")
            self.system_prompt = "당신은 Social Perspective Taking 전문가입니다."

    def classify_response_type(self, user_input: str) -> str:
        """Classify user response type: rule-based first, then LLM fallback."""
        text = user_input.strip()
        text_lower = text.lower().replace(" ", "")

        rule_result = self._classify_by_rules(text, text_lower)
        if rule_result:
            return rule_result

        logger.info(f"[SPT] Rule-based classification failed, using LLM for: '{text[:30]}...'")
        return self._classify_by_llm(user_input)

    def _classify_by_rules(self, text: str, text_lower: str) -> Optional[str]:
        """Rule-based classification (returns None if uncertain)."""

        if text.endswith('?') or re.search(r'(왜|어떻게|뭐|뭔|무슨|어디|언제|누가|뭘)\s*(\?|$)', text):
            logger.info(f"[SPT] ResponseType (rule): '{text[:20]}...' -> QUESTION")
            return ResponseType.QUESTION

        if len(text) <= 5 and text_lower in ['ㅎㅎ', 'ㅋㅋ', '응', '네', '어', '음', '흠', 'ㅇㅇ', 'ㄴㄴ']:
            logger.info(f"[SPT] ResponseType (rule): '{text}' -> SHORT")
            return ResponseType.SHORT

        strong_agree = ['맞아', '인정', '동의해', '그렇지', '맞는말', '맞는 말']
        for p in strong_agree:
            if text_lower.startswith(p) or text_lower == p:
                logger.info(f"[SPT] ResponseType (rule): '{text[:20]}...' -> AGREE")
                return ResponseType.AGREE

        strong_disagree = ['반대', '아니야', '안돼', '싫어', '절대']
        for p in strong_disagree:
            if text_lower.startswith(p) or text_lower == p:
                logger.info(f"[SPT] ResponseType (rule): '{text[:20]}...' -> DISAGREE")
                return ResponseType.DISAGREE

        strong_uncertain = ['글쎄', '모르겠', '애매해', '잘모르']
        for p in strong_uncertain:
            if p in text_lower:
                logger.info(f"[SPT] ResponseType (rule): '{text[:20]}...' -> UNCERTAIN")
                return ResponseType.UNCERTAIN

        return None

    def _classify_by_llm(self, user_input: str) -> str:
        try:
            classifier_prompt_path = Path(__file__).parent / "prompts" / "response_type_classifier.txt"
            with open(classifier_prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            prompt = prompt_template.format(user_input=user_input)

            response = self.classifier_llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip() if hasattr(response, 'content') else str(response)

            if '|' in result:
                label = result.split('|')[0].strip().upper()
            else:
                label = result.strip().upper()

            valid_labels = [ResponseType.AGREE, ResponseType.DISAGREE, ResponseType.UNCERTAIN,
                          ResponseType.SHORT, ResponseType.QUESTION, ResponseType.OTHER]
            if label in valid_labels:
                logger.info(f"[SPT] ResponseType (LLM): '{user_input[:20]}...' -> {label}")
                return label
            else:
                logger.warning(f"[SPT] LLM returned invalid label: {label}, defaulting to OTHER")
                return ResponseType.OTHER

        except Exception as e:
            logger.error(f"[SPT] LLM classification failed: {e}, defaulting to OTHER")
            return ResponseType.OTHER

    def get_dynamic_instruction(self, response_type: str) -> str:
        return RESPONSE_TYPE_INSTRUCTIONS.get(response_type, RESPONSE_TYPE_INSTRUCTIONS[ResponseType.OTHER])

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
            logger.info(f"[SPT] Created new session: {session_id}")
        return self.session_store[session_id]

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info(f"[SPT] Cleared session: {session_id}")
            return True
        return False

    @staticmethod
    def _extract_last_user_message(messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"].strip()
        return ""

    @staticmethod
    def _extract_previous_ai_messages(messages: List[Dict[str, str]]) -> List[str]:
        ai_messages = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                ai_messages.append(msg["content"].strip())
        return ai_messages

    def _build_messages(self, messages: List[Dict[str, str]], dynamic_instruction: Optional[str] = None):
        if dynamic_instruction:
            combined_system_prompt = f"{self.system_prompt}\n\n{dynamic_instruction}"
            logger.info(f"[SPT] Dynamic instruction added: {dynamic_instruction[:50]}...")
        else:
            combined_system_prompt = self.system_prompt

        lc_messages = [SystemMessage(content=combined_system_prompt)]
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

    def _create_llm(self, temperature: float, max_tokens: int, streaming: bool) -> ChatOpenAI:
        kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "streaming": streaming
        }
        # gpt-5 only supports temperature=1 and uses max_completion_tokens
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
        max_tokens: int = 500,
        session_id: str = "default",
        use_response_type: bool = True
    ) -> str:
        try:
            logger.info(f"[SPT] Starting chat with {len(messages)} messages, model={self.model}")

            last_user_msg = self._extract_last_user_message(messages)
            dynamic_instruction = None

            if use_response_type and last_user_msg:
                response_type = self.classify_response_type(last_user_msg)
                dynamic_instruction = self.get_dynamic_instruction(response_type)
                logger.info(f"[SPT] Response type: {response_type}")

            llm = self._create_llm(temperature, max_tokens, streaming=False)
            lc_messages = self._build_messages(messages, dynamic_instruction)
            logger.info(f"[SPT] Built {len(lc_messages)} LangChain messages, calling LLM...")

            result = await llm.ainvoke(lc_messages)
            raw_content = result.content if hasattr(result, "content") else str(result)
            logger.info(f"[SPT] Raw response ({len(raw_content)} chars): '{raw_content if raw_content else 'EMPTY'}'")

            cleaned_content = clean_gpt_response(raw_content)

            history = self.get_session_history(session_id)
            if last_user_msg:
                history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_content)

            logger.info(f"[SPT] session_id={session_id}, response: '{cleaned_content}'")
            return cleaned_content
        except Exception as e:
            logger.error(f"[SPT] Error generating response: {e}", exc_info=True)
            return "죄송해요, 다시 한번 말씀해주시겠어요?"

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        session_id: str = "default"
    ):
        try:
            llm = self._create_llm(temperature, max_tokens, streaming=True)
            lc_messages = self._build_messages(messages)

            full_response = ""
            async for chunk in llm.astream(lc_messages):
                chunk_text = ""
                if hasattr(chunk, "content"):
                    if isinstance(chunk.content, list):
                        chunk_text = "".join(part["text"] for part in chunk.content if isinstance(part, dict) and part.get("text"))
                    else:
                        chunk_text = str(chunk.content)
                else:
                    chunk_text = str(chunk)

                if chunk_text:
                    full_response += chunk_text
                    for char in chunk_text:
                        yield char

            cleaned_response = clean_gpt_response(full_response)

            history = self.get_session_history(session_id)
            if messages:
                last_user_msg = self._extract_last_user_message(messages)
                if last_user_msg:
                    history.add_user_message(last_user_msg)
            history.add_ai_message(cleaned_response)

            logger.info(f"[SPT] session_id={session_id}, streamed: '{cleaned_response}'")

        except Exception as e:
            logger.error(f"[SPT] Error streaming response: {e}", exc_info=True)
            error_message = "죄송해요, 다시 한번 말씀해주시겠어요?"
            for char in error_message:
                yield char

    def get_initial_message(self) -> str:
        return "안녕하세요! 어떻게 도와드릴까요?"
