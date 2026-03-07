from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
import sys

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    ChatGoogleGenerativeAI = None

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_all_prompts, load_json_data

from .stages import Stage1Handler, Stage2Handler, Stage3Handler
from .routing import ConversationRouter
from .persona_llm import PersonaLLM

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    stage: str
    previous_stage: str
    messages: List[Dict[str, str]]
    covered_topics: List[str]
    current_asking_topic: str
    message_count: int
    last_response: str
    artist_character_set: bool
    should_end: bool
    stage1_attempts: int
    stage2_question_asked: bool
    stage2_complete: bool
    stage2_completed: bool  # guards against duplicate stage2 processing
    current_question_index: int
    variation_index: int


class ConversationAgent:
    """3-stage conversation agent: Stage1 (artist character) -> Stage2 (AI art stance) -> Stage3 (ethics)."""

    def __init__(self, api_key: str, model: str = "gpt-5.1"):
        prompts_path = Path(__file__).parent / "prompts"
        self.prompts = load_all_prompts(str(prompts_path))
        logger.info(f"Loaded {len(self.prompts)} prompts: {list(self.prompts.keys())}")

        persona_prompt = self.prompts.get("persona", "당신은 예의 바른 화가 지망생입니다.")
        logger.info(f"Loaded persona prompt: {persona_prompt[:50]}...")

        # max_completion_tokens accounts for reasoning + output tokens in gpt-5 models
        base_llm = ChatOpenAI(
            model=model,
            temperature=1.0,
            max_completion_tokens=1000,
            api_key=api_key
        )

        self.llm = PersonaLLM(base_llm, persona_prompt)
        logger.info("Created PersonaLLM with embedded persona")

        # Prefer Gemini for intent detection (faster, cheaper); fall back to GPT
        gemini_api_key = os.getenv("GOOGLE_API_KEY", "")
        if HAS_GEMINI and gemini_api_key:
            self.analyzer = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                google_api_key=gemini_api_key
            )
            logger.info("Using Gemini (gemini-2.0-flash) for intent detection")
        else:
            self.analyzer = ChatOpenAI(
                model=model,
                temperature=1.0,
                max_completion_tokens=500,
                api_key=api_key
            )
            logger.info("Using GPT for intent detection (Gemini not available)")

        self.ethics_topics = load_json_data("ethics_topics.json", str(prompts_path))
        logger.info(f"Loaded {len(self.ethics_topics)} questions from ethics_topics")

        # Persona is already baked into self.llm, so pass empty persona_prompt
        self.stage1_handler = Stage1Handler(
            self.llm, self.prompts, persona_prompt="", analyzer=self.analyzer
        )
        self.stage2_handler = Stage2Handler(
            self.llm, self.prompts, persona_prompt="", analyzer=self.analyzer
        )
        self.stage3_handler = Stage3Handler(
            self.llm, self.analyzer, self.prompts, self.ethics_topics, persona_prompt=""
        )

        self.router = ConversationRouter(self.analyzer)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

        logger.info(f"ConversationAgent initialized with LangGraph + MemorySaver, model={model}")

    def _build_graph(self) -> StateGraph:
        """Each node produces one response then exits to END; stage transitions happen at entry."""
        workflow = StateGraph(ConversationState)

        workflow.add_node("stage1", self.stage1)
        workflow.add_node("stage2", self.stage2)
        workflow.add_node("stage3", self.stage3)

        workflow.set_conditional_entry_point(
            self.determine_stage,
            {
                "stage1": "stage1",
                "stage2": "stage2",
                "stage3": "stage3",
                "end": END
            }
        )

        workflow.add_edge("stage1", END)
        workflow.add_edge("stage2", END)
        workflow.add_edge("stage3", END)

        return workflow.compile(checkpointer=self.memory)

    def stage1(self, state: ConversationState) -> ConversationState:
        return self.stage1_handler.handle(state)

    def stage2(self, state: ConversationState) -> ConversationState:
        return self.stage2_handler.handle(state)

    def stage3(self, state: ConversationState) -> ConversationState:
        return self.stage3_handler.handle(state)

    def determine_stage(self, state: ConversationState) -> str:
        """Route to the appropriate stage based on current state flags."""
        logger.info(f"[ROUTER] determine_stage ENTRY")
        logger.info(f"[ROUTER] stage={state.get('stage')}, previous_stage={state.get('previous_stage')}")
        logger.info(f"[ROUTER] artist_character_set={state.get('artist_character_set')}, stage1_attempts={state.get('stage1_attempts')}")
        logger.info(f"[ROUTER] stage2_question_asked={state.get('stage2_question_asked')}, stage2_complete={state.get('stage2_complete')}, stage2_completed={state.get('stage2_completed')}")
        logger.info(f"[ROUTER] current_question_index={state.get('current_question_index')}, should_end={state.get('should_end')}")
        logger.info(f"[ROUTER] message_count={state.get('message_count')}, messages_len={len(state.get('messages', []))}")

        if state.get("should_end", False):
            logger.info("Routing to: end")
            return "end"

        current_stage = state.get("stage", "stage1")

        if current_stage == "stage1":
            if state.get("artist_character_set", False) or state.get("stage1_attempts", 0) >= 3:
                logger.info(f"[ROUTER] Stage 1 COMPLETE -> routing to: stage2")
                return "stage2"
            logger.info(f"[ROUTER] Stage 1 NOT complete -> routing to: stage1")
            return "stage1"

        elif current_stage == "stage2":
            if state.get("stage2_complete", False):
                logger.info(f"[ROUTER] Stage 2 COMPLETE -> routing to: stage3")
                return "stage3"
            logger.info(f"[ROUTER] Stage 2 NOT complete -> routing to: stage2")
            return "stage2"

        elif current_stage == "stage3":
            if state.get("should_end", False):
                logger.info(f"[ROUTER] Stage 3 COMPLETE -> routing to: end")
                return "end"
            logger.info(f"[ROUTER] Stage 3 NOT complete -> routing to: stage3")
            return "stage3"

        logger.info(f"[ROUTER] Fallback -> returning current_stage={current_stage}")
        return current_stage

    def process(self, state_dict: Dict[str, Any], user_input: str, thread_id: str = "default") -> Dict[str, Any]:
        logger.info(f"[PROCESS] ENTRY - user_input='{user_input[:50]}...', thread_id={thread_id}")
        logger.info(f"[PROCESS] ENTRY - stage={state_dict.get('stage')}, message_count={state_dict.get('message_count')}, messages_len={len(state_dict.get('messages', []))}")

        if "messages" not in state_dict:
            state_dict["messages"] = []
        state_dict["messages"].append({"role": "user", "content": user_input})

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(state_dict, config=config)

        logger.info(f"[PROCESS] EXIT - stage={result.get('stage')}, message_count={result.get('message_count')}, messages_len={len(result.get('messages', []))}")
        logger.info(f"[PROCESS] EXIT - stage2_complete={result.get('stage2_complete')}, stage2_completed={result.get('stage2_completed')}")
        logger.info(f"[PROCESS] EXIT - response='{result.get('last_response', '')[:80]}...'")

        return result

    def get_initial_message(self) -> str:
        return "선생님, 요새 잘 지내세요? 지난번 진로 조언 해주신 덕에 저도 열심히 연습 중입니다! 요즘 작품활동은 좀 어떠세요?"
