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
    stage1_greeting_sent: bool
    stage1_follow_up_asked: bool
    stage2_question_asked: bool
    stage2_complete: bool
    stage2_completed: bool
    current_question_index: int
    dont_know_count: int


class ConversationAgent:
    """3-stage conversation agent: Stage1 (character setup) -> Stage2 (AI art stance) -> Stage3 (ethics exploration)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        prompts_path = Path(__file__).parent / "prompts"
        self.prompts = load_all_prompts(str(prompts_path))
        logger.info(f"Loaded {len(self.prompts)} prompts: {list(self.prompts.keys())}")

        persona_prompt = self.prompts.get("persona", "당신은 예의 바른 화가 지망생입니다.")
        logger.info(f"Loaded persona prompt: {persona_prompt[:50]}...")

        base_llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            api_key=api_key
        )

        self.llm = PersonaLLM(base_llm, persona_prompt)
        logger.info("Created PersonaLLM with embedded persona")

        # Intent analysis LLM - prefer Gemini (faster, cheaper)
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
                model="gpt-5-nano",
                temperature=1.0,
                max_completion_tokens=200,
                api_key=api_key
            )
            logger.info("Using gpt-5-nano for intent detection (Gemini not available)")

        self.ethics_topics = load_json_data("ethics_topics.json", str(prompts_path))
        logger.info(f"Loaded {len(self.ethics_topics)} questions from ethics_topics")

        # Persona is already baked into self.llm, so pass empty persona_prompt
        self.stage1_handler = Stage1Handler(
            self.llm,
            self.prompts,
            persona_prompt="",
            analyzer=self.analyzer
        )
        self.stage2_handler = Stage2Handler(
            self.llm,
            self.analyzer,
            self.prompts,
            persona_prompt=""
        )
        self.stage3_handler = Stage3Handler(
            self.llm,
            self.analyzer,
            self.prompts,
            self.ethics_topics,
            persona_prompt=""
        )

        self.router = ConversationRouter(self.analyzer)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

        logger.info(f"ConversationAgent initialized with LangGraph + MemorySaver, model={model}")

    def _build_graph(self) -> StateGraph:
        """Each node produces a response then terminates; stage transitions happen at next invoke."""
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
        logger.info(f"determine_stage - stage={state.get('stage')}, covered_topics={state.get('covered_topics', [])}, current_asking_topic={state.get('current_asking_topic', '')}")

        if state.get("should_end", False):
            logger.info("Routing to: end")
            return "end"

        current_stage = state.get("stage", "stage1")

        if current_stage == "stage1":
            if state.get("artist_character_set", False) or state.get("stage1_attempts", 0) >= 2:
                logger.info("Stage 1 complete - routing to: stage2")
                return "stage2"
            logger.info("Routing to: stage1")
            return "stage1"

        elif current_stage == "stage2":
            if state.get("stage2_complete", False):
                logger.info("Stage 2 complete - routing to: stage3")
                return "stage3"
            logger.info("Routing to: stage2")
            return "stage2"

        elif current_stage == "stage3":
            if state.get("should_end", False):
                logger.info("Routing to: end")
                return "end"
            logger.info("Routing to: stage3")
            return "stage3"

        return current_stage

    def process(self, state_dict: Dict[str, Any], user_input: str, thread_id: str = "default") -> Dict[str, Any]:
        if "messages" not in state_dict:
            state_dict["messages"] = []
        state_dict["messages"].append({"role": "user", "content": user_input})

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(state_dict, config=config)

        logger.info(f"[PROCESS] thread_id={thread_id}, stage={result.get('stage')}")

        return result

    def get_initial_message(self) -> str:
        return "너 요즘 그림도 잘 안 그린다며? 요새 뭐 하면서 지내?"
