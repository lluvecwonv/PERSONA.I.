"""
SPT (Social Perspective Taking) Agent

V1: 기존 SPTAgent - ResponseType 분류 + Dynamic Instruction
V2: 새 아키텍처 - DST + Planner + Controller
"""
from .conversation_agent import SPTAgent

# V2 아키텍처 컴포넌트
from .dst import DialogueStateTracker, DialogueState, UserResponseStatus, SPTFrame
from .spt_controller import SPTController, ControllerDecision, ResponseStrategy
from .spt_planner import SPTPlanner, QuestionCandidate, SPTStrategy
from .spt_agent_v2 import SPTAgentV2, SPTAgentV2Wrapper

__all__ = [
    # V1 (기존)
    "SPTAgent",
    # V2 (새 아키텍처)
    "SPTAgentV2",
    "SPTAgentV2Wrapper",
    # DST
    "DialogueStateTracker",
    "DialogueState",
    "UserResponseStatus",
    "SPTFrame",
    # Controller
    "SPTController",
    "ControllerDecision",
    "ResponseStrategy",
    # Planner
    "SPTPlanner",
    "QuestionCandidate",
    "SPTStrategy",
]
