"""
SPT (Social Perspective Taking) Agent

V1: SPTAgent - ResponseType classification + Dynamic Instruction
V2: Simplified architecture - LLM-based Instruction generation (DST removed)
"""
from .conversation_agent import SPTAgent
from .spt_agent_v2 import SPTAgentV2

__all__ = [
    "SPTAgent",
    "SPTAgentV2",
]
