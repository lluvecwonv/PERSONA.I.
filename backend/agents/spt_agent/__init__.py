"""
SPT (Social Perspective Taking) Agent

V1: SPTAgent - ResponseType classification + Dynamic Instruction
V2: SPTAgentV2 - Deprecated (SPT Reflection Framework now embedded in persona prompts)

Current Architecture:
- Persona agents load SPT Reflection Framework directly from spt_reflection_framework.txt
- No separate SPT agent call needed - single LLM call per turn
"""
from .conversation_agent import SPTAgent
from .spt_agent_v2 import SPTAgentV2  # Kept for backward compatibility

__all__ = [
    "SPTAgent",
    "SPTAgentV2",
]
