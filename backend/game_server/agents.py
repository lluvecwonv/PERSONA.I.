"""Agent imports, factory, and routing."""
import sys
from pathlib import Path
import logging

# Ensure parent directory is on sys.path for agent imports
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from agents.facilitator_agent import FacilitatorAgent
from agents.spt_agent import SPTAgent
from agents.colleague1_agent import Colleague1Agent
from agents.colleague2_agent import Colleague2Agent

try:
    from agents.jangmo_agent import JangmoAgent
except ImportError:
    JangmoAgent = None

try:
    from agents.son_agent import SonAgent
except ImportError:
    SonAgent = None

try:
    from utils.persona_validator import validate_and_fix_persona
    HAS_PERSONA_VALIDATOR = True
except ImportError:
    HAS_PERSONA_VALIDATOR = False
    validate_and_fix_persona = None

logger = logging.getLogger(__name__)


def create_agents(api_key: str, model: str = "gpt-4o") -> dict:
    """Instantiate all agents and return as a dict keyed by model name."""
    agents = {
        "artist-apprentice": FacilitatorAgent(api_key=api_key, model=model, persona_type="artist_apprentice"),
        "friend-agent": FacilitatorAgent(api_key=api_key, model=model, persona_type="friend"),
        "moral-agent-spt": SPTAgent(api_key=api_key),
        "colleague1": Colleague1Agent(api_key=api_key, model=model),
        "colleague2": Colleague2Agent(api_key=api_key, model=model),
    }
    if JangmoAgent:
        agents["jangmo"] = JangmoAgent(api_key=api_key, model=model)
    if SonAgent:
        agents["son"] = SonAgent(api_key=api_key, model=model)

    logger.info(f"Created {len(agents)} agents")
    return agents


def get_agent(agents: dict, model_key: str):
    """Resolve a model key to an agent instance."""
    if model_key in agents:
        return agents[model_key]
    # Aliases
    if model_key == "jangmo-agent" and "jangmo" in agents:
        return agents["jangmo"]
    if model_key == "son-agent" and "son" in agents:
        return agents["son"]
    # Default fallback
    return agents.get("artist-apprentice")
