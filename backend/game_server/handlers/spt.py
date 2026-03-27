"""SPT agent chat and stream handlers."""
import logging
from typing import Dict, Any, AsyncGenerator

from ..tts import text_to_speech
from ..exceptions import ChainExecutionError

logger = logging.getLogger(__name__)


async def handle_spt_chat(
    service, message: str, session_id: str,
    temperature: float, max_tokens: int,
    is_first_message: bool, include_audio: bool, voice: str,
) -> Dict[str, Any]:
    sessions = service.session_mgr.sessions
    if session_id not in sessions:
        sessions[session_id] = {"messages": []}

    session_data = sessions[session_id]

    if is_first_message:
        initial_message = service.agents["moral-agent-spt"].get_initial_message()
        session_data["messages"].append({"role": "assistant", "content": initial_message})
        result_dict = {
            "response": initial_message,
            "session_id": session_id,
            "metadata": {"stage": "spt", "message_count": 1, "is_end": False, "is_first": True},
        }
        if include_audio:
            try:
                result_dict["audio"] = await text_to_speech(service.openai_client, initial_message, voice=voice)
            except Exception:
                pass
        return result_dict

    session_data["messages"].append({"role": "user", "content": message})

    try:
        response_text = await service.agents["moral-agent-spt"].chat(
            messages=session_data["messages"], temperature=temperature, max_tokens=max_tokens,
        )
        session_data["messages"].append({"role": "assistant", "content": response_text})

        result_dict = {
            "response": response_text,
            "session_id": session_id,
            "metadata": {"stage": "spt", "message_count": len(session_data["messages"]), "is_end": False},
        }
        if include_audio:
            try:
                result_dict["audio"] = await text_to_speech(service.openai_client, response_text, voice=voice)
            except Exception:
                pass
        return result_dict

    except Exception as e:
        logger.error(f"SPT error: {e}")
        raise ChainExecutionError(f"SPT failed: {str(e)}")


async def handle_spt_stream(
    service, message: str, session_id: str,
    temperature: float, max_tokens: int,
    is_first_message: bool,
) -> AsyncGenerator[str, None]:
    sessions = service.session_mgr.sessions
    if session_id not in sessions:
        sessions[session_id] = {"messages": []}

    session_data = sessions[session_id]

    if is_first_message:
        initial_message = service.agents["moral-agent-spt"].get_initial_message()
        session_data["messages"].append({"role": "assistant", "content": initial_message})
        for char in initial_message:
            yield char
        return

    session_data["messages"].append({"role": "user", "content": message})

    try:
        response_text = await service.agents["moral-agent-spt"].chat(
            messages=session_data["messages"], temperature=temperature, max_tokens=max_tokens,
        )
        session_data["messages"].append({"role": "assistant", "content": response_text})
        for char in response_text:
            yield char
    except Exception as e:
        logger.error(f"SPT stream error: {e}")
        error_msg = "죄송해요, 다시 한번 말씀해주시겠어요?"
        for char in error_msg:
            yield char
