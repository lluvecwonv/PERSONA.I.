"""Persona agent (colleague1/2, jangmo, son) chat and stream handlers."""
import logging
from typing import Dict, Any, List, AsyncGenerator

from ..tts import text_to_speech
from ..exceptions import ChainExecutionError
from ..agents import HAS_PERSONA_VALIDATOR, validate_and_fix_persona

logger = logging.getLogger(__name__)


async def handle_profile_chat(
    service, agent_key: str, agent, error_message: str,
    message: str, session_id: str, temperature: float, max_tokens: int,
    is_first_message: bool, include_audio: bool, voice: str,
    is_game_server: bool = False,
    external_messages: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    session_data = service.session_mgr.get_profile_session(agent_key, session_id, is_game_server)

    # Initialize from external messages if session is empty
    if external_messages and len(session_data["messages"]) == 0:
        filtered_messages = [
            msg for msg in external_messages
            if msg.get("role") in ["user", "assistant"]
            and not any(pattern in msg.get("content", "").lower() for pattern in [
                "### task:", "suggest 3-5", "generate a concise", "generate 1-3"
            ])
        ]
        if filtered_messages:
            if filtered_messages[-1].get("role") == "user":
                session_data["messages"] = filtered_messages[:-1]
            else:
                session_data["messages"] = filtered_messages
            user_count = len([m for m in session_data["messages"] if m.get("role") == "user"])
            session_data["turn_count"] = user_count + 1

    if is_first_message:
        initial_message = service._get_initial_message(agent, agent_key)
        session_data["messages"].append({"role": "assistant", "content": initial_message})

        if not message:
            session_data["turn_count"] = 0
            result_dict = {
                "response": initial_message,
                "session_id": session_id,
                "metadata": {
                    "stage": agent_key,
                    "message_count": len(session_data["messages"]),
                    "turn_count": session_data["turn_count"],
                    "is_end": False, "is_first": True,
                },
            }
            if include_audio:
                try:
                    result_dict["audio"] = await text_to_speech(service.openai_client, initial_message, voice=voice)
                except Exception:
                    pass
            return result_dict

    session_data["messages"].append({"role": "user", "content": message})
    if is_first_message:
        session_data["turn_count"] = 1
    else:
        session_data["turn_count"] += 1
    session_data["messages"] = service.session_mgr.trim_profile_history(session_data["messages"])

    # End conversation at turn 8 for persona agents
    if session_data["turn_count"] >= 8 and agent_key not in ["friend", "artist_apprentice"]:
        final_message = service._get_final_message(agent, agent_key)
        session_data["messages"].append({"role": "assistant", "content": final_message})
        result_dict = {
            "response": final_message,
            "session_id": session_id,
            "metadata": {
                "stage": agent_key,
                "message_count": len(session_data["messages"]),
                "turn_count": session_data["turn_count"],
                "is_end": True,
            },
        }
        if include_audio:
            try:
                result_dict["audio"] = await text_to_speech(service.openai_client, final_message, voice=voice)
            except Exception:
                pass
        return result_dict

    try:
        response_text = await agent.chat(
            messages=session_data["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=session_id,
        )

        if HAS_PERSONA_VALIDATOR and agent_key not in ["friend", "artist_apprentice"]:
            from ..config import settings
            response_text, _ = await validate_and_fix_persona(
                response_text, agent_key, settings.openai_api_key,
            )

        if not response_text or len(response_text.strip()) < 5 or response_text.strip() in ["...", "..."]:
            response_text = error_message

        session_data["messages"].append({"role": "assistant", "content": response_text})

        is_end = False if agent_key in ["friend", "artist_apprentice"] else session_data["turn_count"] >= 8

        result_dict = {
            "response": response_text,
            "session_id": session_id,
            "metadata": {
                "stage": agent_key,
                "message_count": len(session_data["messages"]),
                "turn_count": session_data["turn_count"],
                "is_end": is_end,
            },
        }
        if include_audio:
            try:
                result_dict["audio"] = await text_to_speech(service.openai_client, response_text, voice=voice)
            except Exception:
                pass
        return result_dict

    except Exception as e:
        logger.error(f"{agent_key} error: {e}")
        raise ChainExecutionError(f"{agent_key} failed: {str(e)}")


async def handle_profile_stream(
    service, agent_key: str, agent, error_message: str,
    message: str, session_id: str, temperature: float, max_tokens: int,
    is_first_message: bool, is_game_server: bool = False,
    external_messages: List[Dict[str, str]] = None,
    include_audio: bool = False, voice: str = "alloy",
) -> AsyncGenerator[str, None]:
    session_data = service.session_mgr.get_profile_session(agent_key, session_id, is_game_server)

    # Initialize from external messages if session is empty
    if external_messages and len(session_data["messages"]) == 0:
        filtered_messages = [
            msg for msg in external_messages
            if msg.get("role") in ["user", "assistant"]
            and not any(pattern in msg.get("content", "").lower() for pattern in [
                "### task:", "suggest 3-5", "generate a concise", "generate 1-3"
            ])
        ]
        if filtered_messages:
            if filtered_messages[-1].get("role") == "user":
                session_data["messages"] = filtered_messages[:-1]
            else:
                session_data["messages"] = filtered_messages
            user_count = len([m for m in session_data["messages"] if m.get("role") == "user"])
            session_data["turn_count"] = user_count + 1

    if is_first_message:
        initial_message = service._get_initial_message(agent, agent_key)
        session_data["messages"].append({"role": "assistant", "content": initial_message})

        if not message:
            session_data["turn_count"] = 0
            for char in initial_message:
                yield char
            if include_audio:
                try:
                    audio_data = await text_to_speech(service.openai_client, initial_message, voice=voice)
                    yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
                except Exception as e:
                    logger.warning(f"TTS failed: {e}")
            return

    session_data["messages"].append({"role": "user", "content": message})
    if is_first_message:
        session_data["turn_count"] = 1
    else:
        session_data["turn_count"] += 1
    session_data["messages"] = service.session_mgr.trim_profile_history(session_data["messages"])

    # End conversation at turn 8 for persona agents
    if session_data["turn_count"] >= 8 and agent_key not in ["friend", "artist_apprentice"]:
        final_message = service._get_final_message(agent, agent_key)
        session_data["messages"].append({"role": "assistant", "content": final_message})
        for char in final_message:
            yield char
        if include_audio:
            try:
                audio_data = await text_to_speech(service.openai_client, final_message, voice=voice)
                yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
            except Exception as e:
                logger.warning(f"TTS failed: {e}")
        return

    if not max_tokens or max_tokens > 300:
        max_tokens = 300

    try:
        response = await agent.chat(
            messages=session_data["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=session_id,
        )

        if HAS_PERSONA_VALIDATOR and agent_key not in ["friend", "artist_apprentice"]:
            from ..config import settings
            response, _ = await validate_and_fix_persona(
                response, agent_key, settings.openai_api_key,
            )

        if not response or len(response.strip()) < 5 or response.strip() in ["...", "..."]:
            response = error_message

        for char in response:
            yield char

        if include_audio:
            try:
                audio_data = await text_to_speech(service.openai_client, response, voice=voice)
                yield f"[AUDIO_START]{audio_data}[AUDIO_END]"
            except Exception as e:
                logger.warning(f"TTS failed: {e}")

        session_data["messages"].append({"role": "assistant", "content": response})
    except Exception as e:
        logger.error(f"{agent_key} stream error: {e}")
        for char in error_message:
            yield char
