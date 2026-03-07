"""Voice service using OpenAI TTS for the Friend Agent."""
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import base64

logger = logging.getLogger(__name__)


class FriendVoiceService:
    """Converts text responses to speech via OpenAI TTS and manages session state."""

    def __init__(self, conversation_agent, api_key: str):
        self.conversation_agent = conversation_agent
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("FriendVoiceService initialized")

    def get_or_create_state(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "stage": "stage1",
                "previous_stage": "",
                "messages": [],
                "covered_topics": [],
                "current_asking_topic": "",
                "message_count": 0,
                "last_response": "",
                "artist_character_set": False,
                "should_end": False,
                "stage1_attempts": 0,
                "stage2_question_asked": False,
                "stage2_complete": False,
                "stage2_completed": False
            }
            logger.info(f"Created new voice session: {session_id}")

        return self.sessions[session_id]

    async def text_to_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1"
    ) -> bytes:
        try:
            response = await self.openai_client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )

            audio_data = response.content
            logger.info(f"TTS generated: {len(audio_data)} bytes, voice={voice}")
            return audio_data

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            raise

    async def chat(
        self,
        message: str,
        session_id: str,
        voice: str = "alloy",
        include_audio: bool = True
    ) -> Dict[str, Any]:
        try:
            state = self.get_or_create_state(session_id)
            result = self.conversation_agent.process(state, message)
            response_text = result.get("last_response", "")

            if not response_text:
                logger.warning("Empty response from conversation agent")
                response_text = "죄송해요, 응답을 생성할 수 없었어요."

            self.sessions[session_id] = result

            response_dict = {
                "response": response_text,
                "session_id": session_id,
                "metadata": {
                    "stage": result.get("stage", "unknown"),
                    "message_count": result.get("message_count", 0),
                    "covered_topics": result.get("covered_topics", []),
                    "should_end": result.get("should_end", False)
                }
            }

            if include_audio:
                audio_data = await self.text_to_speech(response_text, voice=voice)
                response_dict["audio"] = base64.b64encode(audio_data).decode()
                response_dict["audio_bytes"] = audio_data
                logger.info(f"Audio included: {len(audio_data)} bytes")

            return response_dict

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            raise

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared voice session: {session_id}")
            return True
        return False

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)
