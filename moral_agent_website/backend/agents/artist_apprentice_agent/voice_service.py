"""
Artist Apprentice Agent Voice Service
OpenAI TTS API를 사용한 음성 응답 서비스
"""
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import base64

logger = logging.getLogger(__name__)


class ArtistApprenticeVoiceService:
    """
    Artist Apprentice Agent를 위한 음성 서비스
    - 텍스트 응답을 OpenAI TTS로 음성 변환
    - 세션 기반 대화 상태 관리
    """

    def __init__(self, conversation_agent, api_key: str):
        """
        음성 서비스 초기화

        Args:
            conversation_agent: Artist Apprentice Agent ConversationAgent 인스턴스
            api_key: OpenAI API 키
        """
        self.conversation_agent = conversation_agent
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ ArtistApprenticeVoiceService initialized")

    def get_or_create_state(self, session_id: str) -> Dict[str, Any]:
        """
        세션 상태 가져오기 또는 새로 생성

        Args:
            session_id: 세션 ID

        Returns:
            세션 상태 딕셔너리
        """
        if session_id not in self.sessions:
            # 새 세션 생성 (초기 상태)
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
        """
        텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            voice: 음성 종류 (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS 모델 (tts-1, tts-1-hd)

        Returns:
            MP3 오디오 데이터 (bytes)
        """
        try:
            response = await self.openai_client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )

            audio_data = response.content
            logger.info(f"✅ TTS generated: {len(audio_data)} bytes, voice={voice}")
            return audio_data

        except Exception as e:
            logger.error(f"❌ TTS error: {e}", exc_info=True)
            raise

    async def chat(
        self,
        message: str,
        session_id: str,
        voice: str = "alloy",
        include_audio: bool = True
    ) -> Dict[str, Any]:
        """
        사용자 메시지 처리 및 음성 응답 생성

        Args:
            message: 사용자 메시지
            session_id: 세션 ID
            voice: TTS 음성 종류
            include_audio: 오디오 데이터 포함 여부

        Returns:
            응답 딕셔너리 (텍스트 + 오디오 데이터)
        """
        try:
            # 세션 상태 가져오기
            state = self.get_or_create_state(session_id)

            # Artist Apprentice Agent로 응답 생성
            result = self.conversation_agent.process(state, message)

            # 응답 텍스트 추출
            response_text = result.get("last_response", "")

            if not response_text:
                logger.warning("Empty response from conversation agent")
                response_text = "죄송해요, 응답을 생성할 수 없었어요."

            # 세션 상태 업데이트
            self.sessions[session_id] = result

            # 응답 딕셔너리 생성
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

            # TTS 음성 생성 (옵션)
            if include_audio:
                audio_data = await self.text_to_speech(response_text, voice=voice)
                response_dict["audio"] = base64.b64encode(audio_data).decode()
                response_dict["audio_bytes"] = audio_data
                logger.info(f"✅ Audio included: {len(audio_data)} bytes")

            return response_dict

        except Exception as e:
            logger.error(f"❌ Chat error: {e}", exc_info=True)
            raise

    def clear_session(self, session_id: str) -> bool:
        """
        세션 삭제

        Args:
            session_id: 세션 ID

        Returns:
            삭제 성공 여부
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared voice session: {session_id}")
            return True
        return False

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 상태 조회

        Args:
            session_id: 세션 ID

        Returns:
            세션 상태 또는 None
        """
        return self.sessions.get(session_id)
