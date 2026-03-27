"""Text-to-speech via OpenAI TTS API."""
import base64
import logging

logger = logging.getLogger(__name__)


async def text_to_speech(client, text: str, voice: str = "alloy", model: str = "tts-1") -> str:
    """Convert text to speech. Returns base64-encoded MP3."""
    if not text or not text.strip():
        raise ValueError("Empty text cannot be converted to speech")

    try:
        response = await client.audio.speech.create(model=model, voice=voice, input=text)
        audio_bytes = response.content
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"[TTS] Error: {type(e).__name__}: {e}")
        raise
