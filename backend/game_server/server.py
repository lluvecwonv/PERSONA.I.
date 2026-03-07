import sys
from pathlib import Path
import os
from typing import Optional

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import asyncpg
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

db_pool = None

async def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", 5432)),
            user=os.environ.get("DB_USER", "persona_i_admin"),
            password=os.environ.get("DB_PASSWORD", "persona_i_admin_11_17!!"),
            database=os.environ.get("DB_NAME", "persona_i"),
            min_size=1,
            max_size=5
        )
    return db_pool

async def get_conversation_history(context_id: str):
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT user_message, ai_message
                FROM conversations
                WHERE context_id = $1
                ORDER BY created_at ASC
                """,
                context_id
            )
            messages = []
            for row in rows:
                messages.append({"role": "user", "content": row["user_message"]})
                messages.append({"role": "assistant", "content": row["ai_message"]})
            return messages
    except Exception as e:
        logger.error(f"[DB] Failed to get conversation history: {e}")
        return []

try:
    from .langchain_service import LangChainService
except ImportError:
    from langchain_service import LangChainService

langchain_service = LangChainService()

# Track last context_id per (user_id, agent_id) to detect context changes
user_agent_last_context: dict[tuple[str, str], str] = {}

AGENTS = {
    "agent-1": {
        "id": "agent-1",
        "name": "Artist Apprentice",
        "model": "artist-apprentice",
        "description": "AI 화가 윤리 대화 게임 - 화가 지망생 에이전트",
        "voice": "alloy",
        "status": "active"
    },
    "agent-2": {
        "id": "agent-2",
        "name": "Friend Agent",
        "model": "friend-agent",
        "description": "AI 윤리 대화 게임 - 친구 에이전트 (죽은 사람을 재현하는 AI)",
        "voice": "nova",
        "status": "active"
    },
    "agent-3": {
        "id": "agent-3",
        "name": "Colleague 1 (동료 화가)",
        "model": "colleague1",
        "description": "AI 화가 윤리 대화 게임 - 의무론적 관점에서 AI 예술을 반대하는 동료 화가",
        "voice": "shimmer",
        "status": "active"
    },
    "agent-4": {
        "id": "agent-4",
        "name": "Colleague 2 (동료 화가)",
        "model": "colleague2",
        "description": "AI 화가 윤리 대화 게임 - 공리주의 관점에서 AI 예술을 찬성하는 동료 화가",
        "voice": "echo",
        "status": "active"
    },
    "agent-5": {
        "id": "agent-5",
        "name": "Jangmo (장모)",
        "model": "jangmo",
        "description": "AI 복원 윤리 대화 게임 - 의무론적 관점에서 AI 복원을 반대하는 장모",
        "voice": "fable",
        "status": "active"
    },
    "agent-6": {
        "id": "agent-6",
        "name": "Son (아들)",
        "model": "son",
        "description": "AI 복원 윤리 대화 게임 - 공리주의 관점에서 AI 복원을 찬성하는 아들",
        "voice": "onyx",
        "status": "active"
    }
}

app = FastAPI(
    title="Persona-Agent Game Server",
    description="AI ethics dialogue game backend server (REST API)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing Moral Agent Game Server...")
        logger.info(f"{len([a for a in AGENTS.values() if a['status'] == 'active'])} active agents loaded")
        logger.info("Moral Agent Game Server started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}", exc_info=True)



class ChatMessageRequest(BaseModel):
    agent_id: str
    message: str
    user_id: str
    stream: bool = False
    include_audio: bool = False
    voice: Optional[str] = None


class ChatMetadata(BaseModel):
    stage: str = "stage1"
    message_count: int = 0
    turn_count: int = 0
    is_end: bool = False
    is_first: bool = False


class ChatResponseData(BaseModel):
    response: str
    metadata: ChatMetadata


class ChatResponse(BaseModel):
    status: str
    data: ChatResponseData


@app.get("/health")
async def health_check():
    return {"status": "ok"}



@app.post("/api/game/context/{context_id}/chat")
async def chat_message(context_id: str, request: ChatMessageRequest):
    logger.info(f"=" * 60)
    logger.info(f"[API] /chat ENTRY (stream={request.stream})")
    logger.info(f"[API] user_id={request.user_id}")
    logger.info(f"[API] context_id={context_id}")
    logger.info(f"[API] agent_id={request.agent_id}")
    logger.info(f"[API] message='{request.message}'")

    # Detect context_id change -> start new session (clear previous conversation)
    global user_agent_last_context
    key = (request.user_id, request.agent_id)
    last_context = user_agent_last_context.get(key)
    if last_context and last_context != context_id:
        logger.info(f"[API] context_id CHANGED for {request.agent_id}: {last_context[:8]}... -> {context_id[:8]}...")
        langchain_service.clear_game_session(last_context)
        logger.info(f"[API] Starting fresh session for new context_id: {context_id[:8]}...")
    user_agent_last_context[key] = context_id

    if request.agent_id not in AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
    agent_info = AGENTS[request.agent_id]



    if agent_info["status"] != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Agent {request.agent_id} is not active (status: {agent_info['status']})"
        )

    voice = request.voice or agent_info.get("voice", "alloy")

    # <skip> command - end conversation immediately
    if request.message.strip().lower() == "<skip>":
        logger.info(f"[API] <skip> command received - ending conversation immediately")
        skip_response = "대화가 종료되었습니다."
        return {
            "status": "success",
            "data": {
                "response": skip_response,
                "metadata": {
                    "stage": "ended",
                    "message_count": 0,
                    "turn_count": 0,
                    "is_end": True,
                    "is_first": False
                }
            }
        }

    # Streaming mode
    if request.stream:
        async def generate():
            try:
                full_response = ""
                full_response = ""
                async for token in langchain_service.chat_stream(
                    message=request.message,
                    session_id=context_id,
                    model=agent_info["model"],
                    is_first_message=False,
                    is_game_server=True
                ):
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

                state = langchain_service.get_or_create_state(context_id, is_game_server=True)
                metadata = {
                    "stage": state.get("stage", "unknown"),
                    "message_count": state.get("message_count", 0),
                    "is_end": state.get("should_end", False)
                }
                final_payload = {
                    "done": True,
                    "metadata": metadata,
                    "full_response": full_response
                }
                if request.include_audio and full_response.strip():
                    try:
                        audio_b64 = await langchain_service.text_to_speech(full_response, voice=voice)
                        final_payload["audio"] = audio_b64
                    except Exception as audio_err:
                        logger.error(f"[STREAM] Failed to synthesize audio: {audio_err}", exc_info=True)

                yield f"data: {json.dumps(final_payload)}\n\n"
                logger.info(f"[STREAM] Completed - response length: {len(full_response)}")

            except Exception as e:
                logger.error(f"Stream error: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming mode
    try:
        logger.info(f"[API] Calling langchain_service.chat()")
        result = await langchain_service.chat(
            message=request.message,
            session_id=context_id,
            model=agent_info["model"],
            is_first_message=False,
            is_game_server=True,
            include_audio=request.include_audio,
            voice=voice
        )

        logger.info(f"[API] RESULT stage={result.get('metadata', {}).get('stage')}")
        logger.info(f"[API] RESULT is_end={result.get('metadata', {}).get('is_end')}")
        logger.info(f"[API] RESULT response='{result.get('response', '')}'")
        logger.info(f"=" * 60)

        # Return response only (DB save handled by BE)
        response_payload = {
            "response": result["response"],
            "metadata": result.get("metadata", {})
        }

        if request.include_audio and "audio" in result:
            response_payload["audio"] = result["audio"]

        return {
            "status": "success",
            "data": response_payload
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts")
async def text_to_speech_api(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        voice = body.get("voice", "alloy")
        model = body.get("model", "tts-1")

        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        if len(text) > 4096:
            raise HTTPException(status_code=400, detail="text too long (max 4096 characters)")

        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice not in valid_voices:
            voice = "alloy"

        logger.info(f"[TTS] Generating audio: text_length={len(text)}, voice={voice}, model={model}")

        audio_base64 = await langchain_service.text_to_speech(text, voice=voice, model=model)

        return {
            "status": "success",
            "audio": audio_base64,
            "format": "mp3",
            "voice": voice,
            "model": model
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TTS] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8002))
    logger.info(f"Starting Moral Agent Game Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
