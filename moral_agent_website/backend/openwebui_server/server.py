from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import json

# 로깅 설정 (먼저 설정)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_service import LangChainService
from config import settings
from db import init_db
from pathlib import Path

# LangChain 서비스 인스턴스 생성
langchain_service = LangChainService()

app = FastAPI(
    title="LangChain Backend for Open WebUI",
    description="Open WebUI와 LangChain을 연결하는 백엔드 API",
    version="1.0.0"
)

# CORS 설정 (Open WebUI와 통신을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 애플리케이션 시작 시 데이터베이스 초기화
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 데이터베이스 초기화"""
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)


# 요청/응답 모델 정의
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    model: Optional[str] = "gpt-5-nano-2025-08-07"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    force_reset: Optional[bool] = False


class  ChatResponse(BaseModel):
    response: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None


# Health check 엔드포인트
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "LangChain Backend is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# OpenAI 호환 채팅 완료 엔드포인트 (Open WebUI용) - 스트리밍 지원
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat_completions_openai_format(request: Dict[str, Any]):
    """
    OpenAI API 호환 형식의 채팅 완료 엔드포인트 (스트리밍 지원)
    """
    try:
        messages = request.get("messages", [])
        if not messages:
            raise ValueError("Messages cannot be empty")

        # 마지막 메시지 가져오기
        last_message = messages[-1]
        user_message = last_message.get("content", "")

        # 테스트용: <skip> 입력 시 대화 종료
        if user_message.strip().lower() == "<skip>":
            logger.info("[TEST] Skip command received - ending conversation")
            skip_response = {
                "id": "chatcmpl-skip",
                "object": "chat.completion",
                "created": 1677652288,
                "model": request.get("model", "test"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "[대화 종료] 테스트가 완료되었습니다."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
            return skip_response

        model = request.get("model", "gpt-5-nano-2025-08-07")
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 1000)
        stream = request.get("stream", False)  # 스트리밍 요청 확인
        force_reset = request.get("force_reset", False)  # 강제 리셋 요청 확인

        # 모델 선택
        use_moral_agent = (model in [
            "friend-agent", "friend_agent",
            "artist-apprentice", "artist_apprentice",
            "moral-agent-spt",
            "colleague1", "colleague-1",
            "colleague2", "colleague-2",
            "jangmo", "jangmo-agent",
            "son", "son-agent"
        ])
        # 모든 moral agent는 langchain_service를 통해 처리

        # TTS는 별도 API로 분리 - 채팅 응답에서는 비활성화
        include_audio = False
        voice = request.get("voice", "alloy" if "artist" in model else "nova")  # TTS 음성 선택 (별도 API용)

        # 첫 대화 감지: assistant 메시지가 없으면 첫 대화
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        is_first_message = len(assistant_messages) == 0
        logger.info(f"First message detection: {is_first_message} (assistant_messages: {len(assistant_messages)})")

        # 세션 ID 추출 및 관리 개선
        session_id = request.get("chat_id") or request.get("conversation_id")

        # 디버깅: Open WebUI가 보내는 request 키 확인
        logger.info(f"REQUEST KEYS: {list(request.keys())}")
        logger.info(f"Provided session ID: {session_id}")

        # Open WebUI에서 세션 정보가 제대로 전달되지 않는 경우를 위한 개선
        if not session_id:
            # 메시지 히스토리를 기반으로 안정적인 세션 ID 생성
            import hashlib
            import uuid
            import time

            # Voice Agent: assistant 메시지가 있으면 기존 세션으로 간주
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]

            # 디버깅: 메시지 내용 확인
            logger.info(f"SESSION ID GENERATION - Total messages: {len(messages)}")
            logger.info(f"Assistant messages count: {len(assistant_messages)}")
            logger.info(f"User messages: {[msg.get('content', '')[:30] + '...' for msg in messages if msg.get('role') == 'user']}")
            if assistant_messages:
                logger.info(f"First assistant message: {assistant_messages[0].get('content', '')[:50]}...")

            # 일반 Agent: assistant 메시지 유무로 신규/복원 판단
            user_messages = [msg for msg in messages if msg.get("role") == "user"]

            # 시스템 프롬프트 필터링 (Open WebUI의 시스템 메시지 제외)
            system_patterns = [
                "### Task:",
                "Suggest 3-5 relevant follow-up questions",
                "Generate a concise",
                "Generate 1-3 broad tags",
                "word title with",
                "categorizing the"
            ]

            # 시스템 프롬프트를 제외한 실제 user 메시지만 필터링
            real_user_messages = []
            for msg in user_messages:
                content = msg.get("content", "").lower().strip()
                is_system = any(pattern.lower() in content for pattern in system_patterns)
                if not is_system:
                    real_user_messages.append(msg)

            logger.info(f"Filtered messages: {len(user_messages)} total -> {len(real_user_messages)} real user messages")

            # 새 채팅 감지: user 메시지가 1개뿐이면 새 채팅으로 간주
            # (Open WebUI가 "새 채팅"을 눌렀을 때의 동작)
            is_new_chat = len(real_user_messages) == 1
            logger.info(f"New chat detection: {is_new_chat} (real_user_messages: {len(real_user_messages)})")

            if assistant_messages and not is_new_chat:
                # assistant 메시지가 있고 user 메시지가 2개 이상이면 기존 세션 복원
                # 중요: 첫 user 메시지 + 첫 assistant 응답으로 세션 찾기
                first_user_content = real_user_messages[0].get("content", "") if real_user_messages else ""
                second_user_content = real_user_messages[1].get("content", "") if len(real_user_messages) > 1 else ""
                first_assistant_content = assistant_messages[0].get("content", "")

                # 첫 user + 첫 assistant 메시지 조합으로 세션 검색
                session_found = False
                try:
                    # DB에서 첫 user + 첫 assistant가 일치하는 세션 찾기
                    all_sessions = langchain_service.db_service.get_all_session_ids()
                    logger.info(f"[RESTORE] Searching {len(all_sessions)} sessions for match")
                    logger.info(f"[RESTORE] Target: user='{first_user_content[:30]}', assistant='{first_assistant_content[:30]}'")

                    for potential_session_id in all_sessions:
                        try:
                            db_history = langchain_service.db_service.get_session_history(potential_session_id)
                            if db_history and len(db_history) >= 1:
                                # 첫 user + 첫 assistant가 일치하는지 확인
                                db_first_user = db_history[0]["user_message"].strip()
                                db_first_assistant = db_history[0]["assistant_response"].strip()

                                first_turn_match = (
                                    db_first_user == first_user_content.strip() and
                                    db_first_assistant == first_assistant_content.strip()
                                )

                                if not first_turn_match:
                                    logger.debug(f"[RESTORE] Session {potential_session_id[:20]}: NO MATCH (user='{db_first_user[:20]}', assistant='{db_first_assistant[:20]}')")
                                else:
                                    logger.info(f"[RESTORE] Session {potential_session_id}: FIRST TURN MATCHED!")

                                # 두 번째 user 메시지가 있으면 그것도 확인 (더 정확한 매칭)
                                if len(real_user_messages) > 1 and len(db_history) >= 2:
                                    # 2턴 이상 대화 -> 두 번째 user도 비교
                                    second_turn_match = db_history[1]["user_message"].strip() == second_user_content.strip()
                                    if first_turn_match and second_turn_match:
                                        session_id = potential_session_id
                                        session_found = True
                                        logger.info(f"SESSION RESTORED by 2-turn match - session_id: {session_id}")
                                        break
                                elif len(real_user_messages) == 1 or len(db_history) == 1:
                                    # 1턴만 있거나 두 번째 메시지가 처음 -> 첫 턴만 비교
                                    if first_turn_match:
                                        session_id = potential_session_id
                                        session_found = True
                                        logger.info(f"SESSION RESTORED by 1-turn match - session_id: {session_id}")
                                        break
                        except:
                            continue
                except Exception as e:
                    logger.warning(f"Session search failed: {e}")

                if not session_found:
                    # 일치하는 세션을 찾지 못함 -> UUID로 새 세션 생성
                    session_id = f"chat_{uuid.uuid4().hex[:16]}"
                    logger.info(f"No matching session found - creating new session: {session_id}")
            elif real_user_messages:
                # assistant 메시지가 없으면 새 세션 생성 (UUID 사용)
                # 새 채팅은 항상 고유한 세션 ID 생성
                first_user_content = real_user_messages[0].get("content", "")
                session_id = f"chat_{uuid.uuid4().hex[:16]}"
                logger.info(f"NEW SESSION CREATED (no assistant messages) - first_user: '{first_user_content[:30]}...', session_id: {session_id}")
            else:
                # user 메시지가 없는 경우: 새 세션 생성
                session_id = f"chat_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000)}"
                logger.info(f"Generated new session_id (no user message): {session_id}")
        else:
            logger.info(f"Using provided session_id: {session_id}")

        logger.info(f"Using session_id: {session_id} (stream={stream}, messages: {len(messages)}, model={model}) for: {user_message[:30]}...")

        if use_moral_agent:
            if stream:
                # 스트리밍 응답
                async def generate_stream():
                    try:
                        full_response = ""
                        async for token in langchain_service.chat_stream(
                            message=user_message,
                            session_id=session_id,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            force_reset=force_reset,
                            is_first_message=is_first_message,
                            external_messages=messages  # ✨ 전체 메시지 히스토리 전달
                        ):
                            full_response += token
                            # OpenAI 스트리밍 형식
                            chunk = {
                                "id": "chatcmpl-123",
                                "object": "chat.completion.chunk",
                                "created": 1677652288,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": token},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                        # 스트림 종료 메시지
                        final_chunk = {
                            "id": "chatcmpl-123",
                            "object": "chat.completion.chunk",
                            "created": 1677652288,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"

                        if include_audio and full_response.strip():
                            try:
                                audio_b64 = await langchain_service.text_to_speech(full_response, voice=voice)
                                audio_chunk = {
                                    "type": "audio",
                                    "model": model,
                                    "audio": audio_b64
                                }
                                yield f"data: {json.dumps(audio_chunk)}\n\n"
                            except Exception as audio_err:
                                logger.error(f"Error generating TTS audio for stream: {audio_err}", exc_info=True)

                        yield "data: [DONE]\n\n"

                    except Exception as e:
                        logger.error(f"Error in moral agent streaming: {str(e)}", exc_info=True)
                        error_chunk = {
                            "error": {
                                "message": str(e),
                                "type": "server_error"
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"

                return StreamingResponse(generate_stream(), media_type="text/event-stream")
            else:
                # 비스트리밍 응답
                result = await langchain_service.chat(
                    message=user_message,
                    session_id=session_id,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    force_reset=force_reset,
                    include_audio=include_audio,
                    voice=voice,
                    is_first_message=is_first_message,
                    external_messages=messages  # ✨ 전체 메시지 히스토리 전달
                )

                # 응답 구성 (TTS 음성 포함 가능)
                response_data = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result["response"]
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }

                # TTS 음성 데이터가 있으면 추가
                if include_audio and "audio" in result:
                    response_data["audio"] = result["audio"]  # base64 encoded audio
                    logger.info(f"Including audio in response (size: {len(result['audio'])} chars)")

                return response_data

        # 스트리밍 요청 (기존 LangChain Service - 일반 모델)
        if stream:
            async def generate_stream():
                try:
                    full_response = ""
                    async for token in langchain_service.chat_stream(
                        message=user_message,
                        session_id=session_id,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        force_reset=force_reset,
                        is_first_message=is_first_message
                    ):
                        full_response += token
                        # OpenAI 스트리밍 형식
                        chunk = {
                            "id": "chatcmpl-123",
                            "object": "chat.completion.chunk",
                            "created": 1677652288,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # 스트림 종료 메시지
                    final_chunk = {
                        "id": "chatcmpl-123",
                        "object": "chat.completion.chunk",
                        "created": 1677652288,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

                    if include_audio and full_response.strip():
                        try:
                            audio_b64 = await langchain_service.text_to_speech(full_response, voice=voice)
                            audio_chunk = {
                                "type": "audio",
                                "model": model,
                                "audio": audio_b64
                            }
                            yield f"data: {json.dumps(audio_chunk)}\n\n"
                        except Exception as audio_err:
                            logger.error(f"Error generating TTS audio for generic stream: {audio_err}", exc_info=True)

                    yield "data: [DONE]\n\n"

                except Exception as e:
                    logger.error(f"Error in streaming: {str(e)}", exc_info=True)
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "server_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # 비스트리밍 요청 (기존 방식)
        else:
            result = await langchain_service.chat(
                message=user_message,
                session_id=session_id,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                force_reset=force_reset,
                is_first_message=is_first_message,
                include_audio=include_audio,
                voice=voice
            )

            # OpenAI API 호환 응답 형식
            response_payload = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result["response"]
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

            if include_audio and "audio" in result:
                response_payload["audio"] = result["audio"]

            return response_payload

    except Exception as e:
        logger.error(f"Error in chat completions endpoint: {str(e)}", exc_info=True)

        if "API key" in str(e).lower():
            raise HTTPException(status_code=500, detail="API key not configured")
        elif isinstance(e, ValueError):
            raise HTTPException(status_code=400, detail="Invalid request")
        else:
            raise HTTPException(status_code=500, detail="Internal server error")


# 메인 채팅 엔드포인트 (내부용)
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    메인 채팅 API 엔드포인트
    Open WebUI에서 호출됨
    """
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")

        result = await langchain_service.chat(
            message=request.message,
            session_id=request.session_id,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            force_reset=request.force_reset
        )

        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
            metadata=result.get("metadata")
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)

        if "API key" in str(e).lower():
            raise HTTPException(status_code=500, detail="API key not configured")
        elif isinstance(e, ValueError):
            raise HTTPException(status_code=400, detail="Invalid request")
        else:
            raise HTTPException(status_code=500, detail="Internal server error")


# 세션 관리 엔드포인트
@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """
    특정 세션의 대화 기록 삭제
    """
    try:
        logger.info(f"Clearing session: {session_id}")

        # LangChain 서비스에서 세션 삭제
        success = langchain_service.clear_session(session_id)

        if success:
            return {"status": "success", "message": f"Session {session_id} cleared"}
        else:
            return {"status": "warning", "message": f"Session {session_id} not found"}

    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/state")
async def get_session_state(session_id: str):
    """
    세션의 현재 상태 조회
    """
    try:
        state = langchain_service.get_session_state(session_id)

        if state:
            return {
                "status": "success",
                "session_id": session_id,
                "state": state
            }
        else:
            return {
                "status": "not_found",
                "message": f"Session {session_id} not found"
            }

    except Exception as e:
        logger.error(f"Error getting session state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/reset")
async def reset_session(session_id: str):
    """
    세션을 초기 상태로 리셋
    """
    try:
        logger.info(f"Resetting session: {session_id}")

        state = langchain_service.reset_session(session_id)

        return {
            "status": "success",
            "message": f"Session {session_id} reset to initial state",
            "state": state
        }

    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/database/clear")
async def clear_all_database():
    """
    전체 데이터베이스 초기화 (모든 세션 삭제)
    개발/테스트 용도로만 사용
    """
    try:
        logger.warning("Clearing entire database - all sessions will be deleted!")

        # 메모리 세션 초기화
        langchain_service.sessions.clear()
        logger.info("In-memory sessions cleared")

        # DB 모든 레코드 삭제
        deleted_count = langchain_service.db_service.clear_all_sessions()
        logger.info(f"Database cleared: {deleted_count} sessions deleted")

        return {
            "status": "success",
            "message": f"Database cleared successfully",
            "deleted_sessions": deleted_count
        }

    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# TTS API - 텍스트를 음성으로 변환
@app.post("/api/tts")
async def text_to_speech_api(request: Request):
    """
    텍스트를 음성으로 변환하는 API

    Request Body:
    - text: 변환할 텍스트 (필수)
    - voice: 음성 종류 (선택, 기본값: "alloy")
      - alloy, echo, fable, onyx, nova, shimmer
    - model: TTS 모델 (선택, 기본값: "tts-1")
      - tts-1 (빠름), tts-1-hd (고품질)

    Returns:
    - audio: base64 인코딩된 MP3 오디오
    """
    try:
        body = await request.json()
        text = body.get("text", "")
        voice = body.get("voice", "alloy")
        model = body.get("model", "tts-1")

        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        if len(text) > 4096:
            raise HTTPException(status_code=400, detail="text too long (max 4096 characters)")

        logger.info(f"TTS request: text='{text[:50]}...', voice={voice}, model={model}")

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
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# OpenAI API 호환 모델 목록 (Open WebUI용)
@app.get("/models")
@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models_openai_format():
    """
    OpenAI API 호환 형식으로 모델 목록 반환
    /models 및 /v1/models 경로 모두 지원
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "moral-agent-spt",
                "object": "model",
                "created": 1677610602,
                "owned_by": "moral-agent",
                "name": "SPT Agent"
            },
            {
                "id": "friend-agent",
                "object": "model",
                "created": 1677610602,
                "owned_by": "moral-agent",
                "name": "Friend Agent"
            },
            {
                "id": "artist-apprentice",
                "object": "model",
                "created": 1677610602,
                "owned_by": "moral-agent",
                "name": "Artist Apprentice Agent"
            },
            {
                "id": "colleague1",
                "object": "model",
                "created": 1677610602,
                "owned_by": "moral-agent",
                "name": "Colleague 1 Agent"
            },
            {
                "id": "colleague2",
                "object": "model",
                "created": 1677610602,
                "owned_by": "moral-agent",
                "name": "Colleague 2 Agent"
            },
            {
                "id": "jangmo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "moral-agent",
                "name": "Jangmo Agent (장모)"
            },
            {
                "id": "son",
                "object": "model",
                "created": 1677610602,
                "owned_by": "moral-agent",
                "name": "Son Agent (아들)"
            },
        ]
    }


# 사용 가능한 모델 목록 (내부용)
@app.get("/api/models")
async def list_models():
    """
    사용 가능한 LLM 모델 목록 반환
    """
    return {
        "models": [
            {"id": "moral-agent-spt", "name": "SPT Agent"},
            {"id": "friend-agent", "name": "Friend Agent"},
            {"id": "artist-apprentice", "name": "Artist Apprentice Agent"},
            {"id": "colleague1", "name": "Colleague 1 Agent"},
            {"id": "colleague2", "name": "Colleague 2 Agent"},
            {"id": "jangmo", "name": "Jangmo Agent (장모)"},
            {"id": "son", "name": "Son Agent (아들)"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
