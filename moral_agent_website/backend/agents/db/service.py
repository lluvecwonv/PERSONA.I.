"""
Database service for conversation history persistence
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from sqlalchemy.orm import Session
from .models import ConversationHistory, ConversationSession, get_session_maker

logger = logging.getLogger(__name__)


class DatabaseService:
    """대화 기록 영구 저장 서비스"""

    def __init__(self):
        """데이터베이스 세션 메이커 초기화"""
        self.SessionMaker = get_session_maker()
        logger.info("DatabaseService initialized")

    def save_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        stage: str,
        covered_topics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        대화 턴을 데이터베이스에 저장

        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지
            assistant_response: 어시스턴트 응답
            stage: 현재 스테이지 (stage1, stage2, stage3, end)
            covered_topics: 다룬 주제 리스트
            metadata: 추가 메타데이터

        Returns:
            성공 여부
        """
        db: Session = self.SessionMaker()
        try:
            # ConversationHistory 레코드 생성
            conversation = ConversationHistory(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                stage=stage,
                covered_topics=covered_topics or [],
                conversation_metadata=metadata or {}
            )
            db.add(conversation)
            db.commit()

            logger.info(f"Saved conversation turn for session {session_id}, stage={stage}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversation turn: {e}", exc_info=True)
            db.rollback()
            return False

        finally:
            db.close()

    def update_session_metadata(
        self,
        session_id: str,
        stage: str,
        total_messages: int,
        covered_topics_count: int,
        is_completed: bool = False
    ) -> bool:
        """
        세션 메타데이터 업데이트

        Args:
            session_id: 세션 ID
            stage: 현재 스테이지
            total_messages: 총 메시지 수
            covered_topics_count: 다룬 주제 수
            is_completed: 완료 여부

        Returns:
            성공 여부
        """
        db: Session = self.SessionMaker()
        try:
            # 기존 세션 조회
            session = db.query(ConversationSession).filter_by(session_id=session_id).first()

            if session:
                # 기존 세션 업데이트
                session.last_updated_at = datetime.utcnow()
                session.final_stage = stage
                session.total_messages = total_messages
                session.covered_topics_count = covered_topics_count
                session.is_completed = 1 if is_completed else 0
            else:
                # 새 세션 생성
                session = ConversationSession(
                    session_id=session_id,
                    final_stage=stage,
                    total_messages=total_messages,
                    covered_topics_count=covered_topics_count,
                    is_completed=1 if is_completed else 0
                )
                db.add(session)

            db.commit()
            logger.info(f"Updated session metadata for {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update session metadata: {e}", exc_info=True)
            db.rollback()
            return False

        finally:
            db.close()

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        특정 세션의 대화 기록 조회

        Args:
            session_id: 세션 ID

        Returns:
            대화 기록 리스트 (시간순 정렬)
        """
        db: Session = self.SessionMaker()
        try:
            conversations = (
                db.query(ConversationHistory)
                .filter_by(session_id=session_id)
                .order_by(ConversationHistory.created_at)
                .all()
            )

            return [
                {
                    "id": conv.id,
                    "user_message": conv.user_message,
                    "assistant_response": conv.assistant_response,
                    "stage": conv.stage,
                    "covered_topics": conv.covered_topics,
                    "conversation_metadata": conv.conversation_metadata,
                    "created_at": conv.created_at.isoformat()
                }
                for conv in conversations
            ]

        except Exception as e:
            logger.error(f"Failed to get session history: {e}", exc_info=True)
            return []

        finally:
            db.close()

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 메타데이터 조회

        Args:
            session_id: 세션 ID

        Returns:
            세션 정보 딕셔너리 또는 None
        """
        db: Session = self.SessionMaker()
        try:
            session = db.query(ConversationSession).filter_by(session_id=session_id).first()

            if session:
                return {
                    "session_id": session.session_id,
                    "started_at": session.started_at.isoformat(),
                    "last_updated_at": session.last_updated_at.isoformat(),
                    "final_stage": session.final_stage,
                    "total_messages": session.total_messages,
                    "covered_topics_count": session.covered_topics_count,
                    "is_completed": session.is_completed == 1
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get session info: {e}", exc_info=True)
            return None

        finally:
            db.close()

    def get_all_session_ids(self) -> List[str]:
        """
        모든 세션 ID 조회

        Returns:
            세션 ID 리스트
        """
        db: Session = self.SessionMaker()
        try:
            sessions = db.query(ConversationSession.session_id).all()
            return [s[0] for s in sessions]
        except Exception as e:
            logger.error(f"Failed to get all session IDs: {e}", exc_info=True)
            return []
        finally:
            db.close()

    def delete_session(self, session_id: str) -> bool:
        """
        세션과 관련된 모든 데이터 삭제

        Args:
            session_id: 세션 ID

        Returns:
            성공 여부
        """
        db: Session = self.SessionMaker()
        try:
            # 대화 기록 삭제
            db.query(ConversationHistory).filter_by(session_id=session_id).delete()

            # 세션 메타데이터 삭제
            db.query(ConversationSession).filter_by(session_id=session_id).delete()

            db.commit()
            logger.info(f"Deleted all data for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session: {e}", exc_info=True)
            db.rollback()
            return False

        finally:
            db.close()

    def clear_all_sessions(self) -> int:
        """
        모든 세션 및 대화 기록을 삭제하고, 삭제된 세션 수를 반환
        개발/테스트 용도
        """
        db: Session = self.SessionMaker()
        try:
            # 세션 개수 파악
            total_sessions = db.query(ConversationSession).count()
            # 대화 기록 삭제
            db.query(ConversationHistory).delete()
            # 세션 메타데이터 삭제
            db.query(ConversationSession).delete()
            db.commit()
            logger.info(f"Deleted all sessions and conversations: {total_sessions} sessions")
            return total_sessions
        except Exception as e:
            logger.error(f"Failed to clear all sessions: {e}", exc_info=True)
            db.rollback()
            return 0
        finally:
            db.close()
