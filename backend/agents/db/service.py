"""Database service for conversation history persistence."""
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from sqlalchemy.orm import Session
from .models import ConversationHistory, ConversationSession, get_session_maker

logger = logging.getLogger(__name__)


class DatabaseService:

    def __init__(self):
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
        db: Session = self.SessionMaker()
        try:
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
        db: Session = self.SessionMaker()
        try:
            session = db.query(ConversationSession).filter_by(session_id=session_id).first()

            if session:
                session.last_updated_at = datetime.utcnow()
                session.final_stage = stage
                session.total_messages = total_messages
                session.covered_topics_count = covered_topics_count
                session.is_completed = 1 if is_completed else 0
            else:
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
        db: Session = self.SessionMaker()
        try:
            db.query(ConversationHistory).filter_by(session_id=session_id).delete()
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
        """Delete all sessions and conversations. For dev/test use."""
        db: Session = self.SessionMaker()
        try:
            total_sessions = db.query(ConversationSession).count()
            db.query(ConversationHistory).delete()
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
