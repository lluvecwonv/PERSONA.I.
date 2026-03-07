"""
Database models for persistent conversation storage
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class ConversationHistory(Base):
    """Persistent conversation history table."""
    __tablename__ = 'conversation_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), index=True, nullable=False)
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    stage = Column(String(50), nullable=False)
    covered_topics = Column(JSON, nullable=True)
    conversation_metadata = Column(JSON, nullable=True)  # 'metadata' is a SQLAlchemy reserved name
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<ConversationHistory(id={self.id}, session_id='{self.session_id}', stage='{self.stage}')>"


class ConversationSession(Base):
    """Session metadata table."""
    __tablename__ = 'conversation_sessions'

    session_id = Column(String(255), primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    final_stage = Column(String(50), nullable=True)
    total_messages = Column(Integer, default=0, nullable=False)
    covered_topics_count = Column(Integer, default=0, nullable=False)
    is_completed = Column(Integer, default=0, nullable=False)  # 0: in progress, 1: completed

    def __repr__(self):
        return f"<ConversationSession(session_id='{self.session_id}', messages={self.total_messages})>"


# Database connection setup
def get_database_url():
    default_url = os.getenv('DATABASE_URL')

    if not default_url:
        # Default: SQLite
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, 'conversations.db')
        default_url = f'sqlite:///{db_path}'

    return default_url


def init_db():
    database_url = get_database_url()
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session_maker():
    engine = init_db()
    return sessionmaker(bind=engine)
