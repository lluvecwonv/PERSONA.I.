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
    """대화 기록 영구 저장 테이블"""
    __tablename__ = 'conversation_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), index=True, nullable=False)
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    stage = Column(String(50), nullable=False)
    covered_topics = Column(JSON, nullable=True)
    conversation_metadata = Column(JSON, nullable=True)  # 'metadata'는 SQLAlchemy 예약어
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<ConversationHistory(id={self.id}, session_id='{self.session_id}', stage='{self.stage}')>"


class ConversationSession(Base):
    """세션 메타데이터 저장 테이블"""
    __tablename__ = 'conversation_sessions'

    session_id = Column(String(255), primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    final_stage = Column(String(50), nullable=True)
    total_messages = Column(Integer, default=0, nullable=False)
    covered_topics_count = Column(Integer, default=0, nullable=False)
    is_completed = Column(Integer, default=0, nullable=False)  # 0: 진행 중, 1: 완료

    def __repr__(self):
        return f"<ConversationSession(session_id='{self.session_id}', messages={self.total_messages})>"


# Database connection setup
def get_database_url():
    """환경 변수에서 DATABASE_URL 가져오기"""
    default_url = os.getenv('DATABASE_URL')

    if not default_url:
        # 기본값: SQLite (Docker 컨테이너용 경로)
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)  # 디렉토리 자동 생성
        db_path = os.path.join(data_dir, 'conversations.db')
        default_url = f'sqlite:///{db_path}'

    return default_url


def init_db():
    """데이터베이스 초기화"""
    database_url = get_database_url()
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session_maker():
    """SQLAlchemy 세션 메이커 생성"""
    engine = init_db()
    return sessionmaker(bind=engine)
