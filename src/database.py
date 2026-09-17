"""Database engine, session management, and ORM models."""

from collections.abc import Iterator
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from src.config import get_settings

settings = get_settings()

# check_same_thread is a SQLite-only flag; passing it to Postgres raises.
connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow() -> datetime:
    """Timezone-aware UTC. ``datetime.utcnow()`` is deprecated and naive."""
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    optimizations = relationship(
        "OptimizationRun", back_populates="user", cascade="all, delete-orphan"
    )


class OptimizationRun(Base):
    __tablename__ = "optimizations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    task_id = Column(String, unique=True, index=True)

    tickers = Column(JSON, nullable=False)
    algorithm = Column(String, nullable=False)
    start_date = Column(String, nullable=False)
    end_date = Column(String, nullable=False)

    weights = Column(JSON, nullable=False)
    metrics = Column(JSON)

    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    user = relationship("User", back_populates="optimizations")


def get_db() -> Iterator[Session]:
    """FastAPI dependency. Guarantees the session is closed on every path."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
