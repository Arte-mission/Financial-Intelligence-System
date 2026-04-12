from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone
from app.database import Base

class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    sector = Column(String, index=True)
    company_name = Column(String)
    listed_shares = Column(String, nullable=True)
    paidup_value = Column(String, nullable=True)
    total_paidup = Column(String, nullable=True)
    instrument_type = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class NewsArticle(Base):
    """
    SQLAlchemy model for storing news articles.
    """
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    source = Column(String, index=True)
    published_at = Column(String, nullable=True) # changed to String for simplicity, or we can use DateTime
    sentiment_score = Column(Float, nullable=True)
    query_used = Column(String, nullable=True)
    fetched_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class MarketMoodSnapshot(Base):
    __tablename__ = "market_mood_snapshots"
    id = Column(Integer, primary_key=True, index=True)
    average_score = Column(Float)
    mood_label = Column(String)
    headline_count = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
