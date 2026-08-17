# src/database.py

from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# --- 1. Connect to the Vault (PostgreSQL) ---
# "postgresql://用户名:密码@地址:端口/数据库名"
# For local development, no password is usually set.
DATABASE_URL = "postgresql://localhost/portfolio_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 2. Define the Index Cards (Models) ---

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship: One user can have many optimization runs
    optimizations = relationship("OptimizationRun", back_populates="user")

class OptimizationRun(Base):
    __tablename__ = "optimizations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Inputs
    tickers = Column(JSON)  # e.g., ["AAPL", "MSFT"]
    algorithm = Column(String)  # "hrp" or "mvo"
    start_date = Column(String)
    end_date = Column(String)
    
    # Outputs
    weights = Column(JSON)  # e.g., [0.26, 0.12, ...]
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to user
    user = relationship("User", back_populates="optimizations")

# --- 3. Create the Tables (Build the filing system) ---
def init_db():
    Base.metadata.create_all(bind=engine)
    print("📁 Archive tables created successfully!")