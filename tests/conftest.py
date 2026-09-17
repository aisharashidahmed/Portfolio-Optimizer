"""Shared fixtures.

The API tests run against a real FastAPI app wired to a throwaway SQLite
database. Only the network boundary (price fetching) is faked -- everything
else, including auth and persistence, is exercised for real.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database import Base, get_db


@pytest.fixture
def db_session():
    """An isolated in-memory database per test."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # keep one connection so :memory: survives
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db_session):
    from src.api.main import app

    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client):
    """Register a user and return an Authorization header for them."""
    response = client.post(
        "/auth/signup", json={"email": "aisha@example.com", "password": "correct-horse"}
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def fake_market_data(monkeypatch):
    """Replace the network call with deterministic synthetic prices."""

    def _prepare(tickers, start_date, end_date):
        rng = np.random.default_rng(42)
        n_obs = 500
        index = pd.date_range("2022-01-03", periods=n_obs, freq="B")

        daily_returns = pd.DataFrame(
            {
                t: rng.normal(0.0004, 0.01 + 0.004 * i, n_obs)
                for i, t in enumerate(tickers)
            },
            index=index,
        )
        prices = 100 * (1 + daily_returns).cumprod()

        return {
            "prices": prices,
            "daily_returns": daily_returns,
            "expected_annual_returns": daily_returns.mean() * 252,
            "annual_covariance": daily_returns.cov() * 252,
            "missing_tickers": [],
        }

    monkeypatch.setattr("src.services.optimization.prepare_market_data", _prepare)
    return _prepare
