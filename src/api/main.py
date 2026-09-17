"""FastAPI application.

Two optimization paths are exposed deliberately:

* ``POST /optimize/sync``  -- runs inline. Fine for small universes, and it
  means the API is fully usable with no broker running.
* ``POST /optimize``       -- enqueues onto Celery. Used for large universes and
  for the ingestion-backed jobs added in later phases.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.algorithms.mvo import OptimizationFailedError
from src.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    CurrencyRequest,
    CurrencyResponse,
    HealthResponse,
    LoginCredentials,
    NseCoverageResponse,
    OptimizationRunOut,
    OptimizationSyncResponse,
    OptimizeRequest,
    SignupCredentials,
    TaskAccepted,
    TaskStatus,
    TokenResponse,
)
from src.auth import (
    create_access_token,
    get_current_user,
    get_password_hash,
    verify_password,
)
from src.config import get_settings
from src.database import OptimizationRun, User, get_db, init_db
from src.services.analysis import (
    AnalysisError,
    run_backtest_comparison,
    run_currency_analysis,
)
from src.services.optimization import (
    InsufficientDataError,
    UnknownAlgorithmError,
    run_optimization,
)
from src.tasks import celery_app, optimize_task

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(
    title="Portfolio Optimizer",
    description="Portfolio optimization and backtesting across US and Kenyan markets.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
@app.post(
    "/auth/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED
)
def signup(
    credentials: SignupCredentials, db: Session = Depends(get_db)
) -> TokenResponse:
    if db.query(User).filter(User.email == credentials.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )

    user = User(
        email=credentials.email,
        hashed_password=get_password_hash(credentials.password),
    )
    db.add(user)
    db.commit()

    return TokenResponse(access_token=create_access_token({"sub": user.email}))


@app.post("/auth/login", response_model=TokenResponse)
def login(credentials: LoginCredentials, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == credentials.email).first()

    # Same error for "no such user" and "wrong password" so the endpoint cannot
    # be used to enumerate registered email addresses.
    if user is None or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    return TokenResponse(access_token=create_access_token({"sub": user.email}))


# --------------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------------
@app.post("/optimize/sync", response_model=OptimizationSyncResponse)
def optimize_sync(
    request: OptimizeRequest,
    current_user: User = Depends(get_current_user),
) -> OptimizationSyncResponse:
    try:
        result = run_optimization(
            tickers=request.tickers,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            algorithm=request.algorithm,
            risk_free_rate=settings.risk_free_rate,
            max_weight=request.max_weight,
            covariance_estimator=request.covariance_estimator,
        )
    except (UnknownAlgorithmError, InsufficientDataError, OptimizationFailedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return OptimizationSyncResponse(**result.to_dict())


@app.post("/optimize", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
def optimize_async(
    request: OptimizeRequest,
    current_user: User = Depends(get_current_user),
) -> TaskAccepted:
    task = optimize_task.delay(
        tickers=request.tickers,
        start_date=request.start_date.isoformat(),
        end_date=request.end_date.isoformat(),
        algorithm=request.algorithm,
        max_weight=request.max_weight,
        user_id=current_user.id,
    )
    return TaskAccepted(task_id=task.id)


@app.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task_status(
    task_id: str, current_user: User = Depends(get_current_user)
) -> TaskStatus:
    """Read task state from Celery's result backend, the single source of truth."""
    async_result = AsyncResult(task_id, app=celery_app)
    state = async_result.state

    if state == "SUCCESS":
        return TaskStatus(task_id=task_id, state=state, result=async_result.result)

    if state == "FAILURE":
        return TaskStatus(task_id=task_id, state=state, error=str(async_result.result))

    # PENDING also covers unknown task ids -- Celery cannot distinguish them.
    return TaskStatus(task_id=task_id, state=state)


@app.get("/history", response_model=list[OptimizationRunOut])
def get_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[OptimizationRun]:
    return (
        db.query(OptimizationRun)
        .filter(OptimizationRun.user_id == current_user.id)
        .order_by(OptimizationRun.created_at.desc())
        .limit(min(limit, 100))
        .all()
    )


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
@app.post("/backtest", response_model=BacktestResponse)
def backtest(
    request: BacktestRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Walk-forward every strategy over identical data.

    Synchronous: a decade of daily data across three strategies runs in a few
    seconds. Longer universes belong on the queued path.
    """
    try:
        comparison = run_backtest_comparison(
            tickers=request.tickers,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            risk_free_rate=settings.risk_free_rate,
            training_days=request.training_days,
            rebalance_every=request.rebalance_every,
            transaction_cost_bps=request.transaction_cost_bps,
            max_weight=request.max_weight,
            strategies=request.strategies,
        )
    except (AnalysisError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return comparison.to_dict()


@app.post("/currency", response_model=CurrencyResponse)
def currency(
    request: CurrencyRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Split each asset's KES return into asset, currency, and interaction."""
    try:
        return run_currency_analysis(
            tickers=request.tickers,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
        )
    except (AnalysisError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@app.get("/nse", response_model=NseCoverageResponse)
def nse_coverage() -> dict:
    """What the NSE ingestion store currently holds.

    Public: it is market data, not user data, and the landing page shows it
    before anyone signs in.
    """
    from src.ingestion.store import MarketDataStore

    store = MarketDataStore("data/market_data.db")
    coverage = store.coverage()
    latest = store.latest_trade_date()

    listings = []
    if latest is not None:
        frame = store.load_prices(start_date=latest, end_date=latest)
        if not frame.empty:
            row = frame.iloc[-1]
            listings = [
                {
                    "ticker": ticker,
                    "close_kes": float(value),
                    "volume": None,
                    "trade_date": latest.isoformat(),
                }
                for ticker, value in row.items()
                if value == value  # drop NaN
            ]

    usable = 0
    if not coverage.empty:
        usable = int((coverage["observations"] >= 30).sum())

    return {
        "tickers_known": len(store.known_tickers()),
        "price_rows": store.row_count(),
        "latest_trade_date": latest.isoformat() if latest else None,
        "usable_for_covariance": usable,
        "listings": sorted(listings, key=lambda r: -r["close_kes"]),
    }


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", environment=settings.environment)
