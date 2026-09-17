"""Request/response models. Validation lives here, not in the endpoints."""

import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)

TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")

# Below roughly two years of daily observations, a covariance matrix estimated
# from a wide universe is mostly noise.
MIN_HISTORY_DAYS = 90


class SignupCredentials(BaseModel):
    """Signup enforces the password policy."""

    email: EmailStr
    # 72 bytes is bcrypt's hard limit; anything longer is silently truncated.
    password: str = Field(min_length=8, max_length=72)


class LoginCredentials(BaseModel):
    """Login deliberately does NOT enforce the password policy.

    Validating length here would return 422 for a short password and 401 for a
    wrong one, which tells an attacker the policy and breaks any account whose
    password predates a policy change. Every bad login looks identical.
    """

    email: EmailStr
    password: str = Field(max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=50)
    start_date: date
    end_date: date
    algorithm: Literal["hrp", "mvo", "equal_weight"]
    max_weight: float = Field(default=0.20, gt=0.0, le=1.0)
    # Shrinkage is the default: the sample estimator is noisy enough that MVO
    # optimizes against its own estimation error. "sample" is kept so the two
    # can be compared.
    covariance_estimator: Literal["ledoit_wolf", "sample"] = "ledoit_wolf"

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, tickers: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for raw in tickers:
            ticker = raw.strip().upper()
            if not TICKER_PATTERN.match(ticker):
                raise ValueError(f"{raw!r} is not a valid ticker symbol")
            if ticker not in seen:
                seen.add(ticker)
                cleaned.append(ticker)

        if len(cleaned) < 2:
            raise ValueError("Need at least 2 distinct tickers")

        return cleaned

    @model_validator(mode="after")
    def check_date_window(self) -> "OptimizeRequest":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")

        if (self.end_date - self.start_date).days < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Date range must span at least {MIN_HISTORY_DAYS} days to "
                f"estimate a covariance matrix"
            )

        if self.start_date.year < 1970:
            raise ValueError("start_date is before available market history")

        return self

    @model_validator(mode="after")
    def check_weight_cap_is_feasible(self) -> "OptimizeRequest":
        """n assets capped at w can sum to at most n*w, which must reach 1.0."""
        if self.algorithm == "mvo" and len(self.tickers) * self.max_weight < 1.0:
            raise ValueError(
                f"max_weight={self.max_weight} is infeasible for "
                f"{len(self.tickers)} tickers: weights could sum to at most "
                f"{len(self.tickers) * self.max_weight:.2f}"
            )
        return self


class PortfolioMetrics(BaseModel):
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float


class OptimizationSyncResponse(BaseModel):
    tickers: list[str]
    weights: list[float]
    algorithm: str
    metrics: PortfolioMetrics
    missing_tickers: list[str] = Field(default_factory=list)
    covariance_diagnostics: dict[str, Any] = Field(default_factory=dict)


class TaskAccepted(BaseModel):
    task_id: str
    state: str = "PENDING"


class TaskStatus(BaseModel):
    task_id: str
    state: str
    result: dict[str, Any] | None = None
    error: str | None = None


class OptimizationRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: str | None
    tickers: list[str]
    algorithm: str
    start_date: str
    end_date: str
    weights: list[float]
    metrics: dict[str, float] | None
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    environment: str


class BacktestRequest(OptimizeRequest):
    """A backtest needs a longer window than a single optimization."""

    training_days: int = Field(default=252, ge=30, le=1260)
    rebalance_every: int = Field(default=21, ge=1, le=252)
    transaction_cost_bps: float = Field(default=10.0, ge=0.0, le=1000.0)
    strategies: list[Literal["equal_weight", "hrp", "mvo"]] | None = None

    @model_validator(mode="after")
    def check_window_leaves_out_of_sample(self) -> "BacktestRequest":
        """Reject a window too short to leave anything out of sample.

        ~252 trading days per calendar year; require at least 60 trading days
        beyond the training window or the result is a single noisy period.
        """
        calendar_days = (self.end_date - self.start_date).days
        approx_trading_days = calendar_days * 252 / 365

        if approx_trading_days < self.training_days + 60:
            raise ValueError(
                f"A {self.training_days}-day training window needs roughly "
                f"{(self.training_days + 60) * 365 / 252:.0f} calendar days of "
                f"data; this range has {calendar_days}."
            )
        return self


class CurvePoint(BaseModel):
    date: str
    value: float


class StrategyResultOut(BaseModel):
    strategy: str
    curve: list[CurvePoint]
    performance: dict[str, float]
    final_weights: dict[str, float]
    average_turnover: float
    total_cost_drag: float
    error: str | None = None


class BacktestResponse(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
    transaction_cost_bps: float
    strategies: list[StrategyResultOut]


class CurrencyRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=20)
    start_date: date
    end_date: date

    @field_validator("tickers")
    @classmethod
    def normalise(cls, tickers: list[str]) -> list[str]:
        cleaned, seen = [], set()
        for raw in tickers:
            ticker = raw.strip().upper()
            if not TICKER_PATTERN.match(ticker):
                raise ValueError(f"{raw!r} is not a valid ticker symbol")
            if ticker not in seen:
                seen.add(ticker)
                cleaned.append(ticker)
        if not cleaned:
            raise ValueError("At least one ticker is required")
        return cleaned

    @model_validator(mode="after")
    def check_window(self) -> "CurrencyRequest":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if (self.end_date - self.start_date).days < MIN_HISTORY_DAYS:
            raise ValueError(f"Date range must span at least {MIN_HISTORY_DAYS} days")
        return self


class FxSummary(BaseModel):
    start_rate: float
    end_rate: float
    total_depreciation: float
    curve: list[CurvePoint]


class CurrencyResponse(BaseModel):
    start_date: str
    end_date: str
    fx: FxSummary
    assets: list[dict[str, Any]]


class NseListingOut(BaseModel):
    ticker: str
    close_kes: float
    volume: float | None = None
    trade_date: str


class NseCoverageResponse(BaseModel):
    tickers_known: int
    price_rows: int
    latest_trade_date: str | None
    usable_for_covariance: int
    listings: list[NseListingOut]
