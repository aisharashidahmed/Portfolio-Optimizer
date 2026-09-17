# Portfolio Optimizer

Portfolio construction and backtesting for investors holding assets across
**two markets at once** — the Nairobi Securities Exchange and US equities.

Most portfolio tools assume you earn, spend, and invest in one currency. A
Kenyan investor holding US assets does not: their real return is the asset
return *and* the KES/USD move, and the second term frequently dominates the
first. This project treats that as a first-class modelling problem rather than
a footnote.

> **Status: active development.** The optimization engine, API, and test suite
> are working. NSE ingestion, the web UI, and the retrieval layer are in
> progress — see the [roadmap](#roadmap).

---

## What works today

- **Three allocation strategies** behind one interface — Hierarchical Risk
  Parity, Mean-Variance Optimization, and an equal-weight benchmark
- **Out-of-sample backtesting** — weights are learned on a training window and
  applied, frozen, to unseen data
- **REST API** with JWT auth, request validation, and per-user run history
- **Async job queue** for optimizations too slow to run inline
- **58 tests**, 89% line coverage, no network access required to run them

## Quickstart

Requires Python 3.11+. No database or Redis needed — it defaults to SQLite and
runs the optimizer inline.

```bash
git clone https://github.com/aisharashidahmed/Portfolio-Optimizer.git
cd Portfolio-Optimizer

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env
pytest                                    # 58 tests, ~10s, no network
uvicorn src.api.main:app --reload         # API docs at localhost:8000/docs
```

Run the engine directly against live market data:

```bash
python run_engine.py      # compare MVO against equal-weight
python run_backtest.py    # train/test split, out-of-sample results
```

## Architecture

```
                 ┌──────────────┐
  HTTP  ────────▶│  FastAPI     │  auth · validation · routing
                 │  src/api     │
                 └──────┬───────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
      ┌───────────────┐    ┌──────────────┐
      │ inline path   │    │ Celery queue │  long-running jobs
      │ /optimize/sync│    │  src/tasks   │
      └───────┬───────┘    └──────┬───────┘
              └─────────┬─────────┘
                        ▼
              ┌──────────────────────┐
              │ src/services         │  one orchestration path
              │   optimization.py    │
              └──────────┬───────────┘
                         │
         ┌───────────────┼────────────────┐
         ▼               ▼                ▼
  ┌────────────┐  ┌─────────────┐  ┌────────────┐
  │data_fetcher│  │ algorithms  │  │  metrics   │
  │ (provider  │  │ hrp·mvo·1/N │  │ return·vol │
  │  boundary) │  │             │  │  ·sharpe   │
  └────────────┘  └─────────────┘  └────────────┘
```

Two design decisions worth calling out:

**One data path.** Every price fetch goes through `src/data_fetcher.py`. The
API and the Celery worker both call `src/services/optimization.py`, so there is
no route through the system that bypasses validation or normalisation.

**Sync and async, deliberately.** `/optimize/sync` runs inline so the API is
fully usable with no broker installed. `/optimize` enqueues onto Celery for
large universes and for the ingestion-backed jobs coming in Phase 2. Queuing
work that takes 200ms would be architecture theatre; queuing a multi-year
multi-market backtest is not.

## The algorithms

| Strategy | Approach | When it wins |
|---|---|---|
| **Equal weight (1/N)** | Uniform allocation | The benchmark. DeMiguel, Garlappi & Uppal (2009) found it beats most optimizers out-of-sample once estimation error is counted. Any strategy that can't beat it isn't earning its complexity. |
| **Mean-Variance** | Maximises Sharpe via SLSQP under a per-asset weight cap | Well-conditioned covariance and genuine views on expected returns |
| **Hierarchical Risk Parity** | Clusters by correlation distance, splits capital down the dendrogram by inverse variance | Ill-conditioned covariance, or more assets than clean observations — it never inverts the covariance matrix |

### On correctness

HRP's correlation matrix is derived from the covariance matrix analytically:

```
corr = D⁻¹ Σ D⁻¹     where D = diag(√diag(Σ))
```

An earlier version called `cov_matrix.corr()`, which treats the covariance
matrix's *columns* as observations and returns something meaningless. On
synthetic data with true pairwise correlations near zero, it produced
correlations near −0.4, so every allocation was clustering on noise.
`tests/test_hrp.py::test_recovers_the_true_correlation_matrix` pins the fix.

## Testing

```bash
pytest                              # everything, no network
pytest --cov=src --cov-report=term  # with coverage
pytest tests/test_hrp.py -v         # one module
ruff check src tests                # lint
```

The suite makes no network calls — the yfinance boundary is faked in
`tests/conftest.py`, and the API tests run against a real FastAPI app wired to
an in-memory SQLite database. Auth and persistence are exercised for real
rather than mocked, so the tests catch integration bugs and not just unit ones.

Tests are organised by behaviour, not by coverage target:

- **Financial invariants** — weights sum to 1, no shorts, the weight cap is
  respected, riskier assets get less
- **Provider hostility** — delisted tickers, all-NaN columns, sparse history,
  yfinance's shape changing between one ticker and many, mismatched trading
  calendars
- **Security** — login returns an identical response for an unknown email and a
  wrong password, so the endpoint can't be used to enumerate users
- **Infeasible input** — 3 assets capped at 20% can never sum to 100%, and the
  API rejects that before it reaches the solver

## Configuration

All configuration is environment-driven via `src/config.py`; nothing secret is
hardcoded. Setting `ENVIRONMENT=production` while the JWT signing key is still
the development default raises at startup rather than silently shipping a
known key.

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` enforces a real `SECRET_KEY` |
| `SECRET_KEY` | dev placeholder | `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `DATABASE_URL` | `sqlite:///./portfolio.db` | Postgres: `postgresql+psycopg://…` |
| `REDIS_URL` | `redis://localhost:6379/0` | Only needed for the async path |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | JSON list |

## Roadmap

- [x] **Phase 0** — Correct HRP, single data path, environment config, API test suite
- [ ] **Phase 1** — Covariance shrinkage, transaction costs, property-based tests
- [ ] **Phase 2** — NSE ingestion pipeline and KES/USD currency-risk decomposition
- [ ] **Phase 3** — Alembic migrations, Docker Compose, structured logging
- [ ] **Phase 4** — Next.js + TypeScript frontend
- [ ] **Phase 5** — Retrieval over SEC EDGAR filings, then NSE annual reports

## Disclaimer

Educational software, not investment advice. Historical returns are a poor
predictor of future ones, and every optimizer here will happily produce a
confident allocation from a covariance matrix that is mostly estimation noise.
