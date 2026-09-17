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
- **Ledoit-Wolf covariance shrinkage**, implemented from the paper rather than
  imported, with the optimal shrinkage intensity derived analytically
- **Walk-forward backtesting** with weight drift, per-rebalance transaction
  costs, and structurally enforced no-lookahead
- **NSE ingestion pipeline** — all 71 Nairobi-listed companies, idempotent
  storage, schema-change detection, and a robots.txt-compliant crawl delay
- **Currency risk decomposition** — splits a Kenyan investor's return on a US
  asset into the asset, the shilling, and the interaction between them
- **Next.js + TypeScript frontend** with hand-rolled SVG charts on a
  colour-vision-validated palette, a data table behind every chart, and
  working light and dark modes
- **REST API** with JWT auth, request validation, and per-user run history
- **Async job queue** for optimizations too slow to run inline
- **277 tests**, 90% line coverage, no network access required to run them

## Quickstart

Requires Python 3.11+. No database or Redis needed — it defaults to SQLite and
runs the optimizer inline.

**Backend:**

```bash
git clone https://github.com/aisharashidahmed/Portfolio-Optimizer.git
cd Portfolio-Optimizer

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env
pytest                                    # 277 tests, ~20s, no network
uvicorn src.api.main:app --reload         # API docs at localhost:8000/docs
```

**Frontend** (in a second terminal):

```bash
cd frontend
npm install
npm run dev                               # http://localhost:3000
```

The UI signs in with a pre-filled demo account created on first use. The NSE
page is public and needs no sign-in.

Run the engine directly against live market data:

```bash
python run_engine.py      # compare MVO against equal-weight
python run_backtest.py    # train/test split, out-of-sample results
```

## The Kenya problem

A Kenyan investor holding US assets does not earn the asset's return. They earn
the asset's return compounded with the KES/USD move, and over the last decade
the currency term was the larger of the two for several assets.

**S&P 500, 2015–2024, for a Kenyan investor:**

| | |
|---|---:|
| Asset return (USD) | 245.8% |
| Currency return | 41.2% |
| Interaction | 101.4% |
| **Total return (KES)** | **388.4%** |

The shilling went from 90.5 to 128.2 per dollar. Holding dollars returned 41%
before any asset appreciation at all.

Risk moves too. Measured in shillings, the S&P's volatility rises from 17.6% to
19.7%, and **20.9% of total variance is currency**, at a correlation of −0.017
with the asset — essentially independent, so it *adds* risk rather than hedging
it.

The pattern across assets is the interesting part:

| | Return (USD) | Return (KES) | Vol (USD) | Vol (KES) | FX share of risk |
|---|---|---|---|---|---|
| SPY | 245.8% | 388.4% | 17.6% | 19.7% | **20.9%** |
| JNJ | 83.6% | 159.3% | 18.1% | 19.9% | 19.5% |
| PG | 146.7% | 248.5% | 18.5% | 20.2% | 18.4% |
| MSFT | 959.5% | 1396.4% | 27.1% | 28.8% | 10.6% |
| AAPL | 958.4% | 1394.8% | 28.5% | 29.9% | 9.4% |

**The safer the asset, the more the currency dominates.** Currency is 21% of the
risk in an S&P index fund but only 9% in Apple. A Kenyan investor reaching for
"safe" US assets is, in shilling terms, mostly buying an FX position.

Reproduce with `python run_currency_analysis.py`.

### Where the data comes from

Verified rather than assumed: Yahoo Finance has **no** NSE coverage under any
ticker-suffix convention, and the exchange sells its historical data rather than
publishing it. What is free:

| Source | Provides | Depth | Constraint |
|---|---|---|---|
| [AFX](https://afx.kwayisi.org/nse/) | All 71 NSE listings, price, volume | ~10 sessions | `Crawl-delay: 60` |
| Yahoo `KES=X` | Daily KES per USD | 10 years | — |
| Yahoo (US equities) | US prices | 10 years+ | — |

The crawl delay shaped the design rather than being worked around: the listing
page carries every ticker, so a daily run costs the source **one request**. The
client identifies itself honestly and backs off exponentially on failure.

**Deep NSE history is not freely available.** The store therefore accumulates
forward — each run adds a day — and says so rather than pretending otherwise:

```
$ python run_ingestion.py --status
71 tickers have under 30 observations and are not yet usable
for covariance estimation.
```

The currency analysis above needs only FX and US data, so it works today across
a full decade.

```bash
python run_ingestion.py                          # fetch and store (idempotent)
python run_ingestion.py --status                 # coverage report
python run_ingestion.py --export data/nse.csv    # portable snapshot
```

## What the backtest actually says

Ten years of daily data, eight US large-caps, monthly rebalancing, 10bp costs:

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Avg turnover | Cost drag |
|---|---|---|---|---|---|---|
| **Equal weight** | 20.48% | 18.31% | **0.900** | −30.08% | 5.24% | 3.02% |
| HRP | 14.83% | 15.59% | 0.695 | −28.65% | 16.16% | 6.09% |
| MVO | 17.22% | 21.15% | 0.625 | −27.16% | 37.02% | 17.00% |

**The naive baseline wins.** This is not a bug, and it is the most useful thing
the project has to say. It reproduces DeMiguel, Garlappi & Uppal (2009): once
you account for estimation error and trading costs, 1/N is very hard to beat.

The mechanism is visible in the last two columns. MVO turns over 37% of the
portfolio per rebalance because small changes in estimated expected returns
produce large changes in optimal weights, and that instability costs 17% of
cumulative return. Watch it degrade as costs rise:

| Cost | Equal weight | HRP | MVO |
|---|---|---|---|
| 0bp | 0.904 | 0.709 | 0.650 |
| 25bp | 0.894 | 0.673 | 0.588 |
| 100bp | 0.862 | 0.565 | **0.405** |

A backtest that ignored transaction costs and weight drift would have reported
MVO's gross numbers and called it a success.

Reproduce with `python run_backtest.py`.

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
    ┌──────┬─────────┼──────────┬──────────────┐
    ▼      ▼         ▼          ▼              ▼
┌────────┐┌────────┐┌────────┐┌──────────┐┌───────────┐
│  data  ││  risk  ││ algor- ││ backtest ││  metrics  │
│fetcher ││covari- ││ ithms  ││ walk-fwd ││ return·   │
│(provi- ││ance    ││hrp·mvo ││ +costs   ││ vol·sharpe│
│der bdy)││shrink  ││·1/N    ││ +drift   ││           │
└────────┘└────────┘└────────┘└──────────┘└───────────┘
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

### Covariance estimation

With N assets you estimate N(N+1)/2 covariance parameters; at N=50 that is
1,275 numbers from maybe 500 observations. Mean-variance optimization then
actively seeks out the noise, loading up on whichever pair *appears* most
negatively correlated — usually the pair estimated worst.

`src/risk/covariance.py` implements Ledoit-Wolf shrinkage toward a
constant-correlation target. The shrinkage intensity is derived, not tuned. On
simulated data drawn from a known covariance matrix, it is closer to the truth
than the sample estimator on **≥80% of individual draws**, and it repairs
matrices the sample estimator leaves singular:

| Assets | Observations | δ | cond(sample) | cond(shrunk) |
|---|---|---|---|---|
| 10 | 1000 | 0.096 | 20.5 | 20.0 |
| 50 | 300 | 0.185 | 220.6 | 171.5 |
| 100 | 110 | 0.373 | 247,809 | **566** |

The vectorised implementation is cross-checked against a deliberately slow,
literal transcription of the paper's formulas, agreeing to 1e-12 across five
different matrix shapes (`tests/test_covariance.py`).

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
- **No-lookahead** — corrupt every observation after a cutoff, re-run, and
  assert no weight before the cutoff moved. If one did, the optimizer saw its
  own future and every performance number is fiction
- **Provider hostility** — delisted tickers, all-NaN columns, sparse history,
  yfinance's shape changing between one ticker and many, mismatched calendars
- **Security** — login returns an identical response for an unknown email and a
  wrong password, so the endpoint can't be used to enumerate users
- **Reference cross-check** — the vectorised Ledoit-Wolf code is verified
  against a literal loop-by-loop transcription of the published formulas
- **Hostile scraped input** — malformed prices, suspended counters, thousands
  separators, volume shorthand, `nan`/`inf` strings, future-dated rows, and a
  site redesign (which must raise, never silently return zero rows)
- **Idempotency** — running the ingestion five times leaves the store exactly
  as running it once did

### Property-based testing

`tests/test_properties.py` uses Hypothesis to search for counterexamples rather
than checking cases we thought of. It found three real problems:

1. **HRP was not permutation-invariant.** Given three assets whose pairwise
   correlation distances were all exactly equal, there is no canonical
   dendrogram — scipy breaks the tie by index order, so reordering the input
   changed the allocation. `HRPOptimizer` now detects total degeneracy and
   falls back to inverse-variance weighting, which is the order-independent
   limit of HRP. Partial ties remain order-dependent; that is a property of
   hierarchical clustering itself and is documented rather than hidden.
2. **MVO produced NaN objectives.** SLSQP's line search probes just outside the
   feasible set, where floating-point error makes `w'Σw` slightly negative;
   `sqrt` of that is NaN, which poisoned the objective and caused spurious
   convergence failures. The quadratic form is now floored at zero, and the
   solver retries from several feasible starting points before giving up.
3. **"Shrinkage always improves conditioning" is false.** Hypothesis disproved
   the intuitive claim twice — once on an already well-conditioned matrix
   (1.5622 → 1.5843) and, more interestingly, on one whose ill-conditioning came
   from variances spanning three orders of magnitude rather than from
   correlation structure (260.93 → 263.02). The constant-correlation target
   preserves variances exactly, so it cannot repair that. The test now asserts
   what shrinkage *does* guarantee: a non-singular result.

## Documentation

- **[docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md)** — how the non-obvious parts
  work and why: HRP's avoidance of matrix inversion, what the Ledoit-Wolf
  shrinkage intensity means, why no-lookahead needs structural enforcement, the
  bugs property-based testing found, and the project's known limitations.

## Frontend

Next.js App Router with TypeScript, no UI framework and no chart library. The
charts are hand-written SVG because the specifications this project follows are
easier to hit exactly than to configure: 2px strokes, a 2px surface ring on
overlapping markers, hairline solid gridlines, endpoint-only direct labels.

Three decisions worth explaining:

**The palette is validated, not chosen by eye.** The three categorical series
slots were run through a colour-vision-deficiency checker against the exact
light and dark surfaces used. Worst all-pairs separation is ΔE 9.2 light / 9.4
dark against a floor of 8, and 24.0 / 20.9 against a normal-vision floor of 15.
Aqua measures 2.74:1 on the light surface — below the 3:1 bar — so every chart
using it also carries direct labels and a table view rather than letting colour
alone carry the value.

**Colour follows the entity, not its rank.** Each strategy is pinned to a slot
in `lib/format.ts`, so filtering one out never repaints the survivors. A reader
who learned that equal weight is blue keeps that.

**Every chart has a table twin.** A tooltip enhances a value; it never gates
it. `components/TableView.tsx` is the WCAG-clean equivalent of each chart and
is one click away.

There is no dual-axis chart anywhere. Two measures of different scale get two
charts.

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
- [x] **Phase 1** — Ledoit-Wolf shrinkage, walk-forward backtesting with costs,
      property-based tests
- [x] **Phase 2** — NSE ingestion pipeline and KES/USD currency-risk decomposition
- [ ] **Phase 3** — Alembic migrations, Docker Compose, structured logging
      *(deferred: lower value than the UI for a reviewer who skims)*
- [x] **Phase 4** — Next.js + TypeScript frontend
- [ ] **Phase 5** — Retrieval over SEC EDGAR filings, then NSE annual reports

## Disclaimer

Educational software, not investment advice. Historical returns are a poor
predictor of future ones, and every optimizer here will happily produce a
confident allocation from a covariance matrix that is mostly estimation noise.
