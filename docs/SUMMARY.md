# How to describe this project

Four versions, shortest first. Pick by context.

---

## 1. Résumé bullets

> **Portfolio Optimizer** — Python (FastAPI, pandas, SciPy), TypeScript (Next.js), PostgreSQL/SQLite, Celery, GitHub Actions
>
> - Built a full-stack portfolio construction and backtesting platform for
>   investors holding both Nairobi Securities Exchange and US equities, with
>   explicit decomposition of asset return versus KES/USD currency risk.
> - Implemented Hierarchical Risk Parity and Ledoit-Wolf covariance shrinkage
>   from the source papers; validated the shrinkage estimator against a
>   loop-by-loop reference implementation (agreement to 1e-12) and against
>   simulation from a known covariance matrix.
> - Built an idempotent ingestion pipeline for NSE prices with schema-change
>   detection, exponential-backoff retries, and robots.txt-compliant rate
>   limiting; upserts on a natural key so re-runs are provable no-ops.
> - Wrote 277 tests at 90% coverage including property-based tests
>   (Hypothesis), which found three correctness bugs — among them a
>   non-permutation-invariant clustering path and a NaN-producing objective
>   function.

---

## 2. One paragraph (portfolio site, cover letter, README intro)

> Portfolio Optimizer is a full-stack tool for investors who hold assets in two
> currencies at once — specifically Kenyans holding both NSE and US equities.
> Most portfolio software assumes you earn, spend, and invest in one currency;
> a Kenyan holding US assets does not, and over the last decade the KES/USD
> move contributed more to their realised return than several of the assets
> did. The backend (Python, FastAPI) implements three allocation strategies
> including Hierarchical Risk Parity and Ledoit-Wolf covariance shrinkage
> written from the source papers, a walk-forward backtester that models weight
> drift and transaction costs with structurally enforced no-lookahead, and an
> idempotent ingestion pipeline for Nairobi Securities Exchange prices, which
> no free API covers. The frontend is Next.js and TypeScript with hand-written
> SVG charts on a colour-vision-validated palette. It has 277 tests at 90%
> coverage, and the property-based suite found three real bugs during
> development. Its headline result is deliberately unflattering: the naive
> equal-weight baseline beats both optimizers out-of-sample once transaction
> costs are charged, which reproduces DeMiguel, Garlappi & Uppal (2009) and is
> documented rather than hidden.

---

## 3. Spoken — "tell me about a project" (about 90 seconds)

> I built a portfolio optimizer for a problem I hadn't seen handled properly:
> a Kenyan investor holding US assets doesn't earn the asset's return, they
> earn it compounded with the shilling. Over 2015 to 2024 the shilling went
> from 90 to 128 per dollar, so the S&P returned 246% in dollars but 388% in
> shillings — and about a fifth of the risk in that position is currency, not
> equity.
>
> Backend is Python and FastAPI. I implemented three allocation strategies —
> Hierarchical Risk Parity, mean-variance, and equal weight as the benchmark —
> plus Ledoit-Wolf covariance shrinkage written from the paper rather than
> imported, because the sample covariance matrix is noisy enough that the
> optimizer ends up fitting its own estimation error.
>
> The part I'd point at for a backend role is the NSE ingestion pipeline.
> There's no free API for Nairobi market data — I verified Yahoo covers nothing
> under any ticker convention, and the exchange sells its history — so I scrape
> a free aggregator whose robots.txt asks for a sixty-second crawl delay. That
> constraint shaped the design: the listing page carries all seventy-one
> tickers, so a daily run costs the source exactly one request. Storage is
> idempotent on a ticker-plus-date key, so a re-run reports zero inserts rather
> than duplicating, and the run aborts without writing if more than twenty
> percent of rows fail validation — because that pattern means the site changed
> shape, and a partial universe in the database is worse than no new data.
>
> The thing I'd actually want to talk about is what the tests caught. I used
> Hypothesis for property-based testing and it found three real bugs, including
> that my HRP implementation wasn't permutation-invariant — if all pairwise
> correlation distances tie, there's no canonical dendrogram and scipy breaks
> the tie by index order, so reordering the input changed the portfolio. It
> also disproved one of my own assumptions: I'd asserted shrinkage always
> improves the covariance matrix's condition number, and it found a case where
> the ill-conditioning came from the variance spread rather than the
> correlations, which that estimator structurally can't fix.
>
> And the honest headline: equal weight beats both optimizers. Mean-variance
> turns over 37% of the portfolio per rebalance and loses 17% of cumulative
> return to costs. That's a known result in the literature, and I'd rather
> report it than tune until the chart looked good.

---

## 4. The follow-ups worth rehearsing

| If they ask | Lead with |
|---|---|
| "Does your optimizer work?" | "No, and the backtest is honest enough to show it. I can point at turnover as the mechanism." |
| "What's the hardest bug you've fixed?" | The FX calendar-alignment bias — inner-joining FX returns onto equity dates silently dropped 93 days of real currency movement and overstated depreciation from 41.6% to 67.7%. |
| "How do you know your maths is right?" | Two independent checks: a slow reference transcription of the paper's formulas, and simulation from a known covariance matrix. |
| "What would you do next?" | Alembic migrations and Docker Compose — deliberately deferred, and I know it's the gap. |
| "What are the limitations?" | Survivorship bias in the ticker list; NSE history accumulates forward because no free archive exists; expected returns are historical means, which is why MVO underperforms. |

Full technical detail for each of these is in [STUDY_GUIDE.md](STUDY_GUIDE.md).
