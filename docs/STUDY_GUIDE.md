# Study guide

The parts of this codebase that carry real interview risk, in priority order.
Each section says what the concept is, why the code does it that way, and the
follow-up question you should be ready for.

Rule of thumb: you do not need to recite formulas. You need to explain **what
problem the code is solving and why the obvious alternative is worse**.

---

## 1. Why `cov_matrix.corr()` was wrong

**Priority: highest.** This is in your commit history and your README, so it
will get asked.

### The concept

A covariance matrix Σ holds how each pair of assets moves together, in squared
units. A correlation matrix holds the same relationships rescaled to [−1, 1].
To convert, divide each entry by the two assets' standard deviations:

```
corr(i,j) = Σ(i,j) / (σᵢ · σⱼ)         where σᵢ = √Σ(i,i)
```

In matrix form, `corr = D⁻¹ Σ D⁻¹` where `D = diag(√diag(Σ))`.

### What the bug was

`cov_matrix.corr()` does something completely different. Pandas' `.corr()`
expects a table of **observations** — rows are days, columns are variables. Given
a covariance matrix it treats each *column of the matrix* as if it were a time
series of measurements and correlates those columns with each other.

The output is a well-formed matrix of plausible-looking numbers that means
nothing. That's what made it survive: it didn't crash, and the weights summed
to 1.

### The evidence

On synthetic data where the true pairwise correlations are ~0:

| pair | true | what the bug produced |
|---|---|---|
| A–B | −0.008 | −0.430 |
| A–C | 0.016 | −0.339 |

HRP clusters assets by correlation distance. Feed it invented correlations and
every cluster — and therefore every allocation — is noise.

### Be ready for

> *"How did you find it?"*

Honestly: it was found by reading the code, then confirmed by constructing data
with known-zero correlations and showing the function returned ~−0.4. The
regression test is `tests/test_hrp.py::test_recovers_the_true_correlation_matrix`.

> *"Why didn't your tests catch it earlier?"*

Because the original tests only checked that weights summed to 1 and were
non-negative — invariants the bug satisfied perfectly. That's the lesson:
testing output *shape* is not testing output *correctness*.

---

## 2. What HRP actually does, and why it avoids matrix inversion

**Priority: highest.** You named the file; you own the algorithm.

### The problem HRP solves

Mean-variance optimization needs Σ⁻¹. When Σ is estimated from limited data it
is close to singular, and inverting a near-singular matrix amplifies estimation
error enormously. The condition number measures this: it is the ratio of largest
to smallest eigenvalue, and it tells you how much the inversion multiplies your
errors. At 247,809 (a real number from this repo) the output is essentially noise.

HRP never inverts anything.

### The three steps

1. **Distance.** Convert correlations to a distance metric:
   `d(i,j) = √(0.5 · (1 − ρ(i,j)))`. Perfectly correlated assets → distance 0.
   Perfectly anticorrelated → distance 1.

2. **Cluster.** Build a dendrogram (tree) by repeatedly joining the two closest
   assets or groups. Assets that move together end up on the same branch.

3. **Recursive bisection.** Walk down the tree from the root. At each split,
   divide capital between the two branches **inversely to their variance** — the
   calmer branch gets more. Recurse until you reach individual assets.

The intuition: instead of asking "what is the single optimal weight vector"
(which needs Σ⁻¹), ask "how should I split money between these two groups"
repeatedly. Each local decision only needs variances, never an inverse.

### Why `single` linkage, not `ward`

`ward` minimises within-cluster variance and assumes the distances live in
Euclidean space. Correlation distances do not satisfy that geometry, so `ward`
is applying a criterion its assumptions don't support. López de Prado uses
single linkage. The original code used `ward`; it was changed.

### Be ready for

> *"Why inverse variance and not equal weights within a cluster?"*

Because a cluster's risk is driven by its noisiest member. Inverse-variance
weighting is the minimum-variance solution when you ignore correlations — which
is exactly the right approximation to make *inside* a cluster, since by
construction the cluster's members are already similar.

> *"What's the weakness of HRP?"*

It ignores expected returns entirely. It is a pure risk-allocation method. If
you have genuine views on which assets will outperform, HRP cannot express them.
It also inherits hierarchical clustering's tie-breaking problem (see §5).

---

## 3. Covariance shrinkage: what δ means

**Priority: high.** This is the most statistically sophisticated thing in the
repo.

### The problem

With N assets you must estimate N(N+1)/2 covariance parameters. At N=50 that's
1,275 numbers, often from ~500 observations. The estimates are noisy.

This is worse than it sounds, because optimizers *actively seek out* the noise.
MVO looks for the most negatively-correlated pair to hedge with — and the pair
that *appears* most negatively correlated is usually the pair whose correlation
was estimated worst. The optimizer is drawn to your largest errors.

### The fix

Blend the noisy sample estimate `S` with a structured target `F`:

```
Σ̂ = δ·F + (1 − δ)·S
```

`F` is the **constant-correlation target**: keep each asset's own variance, but
replace every pairwise correlation with the average correlation across the
universe. That's N+1 parameters instead of 1,275 — heavily biased, but almost
noise-free.

This is the bias-variance tradeoff, concretely. `S` is unbiased but
high-variance. `F` is biased but low-variance. The blend beats both.

### The key point about δ

**δ is derived, not tuned.** Ledoit-Wolf gives a closed form for the δ that
minimises expected squared error from the true covariance matrix:

```
δ* = (π − ρ) / γ / T        clipped to [0, 1]
```

- `π` — total estimation variance in the sample covariance entries
- `ρ` — covariance between the estimation errors of S and F
- `γ` — how wrong the target is (its squared distance from S)

The intuition is the whole method: **shrink hard when the sample estimate is
noisy (large π) and the target isn't badly misspecified (small γ).**

You do not need to reproduce these formulas. You need to say that δ is solved
for, not chosen — that's what distinguishes this from an arbitrary 50/50 blend.

### The numbers from this repo

| Assets | Obs | δ | cond(sample) | cond(shrunk) |
|---|---|---|---|---|
| 10 | 1000 | 0.096 | 20.5 | 20.0 |
| 50 | 300 | 0.185 | 220.6 | 171.5 |
| 100 | 110 | 0.373 | 247,809 | 566 |

δ rises as data gets scarcer. That's the method working.

### Be ready for

> *"How do you know your implementation is right?"*

Two ways. First, a deliberately slow loop-by-loop transcription of the paper's
formulas is in the test file, and the fast vectorised version matches it to
1e-12 across five matrix shapes. Second, simulating from a *known* covariance
matrix, the shrunk estimate is closer to truth than the sample estimate on ≥80%
of individual draws.

> *"Why not just use sklearn?"*

sklearn's `LedoitWolf` shrinks toward a scaled identity matrix — it assumes all
assets are uncorrelated with equal variance. For equities that target is badly
misspecified; the constant-correlation target is the standard choice in finance.

---

## 4. Why no-lookahead needs structural enforcement

**Priority: high.** This is the difference between a backtest and a fantasy.

### The concept

Lookahead bias is using information that wasn't available at decision time. It
is the single most common way backtests lie, and it is usually accidental —
an off-by-one in a slice, a `.fillna()` that back-propagates, normalising with
statistics computed over the whole dataset.

A backtest with lookahead will look *excellent*, which is why it's dangerous.

### How this code prevents it

Weights used from day `t` onward are fitted on `daily_returns.iloc[t-training_days : t]`.
That slice is exclusive of `t`. It is structurally impossible for the optimizer
to see the return it is about to earn.

### How the test proves it

```python
corrupted.iloc[cutoff:] = <garbage>
```

Corrupt every observation after a cutoff, re-run the whole backtest, and assert
that **no weight before the cutoff changed**. If any did, information flowed
backwards in time and every number the engine produces is fiction.

This is much stronger than inspecting the slice by eye. It tests the property,
not the implementation.

### Be ready for

> *"What other biases affect backtests?"*

Survivorship bias (only testing companies that still exist today — your ticker
list is drawn from current large-caps, so this project has it). Data-snooping
(trying many strategies and reporting the best). Ignoring transaction costs and
liquidity. This repo handles costs; it does *not* handle survivorship, and
saying so is better than being caught out.

---

## 5. What property-based testing found

**Priority: high.** This is your best "tell me about a bug you found" story.

Hypothesis generates thousands of inputs looking for a counterexample, instead
of checking the cases you thought of.

### Finding 1 — HRP was not permutation-invariant

With three assets whose pairwise correlation distances were **all exactly equal**
(0.713176), there is no canonical dendrogram. Every asset is equidistant from
every other. scipy breaks the tie by index order — so reordering the columns of
your input changed the portfolio: whichever asset landed alone got 0.3295, the
paired two got 0.3353.

**Fix:** detect total degeneracy and fall back to inverse-variance weighting,
which depends only on the diagonal and is therefore order-independent. It is
also the correct limit of HRP when there's no clustering signal.

**Be honest about the limit:** *partial* ties (some pairs equidistant, others
not) are still order-dependent. That is a property of hierarchical clustering
itself, not of this code. It's documented rather than hidden.

### Finding 2 — MVO produced NaN objectives

SLSQP's line search probes points slightly *outside* the feasible region. There,
floating-point error can make the quadratic form `w'Σw` a tiny negative number.
`√(negative)` is NaN, the objective becomes NaN, and the solver fails with a
misleading "positive directional derivative" error.

**Fix:** floor the variance at zero before the square root, and retry from
several feasible starting points before giving up (SLSQP is a *local* method —
a different start often succeeds).

When it still fails, it raises a typed `OptimizationFailedError` telling the
caller to try HRP. **Failing loudly is correct here**; silently returning
inverse-variance weights while labelling them "mvo" would be worse.

### Finding 3 — a riskless portfolio had a Sharpe ratio of 4×10¹⁷

The guard was `if volatility == 0`. But a constant return series has a
floating-point standard deviation of ~1e-18, not exactly zero. The guard missed,
and the division produced an astronomical number.

**Fix:** compare against a floor (`1e-12`), never against exact zero. This is
the general lesson about float comparisons, in a finance-flavoured costume.

### Finding 4 — one of my own tests was wrong

I asserted "shrinkage never worsens the condition number." Hypothesis disproved
it twice:

- On an already well-conditioned matrix: 1.5622 → 1.5843. Meaningless drift on a
  matrix nobody needed to fix.
- **More interestingly:** on a matrix whose ill-conditioning came from variances
  spanning three orders of magnitude rather than from correlation structure
  (260.93 → 263.02). The constant-correlation target preserves each asset's
  variance *exactly*, so it structurally cannot repair conditioning caused by
  the variance spread.

The test now asserts what shrinkage genuinely guarantees: a non-singular result.

**This is a good story to tell.** It shows you distinguish between "my code is
wrong" and "my assumption was wrong," which is a real skill.

---

## 6. Why equal weight wins

**Priority: high.** It's the headline of your README.

### The result

Ten years, eight US large-caps, monthly rebalancing, 10bp costs:

| Strategy | Sharpe | Avg turnover | Cost drag |
|---|---|---|---|
| Equal weight | **0.900** | 5.24% | 3.02% |
| HRP | 0.695 | 16.16% | 6.09% |
| MVO | 0.625 | 37.02% | 17.00% |

### Why this happens

DeMiguel, Garlappi & Uppal (2009) showed 1/N beats most optimizers
out-of-sample. Two mechanisms:

1. **Estimation error.** Optimizers need expected returns, which are far harder
   to estimate than variances. Small errors in estimated means produce large
   swings in optimal weights. 1/N estimates nothing, so it has no estimation
   error to be wrong about.

2. **Turnover.** Because optimal weights swing, MVO trades constantly — 37% of
   the portfolio per rebalance here, costing 17% of cumulative return. 1/N only
   trades to undo drift.

### Be ready for

> *"So your project doesn't work?"*

The strongest answer, and a true one: *"The optimizers don't beat the baseline,
and the backtest is built to be honest enough to show that. The interesting part
is the mechanism — I can point at turnover and cost drag as the cause, and show
MVO degrading fastest as costs rise. A backtest that ignored transaction costs
would have reported MVO's gross numbers and called it a win."*

Do not apologise for this result. Reporting it is the most credible thing in the
repo.

---

## 7. Design decisions you should be able to justify

Shorter items, but each is a plausible "why did you do it this way?"

### Why both `/optimize/sync` and `/optimize`

Queuing work that takes 200ms is architecture theatre. The sync endpoint means
the API is fully usable with **no broker installed** — a reviewer can clone and
run it. The async path exists for large universes and the ingestion-backed jobs
coming in Phase 2, which genuinely take minutes.

If asked "why not just async everything?": because it adds an entire class of
failure (broker down, task lost, result expired) for no benefit at small N.

### Why signup and login have different schemas

Login deliberately does **not** enforce the 8-character minimum. If it did, a
short wrong password returns 422 while a long wrong password returns 401 —
which tells an attacker your password policy, and locks out any account created
before a policy change. Every failed login must look identical. Same reason
"unknown email" and "wrong password" return the same response: otherwise the
endpoint is a user-enumeration oracle.

### Why SQLite is the default

A reviewer who has to install Postgres before seeing anything usually closes the
tab. `DATABASE_URL` switches to Postgres with one env var.

### Why config refuses to start in production with the dev key

`src/config.py` raises if `ENVIRONMENT=production` while `SECRET_KEY` is still
the placeholder. Failing at startup is strictly better than silently signing
JWTs with a key that's public in the repo.

### Why the service layer exists

Before it, the API and the Celery worker each had their own copy of the fetch
logic, both using an outdated yfinance access pattern. Two data paths means two
places for validation to be skipped. Now there is exactly one.

### Why 72 bytes is the password cap

bcrypt silently truncates at 72 bytes. Without an explicit check, two different
long passwords can hash identically. Rejecting is better than truncating.

---

## The five sentences to have ready

If you only internalise one thing, make it these:

1. *"HRP clusters assets by correlation distance and splits capital down the
   tree by inverse variance, so it never has to invert the covariance matrix."*
2. *"Ledoit-Wolf blends the noisy sample covariance with a low-variance
   structured target, and the blending weight is solved for analytically rather
   than tuned."*
3. *"The backtest models weight drift and charges transaction costs, and
   no-lookahead is verified by corrupting future data and asserting past weights
   don't move."*
4. *"Property-based testing found three real bugs, including that HRP wasn't
   permutation-invariant when correlation distances tie."*
5. *"Equal weight beats both optimizers, and I can point at turnover as the
   mechanism."*

---

## Known limitations — say these before you're asked

Volunteering a limitation reads as confidence. Being caught out by one doesn't.

- **Survivorship bias.** The ticker list is current large-caps, so the backtest
  implicitly excludes companies that failed.
- **Partial-tie ordering in HRP.** Handled for total degeneracy only.
- **Expected returns are historical means.** These are notoriously poor
  forecasts; it's the main reason MVO underperforms here.
- **US-only so far.** The Kenya/NSE side is Phase 2.
- **No walk-forward hyperparameter selection.** `training_days` and
  `rebalance_every` are fixed, not chosen out-of-sample.
