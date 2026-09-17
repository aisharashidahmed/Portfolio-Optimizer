"""Walk-forward backtest comparing every strategy on the same data.

Unlike a single train/test split, this re-fits the optimizer on a rolling
window and steps forward, so the result reflects many independent decisions
rather than one lucky period. Costs are charged on every rebalance.

    python run_backtest.py
"""

import pandas as pd

from src.algorithms import EqualWeightOptimizer, HRPOptimizer, MVOOptimizer
from src.backtesting import compare_strategies, run_walk_forward
from src.data_fetcher import prepare_market_data

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM", "PG"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
RISK_FREE_RATE = 0.04

TRAINING_DAYS = 252  # one year of history to fit on
REBALANCE_EVERY = 21  # roughly monthly
TRANSACTION_COST_BPS = 10.0  # 10bp per unit traded


def main() -> None:
    print(f"Fetching {len(TICKERS)} tickers, {START_DATE} to {END_DATE}...")
    market_data = prepare_market_data(TICKERS, START_DATE, END_DATE)
    daily_returns = market_data["daily_returns"]

    if market_data["missing_tickers"]:
        print(f"  unavailable: {market_data['missing_tickers']}")
    print(f"  {len(daily_returns)} observations x {daily_returns.shape[1]} assets")

    strategies = {
        "Equal weight": EqualWeightOptimizer(),
        "HRP": HRPOptimizer(),
        "MVO": MVOOptimizer(risk_free_rate=RISK_FREE_RATE, max_weight=0.35),
    }

    print(
        f"\nWalk-forward: fit on {TRAINING_DAYS}d, rebalance every "
        f"{REBALANCE_EVERY}d, {TRANSACTION_COST_BPS:.0f}bp costs\n"
    )

    table = compare_strategies(
        daily_returns,
        strategies,
        training_days=TRAINING_DAYS,
        rebalance_every=REBALANCE_EVERY,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        risk_free_rate=RISK_FREE_RATE,
    )

    display = pd.DataFrame(
        {
            "Ann. return": table["annualised_return"].map("{:.2%}".format),
            "Ann. vol": table["annualised_volatility"].map("{:.2%}".format),
            "Sharpe": table["sharpe_ratio"].map("{:.3f}".format),
            "Sortino": table["sortino_ratio"].map("{:.3f}".format),
            "Max DD": table["max_drawdown"].map("{:.2%}".format),
            "Avg turnover": table["average_turnover"].map("{:.2%}".format),
            "Cost drag": table["total_cost_drag"].map("{:.2%}".format),
        }
    )
    print(display.to_string())

    # Costs are not a rounding error -- show how each strategy degrades.
    print("\nNet Sharpe as trading costs rise")
    print("-" * 52)
    header = "  ".join(f"{name:>14}" for name in strategies)
    print(f"{'cost':>6}  {header}")
    for bps in (0.0, 10.0, 25.0, 50.0, 100.0):
        sharpes = [
            run_walk_forward(
                daily_returns,
                optimizer,
                training_days=TRAINING_DAYS,
                rebalance_every=REBALANCE_EVERY,
                transaction_cost_bps=bps,
                risk_free_rate=RISK_FREE_RATE,
            ).performance.sharpe_ratio
            for optimizer in strategies.values()
        ]
        row = "  ".join(f"{s:>14.3f}" for s in sharpes)
        print(f"{bps:>5.0f}bp  {row}")


if __name__ == "__main__":
    main()
