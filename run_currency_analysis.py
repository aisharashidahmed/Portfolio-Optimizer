"""How much of a Kenyan investor's US-asset return is the asset, and how much
is the shilling?

    python run_currency_analysis.py

Uses ten years of free data: US equity prices from Yahoo, and the KES/USD rate
quoted as KES=X.
"""

import warnings

import pandas as pd

from src.analytics.currency import decompose_currency_risk, summarise_currency_exposure
from src.data_fetcher import prepare_market_data
from src.ingestion.fx import fetch_usd_kes

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "AAPL", "MSFT", "JPM", "JNJ", "PG", "XOM"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"


def main() -> None:
    print(f"Fetching {len(TICKERS)} US tickers and USD/KES, {START_DATE} to {END_DATE}...")

    usd_returns = prepare_market_data(TICKERS, START_DATE, END_DATE)["daily_returns"]
    rate = fetch_usd_kes(START_DATE, END_DATE)

    print(
        f"\nUSD/KES: {rate.iloc[0]:.2f} -> {rate.iloc[-1]:.2f}  "
        f"({rate.iloc[-1] / rate.iloc[0] - 1:+.1%} over the period)"
    )
    print("A weaker shilling means dollar assets are worth more in shillings.\n")

    table = summarise_currency_exposure(usd_returns, rate)
    display = pd.DataFrame(
        {
            "Return (USD)": table["asset_return_usd"].map("{:.1%}".format),
            "Return (KES)": table["total_return_kes"].map("{:.1%}".format),
            "Vol (USD)": table["asset_volatility"].map("{:.1%}".format),
            "Vol (KES)": table["total_volatility_kes"].map("{:.1%}".format),
            "corr(asset, FX)": table["correlation"].map("{:+.3f}".format),
            "FX share of risk": table["currency_share_of_variance"].map("{:.1%}".format),
        }
    )
    print(display.to_string())

    decomposition = decompose_currency_risk(usd_returns["SPY"], rate)
    print(f"\nS&P 500 for a Kenyan investor, {START_DATE[:4]}-{END_DATE[:4]}")
    print("-" * 46)
    print(f"  asset return (USD)     {decomposition.asset_return_usd:>10.1%}")
    print(f"  currency return        {decomposition.currency_return:>10.1%}")
    print(f"  interaction            {decomposition.interaction:>10.1%}")
    print(f"  {'':>22}{'':->10}")
    print(f"  total return (KES)     {decomposition.total_return_kes:>10.1%}")
    print(
        f"\n  Volatility rose from {decomposition.asset_volatility:.1%} to "
        f"{decomposition.total_volatility_kes:.1%} once measured in shillings."
    )
    print(
        f"  Currency accounts for {decomposition.currency_share_of_variance:.0%} "
        f"of total risk, at a correlation of {decomposition.correlation:+.3f} --"
    )
    print("  essentially independent, so it adds risk rather than hedging it.")


if __name__ == "__main__":
    main()
