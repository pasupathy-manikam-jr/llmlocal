"""Copy-trader quality scoring for cp4.

Aggregates each account's closed trades into per-account trading metrics
(done in MySQL so we never pull 18M rows into Python), then computes a
transparent 0-100 "copy-trader quality" score with an explanation of the
main drivers. No black box: every component is an interpretable metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "mysql+pymysql://root:root@127.0.0.1:3306/aimsfx_db3"

# Only score accounts with at least this many closed trades — fewer than this
# and win-rate / profit-factor are statistical noise.
MIN_TRADES = 50

# type 0/1 = real buy/sell trades; type 2 (~8k rows) are balance/credit ops.
FEATURE_SQL = """
SELECT
    MT4                                            AS account,
    COUNT(*)                                       AS n_trades,
    SUM(profit)                                    AS net_profit,
    AVG(profit > 0)                                AS win_rate,
    SUM(CASE WHEN profit > 0 THEN profit ELSE 0 END)      AS gross_win,
    SUM(CASE WHEN profit < 0 THEN -profit ELSE 0 END)     AS gross_loss,
    AVG(CASE WHEN profit > 0 THEN profit END)      AS avg_win,
    AVG(CASE WHEN profit < 0 THEN profit END)      AS avg_loss,
    STDDEV_SAMP(profit)                            AS profit_std,
    AVG(profit)                                    AS expectancy,
    AVG(volume)                                    AS avg_volume,
    AVG(stop_loss  > 0)                            AS sl_usage,
    AVG(take_profit > 0)                           AS tp_usage,
    AVG(TIMESTAMPDIFF(MINUTE, open_datetime, close_datetime)) AS avg_hold_min,
    MIN(close_datetime)                            AS first_close,
    MAX(close_datetime)                            AS last_close,
    COUNT(DISTINCT symbol_ID)                      AS n_symbols
FROM member_mt4_trade
WHERE type IN ('0', '1')
  AND close_datetime IS NOT NULL
  AND volume > 0
GROUP BY MT4
HAVING n_trades >= :min_trades
"""


def load_account_features(min_trades: int = MIN_TRADES) -> pd.DataFrame:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        df = pd.read_sql(text(FEATURE_SQL), conn, params={"min_trades": min_trades})
    return df


def _pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank 0..1, robust to outliers (rank-based, not min-max)."""
    return s.rank(pct=True)


def score_traders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived, interpretable metrics
    df["profit_factor"] = df["gross_win"] / df["gross_loss"].replace(0, np.nan)
    df["profit_factor"] = df["profit_factor"].fillna(df["profit_factor"].max())
    # Sharpe-like: mean trade profit / volatility of trade profit
    df["sharpe"] = df["expectancy"] / df["profit_std"].replace(0, np.nan)
    df["sharpe"] = df["sharpe"].fillna(0)
    df["active_days"] = (
        pd.to_datetime(df["last_close"]) - pd.to_datetime(df["first_close"])
    ).dt.days.clip(lower=1)
    df["trades_per_day"] = df["n_trades"] / df["active_days"]

    # Score components (each 0..1 via percentile rank so scales don't fight).
    # Weights reflect what matters for COPYING a trader: consistent, positive,
    # risk-controlled returns beat one lucky home-run.
    components = {
        "net_profit":     (_pct_rank(df["net_profit"]),    0.25),  # did they make money
        "profit_factor":  (_pct_rank(df["profit_factor"]), 0.20),  # win $ vs lose $
        "sharpe":         (_pct_rank(df["sharpe"]),         0.20),  # consistency
        "win_rate":       (_pct_rank(df["win_rate"]),       0.15),  # hit rate
        "sl_usage":       (df["sl_usage"].clip(0, 1),       0.10),  # risk discipline
        "expectancy":     (_pct_rank(df["expectancy"]),     0.10),  # $ per trade
    }
    score = (
        components["net_profit"][0]    * 0.25 +
        components["profit_factor"][0] * 0.20 +
        components["sharpe"][0]        * 0.20 +
        components["win_rate"][0]      * 0.15 +
        components["sl_usage"][0]      * 0.10 +
        components["expectancy"][0]    * 0.10
    )
    df["quality_score"] = (score * 100).round(1)

    # Human-readable reason: the two strongest positive drivers.
    driver_cols = {
        "net_profit": "strong net profit",
        "profit_factor": "wins outweigh losses",
        "sharpe": "consistent returns",
        "win_rate": "high win rate",
        "sl_usage": "disciplined stop-losses",
        "expectancy": "good profit per trade",
    }
    ranks = pd.DataFrame({k: v[0] for k, v in components.items()})
    df["top_driver"] = ranks.idxmax(axis=1).map(driver_cols)

    df = df.sort_values("quality_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


if __name__ == "__main__":
    feats = load_account_features()
    scored = score_traders(feats)
    print(f"scored {len(scored):,} accounts")
    print(scored.head(10)[["rank", "account", "quality_score", "n_trades",
                            "win_rate", "net_profit", "top_driver"]].to_string(index=False))
