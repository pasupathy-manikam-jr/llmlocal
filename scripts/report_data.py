"""Run scoring and emit a JSON summary for the visual report + save full CSV."""
import json
import os

import numpy as np
import pandas as pd

from app.trader_scoring import load_account_features, score_traders

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)


def j(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else round(float(x), 4)
    return x


def main():
    df = score_traders(load_account_features())
    df.to_csv(f"{OUT_DIR}/leaderboard.csv", index=False)

    n = len(df)
    profitable = df["net_profit"] > 0

    # Score distribution in 10-point buckets
    buckets = pd.cut(df["quality_score"], bins=range(0, 101, 10), right=False)
    dist = buckets.value_counts().sort_index()
    score_hist = [{"band": f"{int(iv.left)}-{int(iv.right)}", "count": int(c)}
                  for iv, c in dist.items()]

    def row(r):
        return {
            "rank": int(r["rank"]),
            "account": int(r["account"]),
            "score": j(r["quality_score"]),
            "n_trades": int(r["n_trades"]),
            "win_rate": j(r["win_rate"]),
            "profit_factor": j(r["profit_factor"]),
            "net_profit": j(r["net_profit"]),
            "sl_usage": j(r["sl_usage"]),
            "driver": r["top_driver"],
        }

    summary = {
        "accounts_scored": n,
        "min_trades": 50,
        "total_trades_analyzed": int(df["n_trades"].sum()),
        "pct_profitable": round(100 * profitable.mean(), 1),
        "median_win_rate": j(df["win_rate"].median()),
        "median_profit_factor": j(df["profit_factor"].median()),
        "median_score": j(df["quality_score"].median()),
        "median_sl_usage": j(df["sl_usage"].median()),
        "top20": [row(r) for _, r in df.head(20).iterrows()],
        "bottom10": [row(r) for _, r in df.tail(10).iloc[::-1].iterrows()],
        "score_hist": score_hist,
        # The headline insight: many win often but still lose money.
        "high_winrate_but_losing": int(((df["win_rate"] > 0.6) & (df["net_profit"] < 0)).sum()),
        "high_winrate_count": int((df["win_rate"] > 0.6).sum()),
    }

    with open(f"{OUT_DIR}/report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:1500])
    print(f"\nsaved {OUT_DIR}/leaderboard.csv ({n:,} rows) and {OUT_DIR}/report.json")


if __name__ == "__main__":
    main()
