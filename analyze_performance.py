"""
analyze_performance.py

Quick performance report for nba_edge_app using model_logs.db.

- Buckets bets by edge size and confidence.
- Splits aligned vs non-aligned (hybrid vs cheatsheet).
- Gives win% (excluding pushes) and basic counts.

Usage (from your project folder, with venv activated):

    python analyze_performance.py
    python analyze_performance.py path\to\model_logs.db
"""
import sys
import os
import math
import sqlite3
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()



def load_logs(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found at: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM logs", conn)
    finally:
        conn.close()
    return df


def summarize_group(df: pd.DataFrame, label: str) -> None:
    """Print win/loss summary for a subset of logs."""
    if df.empty:
        print(f"{label}: no rows")
        return

    is_push = df["spread_covered"] == 2
    is_win = df["spread_covered"] == 1
    is_loss = df["spread_covered"] == 0

    n_push = int(is_push.sum())
    n_win = int(is_win.sum())
    n_loss = int(is_loss.sum())
    eff = n_win + n_loss

    win_pct = (n_win / eff * 100) if eff > 0 else float("nan")
    avg_edge = df["edge"].mean() if "edge" in df.columns else float("nan")

    print(
        f"{label:12} N={len(df):3d}, Eff={eff:3d}, "
        f"Wins={n_win:3d}, Losses={n_loss:3d}, Push={n_push:2d}, "
        f"Win%={win_pct:5.1f}%, AvgEdge={avg_edge:6.2f}"
    )

def inspect_big_edges(df: pd.DataFrame, threshold: float = 6.0) -> None:
    """Print details for games with very large edges (for manual audit)."""
    if "edge" not in df.columns:
        print("\nNo 'edge' column in logs; cannot inspect big edges.")
        return

    big = df[df["edge"].abs() >= threshold].copy()
    if big.empty:
        print(f"\nNo games with abs(edge) >= {threshold}.")
        return

    print(f"\nGames with abs(edge) >= {threshold} (N={len(big)}):")

    # Try to pick useful columns if they exist
    cols_preference = [
        "date",
        "favorite",
        "underdog",
        "pick",
        "edge",
        "confidence",
        "models_aligned",
        "final_score",
        "spread_covered",
        # injury / outs if present
        "fav_out",
        "dog_out",
        # rest / B2B terms if present
        "rest_adj_term",
        "b2b_adj_term",
    ]
    cols = [c for c in cols_preference if c in big.columns]

    # Sort by absolute edge descending
    big = big.sort_values("edge", key=lambda s: s.abs(), ascending=False)

    # Round some numeric columns for readability
    for c in ["edge", "rest_adj_term", "b2b_adj_term"]:
        if c in big.columns:
            big[c] = big[c].round(2)

    print(big[cols].to_string(index=False))


def main(db_path: Optional[str] = None) -> None:
    if db_path is None:
        # default to the same as the app: env var or local file
        env_path = os.getenv("NBA_EDGE_DB_PATH")
        db_path = env_path if env_path else "model_logs.db"

    print(f"Using DB: {db_path}")
    df = load_logs(db_path)

    # Basic filtering: require a pick and a recorded result
    df = df.copy()
    df = df[df["pick"].notna()]
    df = df[df["spread_covered"].notna()]

    if df.empty:
        print("No logged picks with results yet.")
        return

    print(f"Total logged picks with results: {len(df)}")

    # Overall summary
    summarize_group(df, "Overall")

    # Edge buckets by absolute edge
    if "edge" in df.columns:
        print("\nEdge bucket performance (by abs(edge)):")
        abs_edge = df["edge"].abs()
        buckets = [
            (0.0, 2.0),
            (2.0, 4.0),
            (4.0, 6.0),
            (6.0, math.inf),
        ]
        for lo, hi in buckets:
            if hi is math.inf:
                mask = abs_edge >= lo
                label = f">={lo:g}"
            else:
                mask = (abs_edge >= lo) & (abs_edge < hi)
                label = f"[{lo:g}, {hi:g})"
            sub = df[mask]
            if sub.empty:
                continue
            summarize_group(sub, label)

    # By confidence (excluding PASS)
    if "confidence" in df.columns:
        print("\nBy confidence (excluding PASS):")
        for conf in sorted(
            c for c in df["confidence"].dropna().unique() if str(c).upper() != "PASS"
        ):
            sub = df[df["confidence"] == conf]
            summarize_group(sub, str(conf))

    # Models aligned vs not
    if "models_aligned" in df.columns:
        print("\nModels aligned vs not (hybrid & cheatsheet):")
        for val, label in [(1, "Aligned"), (0, "Not aligned")]:
            sub = df[df["models_aligned"] == val]
            summarize_group(sub, label)

    # B2B context for favorite
    if "fav_is_b2b" in df.columns:
        print("\nFavorite B2B context:")
        for val, label in [(1, "Fav B2B"), (0, "Fav not B2B")]:
            sub = df[df["fav_is_b2b"] == val]
            summarize_group(sub, label)

    # B2B context for underdog (optional)
    if "dog_is_b2b" in df.columns:
        print("\nUnderdog B2B context:")
        for val, label in [(1, "Dog B2B"), (0, "Dog not B2B")]:
            sub = df[df["dog_is_b2b"] == val]
            summarize_group(sub, label)

    # Detailed look at the biggest edges
    inspect_big_edges(df, threshold=6.0)


if __name__ == "__main__":
    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg_path)
