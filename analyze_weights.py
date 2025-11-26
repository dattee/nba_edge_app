import sqlite3
import numpy as np
import pandas as pd

from pathlib import Path

# Optional: comment these if you don't want logistic yet
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DB_NAME = "model_logs.db"

EDGE_BUCKETS = [-999, 2.5, 6.0, 10.0, 999]  # PASS, MEDIUM, STRONG, ULTRA-ish
EDGE_BUCKET_LABELS = ["<2.5", "2.5–6", "6–10", ">=10"]


def load_data():
    if not Path(DB_NAME).exists():
        print(f"DB {DB_NAME} not found.")
        return None
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM logs", conn)
    conn.close()
    return df


def basic_edge_stats(df: pd.DataFrame):
    print("\n=== A. Edge vs Cover Rate ===")

    if "spread_covered" not in df.columns:
        print("No 'spread_covered' column yet. Log some games & refresh scores first.")
        return

    df = df[df["spread_covered"].isin([0, 1])]
    if df.empty:
        print("No rows with known spread_covered yet.")
        return

    if "edge" not in df.columns:
        print("No 'edge' column found.")
        return

    df = df.dropna(subset=["edge"])
    if df.empty:
        print("No rows with non-null edge.")
        return

    # Bucket by hybrid edge
    df["edge_bucket"] = pd.cut(
        df["edge"],
        bins=EDGE_BUCKETS,
        labels=EDGE_BUCKET_LABELS,
        right=False,
    )

    stats = (
        df.groupby("edge_bucket")
        .agg(
            n=("id", "count"),
            hit_rate=("spread_covered", "mean"),
            avg_edge=("edge", "mean"),
        )
        .reset_index()
    )

    print("\nEdge bucket performance:")
    for _, row in stats.iterrows():
        bucket = row["edge_bucket"]
        n = int(row["n"])
        hit = row["hit_rate"]
        avg_e = row["avg_edge"]
        print(
            f"  Edge {bucket:>6s}: n={n:3d}, "
            f"hit_rate={hit:5.3f} ({hit*100:5.1f}%), avg_edge={avg_e:5.2f}"
        )


def alignment_stats(df: pd.DataFrame):
    print("\n=== B. Model Alignment (Cheatsheet vs Hybrid) ===")

    if "spread_covered" not in df.columns or "models_aligned" not in df.columns:
        print("Missing 'spread_covered' or 'models_aligned' columns.")
        return

    df = df[df["spread_covered"].isin([0, 1])]
    df = df.dropna(subset=["models_aligned"])
    if df.empty:
        print("No rows with alignment + outcome yet.")
        return

    def summarize(mask, label):
        sub = df[mask]
        if sub.empty:
            print(f"  {label}: n=0")
            return
        n = len(sub)
        hit = sub["spread_covered"].mean()
        avg_edge = sub["edge"].mean() if "edge" in sub.columns else np.nan
        print(
            f"  {label}: n={n:3d}, hit_rate={hit:5.3f} ({hit*100:5.1f}%), "
            f"avg_edge={avg_edge:5.2f}"
        )

    aligned_mask = df["models_aligned"] == 1
    misaligned_mask = df["models_aligned"] == 0

    summarize(aligned_mask, "Aligned (cheatsheet & hybrid agree)")
    summarize(misaligned_mask, "Disagreed")


def weight_learning_stats(df: pd.DataFrame):
    print("\n=== C. Learning Component Weights (Advanced) ===")

    needed_cols = [
        "stat_margin",
        "team_margin",
        "player_adj_eff",
        "pace_adj_term",
        "rest_adj_term",
        "b2b_adj_term",
        "spread_covered",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        print(f"Missing component columns: {missing}")
        print("This will start working once you've logged more games with the new app version.")
        return

    df = df[df["spread_covered"].isin([0, 1])]
    df = df.dropna(subset=needed_cols)
    if len(df) < 10:
        print(f"Only {len(df)} usable rows with full component info — need at least ~10.")
        return

    # Features and labels
    feature_cols = [
        "stat_margin",
        "team_margin",
        "player_adj_eff",
        "pace_adj_term",
        "rest_adj_term",
        "b2b_adj_term",
    ]
    X = df[feature_cols].values
    y = df["spread_covered"].values.astype(int)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000)),
        ]
    )
    pipe.fit(X, y)
    logreg = pipe.named_steps["logreg"]
    coefs = logreg.coef_[0]

    print("\nFitted coefficients (after scaling):")
    for col, c in zip(feature_cols, coefs):
        print(f"  {col:15s}: {c:+.4f}")

    # Turn first 3 into normalized weights
    main_idx = [0, 1, 2]  # stat, team, player
    main_coefs = np.abs(coefs[main_idx])
    main_sum = main_coefs.sum()
    if main_sum == 0:
        print("\nMain coefficients are zero; cannot form weights.")
        return

    w_score, w_team, w_player = (main_coefs / main_sum)

    print("\nSuggested new hybrid weights (normalized from coefficients):")
    print(f"  W_SCORE  (stat_margin)   ≈ {w_score:.3f}")
    print(f"  W_TEAM   (team_margin)   ≈ {w_team:.3f}")
    print(f"  W_PLAYER (player_adj)    ≈ {w_player:.3f}")

    stat_scale = abs(coefs[0]) if coefs[0] != 0 else 1.0
    pace_rel = abs(coefs[3]) / stat_scale
    rest_rel = abs(coefs[4]) / stat_scale
    b2b_rel = abs(coefs[5]) / stat_scale

    print("\nRelative strength vs stat_margin (higher = more important):")
    print(f"  pace_adj_term  ≈ {pace_rel:.3f} × stat_margin")
    print(f"  rest_adj_term  ≈ {rest_rel:.3f} × stat_margin")
    print(f"  b2b_adj_term   ≈ {b2b_rel:.3f} × stat_margin")

    print("\nUse these to adjust W_SCORE, W_TEAM, W_PLAYER and your pace/rest/B2B scales in nba_edge_app.py.")


def main():
    print("=== NBA Edge Analyzer — History Analysis ===")

    df = load_data()
    if df is None or df.empty:
        print("No data in logs yet.")
        return

    # Basic stuff works with minimal data
    basic_edge_stats(df)
    alignment_stats(df)

    # Advanced stuff kicks in once enough component data exists
    weight_learning_stats(df)


if __name__ == "__main__":
    main()
