import sqlite3
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DB_NAME = "model_logs.db"

def load_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM logs", conn)
    conn.close()
    return df

def main():
    print("=== NBA Edge Analyzer — Weight Learning ===")

    df = load_data()

    # Only rows where we know if the spread was covered (0/1)
    df = df[df["spread_covered"].isin([0, 1])]

    # Only rows with all component columns present
    feature_cols = [
        "stat_margin",
        "team_margin",
        "player_adj_eff",
        "pace_adj_term",
        "rest_adj_term",
        "b2b_adj_term",
    ]
    df = df.dropna(subset=feature_cols)

    if len(df) < 30:
        print(f"Only {len(df)} usable rows — you probably want at least ~30–50 before trusting the fit.")
    if len(df) < 10:
        print("Not enough data to fit anything reasonable yet.")
        return

    X = df[feature_cols].values
    y = df["spread_covered"].values.astype(int)

    # Logistic regression: P(cover) = sigmoid(w·x + b)
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

    # Turn the first 3 into model weights (absolute value, normalized)
    main_idx = [0, 1, 2]  # stat_margin, team_margin, player_adj_eff
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

    # Pace / rest / B2B: relative scale compared to stat margin
    stat_scale = abs(coefs[0]) if coefs[0] != 0 else 1.0

    pace_rel = abs(coefs[3]) / stat_scale
    rest_rel = abs(coefs[4]) / stat_scale
    b2b_rel  = abs(coefs[5]) / stat_scale

    print("\nRelative strength vs stat_margin (higher = more important):")
    print(f"  pace_adj_term  ≈ {pace_rel:.3f} × stat_margin")
    print(f"  rest_adj_term  ≈ {rest_rel:.3f} × stat_margin")
    print(f"  b2b_adj_term   ≈ {b2b_rel:.3f} × stat_margin")

    print("\nYou can use these as guidance to tweak:")
    print("  W_SCORE, W_TEAM, W_PLAYER in nba_edge_app.py")
    print("  PACE_DELTA_WEIGHT, REST_GAME_WEIGHT, and your B2B penalty scale.\n")

if __name__ == "__main__":
    main()
