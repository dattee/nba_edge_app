"""
ingest_ctg_onoff.py

Use Cleaning The Glass' exported on/off table to update players_onoff.csv.

Expected input:
    CTG_onoff/ctg_onoff_eff_2025.csv

This CSV should be exported from:
    Leaders -> "On/Off Efficiency & Four Factors" (players)

It must contain at least the columns:
    - Player
    - Team
    - Diff        (on/off net rating in pts/100 poss)
    - MIN         (total minutes)

Behavior:
    - Reads the CTG CSV.
    - Keeps only [Player, Team, Diff, MIN].
    - (No minutes filter by default: we keep **all** CTG players.)
    - If players_onoff.csv exists:
        * Treat CTG as the base set of players.
        * Outer-join old players_onoff on (Player, Team) to bring over
          DaysSinceLastGame, LastGameDate, etc.
        * Prefer CTG Diff and MIN when present.
    - If players_onoff.csv does not exist:
        * Create it from CTG data.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

# Set this to a number (e.g., 300) if you want to filter tiny minutes.
# For "fully extract all players", we leave it as None (no filter).
MIN_MINUTES = None


def load_ctg_onoff(path: Path, min_minutes: int | None = MIN_MINUTES) -> pd.DataFrame:
    """Load CTG on/off CSV and normalize to [Player, Team, Diff, MIN]."""

    df = pd.read_csv(path)

    required_cols = ["Player", "Team", "Diff", "MIN"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CTG on/off file is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    df = df[required_cols].copy()

    # Normalize types
    df["Player"] = df["Player"].astype(str)
    df["Team"] = df["Team"].astype(str).str.upper()
    df["Diff"] = pd.to_numeric(df["Diff"], errors="coerce")
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")

    # Drop rows with missing Diff or MIN
    df = df.dropna(subset=["Diff", "MIN"])

    # Filter by minutes if desired
    if min_minutes is not None:
        df = df[df["MIN"] >= min_minutes]

    # Add LastUpdateDate for bookkeeping (optional)
    df["LastUpdateDate"] = dt.date.today().isoformat()

    return df


def merge_ctg_into_players(players_path: Path, ctg_df: pd.DataFrame) -> pd.DataFrame:
    """Merge CTG Diff into players_onoff.csv.

    Logic:
        - CTG is the base. We want **all** CTG players in the final file.
        - If players_onoff.csv exists:
            * Outer-join CTG and old file on (Player, Team).
            * Prefer CTG Diff and MIN when CTG has them.
            * Preserve other columns from old file (DaysSinceLastGame, etc.)
              when available.
        - If it does not exist:
            * Just use CTG data as the base.
    """

    if players_path.exists():
        old = pd.read_csv(players_path)

        # Normalize Team in old file as well
        if "Team" in old.columns:
            old["Team"] = old["Team"].astype(str).str.upper()

        # Outer join: we want union of (Player, Team) from CTG and old
        merged = pd.merge(
            ctg_df,
            old,
            on=["Player", "Team"],
            how="outer",
            suffixes=("_ctg", ""),
        )

        # Prefer CTG's Diff when it exists
        if "Diff_ctg" in merged.columns:
            if "Diff" in merged.columns:
                merged["Diff"] = merged["Diff_ctg"].combine_first(merged["Diff"])
            else:
                merged["Diff"] = merged["Diff_ctg"]

        # Prefer CTG's MIN when it exists
        if "MIN_ctg" in merged.columns:
            if "MIN" in merged.columns:
                merged["MIN"] = merged["MIN_ctg"].combine_first(merged["MIN"])
            else:
                merged["MIN"] = merged["MIN_ctg"]

        # Prefer CTG's LastUpdateDate when available
        if "LastUpdateDate_ctg" in merged.columns:
            if "LastUpdateDate" in merged.columns:
                merged["LastUpdateDate"] = merged["LastUpdateDate_ctg"].combine_first(
                    merged["LastUpdateDate"]
                )
            else:
                merged["LastUpdateDate"] = merged["LastUpdateDate_ctg"]

        # Drop CTG suffix columns we no longer need
        drop_cols = [c for c in merged.columns if c.endswith("_ctg")]
        merged = merged.drop(columns=drop_cols)
    else:
        # No existing file -> CTG becomes the whole file
        merged = ctg_df.copy()

    return merged


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    ctg_dir = base_dir / "CTG_onoff"
    onoff_file = ctg_dir / "ctg_onoff_eff_2025.csv"

    if not onoff_file.exists():
        print(
            f"CTG on/off file not found at {onoff_file}\n"
            "Make sure you've exported the 'On/Off Efficiency & Four Factors' "
            "leaders table and saved it as ctg_onoff_eff_2025.csv inside CTG_onoff."
        )
        return

    try:
        ctg_df = load_ctg_onoff(onoff_file)
    except Exception as e:
        print("Failed to load CTG on/off file:", e)
        return

    players_path = base_dir / "players_onoff.csv"
    merged = merge_ctg_into_players(players_path, ctg_df)
    merged.to_csv(players_path, index=False)
    print(f"Wrote {len(merged)} rows to {players_path}")


if __name__ == "__main__":
    main()
