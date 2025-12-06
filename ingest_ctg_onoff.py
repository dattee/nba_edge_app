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
    - Filters out players with MIN < MIN_MINUTES (to avoid tiny-sample noise).
    - Loads existing players_onoff.csv if present.
    - Overwrites Diff for any matching (Player, Team) from CTG.
    - Preserves all other columns (DaysSinceLastGame, LastGameDate, etc.).
    - Writes updated players_onoff.csv back to disk.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

# Minimum minutes to trust an on/off diff (tune as needed)
MIN_MINUTES = 300


def load_ctg_onoff(path: Path, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    """Load CTG on/off CSV and normalize to [Player, Team, Diff, MIN].

    This version assumes the CTG file has columns:
        Player, Age, Team, Pos, MIN, MPG, Diff Rank, Diff, ...
    """

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

    # Filter by minutes
    if min_minutes is not None:
        df = df[df["MIN"] >= min_minutes]

    # Add LastUpdateDate for bookkeeping (optional)
    df["LastUpdateDate"] = dt.date.today().isoformat()

    return df


def merge_ctg_into_players(players_path: Path, ctg_df: pd.DataFrame) -> pd.DataFrame:
    """Merge CTG Diff into existing players_onoff.csv (if it exists).

    - If players_onoff.csv exists:
        * Left-join CTG diffs on (Player, Team)
        * Overwrite Diff with CTG's value when available
        * Preserve all other columns (DaysSinceLastGame, etc.)
    - If it does not exist:
        * Build a new DataFrame with just Player, Team, Diff, MIN, LastUpdateDate.
    """

    if players_path.exists():
        old = pd.read_csv(players_path)

        # Ensure Team columns are comparable
        if "Team" in old.columns:
            old["Team"] = old["Team"].astype(str).str.upper()

        merged = old.merge(
            ctg_df[["Player", "Team", "Diff"]],
            on=["Player", "Team"],
            how="left",
            suffixes=("", "_ctg"),
        )

        # If CTG provides a Diff, prefer it; otherwise keep existing Diff
        if "Diff_ctg" in merged.columns:
            if "Diff" in merged.columns:
                merged["Diff"] = merged["Diff_ctg"].combine_first(merged["Diff"])
            else:
                merged["Diff"] = merged["Diff_ctg"]
            merged = merged.drop(columns=["Diff_ctg"])
    else:
        # No existing file -> use CTG data as the base
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
