
import datetime
import time
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd

# This script uses the `nba_api` package:
#   pip install nba_api
#
# It does TWO things:
#   1) Update Team_ratings.csv with fresh ORtg / DRtg / Pace / NetRtg
#   2) Update players_onoff.csv with each player's LastGameDate (season-to-date)
#
# Run it from the same folder as your CSVs:
#   python update_stats_pipeline.py


# ============ CONFIG ============
# Adjust this to the current season in "YYYY-YY" format as used by nba_api.
# Example: "2024-25"
SEASON = "2025-26"

TEAM_RATINGS_CSV = "Team_ratings.csv"
PLAYERS_ONOFF_CSV = "players_onoff.csv"

# If a player hasn't played since more than this many days ago,
# you can treat them as "long-term out" in the app and hide them from
# the Out dropdown.
LONG_TERM_OUT_DAYS = 21


@dataclass
class PlayerLastGameInfo:
    player_name: str
    player_id: Optional[int]
    last_game_date: Optional[datetime.date]


def _ensure_nba_api():
    try:
        import nba_api  # noqa: F401
    except ImportError:
        raise SystemExit(
            "The 'nba_api' package is required. Install it with:\n"
            "  pip install nba_api"
        )


def fetch_team_advanced_stats(season: str) -> pd.DataFrame:
    """
    Fetch advanced team stats (ORtg, DRtg, Pace, NetRtg) for a season.

    Returns a DataFrame with at least:
      TEAM_ID, TEAM_NAME, TEAM_ABBREVIATION,
      OFF_RATING, DEF_RATING, NET_RATING, PACE
    """
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams as static_teams

    # Call NBA endpoint
    resp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    df = resp.get_data_frames()[0]

    # Some versions don't include TEAM_ABBREVIATION, so build it from TEAM_ID
    if "TEAM_ABBREVIATION" not in df.columns:
        teams_list = static_teams.get_teams()
        id_to_abbr = {t["id"]: t["abbreviation"].upper() for t in teams_list}
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(id_to_abbr)

    # Now select the columns we care about
    cols = [
        "TEAM_ID",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "PACE",
    ]
    # Filter to only existing columns in case naming changes slightly
    cols = [c for c in cols if c in df.columns]

    return df[cols].copy()


def build_player_lookup() -> Dict[str, dict]:
    """Build a lookup from UPPERCASE full name to player dict (from nba_api)."""
    from nba_api.stats.static import players

    all_players = players.get_players()
    lookup: Dict[str, dict] = {}
    for p in all_players:
        name_key = p["full_name"].upper().strip()
        # In case of duplicates, just keep the first one
        lookup.setdefault(name_key, p)
    return lookup


def find_player_id_by_name(raw_name: str, name_lookup: Dict[str, dict]) -> Optional[int]:
    """Try to map a name from players_onoff.csv to an nba_api player id.

    This does a few normalizations but is still name-based, so
    some manual corrections may be needed in your CSV if names
    don't match how nba_api stores them.
    """
    name = raw_name.upper().strip()

    # Direct lookup first
    if name in name_lookup:
        return name_lookup[name]["id"]

    # Try some simple cleanup: remove periods (e.g., "Jr.", initials), double spaces
    for ch in [".", ","]:
        name = name.replace(ch, " ")
    name = " ".join(name.split())

    if name in name_lookup:
        return name_lookup[name]["id"]

    # As a last resort, try contains-style matches
    for key, pdata in name_lookup.items():
        if name in key or key in name:
            return pdata["id"]

    return None


def fetch_player_last_game(player_id: int, season: str) -> Optional[datetime.date]:
    """
    Return the most recent game date in the season where the player played.

    If no games are found or the API times out, returns None.
    """
    from nba_api.stats.endpoints import playergamelog

    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gl.get_data_frames()[0]
    except Exception as e:
        # Any timeout / HTTP / parsing error -> just skip this player
        print(f"  [WARN] PlayerGameLog failed for id {player_id}: {e}")
        return None

    if df.empty:
        return None

    # GAME_DATE column is like "OCT 29, 2024"
    last_row = df.iloc[0]  # nba_api returns logs in reverse chronological order
    date_str = last_row.get("GAME_DATE")
    if not date_str:
        return None

    try:
        dt = datetime.datetime.strptime(date_str, "%b %d, %Y").date()
        return dt
    except Exception:
        return None


def update_team_ratings(team_ratings_path: str, season: str) -> None:
    print(f"Loading team ratings from {team_ratings_path} ...")
    tr = pd.read_csv(team_ratings_path)
    if "Team" not in tr.columns:
        raise ValueError("Team_ratings.csv must have a 'Team' column with team abbreviations.")

    tr["Team"] = tr["Team"].str.upper()

    print(f"Fetching team advanced stats for season {season} ...")
    adv = fetch_team_advanced_stats(season)

    # Map from abbr -> row
    adv["TEAM_ABBREVIATION"] = adv["TEAM_ABBREVIATION"].str.upper()
    adv_map = {row["TEAM_ABBREVIATION"]: row for _, row in adv.iterrows()}

    # Ensure ORtg/DRtg/Pace/NetRtg columns exist in Team_ratings
    for col in ["ORtg", "DRtg", "Pace", "NetRtg"]:
        if col not in tr.columns:
            tr[col] = pd.NA

    print("Merging advanced stats into Team_ratings.csv ...")
    updated_rows = 0
    for idx, row in tr.iterrows():
        abbr = str(row["Team"]).upper()
        if abbr not in adv_map:
            # Uncomment if you want to see which teams didn't match
            # print(f"[WARN] No advanced stats for Team '{abbr}'")
            continue

        adv_row = adv_map[abbr]
        tr.at[idx, "ORtg"] = adv_row["OFF_RATING"]
        tr.at[idx, "DRtg"] = adv_row["DEF_RATING"]
        tr.at[idx, "Pace"] = adv_row["PACE"]
        # You can choose whether to overwrite Base Net Rating or store NetRtg separately
        tr.at[idx, "NetRtg"] = adv_row["NET_RATING"]
        updated_rows += 1

    print(f"Updated advanced stats for {updated_rows} teams.")

    backup_path = team_ratings_path.replace(".csv", "_backup.csv")
    print(f"Saving backup to {backup_path} ...")
    tr.to_csv(backup_path, index=False)

    print(f"Writing updated team ratings to {team_ratings_path} ...")
    tr.to_csv(team_ratings_path, index=False)


def update_players_onoff(players_onoff_path: str, season: str) -> None:
    print(f"Loading players_onoff from {players_onoff_path} ...")
    df = pd.read_csv(players_onoff_path)

    if "Player" not in df.columns or "Team" not in df.columns:
        raise ValueError("players_onoff.csv must have 'Player' and 'Team' columns.")

    # Prepare new columns
    if "LastGameDate" not in df.columns:
        df["LastGameDate"] = pd.NA
    if "DaysSinceLastGame" not in df.columns:
        df["DaysSinceLastGame"] = pd.NA

    # Build player lookup from nba_api
    print("Building player name lookup ...")
    name_lookup = build_player_lookup()

    today = datetime.date.today()
    n = len(df)
    print(f"Processing {n} players ...")

    for idx, row in df.iterrows():
    player_name = str(row["Player"])

    # If we already know their last game date, just recompute DaysSinceLastGame locally
    existing_date = row.get("LastGameDate")
    if pd.notna(existing_date):
        try:
            last_game = datetime.date.fromisoformat(str(existing_date))
            df.at[idx, "DaysSinceLastGame"] = (today - last_game).days
            continue  # skip API call for this player
        except Exception:
            # If parsing fails, we'll fall through and try the API fresh
            pass

    # Otherwise, we need to hit the API to discover their last game
    pid = find_player_id_by_name(player_name, name_lookup)
    if pid is None:
        # Could not map this player name
        continue

    last_game = fetch_player_last_game(pid, season)
    if last_game is None:
        continue

    df.at[idx, "LastGameDate"] = last_game.isoformat()
    df.at[idx, "DaysSinceLastGame"] = (today - last_game).days

    if (idx + 1) % 25 == 0 or idx == n - 1:
        print(f"  -> processed {idx + 1}/{n} players")

    backup_path = players_onoff_path.replace(".csv", "_backup.csv")
    print(f"Saving backup to {backup_path} ...")
    df.to_csv(backup_path, index=False)

    print(f"Writing updated players_onoff to {players_onoff_path} ...")
    df.to_csv(players_onoff_path, index=False)


def main():
    _ensure_nba_api()
    print("=== NBA Edge Analyzer Stats Pipeline ===")
    print(f"Season: {SEASON}")
    print()

    update_team_ratings(TEAM_RATINGS_CSV, SEASON)
    print()
    update_players_onoff(PLAYERS_ONOFF_CSV, SEASON)

    print("\nDone. You can now re-run your Streamlit app using the refreshed CSVs.")


if __name__ == "__main__":
    main()
