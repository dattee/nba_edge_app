import numpy as np
import unicodedata
import re
import datetime
import time
from dataclasses import dataclass
from typing import Optional, Dict, Set
from nba_api.stats.endpoints import commonallplayers, playergamelog
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


def fetch_today_slate_teams(game_date: Optional[datetime.date] = None) -> Set[str]:
    # Use scoreboardv2 instead of scoreboard
    from nba_api.stats.endpoints import scoreboardv2
    from nba_api.stats.static import teams as static_teams

    if game_date is None:
        game_date = datetime.date.today()

    date_str = game_date.strftime("%m/%d/%Y")
    print(f"Fetching scoreboard for {date_str} ...")

    # ScoreboardV2 returns game header + other tables
    sb = scoreboardv2.ScoreboardV2(game_date=date_str, league_id="00", day_offset=0)
    games = sb.game_header.get_data_frame()
    if games.empty:
        print("No games found on scoreboard for this date.")
        return set()

    teams_list = static_teams.get_teams()
    id_to_abbr = {t["id"]: t["abbreviation"].upper() for t in teams_list}

    slate: Set[str] = set()
    for _, row in games.iterrows():
        home_id = int(row["HOME_TEAM_ID"])
        away_id = int(row["VISITOR_TEAM_ID"])
        home_abbr = id_to_abbr.get(home_id)
        away_abbr = id_to_abbr.get(away_id)
        if home_abbr:
            slate.add(home_abbr)
        if away_abbr:
            slate.add(away_abbr)

    print(f"Today's slate teams: {', '.join(sorted(slate))}")
    return slate

    
# Map CSV names -> canonical names when they genuinely differ.
# Example: your CSV might say "Cameron Thomas" but NBA uses "Cam Thomas".
NAME_ALIASES = {
    "cameron thomas": "cam thomas",
    "nicolas claxton": "nic claxton",
    "nick claxton": "nic claxton",  # in case you ever type it this way
    # add more if you see mismatches
    
}


def normalize_name(name: str) -> str:
    """
    Normalize player names so CSV names like 'Luka Doncic' match
    API names like 'Luka Dončić' or 'Jaren Jackson Jr.'.
    Also applies manual aliases (e.g. 'Cameron Thomas' -> 'Cam Thomas').
    """
    if not isinstance(name, str):
        return ""

    # 1) strip accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # 2) remove punctuation we don't care about
    name = name.replace(".", "")
    name = name.replace("'", "")
    name = name.replace("-", " ")

    # 3) remove common suffixes like Jr, Sr, II, III, IV
    name = re.sub(r"\s+\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.IGNORECASE)

    # 4) normalize whitespace + lowercase
    name = re.sub(r"\s+", " ", name).strip().lower()

    # 5) apply alias map (e.g. "cameron thomas" -> "cam thomas")
    if name in NAME_ALIASES:
        name = NAME_ALIASES[name]

    return name


def build_player_lookup() -> Dict[str, int]:
    """
    Build a lookup from NORMALIZED name -> NBA player id using nba_api.
    This handles accents, Jr., punctuation, and alias names.
    """
    from nba_api.stats.static import players as nba_players

    all_players = nba_players.get_players()
    lookup: Dict[str, int] = {}

    for p in all_players:
        full_name = p.get("full_name") or ""
        key = normalize_name(full_name)
        if not key:
            continue
        # keep first if duplicates
        lookup.setdefault(key, p["id"])

    return lookup


def find_player_id_by_name(raw_name: str, name_lookup: Dict[str, int]) -> Optional[int]:
    """
    Normalize a CSV name and look it up in the NBA player dict.
    """
    if not isinstance(raw_name, str):
        return None
    key = normalize_name(raw_name)
    return name_lookup.get(key)



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


def update_team_ratings(
    team_ratings_path: str,
    season: str,
    only_teams: Optional[Set[str]] = None,
) -> None:
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

    # Normalize only_teams to uppercase, if provided
    if only_teams is not None:
        only_teams = {t.upper() for t in only_teams}

    # Ensure ORtg/DRtg/Pace/NetRtg columns exist in Team_ratings
    for col in ["ORtg", "DRtg", "Pace", "NetRtg"]:
        if col not in tr.columns:
            tr[col] = pd.NA

    print("Merging advanced stats into Team_ratings.csv ...")
    updated_rows = 0
    for idx, row in tr.iterrows():
        abbr = str(row["Team"]).upper()
        # If only_teams specified, skip others
        if only_teams is not None and abbr not in only_teams:
            continue
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


from typing import Optional, Dict, Set  # make sure Set is imported at top

def update_players_onoff(
    csv_path: str,
    season: str,
    only_teams: Optional[Set[str]] = None,
):
    """
    Update players_onoff.csv with LastGameDate and DaysSinceLastGame
    using nba_api PlayerGameLog.

    - Resolves player_id by normalized name (handles accents, Jr, aliases).
    - If no ID or error: leaves LastGameDate blank and DaysSinceLastGame as NaN.
    - Skips players that were already updated today (LastUpdateDate == today).
    - If only_teams is provided, only players on those teams are refreshed;
      others keep their existing values.
    """
    print(f"Loading players_onoff from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["Team"] = df["Team"].str.upper()

    # Normalize filter set
    if only_teams is not None:
        only_teams = {t.upper() for t in only_teams}

    # Ensure columns exist
    if "LastGameDate" not in df.columns:
        df["LastGameDate"] = ""
    if "DaysSinceLastGame" not in df.columns:
        df["DaysSinceLastGame"] = np.nan
    if "LastUpdateDate" not in df.columns:
        df["LastUpdateDate"] = ""  # YYYY-MM-DD string

    # Build normalized name -> id lookup once
    print("Building player name lookup (normalized) ...")
    name_to_id = build_player_lookup()

    today = datetime.date.today()
    today_str = today.isoformat()

    new_last_dates = []
    new_days_since = []
    new_update_dates = []

    # Count how many we will actually process
    if only_teams is not None:
        to_process = df["Team"].isin(only_teams).sum()
        print(f"Processing {to_process} players (filtered to slate teams).")
    else:
        print(f"Processing {len(df)} players ...")

    for _, row in df.iterrows():
        name = row["Player"]
        team_abbr = str(row["Team"]).upper()

        last_update = str(row.get("LastUpdateDate", "") or "")
        # If this player is not on a slate team, just carry forward existing values
        if only_teams is not None and team_abbr not in only_teams:
            new_last_dates.append(str(row.get("LastGameDate", "") or ""))
            try:
                new_days_since.append(float(row.get("DaysSinceLastGame", np.nan)))
            except Exception:
                new_days_since.append(np.nan)
            new_update_dates.append(last_update)
            continue

        # Skip API call if player was updated within the last N days (including today)
        SKIP_DAYS = 3
        recently_updated = False
        if last_update:
            try:
                last_dt = datetime.date.fromisoformat(last_update)
                delta_days = (today - last_dt).days
                if delta_days >= 0 and delta_days <= SKIP_DAYS:
                    recently_updated = True
            except Exception:
                recently_updated = False

        if recently_updated:
            new_last_dates.append(str(row.get("LastGameDate", "") or ""))
            try:
                new_days_since.append(float(row.get("DaysSinceLastGame", np.nan)))
            except Exception:
                new_days_since.append(np.nan)
            new_update_dates.append(last_update)
            continue

        norm_name = normalize_name(name)
        player_id = name_to_id.get(norm_name)

        if player_id is None:
            print(f"[WARN] Could not resolve id for player '{name}' (norm='{norm_name}')")
            new_last_dates.append(str(row.get("LastGameDate", "") or ""))
            try:
                new_days_since.append(float(row.get("DaysSinceLastGame", np.nan)))
            except Exception:
                new_days_since.append(np.nan)
            new_update_dates.append(last_update)
            continue

        try:
            print(f"Fetching game log for {name} ({team_abbr}) ...")
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season",
            )
            games = gl.get_data_frames()[0]
            if games.empty:
                print(f"  -> no games found for {name} in {season}")
                new_last_dates.append("")
                new_days_since.append(np.nan)
                new_update_dates.append(today_str)
                continue

            gdf = pd.DataFrame(games)
            gdf["GAME_DATE"] = pd.to_datetime(gdf["GAME_DATE"])
            last_date = gdf["GAME_DATE"].max().date()

            days_since = (today - last_date).days
            new_last_dates.append(last_date.isoformat())
            new_days_since.append(float(days_since))
            new_update_dates.append(today_str)

            print(f"  -> {name}: last played {last_date} ({days_since} days ago)")

        except Exception as e:
            print(f"  -> error fetching log for '{name}': {e}")
            new_last_dates.append(str(row.get("LastGameDate", "") or ""))
            try:
                new_days_since.append(float(row.get("DaysSinceLastGame", np.nan)))
            except Exception:
                new_days_since.append(np.nan)
            new_update_dates.append(last_update)

    # Write back into the DataFrame
    df["LastGameDate"] = new_last_dates
    df["DaysSinceLastGame"] = new_days_since
    df["LastUpdateDate"] = new_update_dates

    # Backup and save
    backup_path = csv_path.replace(".csv", "_backup.csv")
    print(f"Saving backup to {backup_path} ...")
    df.to_csv(backup_path, index=False)

    print(f"Writing updated players_onoff to {csv_path} ...")
    df.to_csv(csv_path, index=False)
    print("Done updating players_onoff.")


def main():
    _ensure_nba_api()
    print("=== NBA Edge Analyzer Stats Pipeline ===")
    print(f"Season: {SEASON}")
    print()

    import sys
    mode = "full"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "slate":
        print("[mode] Slate-only update (today's games).")
        slate_teams = fetch_today_slate_teams()
        if not slate_teams:
            print("No slate teams found; nothing to update.")
            return

        update_team_ratings(TEAM_RATINGS_CSV, SEASON, only_teams=slate_teams)
        print()
        update_players_onoff(PLAYERS_ONOFF_CSV, SEASON, only_teams=slate_teams)

    else:
        print("[mode] Full update (all teams + all players).")
        update_team_ratings(TEAM_RATINGS_CSV, SEASON)
        print()
        update_players_onoff(PLAYERS_ONOFF_CSV, SEASON)

    print("\nDone. You can now re-run your Streamlit app using the refreshed CSVs.")


if __name__ == "__main__":
    main()

