import requests
import pandas as pd
import time

# ========= CONFIG =========
NBA_STATS_URL = "https://stats.nba.com/stats/leaguedashteamstats"
TEAM_RATINGS_FILE = "Team_ratings.csv"

# Adjust this if needed
SEASON = "2025-26"         # <- current season
SEASON_TYPE = "Regular Season"

# Stats.nba.com requires "real browser" headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/stats/teams/advanced",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
}

def fetch_advanced_json() -> pd.DataFrame:
    """
    Call the official stats.nba.com endpoint for Advanced team stats,
    and return a DataFrame with all columns.
    Retries a few times on timeouts.
    """
    params = {
        "MeasureType": "Advanced",
        "PerMode": "PerGame",
        "PlusMinus": "N",
        "PaceAdjust": "N",
        "Rank": "N",
        "LeagueID": "00",
        "Season": SEASON,
        "SeasonType": SEASON_TYPE,
        "Outcome": "",
        "Location": "",
        "Month": "0",
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "OpponentTeamID": "0",
        "VsConference": "",
        "VsDivision": "",
        "GameSegment": "",
        "Period": "0",
        "ShotClockRange": "",
        "LastNGames": "0",
        "GameScope": "",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "StarterBench": "",
        "TwoWay": "0",
    }

    print(f"[info] Fetching NBA advanced stats from {NBA_STATS_URL} for {SEASON} {SEASON_TYPE}")

    last_exc = None
    for attempt in range(1, 4):  # up to 3 tries
        try:
            print(f"[info] Attempt {attempt}/3 ...")
            resp = requests.get(
                NBA_STATS_URL,
                headers=HEADERS,
                params=params,
                timeout=40  # increase timeout (seconds)
            )
            resp.raise_for_status()
            data = resp.json()

            rs = data["resultSets"][0]
            headers = rs["headers"]
            rows = rs["rowSet"]

            df = pd.DataFrame(rows, columns=headers)
            print("[ok] Successfully fetched advanced stats.")
            return df

        except requests.exceptions.ReadTimeout as e:
            print(f"[warn] Read timeout on attempt {attempt}: {e}")
            last_exc = e
        except requests.exceptions.ConnectionError as e:
            print(f"[warn] Connection error on attempt {attempt}: {e}")
            last_exc = e
        except Exception as e:
            # Any other error – log and stop immediately
            print(f"[error] Unexpected error on attempt {attempt}: {e}")
            raise

        # small backoff before retry
        time.sleep(3)

    # if we got here, all attempts failed
    raise RuntimeError(f"Failed to fetch NBA advanced stats after retries: {last_exc}")


def normalize_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean DataFrame with:
      Team (abbr),
      ORtg, DRtg, NetRtg, Pace
    using columns from the NBA stats response.
    """
    required_cols = [
        "TEAM_ABBREVIATION",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "PACE",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in NBA JSON: {missing}")

    clean = pd.DataFrame()
    clean["Team"] = df["TEAM_ABBREVIATION"].astype(str).str.upper()
    clean["ORtg"] = pd.to_numeric(df["OFF_RATING"], errors="coerce")
    clean["DRtg"] = pd.to_numeric(df["DEF_RATING"], errors="coerce")
    clean["NetRtg"] = pd.to_numeric(df["NET_RATING"], errors="coerce")
    clean["Pace"] = pd.to_numeric(df["PACE"], errors="coerce")

    return clean


def merge_into_team_ratings(adv_df: pd.DataFrame):
    """
    Load Team_ratings.csv, merge advanced stats in by Team,
    and overwrite the file.
    """
    print(f"[info] Loading team ratings from {TEAM_RATINGS_FILE}")
    ratings = pd.read_csv(TEAM_RATINGS_FILE)

    # Normalize Team column name
    if "Team" not in ratings.columns:
        if "TEAM" in ratings.columns:
            ratings.rename(columns={"TEAM": "Team"}, inplace=True)
        else:
            raise RuntimeError("Team_ratings.csv must have a 'Team' or 'TEAM' column.")

    ratings["Team"] = ratings["Team"].astype(str).str.upper()
    adv_df["Team"] = adv_df["Team"].astype(str).str.upper()

    merged = ratings.merge(adv_df, on="Team", how="left", suffixes=("", "_ADV"))

    print("[info] Sample of merged data:")
    print(merged.head())

    merged.to_csv(TEAM_RATINGS_FILE, index=False)
    print(f"[ok] Updated {TEAM_RATINGS_FILE} with ORtg/DRtg/NetRtg/Pace.")


def main():
    df_json = fetch_advanced_json()
    df_adv = normalize_advanced(df_json)
    merge_into_team_ratings(df_adv)


if __name__ == "__main__":
    main()
