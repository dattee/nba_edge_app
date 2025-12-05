"""
injury_scraper_bref.py

Pull the NBA injury report from Basketball-Reference and convert it into
per-team injury lists that your model can use.

Usage (from your project folder):

    python injury_scraper_bref.py

Requirements:
    pip install pandas requests beautifulsoup4
"""

import requests
import pandas as pd
from typing import Dict, List


BREF_INJURY_URL = "https://www.basketball-reference.com/friv/injuries.fcgi"


def get_bref_injuries() -> pd.DataFrame:
    """
    Fetch the NBA injuries table from Basketball-Reference.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing all current injuries. Expected columns include:
        ['Player', 'Team', 'Update', 'Description', ...]
    """
    print(f"Fetching injuries from {BREF_INJURY_URL} ...")

    # Use pandas to read the table with id="injuries"
    tables = pd.read_html(BREF_INJURY_URL, attrs={"id": "injuries"})
    if not tables:
        raise RuntimeError("No injuries table found on Basketball-Reference page.")

    df = tables[0]
    print(f"Loaded injury table: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", list(df.columns))

    return df


def injuries_by_team(df: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    Group injuries by team name.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame returned by get_bref_injuries().

    Returns
    -------
    by_team : dict
        Example structure:
        {
          'Atlanta Hawks': [
              {
                'player': 'Jalen Johnson',
                'description': 'Out (Calf) - The Hawks ...',
                'update': 'Wed, Dec 3, 2025'
              },
              ...
          ],
          ...
        }
    """
    required_cols = {"Player", "Team", "Description"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in injuries DF: {missing}")

    by_team: Dict[str, List[dict]] = {}

    for _, row in df.iterrows():
        team_name = row["Team"]
        entry = {
            "player": row["Player"],
            "description": row["Description"],
            "update": row.get("Update", None),
        }
        by_team.setdefault(team_name, []).append(entry)

    return by_team


# Mapping from Basketball-Reference full team names -> your abbreviations
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def classify_status(description: str) -> str:
    """
    Classify a Basketball-Reference injury description into
    OUT / DOUBTFUL / QUESTIONABLE / OTHER.

    Very simple keyword-based heuristic that we can refine later.
    """
    desc = (description or "").lower()

    # Strong OUT signals
    if desc.startswith("out ") or desc.startswith("out("):
        return "OUT"
    if " as out " in desc:
        return "OUT"
    if " listed toohey as out " in desc:  # sample phrasing, but generic rule above should hit
        return "OUT"
    if "out for season" in desc:
        return "OUT"

    # Doubtful
    if "doubtful" in desc:
        return "DOUBTFUL"

    # Questionable / Day-to-day bucket
    if "questionable" in desc or "day to day" in desc:
        return "QUESTIONABLE"

    return "OTHER"


def build_team_injury_lists(by_team: Dict[str, List[dict]]) -> Dict[str, dict]:
    """
    Convert the raw 'by_team' dict into per-abbreviation lists:

        {
          'GSW': {
              'out': [...],
              'q': [...],       # questionable + doubtful
              'other': [...],   # anything else
          },
          ...
        }
    """
    result: Dict[str, dict] = {}

    for team_name, entries in by_team.items():
        abbr = TEAM_NAME_TO_ABBR.get(team_name)
        if not abbr:
            # Skip G-League / weird teams if they ever appear
            continue

        out_list: List[str] = []
        q_list: List[str] = []
        other_list: List[str] = []

        for e in entries:
            status = classify_status(e["description"])
            player = e["player"]

            if status == "OUT":
                out_list.append(player)
            elif status in ("DOUBTFUL", "QUESTIONABLE"):
                q_list.append(player)
            else:
                other_list.append(player)

        result[abbr] = {
            "out": sorted(set(out_list)),
            "q": sorted(set(q_list)),
            "other": sorted(set(other_list)),
        }

    return result


def print_example_output(team_lists: Dict[str, dict]) -> None:
    """
    Print a small sample of the structured injury lists for sanity checking.
    """
    for abbr in ["GSW", "PHI"]:
        print(f"\n=== {abbr} ===")
        info = team_lists.get(abbr, {})
        if not info:
            print("  (No injury info.)")
            continue

        print("  OUT:")
        for p in info.get("out", []):
            print(f"    - {p}")

        print("  QUESTIONABLE/DOUBTFUL:")
        for p in info.get("q", []):
            print(f"    - {p}")

        if info.get("other"):
            print("  OTHER:")
            for p in info["other"]:
                print(f"    - {p}")


def main():
    df = get_bref_injuries()

    print("\nFirst 5 rows of raw table:")
    print(df.head())

    by_team = injuries_by_team(df)
    team_lists = build_team_injury_lists(by_team)

    print_example_output(team_lists)
    
def get_team_injury_lists() -> dict:
    """
    Convenience function: fetch the BRef injuries page and return
    the per-team injury lists keyed by team abbreviation.

    Returns
    -------
    {
      'GSW': {'out': [...], 'q': [...], 'other': [...]},
      'PHI': {...},
      ...
    }
    """
    df = get_bref_injuries()
    by_team = injuries_by_team(df)
    team_lists = build_team_injury_lists(by_team)
    return team_lists



if __name__ == "__main__":
    main()
