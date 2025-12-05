"""
rapidapi_injuries_client.py

Client for the "NBA Injuries Reports" API on RapidAPI.

Transforms the API results into:
    { 'MIA': {'out': [...], 'q': [...], 'other': [...]}, ... }

Requires .env with:
    RAPIDAPI_NBA_INJURIES_KEY=your_rapidapi_key_here
"""

import os
from datetime import date
from typing import Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

RAPID_KEY = os.getenv("RAPIDAPI_NBA_INJURIES_KEY")
if not RAPID_KEY:
    raise RuntimeError("RAPIDAPI_NBA_INJURIES_KEY not set in .env")

BASE_URL = "https://nba-injuries-reports.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": RAPID_KEY,
    "X-RapidAPI-Host": "nba-injuries-reports.p.rapidapi.com",
}

# Map full team names from the API -> app team abbreviations
TEAM_NAME_TO_ABBR: Dict[str, str] = {
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
    "LA Clippers": "LAC",
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


def fetch_injuries_for_date(d: date) -> List[dict]:
    """
    Raw call to the RapidAPI endpoint for a specific date.

    Endpoint shape:
        GET /injuries/nba/YYYY-MM-DD
    Returns a list of records like:
        {'date': '2025-12-05', 'team': 'Miami Heat', 'player': 'Tyler Herro',
         'status': 'Doubtful', 'reason': '...', 'reportTime': '07PM'}
    """
    date_str = d.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/injuries/nba/{date_str}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    # This API currently returns a bare list
    if isinstance(data, list):
        return data
    # Fallback in case provider wraps it later
    return data.get("data") or data.get("results") or data.get("response") or []


def _classify_status(status: str) -> str:
    """
    Map API 'status' into logical buckets:
        OUT, Q (questionable/doubtful/probable), OTHER
    """
    s = (status or "").lower()
    if "out" in s:
        return "OUT"
    if "doubtful" in s or "questionable" in s or "probable" in s:
        return "Q"
    if "available" in s:
        return "OTHER"
    return "OTHER"


def build_team_injury_lists_for_date(d: date) -> Dict[str, Dict[str, List[str]]]:
    """
    Return dict for the given date:

        {
          'MIA': {'out': [...], 'q': [...], 'other': [...]},
          'BOS': {...},
          ...
        }

    This structure matches what the app previously used for the
    Basketball-Reference scraper.
    """
    raw = fetch_injuries_for_date(d)

    team_lists: Dict[str, Dict[str, List[str]]] = {}

    for item in raw:
        team_full = item.get("team")
        player = item.get("player")
        status = item.get("status")

        if not team_full or not player:
            continue

        abbr = TEAM_NAME_TO_ABBR.get(team_full)
        if not abbr:
            # Unknown team name; skip
            continue

        bucket = _classify_status(status)
        dct = team_lists.setdefault(abbr, {"out": [], "q": [], "other": []})

        if bucket == "OUT":
            dct["out"].append(player)
        elif bucket == "Q":
            dct["q"].append(player)
        else:
            dct["other"].append(player)

    # De-duplicate + sort
    for abbr, info in team_lists.items():
        for key in ("out", "q", "other"):
            info[key] = sorted(set(info[key]))

    return team_lists
