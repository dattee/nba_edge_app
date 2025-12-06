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


def normalize_injury_status(raw_status: str) -> str:
    """
    Map raw injury 'status' from API to one of:
        IN, PROBABLE, QUESTIONABLE, DOUBTFUL, OUT
    """
    if not raw_status:
        return "IN"

    s = str(raw_status).strip().lower()

    if "out" in s or "will not play" in s or "inactive" in s:
        # 'Out', 'Out (personal)', 'will not play', etc.
        return "OUT"
    if "doubt" in s:
        return "DOUBTFUL"
    if "quest" in s:
        return "QUESTIONABLE"
    if "prob" in s:
        return "PROBABLE"

    # 'Available', 'Active', anything else -> treat as playing
    return "IN"


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


def build_injury_lists_and_status_for_date(
    d: date,
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, str]]]:
    """
    Build both the legacy list-style dict and a detailed status map for a date.

    Returns
    -------
    team_lists : {abbr: {"out": [...], "q": [...], "other": [...]}}
    team_status : {abbr: {player: normalized_status}}
    """
    raw = fetch_injuries_for_date(d)

    team_lists: Dict[str, Dict[str, List[str]]] = {}
    team_status: Dict[str, Dict[str, str]] = {}

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

        norm = normalize_injury_status(status)

        bucket = "OTHER"
        if norm == "OUT":
            bucket = "OUT"
        elif norm in {"DOUBTFUL", "QUESTIONABLE", "PROBABLE"}:
            bucket = "Q"

        lists_dict = team_lists.setdefault(abbr, {"out": [], "q": [], "other": []})

        if bucket == "OUT":
            lists_dict["out"].append(player)
        elif bucket == "Q":
            lists_dict["q"].append(player)
        else:
            lists_dict["other"].append(player)

        status_map = team_status.setdefault(abbr, {})
        status_map[player] = norm

    # De-duplicate + sort for list-style output
    for abbr, info in team_lists.items():
        for key in ("out", "q", "other"):
            info[key] = sorted(set(info[key]))

    return team_lists, team_status


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
    team_lists, _ = build_injury_lists_and_status_for_date(d)
    return team_lists


def build_team_status_for_date(d: date) -> Dict[str, Dict[str, str]]:
    """Return team -> player -> normalized status map for the given date."""
    _, team_status = build_injury_lists_and_status_for_date(d)
    return team_status
