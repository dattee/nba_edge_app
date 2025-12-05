"""
rapidapi_nba_injuries_test.py

Quick test for the "nba-injuries-reports" API on RapidAPI.
"""

import os
import requests
from dotenv import load_dotenv
from datetime import date

load_dotenv()

API_KEY = os.getenv("RAPIDAPI_NBA_INJURIES_KEY")
if not API_KEY:
    raise RuntimeError("RAPIDAPI_NBA_INJURIES_KEY not set in .env")

BASE_URL = "https://nba-injuries-reports.p.rapidapi.com"

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "nba-injuries-reports.p.rapidapi.com",
}


def get_injuries_for_date(d: date):
    date_str = d.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/injuries/nba/{date_str}"
    print("Requesting:", url)

    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def main():
    # use today, or pick a known game date if you want
    today = date.today()
    data = get_injuries_for_date(today)

    print("Raw response type:", type(data))

    if isinstance(data, dict):
        records = data.get("data") or data.get("results") or data.get("response") or []
    else:
        records = data

    print(f"Number of injury records: {len(records)}")
    for item in records[:10]:
        print(item)


if __name__ == "__main__":
    main()
