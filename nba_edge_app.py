import math
import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import os
import re
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

# Optional: Basketball-Reference injury scraper
try:
    from sportsdata_client_test import get_team_injury_lists
    HAS_BREF_INJURIES = True
except Exception:
    get_team_injury_lists = None
    HAS_BREF_INJURIES = False

# Load environment variables from .env
load_dotenv()

# Read key from environment and create client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# CONFIG
# =========================
DB_NAME = os.getenv("NBA_EDGE_DB_PATH", "model_logs.db")
st.sidebar.caption(f"DB in use: {os.path.abspath(DB_NAME)}")

ODDS_API_KEY = "fec9f6305dd7a9785e5f261c03c885d7"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
SCORES_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/scores"
REGION = "us"
MARKETS = "spreads"

K5 = 8.0
STRONG_EDGE = 6.0
MEDIUM_EDGE = 2.5

# Hybrid weights (can tune later or learn from history)
W_SCORE = 0.5      # manual/auto score projection margin
W_TEAM = 0.3       # team strength margin
W_PLAYER = 0.2     # player on/off adjustment

# How long a player can be out before we treat them as "baseline absent"
# and stop showing them in the Out dropdown
MAX_DAYS_ABSENT = 21  # you can tweak this later

# Pace weights (small so they nudge rather than dominate)
PACE_DELTA_WEIGHT = 0.06   # fav pace - dog pace
PACE_ENV_WEIGHT = 0.03     # avg pace vs league average

# Rest / schedule density (games in last 5 days)
REST_GAME_WEIGHT = 0.3     # points per extra game in last 5 days (fav perspective)

ET_TZ = ZoneInfo("America/New_York")

# Standard NBA abbrevs
TEAM_ABBRS = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC",
    "LA CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}

# Session flags for Single Game result UI
if "has_decision" not in st.session_state:
    st.session_state["has_decision"] = False
if "decision_logged" not in st.session_state:
    st.session_state["decision_logged"] = False

# =========================
# DB Setup
# =========================
engine = create_engine(f"sqlite:///{DB_NAME}", echo=False)
with engine.connect() as conn:
    # Base table
    conn.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            favorite TEXT,
            underdog TEXT,
            vegas_line REAL,
            model_line REAL,
            edge REAL,
            pick TEXT,
            confidence TEXT
        )
    """
        )
    )

    # Add new columns if they don't exist yet
    cols = conn.execute(text("PRAGMA table_info(logs)")).fetchall()
    col_names = {c[1] for c in cols}

    if "final_score" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN final_score TEXT"))
    if "spread_covered" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN spread_covered INTEGER"))
        
    # Snapshot columns for team metrics at time of bet
    metric_cols = [
        "fav_pace",
        "dog_pace",
        "fav_ortg",
        "dog_ortg",
        "fav_drtg",
        "dog_drtg",
        "fav_netr",
        "dog_netr",
    ]
    for mc in metric_cols:
        if mc not in col_names:
            conn.execute(text(f"ALTER TABLE logs ADD COLUMN {mc} REAL"))    
            
    # Cheatsheet + alignment columns
    if "cheat_edge" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN cheat_edge REAL"))
    if "cheat_pick" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN cheat_pick TEXT"))
    if "models_aligned" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN models_aligned INTEGER"))        
    if "hybrid_pick" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN hybrid_pick TEXT"))
        
    # Component columns for learning weights later
    component_cols = [
        "stat_margin",      # proj_fav - proj_dog
        "team_margin",      # fav_team_margin_combined
        "player_adj_eff",   # effective_player_adj
        "pace_adj_term",    # pace_adj
        "rest_adj_term",    # rest_adj
        "b2b_adj_term",     # b2b_adj
        "vegas_margin",     # -vegas_line (fav perspective)
        "hybrid_margin",    # hybrid_margin (fav perspective)
        "effective_edge",   # edge actually used for confidence
    ]
    for cc in component_cols:
        if cc not in col_names:
            conn.execute(text(f"ALTER TABLE logs ADD COLUMN {cc} REAL"))
            
    if "ctg_notes" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN ctg_notes TEXT"))
    if "ctg_reason" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN ctg_reason TEXT"))
    if "ctg_summary" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN ctg_summary TEXT"))

    # B2B flags for each team
    if "fav_is_b2b" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN fav_is_b2b INTEGER"))
    if "dog_is_b2b" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN dog_is_b2b INTEGER"))

    # Injury / outs snapshot (favorite / underdog perspective)
    if "fav_out" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN fav_out TEXT"))
    if "dog_out" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN dog_out TEXT"))

    # Injury / outs snapshot (home / away perspective)
    if "home_out" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN home_out TEXT"))
    if "away_out" not in col_names:
        conn.execute(text("ALTER TABLE logs ADD COLUMN away_out TEXT"))

    # Injury impact snapshots
    injury_cols = {
        "player_adj_raw": "REAL",
        "fav_player_adj_raw": "REAL",
        "dog_player_adj_raw": "REAL",
        "effective_player_adj": "REAL",
        "fav_effective_player_adj": "REAL",
        "dog_effective_player_adj": "REAL",
        "injury_impact_flag": "TEXT",
        "injury_heavy": "INTEGER",
        "fav_injury_impact_flag": "TEXT",
        "dog_injury_impact_flag": "TEXT",
        "fav_injury_heavy": "INTEGER",
        "dog_injury_heavy": "INTEGER",
    }
    for ic, col_type in injury_cols.items():
        if ic not in col_names:
            try:
                conn.execute(text(f"ALTER TABLE logs ADD COLUMN {ic} {col_type}"))
            except Exception:
                pass

# =========================
# Log loading helper
# =========================
@st.cache_data
def load_all_logs() -> pd.DataFrame:
    """Central place to read the logs table.

    Both Prediction Log and Slate View should call this so they stay in sync.
    """
    with engine.connect() as conn:
        return pd.read_sql("SELECT * FROM logs ORDER BY id", conn)


def delete_pick_by_id(pick_id: int) -> None:
    """
    Delete a single logged pick from the logs table by its primary key id.
    """
    if pick_id is None:
        return

    try:
        pid = int(pick_id)
    except (TypeError, ValueError):
        return

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM logs WHERE id = :id"),
            {"id": pid},
        )

            
# =========================
# Data Load
# =========================
players_df = pd.read_csv("players_onoff.csv")
players_df["Team"] = players_df["Team"].str.upper()

# Parse DaysSinceLastGame as numeric if present
if "DaysSinceLastGame" in players_df.columns:
    players_df["DaysSinceLastGame"] = pd.to_numeric(
        players_df["DaysSinceLastGame"], errors="coerce"
    )

# Team ratings file (from your path: Team_ratings.csv)
try:
    team_ratings_df = pd.read_csv("Team_ratings.csv")
    team_ratings_df["Team"] = team_ratings_df["Team"].str.upper()
    TEAM_RATINGS_AVAILABLE = True
except Exception:
    TEAM_RATINGS_AVAILABLE = False

# Precompute league-average pace if available
if TEAM_RATINGS_AVAILABLE:
    LEAGUE_AVG_PACE = None
    for _col in ["Pace", "PACE"]:
        if _col in team_ratings_df.columns:
            try:
                LEAGUE_AVG_PACE = float(team_ratings_df[_col].mean(skipna=True))
            except Exception:
                LEAGUE_AVG_PACE = None
            break
else:
    LEAGUE_AVG_PACE = None


@st.cache_data(ttl=600)
def load_bref_injury_lists() -> dict:
    """
    Cached wrapper around the Basketball-Reference injury scraper.

    Returns:
      dict like {'GSW': {'out': [...], 'q': [...], 'other': [...]}, ...}
    """
    if not HAS_BREF_INJURIES or get_team_injury_lists is None:
        return {}
    try:
        return get_team_injury_lists()
    except Exception:
        # Fail quietly; app will just not show auto injuries
        return {}


def get_team_power(team_abbr: str, is_home: bool) -> float | None:
    """
    Uses Team_ratings.csv
    Expected columns: Team,Base Net Rating,Home Adjust,Away Adjust,Adjusted Rating,Weight
    """
    if not TEAM_RATINGS_AVAILABLE:
        return None

    row = team_ratings_df[team_ratings_df["Team"] == team_abbr.upper()]
    if row.empty:
        return None

    base = float(row["Base Net Rating"].iloc[0])
    home_adj = float(row["Home Adjust"].iloc[0])
    away_adj = float(row["Away Adjust"].iloc[0])
    return base + (home_adj if is_home else away_adj)


def get_team_metric(team_abbr: str, col_candidates: list[str]):
    """
    Safely fetch a numeric metric (e.g. Pace, ORtg, DRtg, NetRtg) for a team
    from team_ratings_df, trying multiple possible column names.
    Returns float or None.
    """
    if not TEAM_RATINGS_AVAILABLE:
        return None

    row = team_ratings_df[team_ratings_df["Team"] == team_abbr.upper()]
    if row.empty:
        return None

    for col in col_candidates:
        if col in row.columns:
            try:
                return float(row[col].iloc[0])
            except Exception:
                return None
    return None


# =========================
# Helper Functions
# =========================
def compute_player_injury_terms(out_players, players_df, injury_cap=6.0):
    """
    Compute raw and effective injury adjustment plus flags.

    Parameters
    ----------
    out_players : list[str]
        List of player names selected as OUT for this team.
    players_df : pd.DataFrame
        DataFrame with at least: ['Player', 'Diff', 'DaysSinceLastGame'].
    injury_cap : float
        Hard cap (in points) for the absolute effective injury adjustment.

    Returns
    -------
    player_adj_raw : float
        Sum of recency-weighted on/off Diffs for all OUT players.
    player_adj_eff_capped : float
        Effective injury adjustment after existing transform (if any) and hard cap.
        NOTE: the transformation from raw → effective is done outside this function;
              this helper only handles recency weighting and capping.
    injury_impact_flag : str
        One of {"low", "medium", "high"} based on |player_adj_eff_capped|.
    injury_heavy : int
        1 if |player_adj_eff_capped| >= 4.0, else 0.
    """
    if not out_players:
        return 0.0, 0.0, "low", 0

    # Make sure we have the columns we need
    cols = players_df.columns
    if "Player" not in cols or "Diff" not in cols:
        return 0.0, 0.0, "low", 0

    # DaysSinceLastGame is optional but preferred
    has_days = "DaysSinceLastGame" in cols

    player_adj_raw = 0.0

    for name in out_players:
        row = players_df.loc[players_df["Player"] == name]
        if row.empty:
            continue

        diff = row["Diff"].iloc[0]
        try:
            diff = float(diff)
        except (TypeError, ValueError):
            diff = 0.0

        days = None
        if has_days:
            days = row["DaysSinceLastGame"].iloc[0]

        # Recency-based weight
        if days is None or (isinstance(days, float) and math.isnan(days)):
            w_recency = 1.0
        else:
            try:
                d = float(days)
            except (TypeError, ValueError):
                d = 0.0

            if d <= 3:
                w_recency = 1.0      # fresh absence
            elif d <= 7:
                w_recency = 0.7      # ~1–3 games missed
            elif d <= 14:
                w_recency = 0.4      # ~3–5+ games missed
            else:
                w_recency = 0.0      # long-term out → already baked into line/ratings

        player_adj_raw += diff * w_recency

    # NOTE: DO NOT apply transform here.
    # The caller should:
    #   1) convert player_adj_raw → effective_player_adj (existing logic)
    #   2) then apply the cap and flags using the helper below.
    #
    # So we only return the raw here; the caller will pass the effective value
    # back into another helper that applies the cap & flags.
    return player_adj_raw, 0.0, "low", 0


def cap_and_flag_injury_effect(effective_player_adj, injury_cap=6.0):
    """
    Apply a hard cap to the effective injury adjustment and
    compute injury_impact_flag and injury_heavy.

    Returns
    -------
    player_adj_eff_capped : float
    injury_impact_flag : str  # "low" / "medium" / "high"
    injury_heavy : int        # 1 if high, else 0
    """
    # Hard cap
    if effective_player_adj is None:
        effective_player_adj = 0.0

    try:
        eff = float(effective_player_adj)
    except (TypeError, ValueError):
        eff = 0.0

    player_adj_eff_capped = max(min(eff, injury_cap), -injury_cap)
    abs_injury = abs(player_adj_eff_capped)

    if abs_injury < 2.0:
        injury_impact_flag = "low"
        injury_heavy = 0
    elif abs_injury < 4.0:
        injury_impact_flag = "medium"
        injury_heavy = 0
    else:
        injury_impact_flag = "high"
        injury_heavy = 1

    return player_adj_eff_capped, injury_impact_flag, injury_heavy


def classify_edge(val: float) -> str:
    if val > STRONG_EDGE:
        return "STRONG"
    elif val > MEDIUM_EDGE:
        return "MEDIUM"
    return "PASS"


def confidence_to_color(conf: str) -> str:
    if conf == "STRONG":
        return "green"
    elif conf == "MEDIUM":
        return "orange"
    return "red"


def side_to_pick(side: str, fav: str, dog: str, line: float) -> str:
    if side == "favorite":
        return f"{fav} {line:.1f}"
    else:
        return f"{dog} +{abs(line):.1f}"


def full_to_abbr(name: str) -> str:
    key = name.strip().upper()
    return TEAM_ABBRS.get(key, key[:3])
    
def load_cheatsheet_from_scoreboard(file):
    """
    Parse your SCOREBOARD tab in the cheatsheet Excel file.

    Assumptions (matching your sheet):
    - Column B = team names (away row, then home row)
    - Column AC = projected scores
    - Games are in away/home pairs, but there may be blank spacer rows.
    """
    try:
        # Read the SCOREBOARD sheet
        raw = pd.read_excel(file, sheet_name="SCOREBOARD", header=0)
    except Exception as e:
        st.error(f"Error reading cheatsheet: {e}")
        return None

    # Column B is index 1, AC is index 28  (A=0, B=1, ..., AC=28)
    team_col = raw.columns[1]
    proj_col = raw.columns[28]

    rows = []
    i = 0
    n = len(raw)

    while i + 1 < n:
        team1 = raw.iloc[i][team_col]
        team2 = raw.iloc[i + 1][team_col]
        proj1 = raw.iloc[i][proj_col]
        proj2 = raw.iloc[i + 1][proj_col]

        # If this pair isn't a clean away/home game, skip down 1 row and keep scanning
        if pd.isna(team1) or pd.isna(team2) or pd.isna(proj1) or pd.isna(proj2):
            i += 1
            continue

        away_full = str(team1).strip()
        home_full = str(team2).strip()

        away_abbr = full_to_abbr(away_full)
        home_abbr = full_to_abbr(home_full)

        try:
            proj_away = float(proj1)
            proj_home = float(proj2)
        except Exception:
            i += 1
            continue

        # Decide favorite / dog from projections
        if proj_home > proj_away:
            fav_abbr, dog_abbr = home_abbr, away_abbr
        elif proj_away > proj_home:
            fav_abbr, dog_abbr = away_abbr, home_abbr
        else:
            fav_abbr, dog_abbr = home_abbr, away_abbr

        pair_key = "|".join(sorted([away_abbr, home_abbr]))

        rows.append(
            {
                "away_abbr": away_abbr,
                "home_abbr": home_abbr,
                "proj_away": proj_away,
                "proj_home": proj_home,
                "fav_abbr": fav_abbr,
                "dog_abbr": dog_abbr,
                "pair_key": pair_key,
            }
        )

        i += 2

    if not rows:
        st.warning("Cheatsheet parsed 0 games from SCOREBOARD.")
        return None

    df = pd.DataFrame(rows)
    st.caption(f"Cheatsheet parsed {len(df)} games from SCOREBOARD.")
    return df



def lookup_cheatsheet_projection_for_game(
    cheatsheet_df, favorite_abbr, underdog_abbr, home_abbr, away_abbr
):
    if cheatsheet_df is None:
        return None, None, None

    fav = favorite_abbr.upper()
    dog = underdog_abbr.upper()
    home = home_abbr.upper()
    away = away_abbr.upper()

    # Normalized pair key for this game
    pair_key = "|".join(sorted([home, away]))

    # 1) match by pair_key only (ignore fav/home-away quirks)
    mask = cheatsheet_df["pair_key"].str.upper() == pair_key
    if not mask.any():
        return None, None, None

    row = cheatsheet_df[mask].iloc[0]

    proj_home = row["proj_home"]
    proj_away = row["proj_away"]

    # Return projections in fav/dog order
    if fav == home:
        proj_fav, proj_dog = proj_home, proj_away
    else:
        proj_fav, proj_dog = proj_away, proj_home

    return proj_fav, proj_dog

def extract_text_from_pdf(file) -> str:
    """Extract raw text from an uploaded PDF file."""
    if file is None:
        return ""
    try:
        text_chunks = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                text_chunks.append(t)
        return "\n\n".join(text_chunks)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def analyze_game_with_ctg(game_row, ctg_text: str) -> tuple[str, str]:
    """
    Use OpenAI to classify why a pick covered / didn't cover,
    based on your model info + CTG postgame writeup.
    Returns (reason_label, summary_text).
    """
    # If client isn't set up, bail out cleanly

    if not ctg_text.strip():
        return "", "No CTG text available."

    fav = str(game_row["favorite"])
    dog = str(game_row["underdog"])
    pick = str(game_row.get("pick", ""))
    edge = game_row.get("edge", None)
    conf = str(game_row.get("confidence", ""))
    final_score = str(game_row.get("final_score", ""))
    cov = game_row.get("spread_covered", None)

    outcome_text = "unknown"
    if cov == 1:
        outcome_text = "your pick covered the spread"
    elif cov == 0:
        outcome_text = "your pick did NOT cover the spread"

    system_prompt = (
        "You are helping analyze NBA bets using post-game reports.\n"
        "Given a model's pre-game pick, edge, and outcome, plus a detailed "
        "post-game report, decide whether the model's logic was vindicated "
        "or not, and if not, what the main reason was.\n\n"
        "Choose ONE primary reason label from this list:\n"
        "- shooting_variance\n"
        "- defense_matchup\n"
        "- turnovers\n"
        "- rebounding\n"
        "- fouls_free_throws\n"
        "- injuries_rotations\n"
        "- garbage_time\n"
        "- model_miss\n"
        "- other\n\n"
        "Be conservative: only call it model_miss if the report suggests "
        "the underlying assumptions were wrong, not just variance."
    )

    # Guard against edge being None so we don't crash on formatting
    edge_str = f"{edge:.2f}" if edge is not None else "N/A"

    user_prompt = f"""
Game: {fav} vs {dog}
Pick: {pick}
Model edge: {edge_str} (confidence: {conf})
Outcome: {outcome_text}
Final score: {final_score}

Post-game report text (CTG):
\"\"\"
{ctg_text[:8000]}
\"\"\"

1. Did the game fundamentally play out in line with the model's logic (yes/no)?
2. If the pick missed, what is the MAIN reason, using ONE label from the list?
3. Provide a brief 2-3 sentence explanation referencing the report.
Return a JSON object with keys: "reason", "summary".
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    import json
    try:
        data = json.loads(resp.choices[0].message.content)
        reason = (data.get("reason") or "").strip()
        summary = (data.get("summary") or "").strip()
    except Exception:
        reason = ""
        summary = resp.choices[0].message.content.strip()

    return reason, summary
    
def get_team_recent_games(team_abbr: str, scores: list, game_dt: datetime.datetime):
    """
    From the scores list, return this team's games before `game_dt`,
    sorted most-recent first.
    """
    team_abbr = team_abbr.upper()
    recent = []
    for g in scores:
        dt = g.get("commence_dt")
        if not dt or dt >= game_dt:
            continue
        if team_abbr in (g["home_abbr"], g["away_abbr"]):
            recent.append(g)
    recent.sort(key=lambda gg: gg["commence_dt"], reverse=True)
    return recent


def compute_rest_profile(
    team_abbr: str,
    is_home_now: bool,
    game_dt: datetime.datetime,
    scores: list,
) -> dict:
    """
    Build a schedule/rest profile for one team for this game:

    - days_since_last
    - games_last_5 (last 5 days)
    - is_b2b
    - b2b_penalty  (points, negative = more gassed)
    - travel_penalty (small non-B2B travel tax)
    - desc (for UI)
    """
    recent = get_team_recent_games(team_abbr, scores, game_dt)
    today = game_dt.date()

    last_game_dt = None
    last_is_home = None
    games_last_5 = 0

    for g in recent:
        dt = g.get("commence_dt")
        if not dt:
            continue

        delta_days = (today - dt.date()).days

        # games in last 5 days
        if 1 <= delta_days <= 5:
            games_last_5 += 1

        # first (most recent) game
        if last_game_dt is None:
            last_game_dt = dt
            last_is_home = (g["home_abbr"] == team_abbr.upper())

    days_since = None
    is_b2b = False
    travel_pattern = None
    b2b_penalty = 0.0
    travel_penalty = 0.0

    if last_game_dt is not None:
        days_since = (today - last_game_dt.date()).days
        prev_is_home = bool(last_is_home)
        cur_is_home = bool(is_home_now)

        if days_since == 1:
            # Back-to-back classification
            if prev_is_home and cur_is_home:
                travel_pattern = "H→H"
                b2b_penalty = -1.0
            elif prev_is_home and not cur_is_home:
                travel_pattern = "H→A"
                b2b_penalty = -1.3
            elif (not prev_is_home) and cur_is_home:
                travel_pattern = "A→H"
                b2b_penalty = -1.3
            else:
                travel_pattern = "A→A"
                b2b_penalty = -1.6
            is_b2b = True
        else:
            # Non-B2B travel tax: small nudge only
            if prev_is_home != cur_is_home:
                travel_pattern = "travel"
                travel_penalty = -0.2
            else:
                travel_pattern = "no-travel"

    # Text summary
    desc_parts = []
    if days_since is not None:
        desc_parts.append(f"{days_since} days rest")
    if games_last_5:
        desc_parts.append(f"{games_last_5} games last 5d")
    if is_b2b:
        desc_parts.append(f"B2B {travel_pattern}")
    elif travel_pattern and travel_pattern != "no-travel":
        desc_parts.append(travel_pattern)

    return {
        "days_since": days_since,
        "games_last_5": games_last_5,
        "is_b2b": is_b2b,
        "b2b_penalty": float(b2b_penalty),
        "travel_penalty": float(travel_penalty),
        "desc": ", ".join(desc_parts) if desc_parts else "None",
    }

# =========================
# Odds & Scores Fetch
# =========================
@st.cache_data(ttl=300)
def fetch_games_with_odds():
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": MARKETS,
        "oddsFormat": "american",
    }
    try:
        r = requests.get(ODDS_API_URL, params=params)
        r.raise_for_status()
        data = r.json()

        games = []
        now_et = datetime.datetime.now(ET_TZ)

        for ev in data:
            home_full = ev["home_team"]
            away_full = ev["away_team"]

            home_abbr = full_to_abbr(home_full)
            away_abbr = full_to_abbr(away_full)

            commence_str = ev.get("commence_time")
            commence_dt = None
            if commence_str:
                try:
                    commence_dt = datetime.datetime.fromisoformat(
                        commence_str.replace("Z", "+00:00")
                    ).astimezone(ET_TZ)
                except Exception:
                    commence_dt = None

            completed = ev.get("completed", False)

            if completed:
                status = "FINAL"
            elif commence_dt and now_et >= commence_dt:
                status = "LIVE"
            else:
                status = "UPCOMING"

            if not ev.get("bookmakers"):
                continue
            market = ev["bookmakers"][0]["markets"][0]
            outcomes = market["outcomes"]

            fav_name = None
            dog_name = None
            fav_spread = 0.0

            for o in outcomes:
                pt = o.get("point")
                if pt is None:
                    continue
                if pt < 0:
                    fav_name = o["name"]
                    fav_spread = float(pt)
                else:
                    dog_name = o["name"]

            if not fav_name or not dog_name:
                continue

            fav_abbr = full_to_abbr(fav_name)
            dog_abbr = full_to_abbr(dog_name)

            home_score = None
            away_score = None
            scores = ev.get("scores")
            if scores and isinstance(scores, list):
                try:
                    for s in scores:
                        if s.get("name") == home_full:
                            home_score = s.get("score")
                        elif s.get("name") == away_full:
                            away_score = s.get("score")
                except Exception:
                    pass

            if commence_dt:
                time_label = commence_dt.strftime("%I:%M %p").lstrip("0") + " ET"
            else:
                time_label = "TBD"

            if status == "LIVE":
                display = (
                    f"🔥 LIVE — {away_full} @ {home_full} | {fav_abbr} {fav_spread:.1f}"
                )
            elif status == "FINAL":
                if home_score is not None and away_score is not None:
                    display = (
                        f"✅ FINAL — {away_full} {away_score} @ "
                        f"{home_full} {home_score} | {fav_abbr} {fav_spread:.1f}"
                    )
                else:
                    display = (
                        f"✅ FINAL — {away_full} @ {home_full} | "
                        f"{fav_abbr} {fav_spread:.1f}"
                    )
            else:
                display = (
                    f"{time_label} — {away_full} @ {home_full} | "
                    f"{fav_abbr} {fav_spread:.1f}"
                )

            games.append(
                {
                    "home_full": home_full,
                    "away_full": away_full,
                    "home_abbr": home_abbr,
                    "away_abbr": away_abbr,
                    "favorite_full": fav_name,
                    "underdog_full": dog_name,
                    "favorite_abbr": fav_abbr,
                    "underdog_abbr": dog_abbr,
                    "spread": fav_spread,
                    "status": status,
                    "commence_dt": commence_dt,
                    "display": display,
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )

        status_rank = {"LIVE": 0, "UPCOMING": 1, "FINAL": 2}
        games.sort(
            key=lambda g: (
                status_rank.get(g["status"], 3),
                g["commence_dt"] or datetime.datetime.max.replace(tzinfo=ET_TZ),
            )
        )

        return games

    except Exception as e:
        st.error(f"Odds API error: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_scores(days_from: int = 3):
    """
    Use The Odds API scores endpoint to get final scores
    from the last `days_from` days.

    The Odds API only allows daysFrom up to 3, so we clamp here.
    """
    # Safety clamp so we never send an invalid value
    days_from = max(1, min(int(days_from), 3))

    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": days_from,
    }
    try:
        r = requests.get(SCORES_API_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.HTTPError as e:
        # Show both the message and server response (helps debugging)
        try:
            msg = r.text
        except Exception:
            msg = str(e)
        st.error(f"Scores API error: {e}\n\nResponse: {msg}")
        return []
    except Exception as e:
        st.error(f"Scores API error: {e}")
        return []

    results = []
    for ev in data:
        home_full = ev["home_team"]
        away_full = ev["away_team"]
        home_abbr = full_to_abbr(home_full)
        away_abbr = full_to_abbr(away_full)

        commence_str = ev.get("commence_time")
        commence_dt = None
        if commence_str:
            try:
                commence_dt = datetime.datetime.fromisoformat(
                    commence_str.replace("Z", "+00:00")
                ).astimezone(ET_TZ)
            except Exception:
                pass

        completed = ev.get("completed", False)

        home_score = None
        away_score = None
        scores = ev.get("scores")
        if scores and isinstance(scores, list):
            try:
                for s in scores:
                    if s.get("name") == home_full:
                        home_score = int(s.get("score"))
                    elif s.get("name") == away_full:
                        away_score = int(s.get("score"))
            except Exception:
                pass

        results.append(
            {
                "home_full": home_full,
                "away_full": away_full,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "commence_dt": commence_dt,
                "completed": completed,
                "home_score": home_score,
                "away_score": away_score,
            }
        )

    return results



# =========================
# Score/Result Updater
# =========================
def update_final_scores_from_odds(days_from: int = 3):
    """
    Fill in final_score + spread_covered for logged games using The Odds API scores.

    More forgiving matching:
    - Look back `days_from` days (default 5)
    - Match by team pair first, then choose the event with date closest to log date
    - Don't require `completed=True` as long as scores are present
    - Print debug info to terminal for each row (matched / not matched)
    """
    games = fetch_scores(days_from=days_from)
    if not games:
        st.warning("No scores data available to refresh.")
        return

    # Build index: pair_key -> list of events
    # pair_key is frozenset of team abbrs, ignoring home/away
    pair_to_events: dict[frozenset[str], list[dict]] = {}
    for g in games:
        pair_key = frozenset({g["home_abbr"], g["away_abbr"]})
        pair_to_events.setdefault(pair_key, []).append(g)

    # For convenience, sort each event list by commence_dt (if available)
    for evts in pair_to_events.values():
        evts.sort(
            key=lambda e: e["commence_dt"]
            or datetime.datetime.max.replace(tzinfo=ET_TZ)
        )

    updated_count = 0
    skipped_no_match = 0
    skipped_no_scores = 0

    with engine.begin() as conn:
        logs_df = pd.read_sql("SELECT * FROM logs", conn)

        for _, row in logs_df.iterrows():
            try:
                row_date = datetime.date.fromisoformat(str(row["date"]))
            except Exception:
                print(f"[refresh] Row id={row['id']} has invalid date '{row['date']}', skipping.")
                continue

            fav = str(row["favorite"]).upper()
            dog = str(row["underdog"]).upper()
            vegas_line = float(row["vegas_line"])
            pick = str(row["pick"])

            pair_key = frozenset({fav, dog})
            events = pair_to_events.get(pair_key, [])

            if not events:
                print(f"[refresh] No score events for {fav} vs {dog}, date={row_date}, id={row['id']}")
                skipped_no_match += 1
                continue

            # Pick event whose game_date is closest to row_date
            best_event = None
            best_delta = None
            for ev in events:
                if not ev["commence_dt"]:
                    continue
                game_date = ev["commence_dt"].date()
                delta = abs((game_date - row_date).days)
                if (best_delta is None) or (delta < best_delta):
                    best_delta = delta
                    best_event = ev

            if best_event is None:
                print(f"[refresh] Events found for pair {pair_key}, but no commence_dt, id={row['id']}")
                skipped_no_match += 1
                continue

            home_abbr = best_event["home_abbr"]
            away_abbr = best_event["away_abbr"]
            home_score = best_event["home_score"]
            away_score = best_event["away_score"]

            if home_score is None or away_score is None:
                print(
                    f"[refresh] Event for {fav} vs {dog} has no scores yet "
                    f"(home_score={home_score}, away_score={away_score}), id={row['id']}"
                )
                skipped_no_scores += 1
                continue

            # Map fav/dog to scores
            if fav == home_abbr:
                fav_score = int(home_score)
                dog_score = int(away_score)
            elif fav == away_abbr:
                fav_score = int(away_score)
                dog_score = int(home_score)
            else:
                # Teams don't line up the way we expect
                print(
                    f"[refresh] Team abbr mismatch for row id={row['id']}: "
                    f"fav={fav}, dog={dog}, home={home_abbr}, away={away_abbr}"
                )
                skipped_no_match += 1
                continue

            margin_fav = fav_score - dog_score
            line_abs = abs(vegas_line)

            # Determine which team was in the pick string
            parts = pick.split()
            if len(parts) >= 2:
                pick_team = parts[0].upper()
            else:
                pick_team = fav

            bet_on_favorite = (pick_team == fav)

            line_abs = abs(vegas_line)
            eps = 1e-6  # tolerance for float comparisons

            # margin_fav = fav_score - dog_score (already computed above)
            # For a fav bet: compare margin_fav vs line_abs
            # For a dog bet: compare line_abs vs margin_fav (dog wants margin_fav smaller)
            if bet_on_favorite:
                diff = margin_fav - line_abs
            else:
                diff = line_abs - margin_fav

            if abs(diff) < eps:
                covered = 2   # ➖ push
            elif diff > 0:
                covered = 1   # ✅ covered
            else:
                covered = 0   # ❌ missed

            final_score_str = f"{fav} {fav_score} - {dog} {dog_score}"

            conn.execute(
                text(
                    """
                    UPDATE logs
                    SET final_score = :fs,
                        spread_covered = :cov
                    WHERE id = :id
                    """
                ),
                {"fs": final_score_str, "cov": covered, "id": int(row["id"])},
            )
            load_all_logs.clear()
            updated_count += 1
            print(
                f"[refresh] Updated id={row['id']}: {final_score_str}, "
                f"margin_fav={margin_fav}, pick='{pick}', covered={covered}"
            )

    st.success(
        f"Refreshed scores via Odds API. "
        f"Updated {updated_count} rows, "
        f"{skipped_no_match} had no matching game, "
        f"{skipped_no_scores} had no scores yet."
    )



# =========================
# UI Layout
# =========================
st.set_page_config(page_title="NBA Edge Analyzer", layout="wide")
st.title("NBA Spread Edge Analyzer")

st.markdown(
    "<div style='text-align:right;font-size:12px;'>Made by <b>Calvin Thuong</b></div>",
    unsafe_allow_html=True,
)

tab_single, tab_full, tab_perf, tab_ctg, tab_logs = st.tabs(
    ["Single Game", "Full Slate", "Performance", "CTG Review", "Prediction Log"]
)

# ============================================
# SINGLE GAME ANALYZER
# ============================================
with tab_single:
    st.subheader("Game Selection (with Live Odds, ET)")

    games_data = fetch_games_with_odds()

    manual_mode = False
    if games_data:
        now_et = datetime.datetime.now(ET_TZ)
        today_et = now_et.date()
        # Only keep games whose scheduled start is today in ET
        games_data = [
            g for g in games_data
            if g.get("commence_dt") and g["commence_dt"].date() == today_et
        ]

    if not games_data:
        st.warning("⚠ No games found for today from Odds API — switching to manual mode.")
        manual_mode = True
        games_display = ["Manual Entry"]
    else:
        games_display = [g["display"] for g in games_data]

    selected_game = st.selectbox("Select Game", games_display)

    if manual_mode:
        home_full = st.text_input("Home Team (full name)", "Los Angeles Lakers")
        away_full = st.text_input("Away Team (full name)", "Boston Celtics")
        home_abbr = full_to_abbr(home_full)
        away_abbr = full_to_abbr(away_full)
        favorite_abbr = st.text_input("Favorite (abbr)", home_abbr)
        underdog_abbr = st.text_input("Underdog (abbr)", away_abbr)
        vegas_line = st.number_input(
            "Vegas Spread (favorite only)", -30.0, 30.0, -5.5, 0.5
        )
        game_status = "MANUAL"
    else:
        game = next(g for g in games_data if g["display"] == selected_game)
        home_full = game["home_full"]
        away_full = game["away_full"]
        home_abbr = game["home_abbr"]
        away_abbr = game["away_abbr"]
        favorite_abbr = game["favorite_abbr"]
        underdog_abbr = game["underdog_abbr"]
        vegas_line = game["spread"]
        game_status = game["status"]

        st.caption(
            f"Status: **{game_status}** — {away_full} @ {home_full}, "
            f"line: {favorite_abbr} {vegas_line:.1f}"
        )

        # Auto Schedule / B2B Detection (status-adjacent, no header)
        if game_status != "MANUAL" and game.get("commence_dt"):
            schedule_scores = fetch_scores(days_from=7)
            home_rest = compute_rest_profile(
                home_abbr,
                is_home_now=True,
                game_dt=game["commence_dt"],
                scores=schedule_scores,
            )
            away_rest = compute_rest_profile(
                away_abbr,
                is_home_now=False,
                game_dt=game["commence_dt"],
                scores=schedule_scores,
            )

            st.caption(
                f"Schedule — {home_abbr}: {home_rest['desc']} | "
                f"{away_abbr}: {away_rest['desc']}"
            )

            st.session_state["home_rest_profile"] = home_rest
            st.session_state["away_rest_profile"] = away_rest
        else:
            # Manual / fallback: no schedule data
            home_rest = {
                "days_since": None,
                "games_last_5": 0,
                "is_b2b": False,
                "b2b_penalty": 0.0,
                "travel_penalty": 0.0,
                "desc": "N/A",
            }
            away_rest = home_rest.copy()
            st.session_state["home_rest_profile"] = home_rest
            st.session_state["away_rest_profile"] = away_rest
            st.caption("Schedule info unavailable (manual mode).")

    # =========================
    # Cheatsheet Import (Optional)
    # =========================
    with st.expander("Cheatsheet Import (Optional)", expanded=False):
        uploaded_cheatsheet = st.file_uploader(
            "Upload cheatsheet Excel (SCOREBOARD tab)",
            type=["xlsx", "xls"],
            key="cheatsheet_upload",
        )

        # Load / persist cheatsheet_df in session_state
        if uploaded_cheatsheet is not None:
            cheatsheet_df = load_cheatsheet_from_scoreboard(uploaded_cheatsheet)
            st.session_state["cheatsheet_df"] = cheatsheet_df
        else:
            cheatsheet_df = st.session_state.get("cheatsheet_df")

        if cheatsheet_df is not None:
            st.caption(
                f"Cheatsheet loaded from SCOREBOARD tab for {len(cheatsheet_df)} games "
                "(away/home + projections)."
            )
            st.dataframe(
                cheatsheet_df[
                    [
                        "away_abbr",
                        "home_abbr",
                        "proj_away",
                        "proj_home",
                        "fav_abbr",
                        "dog_abbr",
                        "pair_key",
                    ]
                ],
                use_container_width=True,
            )
            
            # 🔁 NEW: toggle to auto-apply projections
            st.checkbox(
                "Auto-apply cheatsheet projections when available",
                value=st.session_state.get("auto_cheat_apply", False),
                key="auto_cheat_apply",
            )
        else:
            st.caption("No cheatsheet loaded yet.")
    

    # Spread + Score Inputs are now taken directly from the game / manual selection
    favorite = favorite_abbr
    underdog = underdog_abbr
    # vegas_line is already defined earlier from odds or manual input

    # =========================
    # Cheatsheet hook for THIS game + Stat model inputs
    # =========================
    cheatsheet_df = st.session_state.get("cheatsheet_df")
    cheat_pf, cheat_pd = None, None

    if cheatsheet_df is not None:
        try:
            cheat_pf, cheat_pd = lookup_cheatsheet_projection_for_game(
                cheatsheet_df=cheatsheet_df,
                favorite_abbr=favorite,
                underdog_abbr=underdog,
                home_abbr=home_abbr,
                away_abbr=away_abbr,
            )
        except Exception as e:
            st.caption(f"Cheatsheet lookup error: {e}")

    auto_apply = st.session_state.get("auto_cheat_apply", False)

    st.subheader("Projected Scores (Stat Model)")

    # Initialize state once
    if "proj_fav_input" not in st.session_state:
        st.session_state["proj_fav_input"] = 110.0
    if "proj_dog_input" not in st.session_state:
        st.session_state["proj_dog_input"] = 104.0

    # If cheatsheet projections exist, show them and optionally auto-apply
    if (cheat_pf is not None) and (cheat_pd is not None):
        st.info(
            f"Cheatsheet projections found: "
            f"{favorite.upper()} {cheat_pf:.1f} — "
            f"{underdog.upper()} {cheat_pd:.1f}"
        )

        # Unique key for this specific matchup so we don't keep overwriting
        current_game_key = f"{favorite.upper()}_{underdog.upper()}_{home_abbr}_{away_abbr}"
        last_key = st.session_state.get("last_cheat_apply_key")

        # 🔁 Auto-apply once per game when toggle is on
        if auto_apply and current_game_key != last_key:
            st.session_state["proj_fav_input"] = float(cheat_pf)
            st.session_state["proj_dog_input"] = float(cheat_pd)
            st.session_state["last_cheat_apply_key"] = current_game_key
            st.rerun()

        # Manual override button still available
        if st.button("Use cheatsheet projections", key="use_cheatsheet_proj"):
            st.session_state["proj_fav_input"] = float(cheat_pf)
            st.session_state["proj_dog_input"] = float(cheat_pd)
            st.session_state["last_cheat_apply_key"] = current_game_key
            st.rerun()
    else:
        st.caption("No matching cheatsheet row for this game.")

    # Now render the inputs, which read from session_state
    proj_fav = st.number_input(
        f"Projected Score ({favorite.upper()})",
        step=1.0,
        key="proj_fav_input",
    )
    proj_dog = st.number_input(
        f"Projected Score ({underdog.upper()})",
        step=1.0,
        key="proj_dog_input",
    )

    def get_projected_lineup(team_abbr: str, max_players: int = 12, max_days: int = 30):
        """
        Heuristic projected lineup for a team.

        - Filters to players on this team in players_df.
        - Keeps only reasonably recent players (DaysSinceLastGame <= max_days or NaN).
        - Tries to identify rotation players using a minutes column if available.
        - Sorts by:
            1) Rotation flag (True first)
            2) Minutes (descending) if available
            3) Most recent game (DaysSinceLastGame ascending)
            4) Impact (abs(Diff) descending) if Diff exists.
        - Returns:
            starters: first 5 players
            bench: next players up to max_players total
        """
        team_df = players_df[players_df["Team"] == team_abbr].copy()

        # Filter to somewhat recent players
        if "DaysSinceLastGame" in team_df.columns:
            recent_mask = (
                team_df["DaysSinceLastGame"].isna()
                | (team_df["DaysSinceLastGame"] <= max_days)
            )
            team_df = team_df[recent_mask]

        if team_df.empty:
            return [], []

        # Try to find a minutes column
        minutes_col = None
        for col in ["MP", "Minutes", "MIN", "MinPerGame", "MinutesPerGame"]:
            if col in team_df.columns:
                minutes_col = col
                break

        # Rotation flag: guys who play real minutes
        if minutes_col:
            team_df["rotation_flag"] = team_df[minutes_col].fillna(0) >= 10  # 10+ mins
        else:
            team_df["rotation_flag"] = True

        # Impact magnitude for tie-breaking
        if "Diff" in team_df.columns:
            team_df["abs_diff_for_sort"] = team_df["Diff"].abs()
        else:
            team_df["abs_diff_for_sort"] = 0.0

        sort_cols = ["rotation_flag"]
        ascending = [False]  # rotation players first

        if minutes_col:
            sort_cols.append(minutes_col)
            ascending.append(False)  # more minutes first

        if "DaysSinceLastGame" in team_df.columns:
            sort_cols.append("DaysSinceLastGame")
            ascending.append(True)  # more recent first

        sort_cols.append("abs_diff_for_sort")
        ascending.append(False)  # bigger impact first

        team_df = team_df.sort_values(sort_cols, ascending=ascending)

        players = team_df["Player"].dropna().tolist()
        starters = players[:5]
        bench = players[5:max_players]

        return starters, bench

    def render_projected_lineup(team_abbr: str, inj_info: dict, label: str):
        """
        Show a simple projected lineup card for one team.

        inj_info is something like:
            {'out': [...], 'q': [...], 'other': [...]}
        """
        # Get heuristic starters/bench from players_df
        starters, bench = get_projected_lineup(team_abbr)

        out_set = set(inj_info.get("out") or [])
        q_set = set(inj_info.get("q") or [])

        # Remove confirmed OUT players from the projected rotation
        starters = [p for p in starters if p not in out_set]
        bench = [p for p in bench if p not in out_set]

        def status_tag(name: str) -> str:
            if name in out_set:
                return "OUT"
            if name in q_set:
                return "Q"
            return ""

        st.markdown(f"**{label} {team_abbr} – Projected lineup (v1)**")

        if not starters and not bench:
            st.caption("No recent players found for this team.")
            return

        st.caption("Starters (heuristic):")
        for name in starters:
            tag = status_tag(name)
            if tag:
                st.markdown(f"- {name}  · **{tag}**")
            else:
                st.markdown(f"- {name}")

        if bench:
            st.caption("Bench / others:")
            for name in bench:
                tag = status_tag(name)
                if tag:
                    st.markdown(f"- {name}  · **{tag}**")
                else:
                    st.markdown(f"- {name}")

    st.subheader("Player Availability (On/Off Impact)")

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        hide_long_out = st.checkbox("Hide long-term out players", value=True)
    with col_filter2:
        max_days_out = st.number_input(
            "Max days since last game to show",
            min_value=1,
            max_value=90,
            value=21,
            step=1,
        )

    def get_player_pool(team_abbr: str):
        team_df = players_df[players_df["Team"] == team_abbr]
        long_out_names = []

        if hide_long_out and "DaysSinceLastGame" in team_df.columns:
            mask_long = (
                team_df["DaysSinceLastGame"].notna()
                & (team_df["DaysSinceLastGame"] > max_days_out)
            )
            long_out_names = (
                team_df.loc[mask_long, "Player"].dropna().unique().tolist()
            )
            team_df = team_df.loc[~mask_long]

        return team_df["Player"].dropna().unique().tolist(), long_out_names

    # Base player pools from your on/off data
    home_players, home_long_out = get_player_pool(home_abbr)
    away_players, away_long_out = get_player_pool(away_abbr)

    # Optional: auto injuries from Basketball-Reference
    bref_team_lists = load_bref_injury_lists()
    home_inj = bref_team_lists.get(home_abbr, {"out": [], "q": [], "other": []})
    away_inj = bref_team_lists.get(away_abbr, {"out": [], "q": [], "other": []})

    # Let user see + optionally apply BRef "OUT" lists
    apply_auto = False
    st.write("BRef injury lists loaded:", bool(bref_team_lists))
    if bref_team_lists:
        with st.expander("Auto injuries from Basketball-Reference", expanded=False):
            col_hi, col_ai = st.columns(2)

            with col_hi:
                st.markdown(f"**{home_abbr}**")
                if home_inj["out"]:
                    st.caption("Out:")
                    st.write(", ".join(home_inj["out"]))
                if home_inj["q"]:
                    st.caption("Questionable / Doubtful:")
                    st.write(", ".join(home_inj["q"]))

            with col_ai:
                st.markdown(f"**{away_abbr}**")
                if away_inj["out"]:
                    st.caption("Out:")
                    st.write(", ".join(away_inj["out"]))
                if away_inj["q"]:
                    st.caption("Questionable / Doubtful:")
                    st.write(", ".join(away_inj["q"]))

            apply_auto = st.button(
                "Apply 'Out' lists above to selectors for this game",
                key="apply_bref_auto_out",
            )

    # --- Projected lineup v1 (read-only helper) ---
    with st.expander("Projected lineups (v1)", expanded=False):
        col_pl_home, col_pl_away = st.columns(2)

        with col_pl_home:
            render_projected_lineup(home_abbr, home_inj, label="Home")

        with col_pl_away:
            render_projected_lineup(away_abbr, away_inj, label="Away")

    # When user clicks apply_auto, pre-fill the selectboxes via session_state
    if apply_auto:
        max_slots = 5

        # Home: keep only OUT players that are actually in the current pool
        home_candidates = [p for p in home_inj["out"] if p in home_players]

        for i in range(max_slots):
            default_name = home_candidates[i] if i < len(home_candidates) else "None"
            st.session_state[f"home_out_{i}"] = default_name

        # Away: same logic
        away_candidates = [p for p in away_inj["out"] if p in away_players]

        for i in range(max_slots):
            default_name = away_candidates[i] if i < len(away_candidates) else "None"
            st.session_state[f"away_out_{i}"] = default_name

    # Manual / final selection of out players (drives the model)
    home_out, away_out = [], []
    colH, colA2 = st.columns(2)

    with colH:
        st.markdown(f"**Home Out: {home_abbr}**")
        for i in range(5):
            choice = st.selectbox(
                f"Out Player {i+1}",
                ["None"] + home_players,
                key=f"home_out_{i}",
            )
            if choice != "None":
                home_out.append(choice)

        if hide_long_out and home_long_out:
            st.caption(
                "Long-term out (hidden): "
                + ", ".join(sorted(home_long_out))
            )

    with colA2:
        st.markdown(f"**Away Out: {away_abbr}**")
        for i in range(5):
            choice = st.selectbox(
                f"Out Player {i+1}",
                ["None"] + away_players,
                key=f"away_out_{i}",
            )
            if choice != "None":
                away_out.append(choice)

        if hide_long_out and away_long_out:
            st.caption(
                "Long-term out (hidden): "
                + ", ".join(sorted(away_long_out))
            )


    # --- Compute + Model Logic ---
    compute_btn = st.button("Compute Edge")

    if compute_btn:
        vegas_margin = -vegas_line
        stat_margin = proj_fav - proj_dog

        # --- Cheatsheet-only model (Model A) ---
        cheat_diff = stat_margin - vegas_margin
        cheat_edge = abs(cheat_diff)
        cheat_side = "favorite" if cheat_diff > 0 else "underdog"
        cheat_pick = side_to_pick(cheat_side, favorite, underdog, vegas_line)

        # --- Player Impact ---
        fav_is_home = favorite.upper() == home_abbr

        fav_out_players = home_out if fav_is_home else away_out
        dog_out_players = away_out if fav_is_home else home_out

        fav_player_adj_raw, _, _, _ = compute_player_injury_terms(
            out_players=fav_out_players, players_df=players_df
        )
        dog_player_adj_raw, _, _, _ = compute_player_injury_terms(
            out_players=dog_out_players, players_df=players_df
        )

        def transform_player_adj(val: float) -> float:
            abs_player = abs(val)
            if abs_player == 0:
                return 0.0
            eff_val = abs_player + (
                abs_player * (abs_player / (abs_player + (K5 * 2)))
            )
            return -eff_val if val < 0 else eff_val

        fav_effective_player_adj_raw = transform_player_adj(-fav_player_adj_raw)
        dog_effective_player_adj_raw = transform_player_adj(dog_player_adj_raw)

        fav_effective_player_adj, fav_injury_flag, fav_injury_heavy = cap_and_flag_injury_effect(
            fav_effective_player_adj_raw, injury_cap=6.0
        )
        dog_effective_player_adj, dog_injury_flag, dog_injury_heavy = cap_and_flag_injury_effect(
            dog_effective_player_adj_raw, injury_cap=6.0
        )

        player_adj = dog_player_adj_raw - fav_player_adj_raw
        effective_player_adj = dog_effective_player_adj + fav_effective_player_adj

        injury_heavy = 1 if (fav_injury_heavy or dog_injury_heavy) else 0
        flag_order = ["low", "medium", "high"]
        fav_rank = flag_order.index(fav_injury_flag) if fav_injury_flag in flag_order else 0
        dog_rank = flag_order.index(dog_injury_flag) if dog_injury_flag in flag_order else 0
        injury_impact_flag = flag_order[max(fav_rank, dog_rank)]

        # --- Team Strength ---
        if TEAM_RATINGS_AVAILABLE:
            home_power = get_team_power(home_abbr, is_home=True)
            away_power = get_team_power(away_abbr, is_home=False)
        else:
            home_power = None
            away_power = None

        if (home_power is not None) and (away_power is not None):
            team_margin_home = home_power - away_power
            if favorite.upper() == home_abbr:
                fav_team_margin = team_margin_home
            else:
                fav_team_margin = -team_margin_home
        else:
            fav_team_margin = 0.0

        fav_team_margin_combined = fav_team_margin

        # --- Pace-based adjustment ---
        pace_adj = 0.0
        fav_pace = None
        dog_pace = None

        if TEAM_RATINGS_AVAILABLE:
            fav_pace = get_team_metric(favorite.upper(), ["Pace", "PACE"])
            dog_pace = get_team_metric(underdog.upper(), ["Pace", "PACE"])

        if (
            (fav_pace is not None)
            and (dog_pace is not None)
            and (LEAGUE_AVG_PACE is not None)
        ):
            pace_delta = fav_pace - dog_pace
            env_pace = ((fav_pace + dog_pace) / 2.0) - LEAGUE_AVG_PACE
            pace_adj = pace_delta * PACE_DELTA_WEIGHT + env_pace * PACE_ENV_WEIGHT

        # --- Rest / B2B adjustments (fav perspective) ---
        b2b_adj = 0.0
        rest_adj = 0.0
        b2b_desc = "None"
        rest_desc = "None"

        home_rest = st.session_state.get("home_rest_profile")
        away_rest = st.session_state.get("away_rest_profile")

        if home_rest and away_rest and not manual_mode:
            fav_is_home = favorite.upper() == home_abbr

            if fav_is_home:
                fav_b2b_pen = home_rest["b2b_penalty"]
                dog_b2b_pen = away_rest["b2b_penalty"]
                fav_games5 = home_rest["games_last_5"]
                dog_games5 = away_rest["games_last_5"]
            else:
                fav_b2b_pen = away_rest["b2b_penalty"]
                dog_b2b_pen = home_rest["b2b_penalty"]
                fav_games5 = away_rest["games_last_5"]
                dog_games5 = home_rest["games_last_5"]

            # B2B adj: difference in B2B penalties (fav perspective)
            b2b_adj = fav_b2b_pen - dog_b2b_pen

            # Rest adj: difference in games in last 5 days (fav perspective)
            games_diff = fav_games5 - dog_games5
            if games_diff != 0:
                rest_adj = -games_diff * REST_GAME_WEIGHT

            # Descriptions for UI
            b2b_desc_parts = []
            if home_rest["is_b2b"]:
                b2b_desc_parts.append(
                    f"{home_abbr} B2B ({home_rest['days_since']}d rest, "
                    f"{home_rest['games_last_5']}g/5d)"
                )
            if away_rest["is_b2b"]:
                b2b_desc_parts.append(
                    f"{away_abbr} B2B ({away_rest['days_since']}d rest, "
                    f"{away_rest['games_last_5']}g/5d)"
                )
            b2b_desc = "; ".join(b2b_desc_parts) if b2b_desc_parts else "None"
            rest_desc = f"{fav_games5}–{dog_games5} (fav–dog games last 5d)"
        else:
            b2b_adj = 0.0
            rest_adj = 0.0
            b2b_desc = "None"
            rest_desc = "N/A"

        # --- Per-team B2B flags used for learning/logging ---
        fav_is_b2b = 0
        dog_is_b2b = 0
        if home_rest and away_rest and not manual_mode:
            fav_is_home = favorite.upper() == home_abbr
            if fav_is_home:
                fav_is_b2b = 1 if home_rest.get("is_b2b") else 0
                dog_is_b2b = 1 if away_rest.get("is_b2b") else 0
            else:
                fav_is_b2b = 1 if away_rest.get("is_b2b") else 0
                dog_is_b2b = 1 if home_rest.get("is_b2b") else 0

        # --- Hybrid margin ---
        hybrid_margin = (
            stat_margin * W_SCORE
            + fav_team_margin_combined * W_TEAM
            + effective_player_adj * W_PLAYER
            + pace_adj
            + b2b_adj
            + rest_adj
        )

        diff_hybrid = hybrid_margin - vegas_margin
        edge = abs(diff_hybrid)
        side = "favorite" if diff_hybrid > 0 else "underdog"

        # alignment logic
        aligned = (cheat_side == side)
        if aligned:
            effective_edge = max(edge, cheat_edge)
        else:
            effective_edge = edge

        conf = classify_edge(effective_edge)
        hybrid_pick = side_to_pick(side, favorite, underdog, vegas_line)
        
        color = confidence_to_color(conf)

        # Snapshot ORtg/DRtg/NetRtg for logging
        fav_ortg = fav_drtg = fav_netr = None
        dog_ortg = dog_drtg = dog_netr = None

        if TEAM_RATINGS_AVAILABLE:
            fav_ortg = get_team_metric(
                favorite.upper(), ["ORtg", "ORTG", "OffRtg", "OFF_RTG"]
            )
            fav_drtg = get_team_metric(
                favorite.upper(), ["DRtg", "DRTG", "DefRtg", "DEF_RTG"]
            )
            fav_netr = get_team_metric(
                favorite.upper(), ["NetRtg", "NETRTG", "Base Net Rating", "Base net"]
            )

            dog_ortg = get_team_metric(
                underdog.upper(), ["ORtg", "ORTG", "OffRtg", "OFF_RTG"]
            )
            dog_drtg = get_team_metric(
                underdog.upper(), ["DRtg", "DRTG", "DefRtg", "DEF_RTG"]
            )
            dog_netr = get_team_metric(
                underdog.upper(), ["NetRtg", "NETRTG", "Base Net Rating", "Base net"]
            )

        # Store everything needed to re-render result on future reruns
        st.session_state.update(
            {
                "favorite": favorite,
                "underdog": underdog,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "vegas_line": float(vegas_line),
                "game_status": game_status,
                "stat_margin": float(stat_margin),
                "vegas_margin": float(vegas_margin),
                "player_adj": float(player_adj),
                "player_adj_raw": float(player_adj),
                "fav_player_adj_raw": float(fav_player_adj_raw),
                "dog_player_adj_raw": float(dog_player_adj_raw),
                "effective_player_adj": float(effective_player_adj),
                "fav_effective_player_adj": float(fav_effective_player_adj),
                "dog_effective_player_adj": float(dog_effective_player_adj),
                "injury_impact_flag": injury_impact_flag,
                "fav_injury_impact_flag": fav_injury_flag,
                "dog_injury_impact_flag": dog_injury_flag,
                "injury_heavy": int(injury_heavy),
                "fav_injury_heavy": int(fav_injury_heavy),
                "dog_injury_heavy": int(dog_injury_heavy),
                "home_power": home_power,
                "away_power": away_power,
                "fav_team_margin_combined": float(fav_team_margin_combined),
                "home_out_list": home_out,
                "away_out_list": away_out,
                "fav_pace": float(fav_pace) if fav_pace is not None else None,
                "dog_pace": float(dog_pace) if dog_pace is not None else None,
                "pace_adj": float(pace_adj),
                "b2b_adj": float(b2b_adj),
                "b2b_desc": b2b_desc,
                "rest_adj": float(rest_adj),
                "rest_desc": rest_desc,
                "hybrid_margin": float(hybrid_margin),
                "edge": float(edge),
                "effective_edge": float(effective_edge),
                "conf": conf,
                "hybrid_pick": hybrid_pick,
                "cheat_edge": float(cheat_edge),
                "cheat_pick": cheat_pick,
                "aligned": 1 if aligned else 0,
                "fav_ortg": float(fav_ortg) if fav_ortg is not None else None,
                "dog_ortg": float(dog_ortg) if dog_ortg is not None else None,
                "fav_drtg": float(fav_drtg) if fav_drtg is not None else None,
                "dog_drtg": float(dog_drtg) if dog_drtg is not None else None,
                "fav_netr": float(fav_netr) if fav_netr is not None else None,
                "dog_netr": float(dog_netr) if dog_netr is not None else None,
                "fav_is_b2b": int(fav_is_b2b),
                "dog_is_b2b": int(dog_is_b2b),
                "color": color,
            }
        )
        st.session_state["has_decision"] = True
        st.session_state["decision_logged"] = False

    # --- Result view + logging (persists after button click) ---
    if st.session_state.get("has_decision", False) and not st.session_state.get(
        "decision_logged", False
    ):
        fav = st.session_state["favorite"]
        dog = st.session_state["underdog"]
        home_abbr = st.session_state["home_abbr"]
        away_abbr = st.session_state["away_abbr"]
        vegas_line = st.session_state["vegas_line"]
        game_status = st.session_state["game_status"]

        stat_margin = st.session_state["stat_margin"]
        vegas_margin = st.session_state["vegas_margin"]
        player_adj = st.session_state["player_adj"]
        effective_player_adj = st.session_state["effective_player_adj"]
        player_adj_raw = st.session_state.get("player_adj_raw", player_adj)
        fav_player_adj_raw = st.session_state.get("fav_player_adj_raw", 0.0)
        dog_player_adj_raw = st.session_state.get("dog_player_adj_raw", 0.0)
        fav_effective_player_adj = st.session_state.get("fav_effective_player_adj", 0.0)
        dog_effective_player_adj = st.session_state.get("dog_effective_player_adj", 0.0)
        injury_impact_flag = st.session_state.get("injury_impact_flag", "low")
        injury_heavy = st.session_state.get("injury_heavy", 0)
        fav_injury_impact_flag = st.session_state.get("fav_injury_impact_flag", "low")
        dog_injury_impact_flag = st.session_state.get("dog_injury_impact_flag", "low")
        fav_injury_heavy = st.session_state.get("fav_injury_heavy", 0)
        dog_injury_heavy = st.session_state.get("dog_injury_heavy", 0)
        home_power = st.session_state["home_power"]
        away_power = st.session_state["away_power"]
        fav_team_margin_combined = st.session_state["fav_team_margin_combined"]
        home_out = st.session_state["home_out_list"]
        away_out = st.session_state["away_out_list"]
        fav_pace = st.session_state["fav_pace"]
        dog_pace = st.session_state["dog_pace"]
        pace_adj = st.session_state["pace_adj"]
        b2b_adj = st.session_state["b2b_adj"]
        b2b_desc = st.session_state["b2b_desc"]
        rest_adj = st.session_state["rest_adj"]
        rest_desc = st.session_state["rest_desc"]
        hybrid_margin = st.session_state["hybrid_margin"]
        edge = st.session_state["edge"]
        conf = st.session_state["conf"]
        hybrid_pick = st.session_state["hybrid_pick"]
        cheat_edge = st.session_state["cheat_edge"]
        cheat_pick = st.session_state["cheat_pick"]
        aligned = st.session_state["aligned"] == 1
        fav_ortg = st.session_state["fav_ortg"]
        dog_ortg = st.session_state["dog_ortg"]
        fav_drtg = st.session_state["fav_drtg"]
        dog_drtg = st.session_state["dog_drtg"]
        fav_netr = st.session_state["fav_netr"]
        dog_netr = st.session_state["dog_netr"]
        fav_is_b2b = st.session_state.get("fav_is_b2b", 0)
        dog_is_b2b = st.session_state.get("dog_is_b2b", 0)

        color = st.session_state["color"]

        fav_pick_str = side_to_pick("favorite", fav, dog, vegas_line)
        dog_pick_str = side_to_pick("underdog", fav, dog, vegas_line)
        default_index = 0 if hybrid_pick == fav_pick_str else 1

        st.markdown("---")
        st.subheader("Pick & Log")

        pick_to_log = st.radio(
            "Pick to log into database",
            [fav_pick_str, dog_pick_str],
            index=default_index,
            key="pick_to_log",
        )

        # 3-column breakdown
        model_cols = st.columns(3)

        with model_cols[0]:
            st.markdown("### Cheatsheet vs Hybrid")
            st.metric("Stat Margin (fav - dog)", round(stat_margin, 2))
            st.metric("Cheatsheet Edge vs Line", round(cheat_edge, 2))
            st.metric("Hybrid Edge vs Line", round(edge, 2))
            st.metric("Hybrid Pick", hybrid_pick)

            if aligned:
                st.success(f"Models aligned on {hybrid_pick} "
                           f"(Cheatsheet: {cheat_pick})")
            else:
                st.warning(
                    f"Models disagree: Hybrid → {hybrid_pick}, "
                    f"Cheatsheet → {cheat_pick}"
                )

        with model_cols[1]:
            st.markdown("### Player Impact Detail")
            st.write(
                f"Raw Player Adj (fav perspective, recency-weighted): **{player_adj_raw:.2f}**"
            )
            st.write(
                f"Effective Player Adj (curved & capped): **{effective_player_adj:.2f}** "
                f"({injury_impact_flag}, heavy={injury_heavy})"
            )
            st.write(
                f"Fav outs adj: **{fav_effective_player_adj:.2f}** (raw {-fav_player_adj_raw:.2f})"
            )
            st.write(
                f"Dog outs adj: **{dog_effective_player_adj:.2f}** (raw {dog_player_adj_raw:.2f})"
            )
            st.write(f"{home_abbr} outs: {', '.join(home_out) if home_out else 'None'}")
            st.write(f"{away_abbr} outs: {', '.join(away_out) if away_out else 'None'}")

        with model_cols[2]:
            st.markdown("### Team Strength Detail")
            if (home_power is not None) and (away_power is not None):
                st.write(f"{home_abbr} Team Power: **{home_power:.2f}**")
                st.write(f"{away_abbr} Team Power: **{away_power:.2f}**")
                st.write(
                    f"Combined Team Margin Used: **{fav_team_margin_combined:.2f}**"
                )
            else:
                st.write("No team ratings found for one or both teams.")

            if (
                (fav_pace is not None)
                and (dog_pace is not None)
                and (LEAGUE_AVG_PACE is not None)
            ):
                st.write(
                    f"Pace (fav / dog / league): "
                    f"**{fav_pace:.1f} / {dog_pace:.1f} / {LEAGUE_AVG_PACE:.1f}**"
                )
            else:
                st.write("Pace data not available for one or both teams.")

            st.write(f"Pace Adjustment Applied: **{pace_adj:.2f}**")
            st.write(f"B2B Adjustment: **{b2b_desc}** ({b2b_adj:+.2f})")
            st.write(
                f"Rest / Schedule Adjustment: **{rest_adj:+.2f}** "
                f"(fav games last 5d vs dog: {rest_desc})"
            )

        # Big decision box that flips perspective with radio
        view_side = "favorite" if pick_to_log == fav_pick_str else "underdog"
        sign = 1 if view_side == "favorite" else -1
        label_team = fav if view_side == "favorite" else dog

        view_stat_margin = stat_margin * sign
        view_team_margin = fav_team_margin_combined * sign
        view_player_adj = effective_player_adj * sign
        view_pace_adj = pace_adj * sign
        view_b2b_adj = b2b_adj * sign
        view_rest_adj = rest_adj * sign
        view_hybrid_margin = hybrid_margin * sign
        view_vegas_margin = vegas_margin * sign

        pick_label = pick_to_log   # this is ORL -4.0 or PHI +4.0, etc.
        
        st.markdown("## 📌 Final Model Decision")
        st.markdown(
            f"""
            <div style='margin-top:10px; padding:15px;
                        background-color:#111; border-radius:8px;
                        border:1px solid {color}; text-align:center;'>
                <div style='font-size:26px; font-weight:bold; color:{color};'>
                    {pick_label}
                </div>
                <div style='font-size:18px; color:white; margin-top:4px;'>
                    Edge vs line (hybrid): {edge:.2f} pts — Confidence: {conf}
                </div>
                <div style='font-size:13px; color:#ccc; margin-top:8px; text-align:left;'>
                    <b>Breakdown ({label_team} perspective):</b><br/>
                    • Stat projection margin: {view_stat_margin:.2f}<br/>
                    • Cheatsheet edge vs line: {cheat_edge:.2f} ({cheat_pick})<br/>
                    • Team strength margin: {view_team_margin:.2f}<br/>
                    • Player impact (effective): {view_player_adj:.2f}<br/>
                    • Pace adjustment: {view_pace_adj:.2f}<br/>
                    • B2B adjustment: {view_b2b_adj:.2f} ({b2b_desc})<br/>
                    • Rest / schedule adjustment: {view_rest_adj:.2f} ({rest_desc})<br/>
                    • Hybrid margin vs opponent: {view_hybrid_margin:.2f}<br/>
                    • Vegas implied margin: {view_vegas_margin:.2f}
                </div>
                <div style='font-size:12px; color:#888; margin-top:6px;'>
                    Game status: {game_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Log this pick into DB ---
        if st.button("💾 Log This Pick"):
            logged_pick = pick_to_log

            # Map home/away outs to favorite/underdog outs for logging
            fav_is_home = favorite.upper() == home_abbr.upper()
            if fav_is_home:
                fav_out_list = home_out
                dog_out_list = away_out
            else:
                fav_out_list = away_out
                dog_out_list = home_out

            entry = pd.DataFrame(
                [
                    {
                        "date": datetime.date.today().strftime("%Y-%m-%d"),
                        "favorite": fav,
                        "underdog": dog,
                        "vegas_line": float(vegas_line),
                        "model_line": 0.0,
                        "edge": float(edge),
                        "pick": logged_pick,
                        "confidence": conf,
                        "final_score": None,
                        "spread_covered": None,
                        "fav_pace": fav_pace,
                        "dog_pace": dog_pace,
                        "fav_ortg": fav_ortg,
                        "dog_ortg": dog_ortg,
                        "fav_drtg": fav_drtg,
                        "dog_drtg": dog_drtg,
                        "fav_netr": fav_netr,
                        "dog_netr": dog_netr,
                        "fav_out": ", ".join(fav_out_list) if fav_out_list else None,
                        "dog_out": ", ".join(dog_out_list) if dog_out_list else None,
                        "home_out": ", ".join(home_out) if home_out else None,
                        "away_out": ", ".join(away_out) if away_out else None,
                        "cheat_edge": cheat_edge,
                        "cheat_pick": cheat_pick,
                        "models_aligned": 1 if aligned else 0,
                        "hybrid_pick": hybrid_pick,
                        "player_adj_raw": player_adj_raw,
                        "fav_player_adj_raw": fav_player_adj_raw,
                        "dog_player_adj_raw": dog_player_adj_raw,
                        "effective_player_adj": effective_player_adj,
                        "fav_effective_player_adj": fav_effective_player_adj,
                        "dog_effective_player_adj": dog_effective_player_adj,
                        "injury_impact_flag": injury_impact_flag,
                        "injury_heavy": injury_heavy,
                        "fav_injury_impact_flag": fav_injury_impact_flag,
                        "dog_injury_impact_flag": dog_injury_impact_flag,
                        "fav_injury_heavy": fav_injury_heavy,
                        "dog_injury_heavy": dog_injury_heavy,
                        # components for weight learning
                        "stat_margin": st.session_state.get("stat_margin"),
                        "team_margin": st.session_state.get("fav_team_margin_combined"),
                        "player_adj_eff": st.session_state.get("effective_player_adj"),
                        "pace_adj_term": st.session_state.get("pace_adj"),
                        "rest_adj_term": st.session_state.get("rest_adj"),
                        "b2b_adj_term": st.session_state.get("b2b_adj"),
                        "vegas_margin": st.session_state.get("vegas_margin"),
                        "hybrid_margin": st.session_state.get("hybrid_margin"),
                        "effective_edge": st.session_state.get("effective_edge"),
                    }
                ]
            )

            with engine.begin() as conn:
                entry.to_sql("logs", conn, if_exists="append", index=False)

            # Clear cached logs so Slate + Prediction Log see the new row
            load_all_logs.clear()

            st.success("Logged!")
            st.session_state["has_decision"] = False
            st.session_state["decision_logged"] = True
            st.rerun()

# ============================================
# FULL SLATE AUTO VIEW
# ============================================
with tab_full:
    st.subheader("Full Slate (Team-Only Auto View)")

    games_data = fetch_games_with_odds()

    if not games_data:
        st.info("No odds data available for today's slate (or Odds API unavailable).")
    else:
        rows = []
        for g in games_data:
            # Skip completed games; this is for upcoming/live
            if g.get("status") == "FINAL":
                continue

            home_abbr = g.get("home_abbr")
            away_abbr = g.get("away_abbr")
            favorite_abbr = g.get("favorite_abbr")
            underdog_abbr = g.get("underdog_abbr")
            vegas_line = float(g.get("spread", 0.0))
            status = g.get("status")
            commence = g.get("commence_dt")

            if not home_abbr or not away_abbr or not favorite_abbr or not underdog_abbr:
                continue

            # --- Team strength (same helper as Single Game) ---
            home_power = get_team_power(home_abbr, is_home=True)
            away_power = get_team_power(away_abbr, is_home=False)

            if home_power is None or away_power is None:
                # Missing rating, skip
                continue

            # Team margin in home-away coordinates
            team_margin_home = home_power - away_power  # home - away

            if favorite_abbr.upper() == home_abbr.upper():
                fav_team_margin = team_margin_home
            else:
                fav_team_margin = -team_margin_home

            # --- Team-only hybrid margin (no stats, no injuries) ---
            vegas_margin = -vegas_line  # favorite perspective
            stat_margin = 0.0
            effective_player_adj = 0.0
            pace_adj = 0.0
            b2b_adj = 0.0
            rest_adj = 0.0

            hybrid_margin = (
                stat_margin * W_SCORE
                + fav_team_margin * W_TEAM
                + effective_player_adj * W_PLAYER
                + pace_adj
                + b2b_adj
                + rest_adj
            )

            diff = hybrid_margin - vegas_margin
            edge = abs(diff)
            side = "favorite" if diff > 0 else "underdog"

            # Confidence thresholds (same ones you use elsewhere)
            if edge >= STRONG_EDGE:
                conf = "STRONG"
            elif edge >= MEDIUM_EDGE:
                conf = "MEDIUM"
            else:
                conf = "PASS"

            # Tip-off label
            if isinstance(commence, datetime.datetime):
                tip_label = commence.strftime("%Y-%m-%d %I:%M %p").lstrip("0") + " ET"
            else:
                tip_label = "TBD"

            rows.append(
                {
                    "away": away_abbr,
                    "home": home_abbr,
                    "fav": favorite_abbr,
                    "dog": underdog_abbr,
                    "spread": vegas_line,
                    "model_margin_fav": round(hybrid_margin, 2),
                    "edge_pts": round(edge, 2),
                    "model_side": side,
                    "confidence": conf,
                    "status": status,
                    "tip_off_et": tip_label,
                }
            )

        if rows:
            slate_df = pd.DataFrame(rows)
            slate_df = slate_df.sort_values("edge_pts", ascending=False)

            st.dataframe(
                slate_df,
                use_container_width=True,
            )
        else:
            st.info("No games with enough data to show in the auto-slate view.")


# ============================================
# PERFORMANCE OVERVIEW
# ============================================
with tab_perf:
    st.subheader("Performance Overview")

    # Load all logs from the DB
    try:
        logs_all = load_all_logs()
    except Exception as e:
        st.error(f"Error loading logs: {e}")
        logs_all = pd.DataFrame()

    if logs_all.empty:
        st.info("No logged picks yet. Log some games and refresh final scores first.")
    elif "spread_covered" not in logs_all.columns:
        st.info("No graded results yet (spread_covered column missing).")
    else:
        df = logs_all.copy()
        # Keep only rows where we have a result
        df = df[df["spread_covered"].notna()]

        if df.empty:
            st.info(
                "No graded picks yet. Use 'Refresh Final Scores' in Prediction Log after games complete."
            )
        else:
            # ---------------- SNAPSHOT METRICS ----------------
            total = len(df)
            wins = (df["spread_covered"] == 1).sum()
            losses = (df["spread_covered"] == 0).sum()
            pushes = (df["spread_covered"] == 2).sum()
            eff = total - pushes

            avg_edge = float(df["edge"].mean()) if "edge" in df.columns else float("nan")
            win_pct = (wins / eff * 100.0) if eff > 0 else float("nan")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total picks", total)
            with col2:
                st.metric("Win% (graded)", f"{win_pct:.1f}%" if eff > 0 else "—")
            with col3:
                st.metric(
                    "Avg edge (pts)",
                    f"{avg_edge:.2f}" if not pd.isna(avg_edge) else "—",
                )
            with col4:
                st.metric("Eff. picks (W+L)", eff)

            st.markdown("---")

            # ---------------- 1. OVERALL SUMMARY TABLE ----------------
            st.markdown("### 1. Overall Results")

            overall_df = pd.DataFrame(
                [
                    {
                        "Total picks": total,
                        "Eff. picks (W+L)": eff,
                        "Wins": wins,
                        "Losses": losses,
                        "Pushes": pushes,
                        "Win%": round(win_pct, 1) if eff > 0 else None,
                        "Avg edge (pts)": round(avg_edge, 2)
                        if not pd.isna(avg_edge)
                        else None,
                    }
                ]
            )
            st.dataframe(overall_df, use_container_width=True)

            # ---------------- 2. EDGE BUCKETS ----------------
            if "edge" in df.columns:
                st.markdown("### 2. Performance by Edge Bucket (|edge|)")

                dfe = df.copy()
                dfe["abs_edge"] = dfe["edge"].abs()

                def summarize_bucket(name, mask):
                    sub = dfe[mask]
                    if sub.empty:
                        return {
                            "Bucket": name,
                            "Picks": 0,
                            "Wins": 0,
                            "Losses": 0,
                            "Pushes": 0,
                            "Win%": None,
                            "Avg edge (pts)": None,
                        }
                    N = len(sub)
                    W = (sub["spread_covered"] == 1).sum()
                    L = (sub["spread_covered"] == 0).sum()
                    P = (sub["spread_covered"] == 2).sum()
                    effN = N - P
                    winp = (W / effN * 100.0) if effN > 0 else None
                    avg_e = float(sub["edge"].mean())
                    return {
                        "Bucket": name,
                        "Picks": N,
                        "Wins": W,
                        "Losses": L,
                        "Pushes": P,
                        "Win%": round(winp, 1) if winp is not None else None,
                        "Avg edge (pts)": round(avg_e, 2),
                    }

                buckets = [
                    summarize_bucket("[0, 2)", (dfe["abs_edge"] < 2)),
                    summarize_bucket(
                        "[2, 4)", (dfe["abs_edge"] >= 2) & (dfe["abs_edge"] < 4)
                    ),
                    summarize_bucket(
                        "[4, 6)", (dfe["abs_edge"] >= 4) & (dfe["abs_edge"] < 6)
                    ),
                    summarize_bucket(">= 6", (dfe["abs_edge"] >= 6)),
                ]
                bucket_df = pd.DataFrame(buckets)
                st.dataframe(bucket_df, use_container_width=True)

            # ---------------- 3. BY CONFIDENCE ----------------
            if "confidence" in df.columns:
                st.markdown("### 3. By Confidence (excluding PASS)")

                conf_df = df[df["confidence"].isin(["MEDIUM", "STRONG"])].copy()
                if conf_df.empty:
                    st.info("No MEDIUM/STRONG picks yet.")
                else:
                    rows = []
                    for conf_val in ["MEDIUM", "STRONG"]:
                        sub = conf_df[conf_df["confidence"] == conf_val]
                        if sub.empty:
                            continue
                        N = len(sub)
                        W = (sub["spread_covered"] == 1).sum()
                        L = (sub["spread_covered"] == 0).sum()
                        P = (sub["spread_covered"] == 2).sum()
                        effN = N - P
                        winp = (W / effN * 100.0) if effN > 0 else None
                        avg_e = float(sub["edge"].mean())
                        rows.append(
                            {
                                "Confidence": conf_val,
                                "Picks": N,
                                "Wins": W,
                                "Losses": L,
                                "Pushes": P,
                                "Win%": round(winp, 1) if winp is not None else None,
                                "Avg edge (pts)": round(avg_e, 2),
                            }
                        )
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # ---------------- 4. MODELS ALIGNED ----------------
            if "models_aligned" in df.columns:
                st.markdown("### 4. Models Aligned vs Not")

                aligned_rows = []
                for label, mask in [
                    ("Aligned (Hybrid = Cheatsheet)", df["models_aligned"] == 1),
                    ("Not aligned", df["models_aligned"] == 0),
                ]:
                    sub = df[mask]
                    if sub.empty:
                        aligned_rows.append(
                            {
                                "Group": label,
                                "Picks": 0,
                                "Wins": 0,
                                "Losses": 0,
                                "Pushes": 0,
                                "Win%": None,
                                "Avg edge (pts)": None,
                            }
                        )
                        continue

                    N = len(sub)
                    W = (sub["spread_covered"] == 1).sum()
                    L = (sub["spread_covered"] == 0).sum()
                    P = (sub["spread_covered"] == 2).sum()
                    effN = N - P
                    winp = (W / effN * 100.0) if effN > 0 else None
                    avg_e = float(sub["edge"].mean())
                    aligned_rows.append(
                        {
                            "Group": label,
                            "Picks": N,
                            "Wins": W,
                            "Losses": L,
                            "Pushes": P,
                            "Win%": round(winp, 1) if winp is not None else None,
                            "Avg edge (pts)": round(avg_e, 2),
                        }
                    )

                st.dataframe(pd.DataFrame(aligned_rows), use_container_width=True)

            # ---------------- 5. INJURY HEAVY CONTEXT ----------------
            if "injury_heavy" in df.columns:
                st.markdown("### 5. Injury-Heavy vs Normal")

                def summarize_injury(label, mask):
                    sub = df[mask]
                    if sub.empty:
                        return {
                            "Group": label,
                            "Picks": 0,
                            "Wins": 0,
                            "Losses": 0,
                            "Pushes": 0,
                            "Win%": None,
                            "Avg edge (pts)": None,
                        }
                    N = len(sub)
                    W = (sub["spread_covered"] == 1).sum()
                    L = (sub["spread_covered"] == 0).sum()
                    P = (sub["spread_covered"] == 2).sum()
                    effN = N - P
                    winp = (W / effN * 100.0) if effN > 0 else None
                    avg_e = float(sub["edge"].mean())
                    return {
                        "Group": label,
                        "Picks": N,
                        "Wins": W,
                        "Losses": L,
                        "Pushes": P,
                        "Win%": round(winp, 1) if winp is not None else None,
                        "Avg edge (pts)": round(avg_e, 2),
                    }

                injury_rows = [
                    summarize_injury("Injury heavy = 1", df["injury_heavy"] == 1),
                    summarize_injury("Not heavy (0)", df["injury_heavy"] != 1),
                ]
                st.dataframe(pd.DataFrame(injury_rows), use_container_width=True)

            # ---------------- 6. B2B CONTEXT (if available) ----------------
            if "fav_is_b2b" in df.columns and "dog_is_b2b" in df.columns:
                st.markdown("### 6. B2B Context")

                def summarize_b2b(label, mask):
                    sub = df[mask]
                    if sub.empty:
                        return {
                            "Group": label,
                            "Picks": 0,
                            "Wins": 0,
                            "Losses": 0,
                            "Pushes": 0,
                            "Win%": None,
                            "Avg edge (pts)": None,
                        }
                    N = len(sub)
                    W = (sub["spread_covered"] == 1).sum()
                    L = (sub["spread_covered"] == 0).sum()
                    P = (sub["spread_covered"] == 2).sum()
                    effN = N - P
                    winp = (W / effN * 100.0) if effN > 0 else None
                    avg_e = float(sub["edge"].mean())
                    return {
                        "Group": label,
                        "Picks": N,
                        "Wins": W,
                        "Losses": L,
                        "Pushes": P,
                        "Win%": round(winp, 1) if winp is not None else None,
                        "Avg edge (pts)": round(avg_e, 2),
                    }

                b2b_rows = [
                    summarize_b2b("Fav B2B", df["fav_is_b2b"] == 1),
                    summarize_b2b("Fav not B2B", df["fav_is_b2b"] == 0),
                    summarize_b2b("Dog B2B", df["dog_is_b2b"] == 1),
                    summarize_b2b("Dog not B2B", df["dog_is_b2b"] == 0),
                ]
                st.dataframe(pd.DataFrame(b2b_rows), use_container_width=True)



# ============================================
# CTG REVIEW / DAILY REVIEW
# ============================================
with tab_ctg:
    st.subheader("CTG Review & Daily Review")

    # --- CTG PDF upload (one or more files) ---
    st.markdown("### CTG Postgame Reports (Optional)")
    ctg_pdfs = st.file_uploader(
        "Upload CTG PDFs for this slate (e.g. '11_25_25 ORL @ PHI')",
        type=["pdf"],
        accept_multiple_files=True,
        key="ctg_pdf_upload",
    )

    # ctg_docs maps: (date, frozenset({TEAM1, TEAM2})) -> {"text": str, "name": filename}
    ctg_docs = st.session_state.get("ctg_docs", {})

    if ctg_pdfs:
        pattern = re.compile(r"(\d{2}_\d{2}_\d{2})\s+([A-Za-z]{2,4})\s+@\s+([A-Za-z]{2,4})")

        for uploaded in ctg_pdfs:
            base = os.path.splitext(uploaded.name)[0]
            try:
                # Allow extra text before/after, just find "MM_DD_YY ORL @ PHI" anywhere
                m = pattern.search(base)
                if not m:
                    raise ValueError("Could not find 'MM_DD_YY TEAM @ TEAM' pattern")

                date_part = m.group(1)      # "11_25_25"
                away_abbr = m.group(2).upper()
                home_abbr = m.group(3).upper()

                month_s, day_s, year_s = date_part.split("_")
                month = int(month_s)
                day = int(day_s)
                yy = int(year_s)
                year = 2000 + yy if yy < 100 else yy

                game_date = datetime.date(year, month, day)
                key = (game_date, frozenset({away_abbr, home_abbr}))

                # ✅ use ctg_text, don't shadow sqlalchemy.text
                ctg_text = extract_text_from_pdf(uploaded)

                if ctg_text.strip():
                    ctg_docs[key] = {
                        "text": ctg_text,
                        "name": uploaded.name,
                    }
                    st.success(
                        f"Linked CTG PDF '{uploaded.name}' to "
                        f"{away_abbr} @ {home_abbr} on {game_date}."
                    )
                else:
                    st.warning(f"No text found in '{uploaded.name}'.")
            except Exception as e:
                st.warning(
                    f"Could not parse CTG filename '{uploaded.name}'. Expected something containing "
                    f"'MM_DD_YY ORL @ PHI'. Error: {e}"
                )

        # persist mapping in session
        st.session_state["ctg_docs"] = ctg_docs

        
    try:
        all_logs = load_all_logs()  
    except Exception as e:
        st.warning(f"Error reading logs: {e}")
        all_logs = pd.DataFrame()

    if all_logs.empty:
        st.info("No logged picks yet.")
    else:
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)

        st.markdown("### CTG Review Date")

        review_date = st.date_input(
            "Date to review alongside Cleaning The Glass reports",
            value=yesterday,
            max_value=today,
        )

        review_str = review_date.strftime("%Y-%m-%d")
        review_logs = all_logs[all_logs["date"] == review_str]

        # Friendly Spread Covered label
        if "spread_covered" in review_logs.columns:
            review_logs = review_logs.copy()
            review_logs["Spread Covered"] = review_logs["spread_covered"].map(
                {1: "✅ Covered", 0: "❌ Missed", 2: "➖ Push"}
            )
        else:
            review_logs["Spread Covered"] = ""

        st.markdown(f"#### Logged picks for {review_str}")
        if review_logs.empty:
            st.info("No picks logged for this date.")
        else:
            # Show compact table for quick scan while reading CTG
            cols_to_show = [
                c for c in [
                    "favorite",
                    "underdog",
                    "vegas_line",
                    "edge",
                    "pick",
                    "confidence",
                    "final_score",
                    "Spread Covered",
                ] if c in review_logs.columns
            ]
            st.dataframe(review_logs[cols_to_show], use_container_width=True)

            # Optional: per-game expanders with more context
            st.markdown("##### Per-game breakdown")
    for _, row in review_logs.iterrows():
        fav = str(row["favorite"]).upper()
        dog = str(row["underdog"]).upper()
        date_str = str(row["date"])
        edge = row.get("edge", None)
        conf = row.get("confidence", "")
        pick = row.get("pick", "")
        fs = row.get("final_score", None)
        cov = row.get("spread_covered", None)

        title = f"{date_str} — {fav} vs {dog} | Pick: {pick}"
        if edge is not None:
            title += f" (Edge {edge:.2f}, {conf})"

        with st.expander(title):
            st.write(f"**Pick:** {pick}")
            st.write(f"**Vegas line at time:** {row['vegas_line']:+.1f}")
            if edge is not None:
                st.write(f"**Model edge (hybrid):** {edge:.2f} pts")
            st.write(f"**Confidence:** {conf}")

            # Players who were out at log time (from logs.fav_out / logs.dog_out)
            fav_out = row.get("fav_out", None)
            dog_out = row.get("dog_out", None)
            outs_parts = []
            if fav_out:
                outs_parts.append(f"{fav} outs: {fav_out}")
            if dog_out:
                outs_parts.append(f"{dog} outs: {dog_out}")
            if outs_parts:
                st.write("**Players out at log time:** " + " | ".join(outs_parts))

            if fs:
                if cov == 1:
                    st.markdown(f"**Result:** ✅ Covered — Final score: {fs}")
                elif cov == 0:
                    st.markdown(f"**Result:** ❌ Missed — Final score: {fs}")
                elif cov == 2:
                    st.markdown(f"**Result:** ➖ Push - Final Score: {fs}")
                else:
                    st.markdown(f"**Result:** Final score: {fs}")
            else:
                st.markdown("**Result:** Final score not logged yet.")

            # --- CTG-based auto analysis only (no manual notes) ---

            # Build key for this game (date + team pair)
            try:
                row_date = datetime.date.fromisoformat(date_str)
            except Exception:
                row_date = None

            ctg_entry = None
            if row_date is not None:
                game_key = (row_date, frozenset({fav, dog}))
                ctg_entry = ctg_docs.get(game_key)

            if ctg_entry:
                ctg_text_for_row = ctg_entry["text"]
                source_name = ctg_entry["name"]
                st.markdown(f"_CTG PDF linked_: `{source_name}`")

                existing_reason = row.get("ctg_reason") or ""
                existing_summary = row.get("ctg_summary") or ""

                if existing_reason or existing_summary:
                    st.write(f"_Last saved reason_: `{existing_reason}`")
                    if existing_summary:
                        st.write(existing_summary)

                if st.button(
                    "Analyze with CTG",
                    key=f"analyze_ctg_{int(row['id'])}",
                ):
                    reason, summary = analyze_game_with_ctg(row, ctg_text_for_row)
                    if reason or summary:
                        with engine.begin() as conn2:
                            conn2.execute(
                                text(
                                    """
                                    UPDATE logs
                                    SET ctg_reason = :reason,
                                        ctg_summary = :summary
                                    WHERE id = :id
                                    """
                                ),
                                {
                                    "reason": reason,
                                    "summary": summary,
                                    "id": int(row["id"]),
                                },
                            )
                        st.success("CTG analysis saved.")
                        st.write(f"_Latest reason_: `{reason}`")
                        if summary:
                            st.write(summary)
                    else:
                        st.warning("Could not derive a CTG-based reason.")
            else:
                st.caption(
                    "No CTG PDF linked for this matchup. "
                    "Upload a file named like 'MM_DD_YY ORL @ PHI' that matches this date and teams."
                )

                    

    # 👇 DEDENTED: runs once per date, not per row
    # Quick export for CTG day if you want it in Excel
    csv_bytes = review_logs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download this date's picks as CSV",
         data=csv_bytes,
         file_name=f"nba_edge_picks_{review_str}.csv",
         mime="text/csv",
         key=f"download_logs_csv_{review_str}",
    )

    st.markdown("### Today's Logged Picks")

    today_str = today.strftime("%Y-%m-%d")
    today_logs = all_logs[all_logs["date"] == today_str]

    if today_logs.empty:
        st.info("No picks logged yet today.")
    else:
            if "spread_covered" in today_logs.columns:
                today_logs = today_logs.copy()
                today_logs["Spread Covered"] = today_logs["spread_covered"].map(
                    {1: "✅ Covered", 0: "❌ Missed", 2: "➖ Push"}
                )
            else:
                today_logs["Spread Covered"] = ""

            cols_to_show_today = [
                c for c in [
                    "favorite",
                    "underdog",
                    "vegas_line",
                    "edge",
                    "pick",
                    "confidence",
                    "Spread Covered",
                ] if c in today_logs.columns
            ]
            st.dataframe(today_logs[cols_to_show_today], use_container_width=True)



# ============================================
# PREDICTION LOG VIEWER
# ============================================
with tab_logs:
    st.subheader("Logged Predictions")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("🔄 Refresh Final Scores from Odds API"):
            update_final_scores_from_odds()
            st.rerun()

    with col_btn2:
        if st.button("🗑️ Delete Last Logged Pick"):
            logs_df = load_all_logs()
            if logs_df is None or logs_df.empty:
                st.info("No logged picks to delete.")
            else:
                last_id = int(logs_df["id"].max())
                delete_pick_by_id(last_id)
                # Clear cached logs so Single Game / Slate / Prediction Log all reload fresh
                try:
                    load_all_logs.clear()
                except Exception:
                    pass
                st.success("Deleted last logged pick.")
                st.rerun()

    try:
        logs_df = load_all_logs()
        logs = logs_df.sort_values("id", ascending=False).reset_index(drop=True)
        if logs.empty:
            st.info("No logs yet.")
        else:
            # -------- MAIN TABLE WITH DATE SEPARATORS --------
            display_logs = logs.copy()

            # Friendly "Spread Covered" col
            if "spread_covered" in display_logs.columns:
                display_logs["Spread Covered"] = display_logs["spread_covered"].map(
                    {1: "✅ Covered", 0: "❌ Missed", 2: "➖ Push"}
                )
            else:
                display_logs["Spread Covered"] = ""

            # Rename final_score for table
            if "final_score" in display_logs.columns:
                display_logs.rename(columns={"final_score": "Final score"}, inplace=True)

            # Drop internal columns, including vegas_line and id
            for col in [
                "id",
                "vegas_line",
                "spread_covered",
                "fav_pace",
                "dog_pace",
                "fav_ortg",
                "dog_ortg",
                "fav_drtg",
                "dog_drtg",
                "fav_netr",
                "dog_netr",
                "cheat_edge",
                "cheat_pick",
                "models_aligned",
                "hybrid_pick",
                "stat_margin",
                "team_margin",
                "player_adj_eff",
                "pace_adj_term",
                "rest_adj_term",
                "b2b_adj_term",
                "vegas_margin",
                "hybrid_margin",
                "effective_edge",
                "ctg_notes",
                "ctg_reason",
                "ctg_summary",
            ]:
                if col in display_logs.columns:
                    display_logs = display_logs.drop(columns=[col])

            # Columns we actually want in the table
            cols_for_table = [
                "date",
                "favorite",
                "underdog",
                "edge",
                "pick",
                "confidence",
                "Spread Covered",
                "Final score",
            ]
            cols_for_table = [c for c in cols_for_table if c in display_logs.columns]
            display_logs = display_logs[cols_for_table]

            # Sort by date (desc)
            display_logs_sorted = display_logs.sort_values(
                by=["date"], ascending=[False]
            )

            # Build a new DataFrame with bold header rows per date
            rows = []
            for date_val, group in display_logs_sorted.groupby("date", sort=False):
                # Header row for this date
                header_row = {col: "" for col in display_logs_sorted.columns}
                header_row["date"] = str(date_val)
                header_row["_is_header"] = True
                # Make sure edge is truly empty so formatter skips it
                header_row["edge"] = None
                rows.append(header_row)

                # Actual game rows
                for _, r in group.iterrows():
                    d = r.to_dict()
                    d["_is_header"] = False
                    rows.append(d)

            table_df = pd.DataFrame(rows)

            # --- Build header mask, then drop helper column ---
            header_mask = (
                table_df["_is_header"].fillna(False).astype(bool)
                if "_is_header" in table_df.columns
                else pd.Series(False, index=table_df.index)
            )

            if "_is_header" in table_df.columns:
                table_df = table_df.drop(columns=["_is_header"])

            # Robust formatter for edge (handles None/strings)
            format_dict = {}
            if "edge" in table_df.columns:
                def fmt_edge(x):
                    try:
                        if pd.isna(x):
                            return ""
                        return f"{float(x):.2f}"
                    except Exception:
                        return str(x)
                format_dict["edge"] = fmt_edge

            # Styling: bold header rows using the mask
            def style_rows(row):
                # row.name is the index in table_df, which matches header_mask index
                if header_mask.loc[row.name]:
                    return [
                        "font-weight: bold; background-color: #222222; color: #ffffff;"
                    ] * len(row)
                return [""] * len(row)

            styled_table = (
                table_df.style
                .apply(style_rows, axis=1)
                .format(format_dict)
            )

            st.dataframe(styled_table, use_container_width=True, hide_index=True)

            st.subheader("Delete a Logged Pick")

            if logs_df is None or logs_df.empty:
                st.info("No logged picks available to delete.")
            else:
                df_for_select = logs_df.copy()

                # Sort newest-to-oldest so the latest pick is first in the dropdown
                if "id" in df_for_select.columns:
                    df_for_select = df_for_select.sort_values("id", ascending=False)
                elif "date" in df_for_select.columns:
                    df_for_select = df_for_select.sort_values("date", ascending=False)

                def _build_label(row):
                    date_str = str(row.get("date", ""))
                    fav = row.get("favorite", "")
                    dog = row.get("underdog", "")
                    pick = row.get("pick", "")
                    edge = row.get("edge", None)
                    edge_str = (
                        f"{edge:.2f}" if isinstance(edge, (int, float)) else str(edge)
                    )
                    return (
                        f"[id {row['id']}] {date_str} – {fav} vs {dog} – "
                        f"Pick: {pick} – Edge: {edge_str}"
                    )

                df_for_select["label"] = df_for_select.apply(_build_label, axis=1)

                # Default to the most recent pick
                default_index = 0 if len(df_for_select) > 0 else None

                selected_label = st.selectbox(
                    "Select a pick to delete:",
                    options=df_for_select["label"].tolist(),
                    index=default_index,
                )

                selected_row = df_for_select[
                    df_for_select["label"] == selected_label
                ].iloc[0]
                selected_id = int(selected_row["id"])

                st.write("You selected:", selected_label)

                confirm_key = "delete_pick_confirm"
                reset_flag_key = "delete_pick_confirm_reset"

                # Before drawing the checkbox, check if we need to reset it
                if st.session_state.get(reset_flag_key, False):
                    # Remove the old checkbox state so it comes back unchecked
                    st.session_state.pop(confirm_key, None)
                    st.session_state[reset_flag_key] = False

                confirm = st.checkbox(
                    "I understand this will permanently delete this pick from the log, "
                    "Per-Game Detail, and CTG views.",
                    key=confirm_key,
                )

                if st.button("Delete selected pick", type="primary", disabled=not confirm):
                    delete_pick_by_id(selected_id)

                    # Clear cached logs if using st.cache_data
                    try:
                        load_all_logs.clear()
                    except Exception:
                        pass

                    # On the next rerun, clear the checkbox state
                    st.session_state[reset_flag_key] = True

                    st.success(f"Deleted pick with id {selected_id}.")

                    try:
                        st.rerun()
                    except Exception:
                        pass

            # -------- PER-GAME DETAILS, GROUPED BY DATE --------
            st.markdown("### 🔍 Per-Game Details (Pace / Ratings Tags)")

            # Sort logs by date desc, then id desc
            logs_sorted = logs.sort_values(
                by=["date", "id"], ascending=[False, False]
            )

            for date_val, day_group in logs_sorted.groupby("date", sort=False):
                with st.expander(f"{date_val} — {len(day_group)} logged pick(s)"):
                    # --- NEW: per-date CSV download ---
                    csv_day = day_group.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download this date as CSV",
                        data=csv_day,
                        file_name=f"logged_picks_{date_val}.csv",
                        mime="text/csv",
                        key=f"download_logs_{date_val}",
                    )   
                    for _, row in day_group.iterrows():
                        fav = str(row["favorite"]).upper()
                        dog = str(row["underdog"]).upper()
                        date = row["date"]
                        pick = row["pick"]
                        edge = row["edge"]
                        conf = row["confidence"]

                        fav_pace = get_team_metric(fav, ["Pace", "PACE"])
                        dog_pace = get_team_metric(dog, ["Pace", "PACE"])

                        fav_ortg = get_team_metric(
                            fav, ["ORtg", "ORTG", "OffRtg", "OFF_RTG"]
                        )
                        fav_drtg = get_team_metric(
                            fav, ["DRtg", "DRTG", "DefRtg", "DEF_RTG"]
                        )
                        fav_netr = get_team_metric(
                            fav, ["NetRtg", "NETRTG", "Base Net Rating", "Base net"]
                        )

                        dog_ortg = get_team_metric(
                            dog, ["ORtg", "ORTG", "OffRtg", "OFF_RTG"]
                        )
                        dog_drtg = get_team_metric(
                            dog, ["DRtg", "DRTG", "DefRtg", "DEF_RTG"]
                        )
                        dog_netr = get_team_metric(
                            dog, ["NetRtg", "NETRTG", "Base Net Rating", "Base net"]
                        )

                        title = (
                            f"{fav} vs {dog}  |  Pick: {pick}  "
                            f"(Edge {edge:.2f}, {conf})"
                        )

                        with st.expander(title):
                            st.write(f"**Logged pick:** {pick}")
                            st.write(f"**Date:** {date}")
                            if "vegas_line" in row and pd.notna(row["vegas_line"]):
                                st.write(f"**Vegas line at time:** {row['vegas_line']:+.1f}")
                            st.write(f"**Model edge:** {edge:.2f} pts")
                            st.write(f"**Confidence:** {conf}")

                            # Players who were out at log time (from logs.fav_out / logs.dog_out)
                            fav_out = row.get("fav_out", None)
                            dog_out = row.get("dog_out", None)
                            outs_parts = []
                            if fav_out:
                                outs_parts.append(f"{fav} outs: {fav_out}")
                            if dog_out:
                                outs_parts.append(f"{dog} outs: {dog_out}")
                            if outs_parts:
                                st.write("**Players out at log time:** " + " | ".join(outs_parts))

                            # Cheatsheet vs hybrid at log time
                            cheat_edge_row = row.get("cheat_edge", None)
                            cheat_pick_row = row.get("cheat_pick", None)
                            aligned_flag = row.get("models_aligned", None)
                            hybrid_pick_row = row.get("hybrid_pick", None)

                            if pd.notna(cheat_edge_row):
                                cheat_edge_val = float(cheat_edge_row)

                                if aligned_flag == 1:
                                    st.markdown(
                                        f"**Models aligned** at log time. "
                                        f"Cheatsheet: {cheat_pick_row} "
                                        f"(edge {cheat_edge_val:.2f})."
                                    )
                                else:
                                    st.markdown(
                                        f"**Models disagreed** at log time. "
                                        f"Cheatsheet: {cheat_pick_row} "
                                        f"(edge {cheat_edge_val:.2f}). "
                                        f"Hybrid model: {hybrid_pick_row}."
                                    )

                            # Result + which model side covered (when misaligned)
                            fs = row.get("final_score", None)
                            cov = row.get("spread_covered", None)

                            if fs and pd.notna(fs):
                                if cov == 1:
                                    st.markdown(f"**Result:** ✅ Covered — Final score: {fs}")
                                elif cov == 0:
                                    st.markdown(f"**Result:** ❌ Missed — Final score: {fs}")
                                elif cov == 2:
                                    st.markdown(f"**Result:** ➖ Push — Final score: {fs}")
                                else:
                                    st.markdown(f"**Result:** Final score: {fs}")

                                # Extra line ONLY when models disagreed and we know outcome
                                if (
                                    pd.notna(cheat_edge_row)
                                    and aligned_flag == 0
                                    and cov in (0, 1)
                                ):
                                    cheat_pick_str = str(cheat_pick_row) if pd.notna(cheat_pick_row) else ""
                                    hybrid_pick_str = str(hybrid_pick_row) if pd.notna(hybrid_pick_row) else ""
                                    logged_pick_str = str(pick)

                                    which_model = None
                                    # If our logged pick covered, whichever model matched logged pick "covered"
                                    if cov == 1:
                                        if logged_pick_str == cheat_pick_str:
                                            which_model = "Cheatsheet model side"
                                        elif logged_pick_str == hybrid_pick_str:
                                            which_model = "Hybrid model side"
                                    # If our logged pick missed but models disagreed,
                                    # the *other* side would have covered.
                                    elif cov == 0:
                                        if logged_pick_str == cheat_pick_str and hybrid_pick_str:
                                            which_model = "Hybrid model side would have covered"
                                        elif logged_pick_str == hybrid_pick_str and cheat_pick_str:
                                            which_model = "Cheatsheet model side would have covered"

                                    if which_model:
                                        st.markdown(f"**Model outcome note:** {which_model}.")

                            else:
                                st.markdown("**Result:** Final score not logged yet.")

                            st.markdown("**Team Metrics Snapshot (at time of view)**")

                            metrics_rows = [
                                {
                                    "Team": fav,
                                    "Role": "Favorite",
                                    "Pace": f"{fav_pace:.1f}"
                                    if fav_pace is not None
                                    else "—",
                                    "ORtg": f"{fav_ortg:.1f}"
                                    if fav_ortg is not None
                                    else "—",
                                    "DRtg": f"{fav_drtg:.1f}"
                                    if fav_drtg is not None
                                    else "—",
                                    "NetRtg": f"{fav_netr:.1f}"
                                    if fav_netr is not None
                                    else "—",
                                },
                                {
                                    "Team": dog,
                                    "Role": "Underdog",
                                    "Pace": f"{dog_pace:.1f}"
                                    if dog_pace is not None
                                    else "—",
                                    "ORtg": f"{dog_ortg:.1f}"
                                    if dog_ortg is not None
                                    else "—",
                                    "DRtg": f"{dog_drtg:.1f}"
                                    if dog_drtg is not None
                                    else "—",
                                    "NetRtg": f"{dog_netr:.1f}"
                                    if dog_netr is not None
                                    else "—",
                                },
                            ]

                            st.table(pd.DataFrame(metrics_rows))

                            st.caption(
                                "These tags are pulled from the current `Team_ratings.csv` "
                                "(not historically frozen yet). Later we can upgrade this "
                                "to snapshot the values at the time of the bet for true backtesting."
                            )

    except Exception as e:
        st.warning(f"No log table yet or error reading logs: {e}")
