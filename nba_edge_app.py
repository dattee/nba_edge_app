import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text

# =========================
# CONFIG
# =========================
DB_NAME = "model_logs.db"

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
    Read the 'SCOREBOARD' tab where:
      - Column B = team abbr (away on top row, home underneath)
      - Column AC = projected score for that team
    We turn this into a clean DataFrame with:
      away_abbr, home_abbr, proj_away, proj_home
    """
    # Try SCOREBOARD sheet by name; fall back to first sheet if needed
    try:
        raw = pd.read_excel(file, sheet_name="SCOREBOARD", header=None)
    except ValueError:
        raw = pd.read_excel(file, sheet_name=0, header=None)

    team_col_idx = 1   # column B
    proj_col_idx = 28  # column AC (0-based index)

    rows = []
    for i in range(len(raw)):
        # Team cell
        team = raw.iat[i, team_col_idx] if team_col_idx < raw.shape[1] else None
        # Projected score cell
        proj = raw.iat[i, proj_col_idx] if proj_col_idx < raw.shape[1] else None

        if isinstance(team, str):
            team_str = team.strip().upper()
        else:
            team_str = None

        if not team_str:
            continue

        # Filter out non-team garbage (e.g., "GAME 1", headers, etc.)
        if len(team_str) > 4:
            continue

        # Need a numeric projected score
        try:
            proj_val = float(proj)
        except (TypeError, ValueError):
            continue

        rows.append({"team": team_str, "proj": proj_val})

    # Group into pairs: first row = AWAY, second row = HOME
    games = []
    pending = None
    for r in rows:
        if pending is None:
            pending = r
        else:
            away = pending
            home = r
            games.append(
                {
                    "away_abbr": away["team"],
                    "home_abbr": home["team"],
                    "proj_away": away["proj"],
                    "proj_home": home["proj"],
                }
            )
            pending = None

    return pd.DataFrame(games)


def lookup_cheatsheet_projection_for_game(
    cheatsheet_df: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    favorite_abbr: str,
    underdog_abbr: str,
):
    """
    Given:
      - cheatsheet_df with cols: away_abbr, home_abbr, proj_away, proj_home
      - home_abbr / away_abbr from Odds API / manual entry
      - favorite_abbr / underdog_abbr from the current game
    Return (proj_fav, proj_dog) according to who is favorite.
    """
    required_cols = {"away_abbr", "home_abbr", "proj_away", "proj_home"}
    if not required_cols.issubset(cheatsheet_df.columns) or cheatsheet_df.empty:
        # Cheatsheet not parsed correctly or no games detected
        return None, None

    home_abbr = home_abbr.upper()
    away_abbr = away_abbr.upper()
    favorite_abbr = favorite_abbr.upper()
    underdog_abbr = underdog_abbr.upper()

    sub = cheatsheet_df[
        (cheatsheet_df["home_abbr"].str.upper() == home_abbr)
        & (cheatsheet_df["away_abbr"].str.upper() == away_abbr)
    ]

    if sub.empty:
        return None, None

    row = sub.iloc[0]
    proj_home = float(row["proj_home"])
    proj_away = float(row["proj_away"])

    if favorite_abbr == home_abbr:
        proj_fav = proj_home
        proj_dog = proj_away
    else:
        proj_fav = proj_away
        proj_dog = proj_home

    return proj_fav, proj_dog




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
    """
    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": days_from,
    }
    try:
        r = requests.get(SCORES_API_URL, params=params)
        r.raise_for_status()
        data = r.json()
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
def update_final_scores_from_odds():
    """Fill in final_score + spread_covered for logged games using Scores API."""
    games = fetch_scores(days_from=3)
    if not games:
        st.warning("No scores data available to refresh.")
        return

    # Build lookup by (date, sorted team pair)
    index = {}
    for g in games:
        if not g["commence_dt"]:
            continue
        game_date = g["commence_dt"].date()
        team_pair = tuple(sorted([g["home_abbr"], g["away_abbr"]]))
        index.setdefault((game_date, team_pair), []).append(g)

    with engine.begin() as conn:
        logs_df = pd.read_sql("SELECT * FROM logs", conn)

        for _, row in logs_df.iterrows():
            try:
                row_date = datetime.date.fromisoformat(str(row["date"]))
            except Exception:
                continue

            fav = str(row["favorite"]).upper()
            dog = str(row["underdog"]).upper()
            vegas_line = float(row["vegas_line"])
            pick = str(row["pick"])

            key = (row_date, tuple(sorted([fav, dog])))
            candidates = index.get(key, [])

            # Try +/- 1 day if nothing at exact date
            if not candidates:
                for delta in (-1, 1):
                    key_alt = (
                        row_date + datetime.timedelta(days=delta),
                        tuple(sorted([fav, dog])),
                    )
                    if key_alt in index:
                        candidates = index[key_alt]
                        break

            if not candidates:
                continue

            # Prefer completed with valid scores
            event = None
            for g in candidates:
                if (
                    g["completed"]
                    and g["home_score"] is not None
                    and g["away_score"] is not None
                ):
                    event = g
                    break
            if event is None:
                continue

            home_abbr = event["home_abbr"]
            away_abbr = event["away_abbr"]
            home_score = int(event["home_score"])
            away_score = int(event["away_score"])

            if fav == home_abbr:
                fav_score = home_score
                dog_score = away_score
            elif fav == away_abbr:
                fav_score = away_score
                dog_score = home_score
            else:
                # Teams don't match our fav/dog pair
                continue

            margin_fav = fav_score - dog_score

            parts = pick.split()
            if len(parts) >= 2:
                pick_team = parts[0].upper()
            else:
                pick_team = fav

            bet_on_favorite = pick_team == fav
            line_abs = abs(vegas_line)

            if bet_on_favorite:
                covered = 1 if margin_fav > line_abs else 0
            else:
                covered = 1 if margin_fav < line_abs else 0

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


# =========================
# UI Layout
# =========================
st.set_page_config(page_title="NBA Edge Analyzer", layout="wide")
st.title("NBA Spread Edge Analyzer")

st.markdown(
    "<div style='text-align:right;font-size:12px;'>Made by <b>Calvin Thuong</b></div>",
    unsafe_allow_html=True,
)

tab_single, tab_logs = st.tabs(["Single Game", "Prediction Log"])

# ============================================
# SINGLE GAME ANALYZER
# ============================================
with tab_single:
    st.subheader("Game Selection (with Live Odds, ET)")

    games_data = fetch_games_with_odds()

    manual_mode = False
    if not games_data:
        st.warning("⚠ Odds API unavailable — switching to manual mode.")
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

    # Situational flags (Back-to-Back)
    st.subheader("Situational Flags (B2B + Schedule)")
    col_flag1, col_flag2 = st.columns(2)
    with col_flag1:
         b2b_home = st.checkbox(f"{home_abbr} B2B", value=False)
         home_games_last5 = st.number_input(
             f"{home_abbr} games last 5 days",
             min_value=0,
             max_value=5,
             value=2,
             step=1,
         )
    with col_flag2:
         b2b_away = st.checkbox(f"{away_abbr} B2B", value=False)
         away_games_last5 = st.number_input(
             f"{away_abbr} games last 5 days",
             min_value=0,
             max_value=5,
             value=2,
             step=1,
         )

    st.subheader("Spread + Score Inputs")

    colA, colB, colC = st.columns(3)
    with colA:
        favorite = st.text_input("Favorite Team (abbr)", value=favorite_abbr)
    with colB:
        underdog = st.text_input("Underdog Team (abbr)", value=underdog_abbr)
    with colC:
        vegas_line = st.number_input(
            "Vegas Spread (favorite only)", step=0.5, value=float(vegas_line)
        )
        
    # =========================
    # Cheatsheet Import (Optional)
    # =========================
    st.subheader("Cheatsheet Import (Optional)")

    uploaded_cheatsheet = st.file_uploader(
        "Upload cheatsheet Excel (SCOREBOARD tab)",
        type=["xlsx"],
        key="cheatsheet_upload",
    )

    if uploaded_cheatsheet is not None:
        try:
            cheatsheet_games = load_cheatsheet_from_scoreboard(uploaded_cheatsheet)
            st.session_state["cheatsheet_df"] = cheatsheet_games
            st.caption(
                f"Cheatsheet loaded from SCOREBOARD tab "
                f"for {len(cheatsheet_games)} games (away/home + projections)."
            )
            # Optional preview
            st.dataframe(cheatsheet_games.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading cheatsheet: {e}") 

    # Try to pull projections for this specific game from the loaded cheatsheet
    cheat_pf = cheat_pd = None
    cheatsheet_df = st.session_state.get("cheatsheet_df")

    if cheatsheet_df is not None:
        cheat_pf, cheat_pd = lookup_cheatsheet_projection_for_game(
            cheatsheet_df,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            favorite_abbr=favorite,
            underdog_abbr=underdog,
        )

    if (cheat_pf is not None) and (cheat_pd is not None):
        st.info(
            f"Cheatsheet projections found: {favorite} {cheat_pf:.1f} — "
            f"{underdog} {cheat_pd:.1f}"
        )
        if st.button("Use cheatsheet projections", key="use_cheatsheet_proj"):
            st.session_state["proj_fav"] = float(cheat_pf)
            st.session_state["proj_dog"] = float(cheat_pd)        

    st.subheader("Projected Scores (Stat Model)")

    # Favorite
    if "proj_fav" in st.session_state:
        proj_fav = st.number_input(
            f"Projected Score ({favorite})",
            step=1.0,
            key="proj_fav",
        )
    else:
        proj_fav = st.number_input(
            f"Projected Score ({favorite})",
            step=1.0,
            value=110.0,
            key="proj_fav",
        )

    # Underdog
    if "proj_dog" in st.session_state:
        proj_dog = st.number_input(
            f"Projected Score ({underdog})",
            step=1.0,
            key="proj_dog",
        )
    else:
        proj_dog = st.number_input(
            f"Projected Score ({underdog})",
            step=1.0,
            value=104.0,
            key="proj_dog",
        )

    st.subheader("Player Availability (On/Off Impact)")

    # Filter player list to exclude long-term absences
    home_players_df = players_df[players_df["Team"] == home_abbr].copy()
    away_players_df = players_df[players_df["Team"] == away_abbr].copy()

    if "DaysSinceLastGame" in home_players_df.columns:
        mask_home = (
            home_players_df["DaysSinceLastGame"].isna()
            | (home_players_df["DaysSinceLastGame"] <= MAX_DAYS_ABSENT)
        )
        home_players_df = home_players_df[mask_home]

    if "DaysSinceLastGame" in away_players_df.columns:
        mask_away = (
            away_players_df["DaysSinceLastGame"].isna()
            | (away_players_df["DaysSinceLastGame"] <= MAX_DAYS_ABSENT)
        )
        away_players_df = away_players_df[mask_away]

    home_players = home_players_df["Player"].unique()
    away_players = away_players_df["Player"].unique()

    home_out, away_out = [], []
    colH, colA2 = st.columns(2)

    with colH:
        st.markdown(f"**Home Out: {home_abbr}**")
        for i in range(5):
            choice = st.selectbox(
                f"Out Player {i+1}",
                ["None"] + list(home_players),
                key=f"home_out_{i}",
            )
            if choice != "None":
                home_out.append(choice)

    with colA2:
        st.markdown(f"**Away Out: {away_abbr}**")
        for i in range(5):
            choice = st.selectbox(
                f"Out Player {i+1}",
                ["None"] + list(away_players),
                key=f"away_out_{i}",
            )
            if choice != "None":
                away_out.append(choice)

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
        player_adj = 0.0
        for p in home_out:
            row = players_df[
                (players_df["Player"] == p) & (players_df["Team"] == home_abbr)
            ]
            if not row.empty:
                diff_val = float(row["Diff"].iloc[0])
                player_adj += -diff_val if favorite.upper() == home_abbr else diff_val

        for p in away_out:
            row = players_df[
                (players_df["Player"] == p) & (players_df["Team"] == away_abbr)
            ]
            if not row.empty:
                diff_val = float(row["Diff"].iloc[0])
                player_adj += -diff_val if favorite.upper() == away_abbr else diff_val

        abs_player = abs(player_adj)
        if abs_player == 0:
            effective_player_adj = 0.0
        else:
            effective_player_adj = abs_player + (
                abs_player * (abs_player / (abs_player + (K5 * 2)))
            )
            effective_player_adj = (
                -effective_player_adj if player_adj < 0 else effective_player_adj
            )

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

        # --- B2B + schedule density ---
        B2B_PENALTY = 1.5
        b2b_adj = 0.0
        b2b_desc_parts: list[str] = []

        if b2b_home:
            delta_home = -B2B_PENALTY if favorite.upper() == home_abbr else B2B_PENALTY
            b2b_adj += delta_home
            b2b_desc_parts.append(f"{home_abbr} {delta_home:+.2f}")

        if b2b_away:
            delta_away = -B2B_PENALTY if favorite.upper() == away_abbr else B2B_PENALTY
            b2b_adj += delta_away
            b2b_desc_parts.append(f"{away_abbr} {delta_away:+.2f}")

        b2b_desc = ", ".join(b2b_desc_parts) if b2b_desc_parts else "None"

        rest_adj = 0.0
        rest_desc = "None"
        try:
            fav_is_home = favorite.upper() == home_abbr

            home_games = float(home_games_last5)
            away_games = float(away_games_last5)

            if fav_is_home:
                fav_games = home_games
                dog_games = away_games
            else:
                fav_games = away_games
                dog_games = home_games

            games_diff = fav_games - dog_games  # >0 fav more taxed

            if games_diff != 0:
                rest_adj = -games_diff * REST_GAME_WEIGHT
                if abs(games_diff) >= 3:
                    rest_desc = f"{int(fav_games)}–{int(dog_games)} (heavy rest edge)"
                else:
                    rest_desc = f"{int(fav_games)}–{int(dog_games)}"
        except Exception:
            rest_adj = 0.0
            rest_desc = "N/A"

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
                "effective_player_adj": float(effective_player_adj),
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
            st.write(f"Raw Player Adj (fav perspective): **{player_adj:.2f}**")
            st.write(f"Effective Player Adj (curved): **{effective_player_adj:.2f}**")
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

        # Log button (only here, only in Single Game tab)
        if st.button("💾 Log This Pick"):
            logged_pick = pick_to_log

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
                        "cheat_edge": cheat_edge,
                        "cheat_pick": cheat_pick,
                        "models_aligned": 1 if aligned else 0,
                        "hybrid_pick": hybrid_pick,
                        
                        # NEW: components for weight learning
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

            st.success("Logged!")
            st.session_state["has_decision"] = False
            st.session_state["decision_logged"] = True
            st.rerun()




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
            with engine.begin() as conn:
                conn.execute(text(
                    "DELETE FROM logs WHERE id = (SELECT MAX(id) FROM logs)"
                ))
            st.success("Last logged pick deleted.")
            st.rerun()

    #with col_btn3:
     #   if st.button("🔥 Clear ALL Logs"):
      #      with engine.begin() as conn:
       #         conn.execute(text("DELETE FROM logs"))
        #    st.success("All logs cleared.")
         #   st.rerun()

    try:
        logs = pd.read_sql("SELECT * FROM logs ORDER BY id DESC", engine)
        if logs.empty:
            st.info("No logs yet.")
        else:
            display_logs = logs.copy()

            # Friendly "Spread Covered" col
            if "spread_covered" in display_logs.columns:
                display_logs["Spread Covered"] = display_logs["spread_covered"].map(
                    {1: "✅ Covered", 0: "❌ Missed"}
                )
            else:
                display_logs["Spread Covered"] = ""

            if "final_score" in display_logs.columns:
                display_logs.rename(columns={"final_score": "Final score"}, inplace=True)

            # Hide internal columns we don't want in main table
            for col in [
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
                "model_line",
            ]:
                if col in display_logs.columns:
                    display_logs = display_logs.drop(columns=[col])

            st.dataframe(display_logs, use_container_width=True)

            st.markdown("### 🔍 Per-Game Details (Pace / Ratings Tags)")

            for _, row in logs.iterrows():
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

                title = f"{date} — {fav} vs {dog}  |  Pick: {pick}  (Edge {edge:.2f}, {conf})"

                with st.expander(title):
                    st.write(f"**Logged pick:** {pick}")
                    st.write(f"**Vegas line at time:** {row['vegas_line']:+.1f}")
                    st.write(f"**Model edge:** {edge:.2f} pts")
                    st.write(f"**Confidence:** {conf}")
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
                            
                    if "final_score" in row and pd.notna(row["final_score"]):
                        fs = row["final_score"]
                        cov = row.get("spread_covered", None)
                        if cov == 1:
                            st.markdown(f"**Result:** ✅ Covered — Final score: {fs}")
                        elif cov == 0:
                            st.markdown(f"**Result:** ❌ Missed — Final score: {fs}")
                        else:
                            st.markdown(f"**Result:** Final score: {fs}")
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

