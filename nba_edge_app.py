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
REGION = "us"
MARKETS = "spreads"

K5 = 8.0
STRONG_EDGE = 6.0
MEDIUM_EDGE = 2.5

# Hybrid weights (can tune later or learn from history)
W_SCORE = 0.5      # manual/auto score projection margin
W_TEAM = 0.3       # team strength margin
W_PLAYER = 0.2     # player on/off adjustment

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

# =========================
# DB Setup
# =========================
engine = create_engine(f"sqlite:///{DB_NAME}", echo=False)
with engine.connect() as conn:
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

# =========================
# Data Load
# =========================
players_df = pd.read_csv("players_onoff.csv")
players_df["Team"] = players_df["Team"].str.upper()

# Team ratings file (from your path: Team_ratings.csv)
try:
    team_ratings_df = pd.read_csv("Team_ratings.csv")
    team_ratings_df["Team"] = team_ratings_df["Team"].str.upper()
    TEAM_RATINGS_AVAILABLE = True
except Exception:
    TEAM_RATINGS_AVAILABLE = False


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


# =========================
# Odds API Fetch
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
                    # API is UTC; convert to ET
                    commence_dt = datetime.datetime.fromisoformat(
                        commence_str.replace("Z", "+00:00")
                    ).astimezone(ET_TZ)
                except Exception:
                    commence_dt = None

            completed = ev.get("completed", False)

            # Determine status
            if completed:
                status = "FINAL"
            elif commence_dt and now_et >= commence_dt:
                status = "LIVE"
            else:
                status = "UPCOMING"

            # Get spread market
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

            # Try to get final score if available (for FINAL status)
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

            # Time label in ET
            if commence_dt:
                time_label = commence_dt.strftime("%I:%M %p").lstrip("0") + " ET"
            else:
                time_label = "TBD"

            # Build display string
            if status == "LIVE":
                display = f"🔥 LIVE — {away_full} @ {home_full} | {fav_abbr} {fav_spread:.1f}"
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
            else:  # UPCOMING
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

        # Sort: LIVE first, then UPCOMING by time, then FINAL
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


# =========================
# UI Layout
# =========================
st.set_page_config(page_title="NBA Edge Analyzer", layout="wide")
st.title("🏀 NBA Spread Edge Analyzer")

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
            f"Status: **{game_status}** — {away_full} @ {home_full}, line: {favorite_abbr} {vegas_line:.1f}"
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

    st.subheader("Projected Scores (Stat Model)")
    proj_fav = st.number_input("Projected Score (Favorite)", step=1.0, value=110.0)
    proj_dog = st.number_input("Projected Score (Underdog)", step=1.0, value=104.0)

    st.subheader("Player Availability (On/Off Impact)")

    home_players = players_df[players_df["Team"] == home_abbr]["Player"].unique()
    away_players = players_df[players_df["Team"] == away_abbr]["Player"].unique()

    home_out, away_out = [], []
    colH, colA2 = st.columns(2)

    with colH:
        st.markdown(f"**Home Out: {home_abbr}**")
        for i in range(5):
            choice = st.selectbox(
                f"Out Player {i+1}", ["None"] + list(home_players), key=f"home_out_{i}"
            )
            if choice != "None":
                home_out.append(choice)

    with colA2:
        st.markdown(f"**Away Out: {away_abbr}**")
        for i in range(5):
            choice = st.selectbox(
                f"Out Player {i+1}", ["None"] + list(away_players), key=f"away_out_{i}"
            )
            if choice != "None":
                away_out.append(choice)

    compute_btn = st.button("Compute Edge")

    if compute_btn:
        # ===============================
        # Basic values
        # ===============================
        vegas_margin = -vegas_line  # favorite margin implied by line
        stat_margin = proj_fav - proj_dog  # positive means favorite wins in projection

        # ===============================
        # Player Impact
        # ===============================
        player_adj = 0.0

        # Home outs
        for p in home_out:
            row = players_df[
                (players_df["Player"] == p) & (players_df["Team"] == home_abbr)
            ]
            if not row.empty:
                diff_val = float(row["Diff"].iloc[0])
                # If favorite is home, losing a good player hurts them
                player_adj += -diff_val if favorite.upper() == home_abbr else diff_val

        # Away outs
        for p in away_out:
            row = players_df[
                (players_df["Player"] == p) & (players_df["Team"] == away_abbr)
            ]
            if not row.empty:
                diff_val = float(row["Diff"].iloc[0])
                player_adj += -diff_val if favorite.upper() == away_abbr else diff_val

        # Optionally apply the curved edge logic to player_adj as "effective player impact"
        abs_player = abs(player_adj)
        if abs_player == 0:
            effective_player_adj = 0.0
        else:
            effective_player_adj = abs_player + (
                abs_player * (abs_player / (abs_player + (K5 * 2)))
            )
            # keep sign
            effective_player_adj = (
                -effective_player_adj if player_adj < 0 else effective_player_adj
            )

        # ===============================
        # Team Strength (from ratings)
        # ===============================
        if TEAM_RATINGS_AVAILABLE:
            home_power = get_team_power(home_abbr, is_home=True)
            away_power = get_team_power(away_abbr, is_home=False)
        else:
            home_power = None
            away_power = None

        if (home_power is not None) and (away_power is not None):
            # team_margin: from home perspective
            team_margin_home = home_power - away_power
            # from favorite perspective
            if favorite.upper() == home_abbr:
                fav_team_margin = team_margin_home
            else:
                fav_team_margin = -team_margin_home
        else:
            fav_team_margin = 0.0

        # ===============================
        # Situational Adjustments (pace / B2B placeholders)
        # ===============================
        pace_adj = 0.0  # TODO: wire to your daily file later
        b2b_adj = 0.0   # TODO: detect B2B and add penalty

        # ===============================
        # HYBRID MARGIN (single main model)
        # ===============================
        hybrid_margin = (
            stat_margin * W_SCORE
            + fav_team_margin * W_TEAM
            + effective_player_adj * W_PLAYER
            + pace_adj
            + b2b_adj
        )

        diff_hybrid = hybrid_margin - vegas_margin
        edge = abs(diff_hybrid)
        side = "favorite" if diff_hybrid > 0 else "underdog"
        conf = classify_edge(edge)
        final_pick = side_to_pick(side, favorite, underdog, vegas_line)

        # ===============================
        # 3 columns: Hybrid, Player Detail, Team Detail
        # ===============================
        model_cols = st.columns(3)

        with model_cols[0]:
            st.markdown("### Hybrid Model (Score + Team + Players)")
            st.metric("Stat Margin (fav - dog)", round(stat_margin, 2))
            st.metric("Hybrid Margin vs Dog", round(hybrid_margin, 2))
            st.metric("Edge vs Line (pts)", round(edge, 2))
            st.metric("Hybrid Pick", final_pick)

        with model_cols[1]:
            st.markdown("### Player Impact Detail")
            st.write(f"Raw Player Adj (fav perspective): **{player_adj:.2f}**")
            st.write(f"Effective Player Adj (curved): **{effective_player_adj:.2f}**")
            st.write(f"Home outs: {', '.join(home_out) if home_out else 'None'}")
            st.write(f"Away outs: {', '.join(away_out) if away_out else 'None'}")

        with model_cols[2]:
            st.markdown("### Team Strength Detail")
            if (home_power is not None) and (away_power is not None):
                st.write(f"Home Team Power: **{home_power:.2f}**")
                st.write(f"Away Team Power: **{away_power:.2f}**")
                st.write(f"Team Margin (fav - dog): **{fav_team_margin:.2f}**")
            else:
                st.write("No team ratings found for one or both teams.")
            st.write(f"Pace Adj (placeholder): **{pace_adj:.2f}**")
            st.write(f"B2B Adj (placeholder): **{b2b_adj:.2f}**")

        # ===============================
        # FINAL DECISION (full width)
        # ===============================
        color = confidence_to_color(conf)

        st.markdown("---")
        st.markdown("## 📌 Final Model Decision")
        st.markdown(
            f"""
            <div style='margin-top:10px; padding:15px;
                        background-color:#111; border-radius:8px;
                        border:1px solid {color}; text-align:center;'>
                <div style='font-size:26px; font-weight:bold; color:{color};'>
                    {final_pick}
                </div>
                <div style='font-size:18px; color:white; margin-top:4px;'>
                    Edge vs line: {edge:.2f} pts — Confidence: {conf}
                </div>
                <div style='font-size:13px; color:#ccc; margin-top:8px; text-align:left;'>
                    <b>Breakdown (fav perspective):</b><br/>
                    • Stat projection margin: {stat_margin:.2f}<br/>
                    • Team strength margin: {fav_team_margin:.2f}<br/>
                    • Player impact (effective): {effective_player_adj:.2f}<br/>
                    • Pace adjustment: {pace_adj:.2f}<br/>
                    • B2B / situational adjustment: {b2b_adj:.2f}<br/>
                    • Hybrid margin vs dog: {hybrid_margin:.2f}<br/>
                    • Vegas implied margin: {vegas_margin:.2f}
                </div>
                <div style='font-size:12px; color:#888; margin-top:6px;'>
                    Game status: {game_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # store in session for logging
        st.session_state.update(
            {
                "favorite": favorite,
                "underdog": underdog,
                "final_pick": final_pick,
                "final_edge": edge,
                "final_conf": conf,
                "vegas_line": vegas_line,
            }
        )

    # ---------- LOG BUTTON ----------
    if st.button("💾 Log This Pick"):
        if "final_pick" not in st.session_state:
            st.warning("Compute first before logging.")
        else:
            entry = pd.DataFrame(
                [
                    {
                        "date": datetime.date.today().strftime("%Y-%m-%d"),
                        "favorite": st.session_state["favorite"],
                        "underdog": st.session_state["underdog"],
                        "vegas_line": float(st.session_state["vegas_line"]),
                        "model_line": 0.0,
                        "edge": float(st.session_state["final_edge"]),
                        "pick": st.session_state["final_pick"],
                        "confidence": st.session_state["final_conf"],
                    }
                ]
            )
            with engine.begin() as conn:
                entry.to_sql("logs", conn, if_exists="append", index=False)
            st.success("Logged successfully!")

# ============================================
# PREDICTION LOG VIEWER
# ============================================
with tab_logs:
    st.subheader("Logged Predictions")
    try:
        logs = pd.read_sql("SELECT * FROM logs ORDER BY id DESC", engine)
        if logs.empty:
            st.info("No logs yet.")
        else:
            st.dataframe(logs, use_container_width=True)
    except Exception as e:
        st.warning(f"No log table yet or error reading logs: {e}")
