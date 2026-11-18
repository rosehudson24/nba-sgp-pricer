import re
import unicodedata
from urllib.parse import quote_plus
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.stats import norm, gamma
from propline import PropLine

# ==========================================
# BRAND & API CONFIGURATION
# ==========================================
BRAND_NAME = "Hector's Picks"
BRAND_SUBTITLE = "NBA Prop Projections · Monte Carlo Matchup Simulator · Analytics Terminal"
PROPLINE_API_KEY = "b2858a5f3784220e7da1dee8d9916ab7"
pl_client = PropLine(PROPLINE_API_KEY)

st.set_page_config(
    page_title=f"{BRAND_NAME} — NBA Prop & Matchup Analytics",
    page_icon=":material/sports_basketball:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 2rem;
    max-width: 1340px;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }

.brand-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.85rem 1rem;
    background: #111622;
    border: 1px solid #1E2638;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.brand-left { display: flex; align-items: center; gap: 0.7rem; }
.brand-logo {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1rem;
    color: #06100C;
    letter-spacing: -0.5px;
}
.brand-name {
    font-size: 1.25rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.4px;
    margin: 0;
    line-height: 1.15;
}
.brand-sub {
    font-size: 0.75rem;
    color: #8B97AD;
    margin: 0.1rem 0 0 0;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.28);
    color: #34D399;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
}
.status-dot {
    width: 6px;
    height: 6px;
    background: #10B981;
    border-radius: 50%;
}

.table-scroll-wrap {
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

a.player-link {
    display: flex;
    align-items: center;
    gap: 8px;
    overflow: hidden;
    text-decoration: none !important;
    color: inherit;
    cursor: pointer;
}
a.player-link:hover .player-name-txt {
    color: #10B981;
    text-decoration: underline;
}
.headshot-wrap {
    position: relative;
    width: 30px;
    height: 30px;
    flex-shrink: 0;
}
.headshot-img {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    object-fit: cover;
    background: #1E2638;
    border: 1px solid #2D3952;
}
.team-mini-logo {
    position: absolute;
    bottom: -2px;
    right: -3px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #0F1420;
    padding: 1px;
    object-fit: contain;
    border: 1px solid #263147;
}
.team-inline-logo {
    width: 18px;
    height: 18px;
    object-fit: contain;
    vertical-align: middle;
    margin-right: 5px;
}
.player-name-txt {
    font-weight: 600;
    color: #F8FAFC;
    white-space: nowrap;
    font-size: 0.84rem;
}
.pos-pill {
    font-size: 0.65rem;
    font-weight: 700;
    color: #94A3B8;
    background: #1A2234;
    padding: 1px 4px;
    border-radius: 4px;
    flex-shrink: 0;
}
.inj-q-pill {
    font-size: 0.62rem;
    font-weight: 700;
    color: #FBBF24;
    background: rgba(245, 158, 11, 0.16);
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 1px 4px;
    border-radius: 4px;
    flex-shrink: 0;
}
.inj-out-pill {
    font-size: 0.65rem;
    font-weight: 700;
    color: #F87171;
    background: rgba(239, 68, 68, 0.16);
    border: 1px solid rgba(239, 68, 68, 0.38);
    padding: 1px 6px;
    border-radius: 4px;
}

.board-card {
    background: #111622;
    border: 1px solid #1E2638;
    border-radius: 11px;
    margin-bottom: 1rem;
    overflow: hidden;
}
.board-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.85rem;
    background: #151C2C;
    border-bottom: 1px solid #1E2638;
    border-left: 3px solid #10B981;
}
.board-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.2px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.board-count {
    font-size: 0.7rem;
    font-weight: 600;
    color: #94A3B8;
    background: #1E2638;
    padding: 2px 8px;
    border-radius: 999px;
}
table.prop-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.81rem;
    white-space: nowrap;
}
table.prop-table th {
    text-align: left;
    padding: 0.5rem 0.65rem;
    color: #8B97AD;
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    border-bottom: 1px solid #1E2638;
    background: #111622;
}
table.prop-table td {
    padding: 0.44rem 0.65rem;
    color: #E2E8F0;
    border-bottom: 1px solid #182032;
    vertical-align: middle;
}
table.prop-table tr:last-child td { border-bottom: none; }
table.prop-table tr:hover td { background: #161E30; }
.col-right { text-align: right !important; }
.line-cell { font-weight: 600; color: #38BDF8; }
.proj-cell { font-weight: 700; color: #F8FAFC; }
.rating-good { color: #34D399; font-weight: 600; }
.rating-bad { color: #F87171; font-weight: 600; }
.rating-neutral { color: #94A3B8; }

.pick-card {
    background: #111622;
    border: 1px solid #1E2638;
    border-radius: 10px;
    padding: 0.8rem 0.95rem;
    margin-bottom: 0.5rem;
}
.pick-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
}
.pick-line { font-size: 0.9rem; font-weight: 600; color: #10B981; margin-bottom: 0.2rem; }
.pick-details { font-size: 0.76rem; color: #8B97AD; line-height: 1.4; }

.matchup-card {
    background: #111622;
    border: 1px solid #1E2638;
    border-radius: 11px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
}
.matchup-team-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
}
.matchup-team {
    font-size: 0.92rem;
    font-weight: 600;
    color: #F1F5F9;
    display: flex;
    align-items: center;
}
.matchup-prob { font-size: 0.75rem; color: #64748B; margin-left: 4px; }
.matchup-pts { font-size: 1.2rem; font-weight: 800; color: #10B981; }
.matchup-footer {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
    padding-top: 0.45rem;
    border-top: 1px solid #1A2234;
    font-size: 0.75rem;
    color: #8B97AD;
}

.game-hero {
    background: linear-gradient(135deg, #111622 0%, #161F33 100%);
    border: 1px solid #26324B;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1.1rem;
}

.inj-strip {
    background: #111622;
    border: 1px solid #1E2638;
    border-left: 3px solid #EF4444;
    border-radius: 9px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.9rem;
    font-size: 0.78rem;
    color: #CBD5E1;
}

.heatmap-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-top: 0.4rem;
}
.heat-tile {
    border-radius: 9px;
    padding: 0.65rem 0.75rem;
    border: 1px solid #243047;
}
.heat-smash { background: rgba(16, 185, 129, 0.16); border-color: rgba(16, 185, 129, 0.42); }
.heat-neutral { background: rgba(148, 163, 184, 0.08); border-color: #26324B; }
.heat-tough { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.4); }
.heat-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.4px; color: #94A3B8; font-weight: 600; }
.heat-rank { font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin: 2px 0; }
.heat-sub { font-size: 0.72rem; color: #CBD5E1; }

.site-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1E2638;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    font-size: 0.73rem;
    color: #64748B;
}

@media(max-width: 768px) {
    .heatmap-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)

NBA_TEAM_LOGOS = {
    "Atlanta Hawks": "https://a.espncdn.com/i/teamlogos/nba/500/atl.png",
    "Boston Celtics": "https://a.espncdn.com/i/teamlogos/nba/500/bos.png",
    "Brooklyn Nets": "https://a.espncdn.com/i/teamlogos/nba/500/bkn.png",
    "Charlotte Hornets": "https://a.espncdn.com/i/teamlogos/nba/500/cha.png",
    "Chicago Bulls": "https://a.espncdn.com/i/teamlogos/nba/500/chi.png",
    "Cleveland Cavaliers": "https://a.espncdn.com/i/teamlogos/nba/500/cle.png",
    "Dallas Mavericks": "https://a.espncdn.com/i/teamlogos/nba/500/dal.png",
    "Denver Nuggets": "https://a.espncdn.com/i/teamlogos/nba/500/den.png",
    "Detroit Pistons": "https://a.espncdn.com/i/teamlogos/nba/500/det.png",
    "Golden State Warriors": "https://a.espncdn.com/i/teamlogos/nba/500/gs.png",
    "Houston Rockets": "https://a.espncdn.com/i/teamlogos/nba/500/hou.png",
    "Indiana Pacers": "https://a.espncdn.com/i/teamlogos/nba/500/ind.png",
    "LA Clippers": "https://a.espncdn.com/i/teamlogos/nba/500/lac.png",
    "Los Angeles Clippers": "https://a.espncdn.com/i/teamlogos/nba/500/lac.png",
    "Los Angeles Lakers": "https://a.espncdn.com/i/teamlogos/nba/500/lal.png",
    "Memphis Grizzlies": "https://a.espncdn.com/i/teamlogos/nba/500/mem.png",
    "Miami Heat": "https://a.espncdn.com/i/teamlogos/nba/500/mia.png",
    "Milwaukee Bucks": "https://a.espncdn.com/i/teamlogos/nba/500/mil.png",
    "Minnesota Timberwolves": "https://a.espncdn.com/i/teamlogos/nba/500/min.png",
    "New Orleans Pelicans": "https://a.espncdn.com/i/teamlogos/nba/500/no.png",
    "New York Knicks": "https://a.espncdn.com/i/teamlogos/nba/500/ny.png",
    "Oklahoma City Thunder": "https://a.espncdn.com/i/teamlogos/nba/500/okc.png",
    "Orlando Magic": "https://a.espncdn.com/i/teamlogos/nba/500/orl.png",
    "Philadelphia 76ers": "https://a.espncdn.com/i/teamlogos/nba/500/phi.png",
    "Phoenix Suns": "https://a.espncdn.com/i/teamlogos/nba/500/phx.png",
    "Portland Trail Blazers": "https://a.espncdn.com/i/teamlogos/nba/500/por.png",
    "Sacramento Kings": "https://a.espncdn.com/i/teamlogos/nba/500/sac.png",
    "San Antonio Spurs": "https://a.espncdn.com/i/teamlogos/nba/500/sa.png",
    "Toronto Raptors": "https://a.espncdn.com/i/teamlogos/nba/500/tor.png",
    "Utah Jazz": "https://a.espncdn.com/i/teamlogos/nba/500/uta.png",
    "Washington Wizards": "https://a.espncdn.com/i/teamlogos/nba/500/wsh.png",
}

ESPN_TO_ODDS_TEAM = {
    "LA Clippers": "Los Angeles Clippers",
}

POS_FULL_PLURAL = {
    "G": "Guards",
    "F": "Forwards",
    "C": "Centers",
}

DEFAULT_HEADSHOT = "https://a.espncdn.com/combiner/i?img=/i/headshots/nophoto.png&w=96&h=70"

def map_position_group(raw_pos: str) -> str:
    p = str(raw_pos).strip().upper()
    if p in ("C", "CENTER"):
        return "C"
    if p in ("PG", "SG", "G", "GUARD"):
        return "G"
    if p in ("SF", "PF", "F", "FORWARD"):
        return "F"
    return "F"

def get_team_logo(team_name: str) -> str:
    return NBA_TEAM_LOGOS.get(team_name, "https://a.espncdn.com/i/teamlogos/leagues/500/nba.png")

def build_headshot_url(athlete_id, href: str = "") -> str:
    if isinstance(href, str) and href.startswith("http"):
        return href
    if pd.notna(athlete_id):
        try:
            aid = int(float(athlete_id))
            if aid > 0:
                return f"https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/{aid}.png&w=96&h=70"
        except Exception:
            pass
    return DEFAULT_HEADSHOT

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = name.lower().replace(".", "").replace("'", "").replace("-", " ")
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()

def short_matchup(game_label: str) -> str:
    parts = [t.strip().split()[-1] for t in game_label.split("@")]
    return f"{parts[0]} @ {parts[1]}" if len(parts) == 2 else game_label

def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def player_badge_html(player_name: str, headshot_url: str, team_logo_url: str, pos_label: str = "", book: str = "draftkings", from_game: str = "", inj_status: str = "") -> str:
    href = f"?player={quote_plus(player_name)}&book={book}"
    if from_game:
        href += f"&game={quote_plus(from_game)}"
    pos_html = f'<span class="pos-pill">{pos_label}</span>' if pos_label else ""
    inj_html = ""
    if inj_status and any(k in inj_status.lower() for k in ("questionable", "day-to-day", "probable", "gtd")):
        inj_html = '<span class="inj-q-pill">GTD</span>'
    return (
        f'<a class="player-link" href="{href}" target="_self" title="Open {player_name} Scouting Profile">'
        f'<div class="headshot-wrap">'
        f'<img class="headshot-img" src="{headshot_url}" onerror="this.src=\'{DEFAULT_HEADSHOT}\'" />'
        f'<img class="team-mini-logo" src="{team_logo_url}" />'
        f'</div>'
        f'<span class="player-name-txt">{player_name}</span>'
        f'{pos_html}{inj_html}'
        f'</a>'
    )

@st.cache_resource
def get_daily_pregame_vault() -> dict:
    return {}

@st.cache_data(ttl=900)
def fetch_nba_injury_report() -> tuple[dict, dict]:
    player_injuries = {}
    team_out_lists = {}
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        data = requests.get(url, timeout=8).json()
        for team_blk in data.get("injuries", []):
            raw_team = team_blk.get("displayName", "")
            team_name = ESPN_TO_ODDS_TEAM.get(raw_team, raw_team)
            for item in team_blk.get("injuries", []):
                ath = item.get("athlete", {})
                p_name = ath.get("displayName", "")
                status = str(item.get("status", "")).strip()
                short_comment = str(item.get("shortComment", "")).strip()
                if not p_name or not status:
                    continue
                norm_p = normalize_name(p_name)
                st_low = status.lower()
                is_out = any(k in st_low for k in ("out", "suspension", "suspended", "doubtful", "inactive"))
                player_injuries[norm_p] = {
                    "name": p_name,
                    "status": status,
                    "detail": short_comment,
                    "team": team_name,
                    "is_out": is_out,
                }
                if is_out and team_name:
                    team_out_lists.setdefault(team_name, []).append(f"{p_name} ({status})")
    except Exception:
        pass
    return player_injuries, team_out_lists

@st.cache_data(ttl=21600)
def load_nba_data(seasons=(2025, 2026, 2027)):
    frames = []
    for yr in seasons:
        url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{yr}.parquet"
        try:
            part = pd.read_parquet(url)
            part["SEASON"] = yr
            frames.append(part)
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True)
    df = df[df["minutes"] >= 10].copy()
    df["game_id"] = df["game_id"].astype(str)
    if "game_date" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values("GAME_DATE")
    else:
        df["GAME_DATE"] = pd.NaT

    df = df.drop_duplicates(subset=["game_id", "athlete_display_name"])
    rename_map = {
        "game_id": "GAME_ID",
        "athlete_id": "ATHLETE_ID",
        "athlete_headshot_href": "HEADSHOT_HREF",
        "athlete_display_name": "PLAYER_NAME",
        "athlete_position_abbreviation": "RAW_POS",
        "team_display_name": "TEAM_NAME",
        "team_logo": "TEAM_LOGO",
        "opponent_team_display_name": "OPP_NAME",
        "home_away": "HOME_AWAY",
        "minutes": "MIN",
        "points": "PTS",
        "rebounds": "REB",
        "assists": "AST",
        "three_point_field_goals_made": "FG3M",
        "steals": "STL",
        "blocks": "BLK",
        "turnovers": "TOV",
        "field_goals_made": "FGM",
        "field_goals_attempted": "FGA",
        "three_point_field_goals_attempted": "FG3A",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in ["ATHLETE_ID", "FGM", "FGA", "FG3A"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["HEADSHOT_HREF", "TEAM_LOGO", "HOME_AWAY", "RAW_POS"]:
        if col not in df.columns:
            df[col] = ""

    valid_pos = df[df["RAW_POS"].astype(str).str.strip() != ""].drop_duplicates(subset=["PLAYER_NAME"], keep="last")
    pos_map = dict(zip(valid_pos["PLAYER_NAME"], valid_pos["RAW_POS"]))
    df["RAW_POS"] = df["PLAYER_NAME"].map(pos_map).fillna(df["RAW_POS"]).fillna("F")
    df["POS_GROUP"] = df["RAW_POS"].apply(map_position_group)

    df["NORM_NAME"] = df["PLAYER_NAME"].apply(normalize_name)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

@st.cache_data(ttl=21600)
def build_player_media_lookup(df: pd.DataFrame) -> dict:
    latest = df.drop_duplicates(subset=["NORM_NAME"], keep="last")
    media = {}
    for _, r in latest.iterrows():
        t_name = str(r.get("TEAM_NAME", ""))
        raw_tlogo = str(r.get("TEAM_LOGO", ""))
        t_logo = raw_tlogo if raw_tlogo.startswith("http") else get_team_logo(t_name)
        aid = r.get("ATHLETE_ID")
        h_url = build_headshot_url(aid, str(r.get("HEADSHOT_HREF", "")))
        raw_p = str(r.get("RAW_POS", "")).strip() or r.get("POS_GROUP", "F")
        media[r["NORM_NAME"]] = {
            "athlete_id": int(float(aid)) if pd.notna(aid) else 0,
            "headshot": h_url,
            "team_logo": t_logo,
            "team_name": t_name,
            "raw_pos": raw_p,
            "pos_group": r.get("POS_GROUP", "F"),
        }
    return media

@st.cache_data(ttl=21600)
def compute_advanced_team_metrics(df: pd.DataFrame) -> dict:
    stats = ["PTS", "REB", "AST", "FG3M", "PRA"]
    if "OPP_NAME" not in df.columns:
        return {"dvp_mult": {}, "dvp_ranks": {}, "dvp_avg": {}, "last_game": {}}

    max_season = df["SEASON"].max()
    recent_df = df[df["SEASON"] >= max_season - 1].copy()

    pos_game_allowed = recent_df.groupby(["OPP_NAME", "POS_GROUP", "GAME_ID", "SEASON"])[stats].sum().reset_index()
    pos_game_allowed["w"] = np.where(pos_game_allowed["SEASON"] == max_season, 2.0, 1.0)

    def weighted_avg(grp):
        w = grp["w"].values
        return pd.Series({s: np.average(grp[s].values, weights=w) for s in stats})

    dvp_allowed = pos_game_allowed.groupby(["OPP_NAME", "POS_GROUP"]).apply(weighted_avg, include_groups=False)

    dvp_mult = {}
    dvp_ranks = {}
    dvp_avg = {}

    for pos_g in ["G", "F", "C"]:
        if pos_g not in dvp_allowed.index.get_level_values(1):
            continue
        sub_pos = dvp_allowed.xs(pos_g, level="POS_GROUP")
        lg_pos_avg = sub_pos.mean()
        mult_df = (sub_pos / lg_pos_avg).fillna(1.0)

        for team_name in sub_pos.index:
            dvp_mult.setdefault(team_name, {})[pos_g] = mult_df.loc[team_name].to_dict()
            dvp_avg.setdefault(team_name, {})[pos_g] = sub_pos.loc[team_name].round(1).to_dict()

        for s in stats:
            ranked = sub_pos[s].rank(ascending=True, method="min").astype(int)
            for team_name, rk in ranked.items():
                dvp_ranks.setdefault(team_name, {}).setdefault(pos_g, {})[s] = int(rk)

    last_game = {}
    if df["GAME_DATE"].notna().any():
        last_game = df.dropna(subset=["GAME_DATE"]).groupby("TEAM_NAME")["GAME_DATE"].max().to_dict()

    return {
        "dvp_mult": dvp_mult,
        "dvp_ranks": dvp_ranks,
        "dvp_avg": dvp_avg,
        "last_game": last_game,
    }

def format_matchup_rating(mult: float, pos_group: str = "") -> tuple[str, str]:
    score = round(mult * 100, 1)
    pos_tag = f" vs {pos_group}" if pos_group else ""
    if score >= 102.5:
        return f"{score} · Smash{pos_tag}", "rating-good"
    elif score >= 100.8:
        return f"{score} · Favorable{pos_tag}", "rating-good"
    elif score <= 97.5:
        return f"{score} · Tough{pos_tag}", "rating-bad"
    elif score <= 99.2:
        return f"{score} · Below Avg{pos_tag}", "rating-bad"
    return f"{score} · Neutral{pos_tag}", "rating-neutral"

@st.cache_data(ttl=1800)
def fetch_player_shot_chart(athlete_id: int, recent_game_ids: tuple) -> pd.DataFrame:
    if not athlete_id or not recent_game_ids:
        return pd.DataFrame()
    shots = []
    for gid in recent_game_ids[:8]:
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={gid}"
            data = requests.get(url, timeout=6).json()
            for play in data.get("plays", []):
                if not play.get("shootingPlay"):
                    continue
                participants = play.get("participants", [])
                if not participants:
                    continue
                shooter_id = participants[0].get("athlete", {}).get("id")
                if str(shooter_id) != str(athlete_id):
                    continue
                coord = play.get("coordinate", {})
                cx, cy = coord.get("x"), coord.get("y")
                if cx is None or cy is None:
                    continue
                fx, fy = float(cx), float(cy)
                if fy > 47:
                    fy = 94.0 - fy
                    fx = 50.0 - fx
                court_x = fx - 25.0
                court_y = max(0.0, fy)
                made = bool(play.get("scoringPlay"))
                pts_val = int(play.get("scoreValue", 2))
                dist = np.hypot(court_x, court_y - 5.25)
                if pts_val == 3:
                    zone = "3-Point"
                elif dist <= 8.0:
                    zone = "Paint / Rim"
                else:
                    zone = "Mid-Range"
                shots.append({
                    "x": court_x,
                    "y": court_y,
                    "made": made,
                    "zone": zone,
                    "text": play.get("text", ""),
                })
        except Exception:
            continue
    return pd.DataFrame(shots)

def build_shot_chart_figure(shots_df: pd.DataFrame, player_name: str) -> go.Figure:
    fig = go.Figure()
    line_col = "#334155"
    shapes = [
        dict(type="rect", x0=-25, y0=0, x1=25, y1=47, line=dict(color=line_col, width=2)),
        dict(type="rect", x0=-8, y0=0, x1=8, y1=19, line=dict(color=line_col, width=1.5)),
        dict(type="circle", x0=-6, y0=13, x1=6, y1=25, line=dict(color=line_col, width=1.5)),
        dict(type="line", x0=-3, y0=4, x1=3, y1=4, line=dict(color="#94A3B8", width=2.5)),
        dict(type="circle", x0=-0.75, y0=4.5, x1=0.75, y1=6.0, line=dict(color="#F59E0B", width=2)),
        dict(type="line", x0=-22, y0=0, x1=-22, y1=14, line=dict(color=line_col, width=1.8)),
        dict(type="line", x0=22, y0=0, x1=22, y1=14, line=dict(color=line_col, width=1.8)),
    ]

    theta = np.linspace(np.arcsin(8.75 / 23.75), np.pi - np.arcsin(8.75 / 23.75), 100)
    arc_x = 23.75 * np.cos(theta)
    arc_y = 5.25 + 23.75 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y, mode="lines",
        line=dict(color=line_col, width=1.8),
        hoverinfo="skip", showlegend=False
    ))

    if not shots_df.empty:
        made_df = shots_df[shots_df["made"]]
        miss_df = shots_df[~shots_df["made"]]
        fig.add_trace(go.Scatter(
            x=miss_df["x"], y=miss_df["y"], mode="markers",
            name=f"Missed ({len(miss_df)})",
            marker=dict(symbol="x", size=8, color="#EF4444", opacity=0.8),
            text=miss_df["text"], hovertemplate="%{text}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=made_df["x"], y=made_df["y"], mode="markers",
            name=f"Made ({len(made_df)})",
            marker=dict(symbol="circle", size=9, color="#10B981", line=dict(color="#064E3B", width=1)),
            text=made_df["text"], hovertemplate="%{text}<extra></extra>"
        ))

    fig.update_layout(
        shapes=shapes,
        paper_bgcolor="#111622",
        plot_bgcolor="#0D121C",
        margin=dict(l=10, r=10, t=36, b=10),
        height=390,
        title=dict(text=f"{player_name} · Shot Chart (Recent Games)", font=dict(size=14, color="#F8FAFC")),
        xaxis=dict(range=[-25.5, 25.5], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-0.5, 47.5], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1, fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(color="#CBD5E1", size=11))
    )
    return fig

def build_hit_rate_chart(p_df: pd.DataFrame, stat_col: str, line_val: float, player_name: str) -> go.Figure:
    recent = p_df.tail(20).copy()
    dates = recent["GAME_DATE"].dt.strftime("%b %d").fillna("Game")
    opps = recent["OPP_NAME"].apply(lambda x: str(x).split()[-1])
    labels = [f"{d}<br>vs {o}" for d, o in zip(dates, opps)]
    vals = recent[stat_col].values
    colors = ["#10B981" if v > line_val else "#EF4444" for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=vals,
        marker_color=colors,
        text=[f"{v:.0f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#E2E8F0", size=11),
        hovertemplate="<b>%{x}</b><br>" + stat_col + ": %{y:.0f}<extra></extra>"
    ))
    fig.add_hline(
        y=line_val,
        line_dash="dash",
        line_color="#38BDF8",
        line_width=2,
        annotation_text=f"Line: {line_val}",
        annotation_position="top left",
        annotation_font=dict(color="#38BDF8", size=12)
    )
    max_y = max(float(np.max(vals)) * 1.22, line_val * 1.35, 5.0)
    fig.update_layout(
        paper_bgcolor="#111622",
        plot_bgcolor="#111622",
        margin=dict(l=20, r=20, t=38, b=35),
        height=310,
        title=dict(text=f"{player_name} · Last {len(recent)} Games ({stat_col} vs {line_val} Line)", font=dict(size=14, color="#F8FAFC")),
        xaxis=dict(showgrid=False, tickfont=dict(color="#94A3B8", size=10)),
        yaxis=dict(range=[0, max_y], showgrid=True, gridcolor="#1E2638", tickfont=dict(color="#94A3B8", size=10)),
        showlegend=False
    )
    return fig

@st.cache_data(ttl=600)
def fetch_and_lock_propline_slate(bookmaker: str = "draftkings", max_games: int = 12):
    et_tz = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(et_tz)
    today_date_et = now_et.date()
    date_key = f"{today_date_et.isoformat()}_{bookmaker}"

    vault = get_daily_pregame_vault()
    day_bucket = vault.setdefault(date_key, {"games": {}, "props": {}})

    try:
        events = pl_client.get_events("basketball_nba")
        pregame_events = []

        for ev in events:
            c_str = ev.get("commence_time", "")
            if not c_str:
                continue
            tip_utc = datetime.fromisoformat(c_str.replace("Z", "+00:00"))
            tip_et = tip_utc.astimezone(et_tz)
            if tip_et.date() != today_date_et:
                continue

            away = ESPN_TO_ODDS_TEAM.get(ev.get("away_team", ""), ev.get("away_team", ""))
            home = ESPN_TO_ODDS_TEAM.get(ev.get("home_team", ""), ev.get("home_team", ""))
            lbl = f"{away} @ {home}"
            tip_str = tip_et.strftime("%I:%M %p ET").lstrip("0")
            event_id = ev.get("id")

            day_bucket["games"][lbl] = {
                "spread_home": 0.0,
                "total": 228.0,
                "id": event_id,
                "tip_str": tip_str,
                "tip_utc": tip_utc.isoformat(),
            }
            if tip_utc > (now_utc + timedelta(minutes=3)):
                ev["_tip_et"] = tip_et
                pregame_events.append(ev)

        pregame_events.sort(key=lambda x: x["_tip_et"])
        market_list = ["player_points", "player_rebounds", "player_assists", "player_threes", "player_points_rebounds_assists", "spreads", "totals"]

        for ev in pregame_events[:max_games]:
            away = ESPN_TO_ODDS_TEAM.get(ev.get("away_team", ""), ev.get("away_team", ""))
            home = ESPN_TO_ODDS_TEAM.get(ev.get("home_team", ""), ev.get("home_team", ""))
            lbl = f"{away} @ {home}"
            event_id = ev.get("id")

            odds_data = pl_client.get_odds("basketball_nba", event_id=event_id, markets=market_list)
            
            spread_home = 0.0
            total_line = 228.0
            candidates = {}

            for book in odds_data.get("bookmakers", []):
                for market in book.get("markets", []):
                    m_key = market.get("key")
                    if m_key == "spreads":
                        for o in market.get("outcomes", []):
                            if o.get("name") == home and o.get("point") is not None:
                                spread_home = float(o["point"])
                    elif m_key == "totals":
                        for o in market.get("outcomes", []):
                            if o.get("name") == "Over" and o.get("point") is not None:
                                total_line = float(o["point"])
                    else:
                        stat_map = {
                            "player_points": "PTS",
                            "player_rebounds": "REB",
                            "player_assists": "AST",
                            "player_threes": "FG3M",
                            "player_points_rebounds_assists": "PRA"
                        }
                        stat_code = stat_map.get(m_key)
                        if not stat_code:
                            continue
                        for outcome in market.get("outcomes", []):
                            player = outcome.get("description")
                            side = outcome.get("name")
                            line = outcome.get("point")
                            price = outcome.get("price")
                            if not player or side not in ("Over", "Under") or line is None or price is None:
                                continue
                            norm_p = normalize_name(player)
                            p_obj = candidates.setdefault(norm_p, {"display_name": player, "stats": {}})
                            st_obj = p_obj["stats"].setdefault(stat_code, {})
                            line_obj = st_obj.setdefault(float(line), {})
                            line_obj[side] = int(price)

            day_bucket["games"][lbl]["spread_home"] = spread_home
            day_bucket["games"][lbl]["total"] = total_line

            props = {}
            for norm_p, p_obj in candidates.items():
                clean_stats = {}
                for stat_code, lines_dict in p_obj["stats"].items():
                    two_way = [(ln, sides) for ln, sides in lines_dict.items() if "Over" in sides and "Under" in sides]
                    if not two_way:
                        continue
                    best_ln, best_sides = min(two_way, key=lambda item: abs(item[1]["Over"] + 110) + abs(item[1]["Under"] + 110))
                    clean_stats[stat_code] = {
                        "line": best_ln,
                        "Over": best_sides["Over"],
                        "Under": best_sides["Under"],
                    }
                if clean_stats:
                    props[norm_p] = {"display_name": p_obj["display_name"], "stats": clean_stats}

            if props:
                day_bucket["props"][lbl] = props

    except Exception:
        pass

    all_game_labels = sorted(day_bucket["games"].keys(), key=lambda k: day_bucket["games"][k].get("tip_utc", ""))
    slate_date_label = today_date_et.strftime("%A, %b %d")
    fetched_at = now_et.strftime("%I:%M %p ET")
    return all_game_labels, day_bucket["games"], day_bucket["props"], fetched_at, slate_date_label

def american_to_prob(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds: int) -> float:
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

def prob_to_american(p: float) -> int:
    p = np.clip(p, 0.0001, 0.9999)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)

def get_no_vig_prob(odds_selected: int, odds_opposing: int = 0) -> float:
    p_sel = american_to_prob(odds_selected)
    if odds_opposing != 0:
        p_opp = american_to_prob(odds_opposing)
        return p_sel / (p_sel + p_opp)
    return p_sel * 0.955

def exp_weighted_mean(values: np.ndarray, decay: float = 0.94) -> float:
    n = len(values)
    weights = decay ** np.arange(n - 1, -1, -1)
    return float(np.sum(values * weights) / np.sum(weights))

def exp_weighted_std(values: np.ndarray, mean_val: float, decay: float = 0.94) -> float:
    n = len(values)
    weights = decay ** np.arange(n - 1, -1, -1)
    var = np.sum(weights * (values - mean_val) ** 2) / np.sum(weights)
    return float(np.sqrt(max(var, 0.65)))

def skewed_prob_over(line: float, proj_mean: float, proj_std: float, emp_series: pd.Series) -> float:
    emp_recent = (emp_series.tail(15) > line).mean()
    emp_season = (emp_series.tail(40) > line).mean()
    emp_blend = 0.65 * emp_recent + 0.35 * emp_season

    if proj_mean <= 0.1:
        return 0.01

    variance = max(proj_std ** 2, proj_mean * 1.05)
    shape_k = (proj_mean ** 2) / variance
    scale_theta = variance / proj_mean
    dist_over = float(1.0 - gamma.cdf(line, a=shape_k, scale=scale_theta))

    w_dist = min(0.70, max(0.50, len(emp_series) / 80.0))
    return float(np.clip(w_dist * dist_over + (1.0 - w_dist) * emp_blend, 0.02, 0.98))

def evaluate_straight_edges(df: pd.DataFrame, slate_props: dict, game_lines: dict, team_metrics: dict, media_lookup: dict, injury_map: dict) -> pd.DataFrame:
    rows = []
    grouped = {k: v for k, v in df.groupby("NORM_NAME")}
    dvp_mults_all = team_metrics["dvp_mult"]
    last_games = team_metrics["last_game"]
    today_ts = pd.Timestamp.now().normalize()

    team_usage_bump = {}
    for norm_inj, inj_info in injury_map.items():
        if inj_info.get("is_out") and norm_inj in grouped:
            inj_mins = grouped[norm_inj]["MIN"].tail(15).mean()
            if inj_mins >= 24.0:
                tm = inj_info.get("team", "")
                team_usage_bump[tm] = min(1.06, team_usage_bump.get(tm, 1.0) + 0.025)

    for game_label, game_props in slate_props.items():
        teams_in_game = [t.strip() for t in game_label.split("@")]
        if len(teams_in_game) != 2:
            continue
        away_team, home_team = teams_in_game[0], teams_in_game[1]
        g_line_info = game_lines.get(game_label, {"spread_home": 0.0, "total": 228.0})
        vegas_total = g_line_info["total"]
        abs_spread = abs(g_line_info["spread_home"])

        vegas_pace_mult = 1.0 + 0.45 * ((vegas_total - 228.0) / 228.0)

        for norm_p, p_data in game_props.items():
            inj_rec = injury_map.get(norm_p, {})
            if inj_rec.get("is_out"):
                continue

            if norm_p not in grouped:
                continue
            p_df = grouped[norm_p]
            if len(p_df) < 12:
                continue

            player_team = p_df["TEAM_NAME"].iloc[-1]
            is_home = (player_team == home_team)
            opp_team = away_team if is_home else home_team

            p_media = media_lookup.get(norm_p, {})
            pos_group = p_media.get("pos_group", p_df["POS_GROUP"].iloc[-1])
            raw_pos = p_media.get("raw_pos", pos_group)
            headshot_url = p_media.get("headshot", DEFAULT_HEADSHOT)
            team_logo_url = get_team_logo(player_team)

            opp_pos_mults = dvp_mults_all.get(opp_team, {}).get(pos_group, {})

            min_vals = p_df["MIN"].dropna().tail(30).values
            proj_min = exp_weighted_mean(min_vals, decay=0.90)

            if abs_spread >= 11.5 and proj_min >= 28.0:
                proj_min *= 0.96

            b2b_mult = 1.0
            if player_team in last_games:
                days_rest = (today_ts - last_games[player_team]).days
                if days_rest == 1:
                    b2b_mult = 0.975

            loc_mult = 1.012 if is_home else 0.988
            usage_mult = team_usage_bump.get(player_team, 1.0)

            for stat, info in p_data["stats"].items():
                if stat not in p_df.columns or "Over" not in info or "Under" not in info:
                    continue
                line = info["line"]
                valid_df = p_df[["MIN", stat]].dropna().tail(50)
                if len(valid_df) < 12:
                    continue

                per_min_rates = (valid_df[stat] / valid_df["MIN"]).values
                exp_rate = exp_weighted_mean(per_min_rates, decay=0.94)
                base_proj = exp_rate * proj_min

                raw_dvp = opp_pos_mults.get(stat, 1.0)
                damp_dvp = 1.0 + 0.55 * (raw_dvp - 1.0)
                total_env_mult = damp_dvp * vegas_pace_mult * loc_mult * b2b_mult * usage_mult
                proj = base_proj * total_env_mult

                std = exp_weighted_std(valid_df[stat].values, proj, decay=0.94)
                model_over = skewed_prob_over(line, proj, std, valid_df[stat])
                model_under = 1.0 - model_over

                nv_over = get_no_vig_prob(info["Over"], info["Under"])
                nv_under = get_no_vig_prob(info["Under"], info["Over"])

                for side, mod_p, nv_p, odds in [
                    ("Over", model_over, nv_over, info["Over"]),
                    ("Under", model_under, nv_under, info["Under"]),
                ]:
                    dec = american_to_decimal(odds)
                    edge_pct = (mod_p - nv_p) * 100
                    rating_mult = total_env_mult if side == "Over" else (2.0 - total_env_mult)
                    rating_txt, rating_cls = format_matchup_rating(rating_mult, pos_group)

                    rows.append({
                        "Player": p_data["display_name"],
                        "NORM_NAME": norm_p,
                        "Pos": raw_pos,
                        "PosGroup": pos_group,
                        "InjStatus": inj_rec.get("status", ""),
                        "Headshot": headshot_url,
                        "TeamLogo": team_logo_url,
                        "Line": f"{side} {line} {stat}",
                        "RawLine": float(line),
                        "Side": side,
                        "Stat": stat,
                        "Projection": round(float(proj), 1),
                        "Matchup": short_matchup(game_label),
                        "FullMatchup": game_label,
                        "OppTeam": opp_team,
                        "Matchup Rating": rating_txt,
                        "RatingClass": rating_cls,
                        "Odds": int(odds),
                        "Win Prob": round(float(mod_p * 100), 1),
                        "Edge": round(float(edge_pct), 1),
                    })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="Win Prob", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=1800)
def simulate_full_game(df: pd.DataFrame, away_team: str, home_team: str, team_metrics: dict, game_line: dict, media_lookup: dict, out_players_key: tuple = (), n_sims: int = 15000):
    dvp_mults_all = team_metrics["dvp_mult"]
    last_games = team_metrics["last_game"]
    today_ts = pd.Timestamp.now().normalize()
    vegas_total = game_line.get("total", 228.0)
    vegas_spread_home = game_line.get("spread_home", 0.0)
    vegas_pace_mult = 1.0 + 0.35 * ((vegas_total - 228.0) / 228.0)
    max_season = df["SEASON"].max()
    out_set = set(out_players_key)

    recent_season_df = df[df["SEASON"] >= max_season - 1]
    team_game_pts = recent_season_df.groupby(["TEAM_NAME", "GAME_ID"])["PTS"].sum().groupby("TEAM_NAME").mean().to_dict()
    opp_game_pts = recent_season_df.groupby(["OPP_NAME", "GAME_ID"])["PTS"].sum().groupby("OPP_NAME").mean().to_dict()
    lg_avg_pts = float(np.mean(list(team_game_pts.values()))) if team_game_pts else 114.0

    def get_rotation(team: str, opp: str, is_home: bool):
        t_df = df[df["TEAM_NAME"] == team].copy()
        if t_df.empty:
            return np.full(n_sims, 114.0), pd.DataFrame()

        curr_season_df = t_df[t_df["SEASON"] == max_season]
        active_pool = curr_season_df if not curr_season_df.empty else t_df
        active_pool = active_pool[~active_pool["NORM_NAME"].isin(out_set)]

        recent_games = active_pool["GAME_ID"].drop_duplicates().tail(10)
        recent_df = active_pool[active_pool["GAME_ID"].isin(recent_games)]

        rot_players = (
            recent_df.groupby("PLAYER_NAME")["MIN"]
            .agg(["count", "mean"])
            .query("count >= 2")
            .sort_values(by="mean", ascending=False)
            .head(9)
            .index.tolist()
        )
        if not rot_players:
            rot_players = recent_df["PLAYER_NAME"].unique().tolist()[:9]

        sub = t_df[t_df["PLAYER_NAME"].isin(rot_players)]
        loc_mult = 1.012 if is_home else 0.988
        b2b_mult = 0.978 if (team in last_games and (today_ts - last_games[team]).days == 1) else 1.0

        raw_mins = [exp_weighted_mean(sub[sub["PLAYER_NAME"] == p].tail(35)["MIN"].values, decay=0.90) for p in rot_players]
        total_raw_m = sum(raw_mins)
        norm_mins = [(m / total_raw_m) * 240.0 for m in raw_mins] if total_raw_m > 0 else [240.0 / max(len(rot_players), 1)] * len(rot_players)
        norm_mins = [min(39.5, m) for m in norm_mins]
        if sum(norm_mins) > 0:
            norm_mins = [(m / sum(norm_mins)) * 240.0 for m in norm_mins]

        player_params = []
        for p, mins in zip(rot_players, norm_mins):
            p_sub = sub[sub["PLAYER_NAME"] == p].tail(35)
            norm_p = normalize_name(p)
            p_media = media_lookup.get(norm_p, {})
            pos_g = p_media.get("pos_group", p_sub["POS_GROUP"].iloc[-1])
            opp_pos_mults = dvp_mults_all.get(opp, {}).get(pos_g, {})

            stats_proj = {}
            for st_col in ["PTS", "REB", "AST", "FG3M"]:
                rates = (p_sub[st_col] / p_sub["MIN"]).values
                base = exp_weighted_mean(rates, decay=0.94) * mins
                mult = (1.0 + 0.50 * (opp_pos_mults.get(st_col, 1.0) - 1.0)) * vegas_pace_mult * loc_mult * b2b_mult
                mean_val = float(base * mult)
                std_val = exp_weighted_std(p_sub[st_col].values, mean_val, decay=0.94)
                stats_proj[st_col] = (mean_val, std_val)
            player_params.append({"player": p, "min": mins, "stats": stats_proj})

        piv_pts = sub.pivot(index="GAME_ID", columns="PLAYER_NAME", values="PTS").reindex(columns=rot_players)
        corr = piv_pts.corr(min_periods=6).fillna(0.0).to_numpy().copy()
        np.fill_diagonal(corr, 1.0)

        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 0.05)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(corr_psd))
        corr_psd = corr_psd / np.outer(d, d)

        rng = np.random.default_rng(42)
        z = rng.multivariate_normal(np.zeros(len(rot_players)), corr_psd, size=n_sims)
        u = norm.cdf(z)
        sim_pts_matrix = np.zeros((n_sims, len(rot_players)))
        box_rows = []
        team_logo_url = get_team_logo(team)

        for idx, p_info in enumerate(player_params):
            pts_m, pts_s = p_info["stats"]["PTS"]
            var_p = max(pts_s ** 2, pts_m * 1.05)
            sim_p = gamma.ppf(np.clip(u[:, idx], 0.001, 0.999), a=(pts_m ** 2) / var_p, scale=var_p / max(pts_m, 0.1))
            sim_pts_matrix[:, idx] = sim_p

            def sim_gamma_stat(m_val, s_val):
                if m_val <= 0.1:
                    return np.zeros(n_sims)
                v = max(s_val ** 2, m_val * 1.05)
                return rng.gamma(shape=(m_val ** 2) / v, scale=v / m_val, size=n_sims)

            sim_r = sim_gamma_stat(*p_info["stats"]["REB"])
            sim_a = sim_gamma_stat(*p_info["stats"]["AST"])
            sim_3 = sim_gamma_stat(*p_info["stats"]["FG3M"])

            norm_p = normalize_name(p_info["player"])
            p_media = media_lookup.get(norm_p, {})
            box_rows.append({
                "Player": p_info["player"],
                "Pos": p_media.get("raw_pos", "F"),
                "Headshot": p_media.get("headshot", DEFAULT_HEADSHOT),
                "TeamLogo": team_logo_url,
                "MIN": round(p_info["min"], 1),
                "PTS": round(float(np.mean(sim_p)), 1),
                "REB": round(float(np.mean(sim_r)), 1),
                "AST": round(float(np.mean(sim_a)), 1),
                "3PM": round(float(np.mean(sim_3)), 1),
                "PRA": round(float(np.mean(sim_p + sim_r + sim_a)), 1),
            })

        raw_team_sim = sim_pts_matrix.sum(axis=1)
        tm_off = team_game_pts.get(team, lg_avg_pts)
        opp_def = opp_game_pts.get(opp, lg_avg_pts)
        expected_team_pts = ((tm_off + opp_def) / 2.0) * (1.015 if is_home else 0.985)
        if vegas_spread_home != 0.0:
            vegas_implied = (vegas_total / 2.0) - (vegas_spread_home / 2.0) if is_home else (vegas_total / 2.0) + (vegas_spread_home / 2.0)
            expected_team_pts = 0.55 * expected_team_pts + 0.45 * vegas_implied

        shift = expected_team_pts - float(np.mean(raw_team_sim))
        calibrated_scores = raw_team_sim + (0.45 * shift)
        return calibrated_scores, pd.DataFrame(box_rows)

    away_scores, away_box = get_rotation(away_team, home_team, is_home=False)
    home_scores, home_box = get_rotation(home_team, away_team, is_home=True)

    rng = np.random.default_rng(99)
    ties = np.isclose(away_scores, home_scores, atol=0.1)
    away_scores[ties] += rng.normal(0, 3.0, size=ties.sum())

    away_win_prob = float(np.mean(away_scores > home_scores))
    home_win_prob = 1.0 - away_win_prob
    proj_spread = float(np.mean(away_scores - home_scores))
    proj_total = float(np.mean(away_scores + home_scores))

    return {
        "away_score": round(float(np.mean(away_scores)), 1),
        "home_score": round(float(np.mean(home_scores)), 1),
        "away_win_prob": round(away_win_prob * 100, 1),
        "home_win_prob": round(home_win_prob * 100, 1),
        "proj_spread": round(proj_spread, 1),
        "proj_total": round(proj_total, 1),
        "away_box": away_box,
        "home_box": home_box,
    }

def build_auto_sgps(df: pd.DataFrame, game_label: str, game_props: dict, injury_map: dict, min_games: int = 15) -> pd.DataFrame:
    active_norms = [
        np_name for np_name in game_props.keys()
        if np_name in set(df["NORM_NAME"]) and not injury_map.get(np_name, {}).get("is_out")
    ]
    sub = df[df["NORM_NAME"].isin(active_norms)].copy()
    if sub.empty:
        return pd.DataFrame()

    stats = ["PTS", "REB", "AST", "FG3M"]
    pivoted = sub.pivot(index="GAME_ID", columns="NORM_NAME", values=stats)
    pivoted.columns = [f"{p}|{s}" for s, p in pivoted.columns]

    corr = pivoted.corr(min_periods=min_games)
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).unstack().dropna().reset_index()
    pairs.columns = ["B", "A", "Corr"]

    sgp_rows = []
    for _, r in pairs.iterrows():
        p1, s1 = r["A"].split("|")
        p2, s2 = r["B"].split("|")
        if p1 == p2:
            continue
        st1 = game_props.get(p1, {}).get("stats", {}).get(s1, {})
        st2 = game_props.get(p2, {}).get("stats", {}).get(s2, {})
        if not ("Over" in st1 and "Over" in st2 and "Under" in st2):
            continue

        c_val = r["Corr"]
        if c_val >= 0.22:
            p_a = get_no_vig_prob(st1["Over"], st1.get("Under", 0))
            p_b = get_no_vig_prob(st2["Over"], st2.get("Under", 0))
            z = np.random.multivariate_normal([0, 0], [[1, c_val], [c_val, 1]], size=40000)
            u = norm.cdf(z)
            joint_p = float(np.mean((u[:, 0] < p_a) & (u[:, 1] < p_b)))
            sgp_rows.append({
                "Game": game_label,
                "Angle": "Positive Stack",
                "Leg 1": f"{game_props[p1]['display_name']} Over {st1['line']} {s1} ({st1['Over']:+d})",
                "Leg 2": f"{game_props[p2]['display_name']} Over {st2['line']} {s2} ({st2['Over']:+d})",
                "Correlation": round(float(c_val), 3),
                "Book Naive": prob_to_american(p_a * p_b),
                "True Fair": prob_to_american(joint_p),
            })
        elif c_val <= -0.22:
            p_a = get_no_vig_prob(st1["Over"], st1.get("Under", 0))
            p_b = get_no_vig_prob(st2["Under"], st2.get("Over", 0))
            eff_corr = abs(c_val)
            z = np.random.multivariate_normal([0, 0], [[1, eff_corr], [eff_corr, 1]], size=40000)
            u = norm.cdf(z)
            joint_p = float(np.mean((u[:, 0] < p_a) & (u[:, 1] < p_b)))
            sgp_rows.append({
                "Game": game_label,
                "Angle": "Negative Fade",
                "Leg 1": f"{game_props[p1]['display_name']} Over {st1['line']} {s1} ({st1['Over']:+d})",
                "Leg 2": f"{game_props[p2]['display_name']} Under {st2['line']} {s2} ({st2['Under']:+d})",
                "Correlation": round(float(c_val), 3),
                "Book Naive": prob_to_american(p_a * p_b),
                "True Fair": prob_to_american(joint_p),
            })

    if not sgp_rows:
        return pd.DataFrame()
    res_df = pd.DataFrame(sgp_rows)
    return res_df.reindex(res_df["Correlation"].abs().sort_values(ascending=False).index).head(15).reset_index(drop=True)

def price_sgp(df: pd.DataFrame, legs: list, book_sgp_odds: int, n_sims: int = 100000):
    merged = None
    col_names = []
    labels = []
    for i, leg in enumerate(legs):
        col = f"L{i}_{leg['player']}_{leg['stat']}"
        col_names.append(col)
        labels.append(f"{leg['player']} ({leg['side']} {leg['stat']})")
        sub = df[df["PLAYER_NAME"] == leg["player"]][["GAME_ID", leg["stat"]]].rename(columns={leg["stat"]: col})
        if leg["side"] == "Under":
            sub[col] = -sub[col]
        merged = sub if merged is None else pd.merge(merged, sub, on="GAME_ID")

    if len(merged) < 10:
        return {"error": f"Only {len(merged)} shared games found. Pick teammates or frequent opponents."}

    corr_matrix = merged[col_names].corr().to_numpy().copy()
    np.fill_diagonal(corr_matrix, 1.0)
    corr_df = pd.DataFrame(np.round(corr_matrix, 3), index=labels, columns=labels)
    fair_probs = [get_no_vig_prob(leg["odds"], leg.get("opp_odds", 0)) for leg in legs]

    z = np.random.multivariate_normal(np.zeros(len(legs)), corr_matrix, size=n_sims)
    u = norm.cdf(z)

    hits = np.ones(n_sims, dtype=bool)
    for i, p in enumerate(fair_probs):
        hits &= (u[:, i] < p)

    joint_prob = float(np.mean(hits))
    indep_prob = float(np.prod(fair_probs))
    book_dec = american_to_decimal(book_sgp_odds)
    ev_pct = (joint_prob * book_dec - 1) * 100

    return {
        "shared_games": len(merged),
        "corr_df": corr_df,
        "fair_leg_probs": [round(p * 100, 1) for p in fair_probs],
        "naive_odds": prob_to_american(indep_prob),
        "fair_odds": prob_to_american(joint_prob),
        "fair_prob": joint_prob,
        "ev_pct": ev_pct
    }

def render_clean_prop_board(source_df: pd.DataFrame, section_title: str, empty_msg: str, book: str = "draftkings", from_game: str = ""):
    if source_df.empty:
        st.markdown(
            f'<div class="board-card"><div class="board-header"><span class="board-title">{section_title}</span>'
            f'<span class="board-count">0 Plays</span></div>'
            f'<div style="padding: 0.9rem 1rem; color: #8B97AD; font-size: 0.84rem;">{empty_msg}</div></div>',
            unsafe_allow_html=True
        )
        return

    rows_html = []
    for _, r in source_df.iterrows():
        p_cell = player_badge_html(
            r["Player"], r["Headshot"], r["TeamLogo"], r.get("Pos", ""),
            book=book, from_game=from_game, inj_status=r.get("InjStatus", "")
        )
        rows_html.append(
            f'<tr>'
            f'<td style="width:28%;">{p_cell}</td>'
            f'<td style="width:16%;" class="line-cell">{r["Line"]}</td>'
            f'<td style="width:10%;" class="proj-cell col-right">{r["Projection"]:.1f}</td>'
            f'<td style="width:14%;">{r["Matchup"]}</td>'
            f'<td style="width:18%;" class="{r["RatingClass"]}">{r["Matchup Rating"]}</td>'
            f'<td style="width:8%;" class="col-right">{r["Odds"]:+d}</td>'
            f'<td style="width:6%;" class="proj-cell col-right">{r["Win Prob"]:.0f}%</td>'
            f'</tr>'
        )

    table_html = (
        f'<div class="board-card">'
        f'<div class="board-header"><span class="board-title">{section_title}</span><span class="board-count">{len(source_df)} Props · Click Player</span></div>'
        f'<div class="table-scroll-wrap"><table class="prop-table">'
        f'<thead><tr>'
        f'<th style="width:28%;">Player</th>'
        f'<th style="width:16%;">Line</th>'
        f'<th style="width:10%;" class="col-right">Proj</th>'
        f'<th style="width:14%;">Matchup</th>'
        f'<th style="width:18%;">DvP Rating</th>'
        f'<th style="width:8%;" class="col-right">Odds</th>'
        f'<th style="width:6%;" class="col-right">Win %</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

def render_box_score_table(team_name: str, win_prob: float, box_df: pd.DataFrame, injury_map: dict, book: str = "draftkings", from_game: str = ""):
    t_logo = get_team_logo(team_name)
    rows_html = []
    for _, r in box_df.iterrows():
        norm_p = normalize_name(r["Player"])
        inj_st = injury_map.get(norm_p, {}).get("status", "")
        p_cell = player_badge_html(r["Player"], r["Headshot"], r["TeamLogo"], r.get("Pos", ""), book=book, from_game=from_game, inj_status=inj_st)
        rows_html.append(
            f'<tr>'
            f'<td style="width:38%;">{p_cell}</td>'
            f'<td style="width:10%;" class="col-right">{r["MIN"]:.1f}</td>'
            f'<td style="width:10%;" class="proj-cell col-right">{r["PTS"]:.1f}</td>'
            f'<td style="width:10%;" class="col-right">{r["REB"]:.1f}</td>'
            f'<td style="width:10%;" class="col-right">{r["AST"]:.1f}</td>'
            f'<td style="width:10%;" class="col-right">{r["3PM"]:.1f}</td>'
            f'<td style="width:12%;" class="line-cell col-right">{r["PRA"]:.1f}</td>'
            f'</tr>'
        )

    html = (
        f'<div class="board-card">'
        f'<div class="board-header"><span class="board-title"><img class="team-inline-logo" src="{t_logo}" />{team_name}</span><span class="board-count">Win Prob: {win_prob:.1f}%</span></div>'
        f'<div class="table-scroll-wrap"><table class="prop-table">'
        f'<thead><tr>'
        f'<th style="width:38%;">Active Rotation (Click Player)</th>'
        f'<th style="width:10%;" class="col-right">MIN</th>'
        f'<th style="width:10%;" class="col-right">PTS</th>'
        f'<th style="width:10%;" class="col-right">REB</th>'
        f'<th style="width:10%;" class="col-right">AST</th>'
        f'<th style="width:10%;" class="col-right">3PM</th>'
        f'<th style="width:12%;" class="col-right">PRA</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

qp_player = st.query_params.get("player", "")
qp_game = st.query_params.get("game", "")
qp_book = st.query_params.get("book", "draftkings")
if qp_book not in ["draftkings", "fanduel", "betmgm", "caesars"]:
    qp_book = "draftkings"

if "selected_book" not in st.session_state:
    st.session_state["selected_book"] = qp_book
if qp_player:
    st.session_state["active_player_profile"] = qp_player
if qp_game:
    st.session_state["active_game_page"] = qp_game

df = load_nba_data()
injury_map, team_out_lists = fetch_nba_injury_report()
out_players_tuple = tuple(sorted([k for k, v in injury_map.items() if v.get("is_out")]))
media_lookup = build_player_media_lookup(df)
team_metrics = compute_advanced_team_metrics(df)

st.markdown(
    f'<div class="brand-bar"><div class="brand-left"><div class="brand-logo">HP</div>'
    f'<div><p class="brand-name">{BRAND_NAME}</p><p class="brand-sub">{BRAND_SUBTITLE}</p></div></div>'
    f'<span class="status-badge"><span class="status-dot"></span> PROPLINE + DvP ENGINE</span></div>',
    unsafe_allow_html=True
)

all_game_labels, game_lines, slate_props, fetched_at, slate_date_label = fetch_and_lock_propline_slate(
    bookmaker=st.session_state["selected_book"], max_games=12
)

edges_df = evaluate_straight_edges(df, slate_props, game_lines, team_metrics, media_lookup, injury_map) if slate_props else pd.DataFrame()

team_today_opp = {}
for g_lbl in all_game_labels:
    pts_g = [t.strip() for t in g_lbl.split("@")]
    if len(pts_g) == 2:
        team_today_opp[pts_g[0]] = (pts_g[1], "Away", g_lbl)
        team_today_opp[pts_g[1]] = (pts_g[0], "Home", g_lbl)

# =========================================================
# SAME-TAB PLAYER SCOUTING PROFILE
# =========================================================
scout_player = st.session_state.get("active_player_profile")
if scout_player:
    nav_p1, _ = st.columns([1.6, 4.4])
    with nav_p1:
        back_lbl = "← Back to Matchup Box Score" if st.session_state.get("active_game_page") else "← Back to Full Slate"
        if st.button(back_lbl, icon=":material/arrow_back:", width="stretch"):
            st.session_state["active_player_profile"] = None
            if "player" in st.query_params:
                del st.query_params["player"]
            st.rerun()

    norm_scout = normalize_name(scout_player)
    p_hist = df[df["NORM_NAME"] == norm_scout].copy()
    if p_hist.empty:
        st.warning(f"No historical game logs found for {scout_player}.")
        st.stop()

    p_display = p_hist["PLAYER_NAME"].iloc[-1]
    p_team = p_hist["TEAM_NAME"].iloc[-1]
    p_media = media_lookup.get(norm_scout, {})
    h_url = p_media.get("headshot", DEFAULT_HEADSHOT)
    t_logo = get_team_logo(p_team)
    ath_id = p_media.get("athlete_id", 0)
    pos_group = p_media.get("pos_group", p_hist["POS_GROUP"].iloc[-1])
    raw_pos = p_media.get("raw_pos", pos_group)
    pos_plural = POS_FULL_PLURAL.get(pos_group, "Players")
    p_inj = injury_map.get(norm_scout, {})

    if p_team in team_today_opp:
        opp_team, loc_label, matchup_lbl = team_today_opp[p_team]
    else:
        opp_team = str(p_hist["OPP_NAME"].iloc[-1])
        loc_label = "Matchup"
        matchup_lbl = f"{p_team} vs {opp_team}"

    opp_logo = get_team_logo(opp_team)
    p_lines_df = edges_df[edges_df["NORM_NAME"] == norm_scout] if not edges_df.empty else pd.DataFrame()

    inj_banner_tag = ""
    if p_inj:
        cls_inj = "inj-out-pill" if p_inj.get("is_out") else "inj-q-pill"
        inj_banner_tag = f' &nbsp;<span class="{cls_inj}">INJURY REPORT: {p_inj["status"].upper()}</span>'

    st.markdown(
        f'<div class="game-hero"><div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">'
        f'<div style="display:flex; align-items:center; gap:14px;">'
        f'<img src="{h_url}" onerror="this.src=\'{DEFAULT_HEADSHOT}\'" style="width:68px; height:68px; border-radius:50%; object-fit:cover; background:#1E2638; border:2px solid #10B981;" />'
        f'<div><div style="font-size:0.78rem; color:#10B981; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Player Prop Scouting · Position: {raw_pos} ({pos_plural}){inj_banner_tag}</div>'
        f'<div style="font-size:1.65rem; font-weight:800; color:#F8FAFC; line-height:1.2;">{p_display}</div>'
        f'<div style="font-size:0.86rem; color:#94A3B8; margin-top:3px; display:flex; align-items:center; gap:6px;">'
        f'<img src="{t_logo}" style="width:18px; height:18px;" /> <b>{p_team}</b> &nbsp;·&nbsp; {loc_label} vs '
        f'<img src="{opp_logo}" style="width:18px; height:18px;" /> <b>{opp_team}</b></div></div></div>'
        f'<div style="text-align:right; font-size:0.84rem; color:#CBD5E1;">'
        f'<div>Season Avg: <b>{p_hist.tail(40)["PTS"].mean():.1f} PTS · {p_hist.tail(40)["REB"].mean():.1f} REB · {p_hist.tail(40)["AST"].mean():.1f} AST</b></div>'
        f'<div style="margin-top:3px;">Last 10 Avg: <b>{p_hist.tail(10)["PTS"].mean():.1f} PTS · {p_hist.tail(10)["REB"].mean():.1f} REB · {p_hist.tail(10)["AST"].mean():.1f} AST</b></div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    if not p_lines_df.empty:
        render_clean_prop_board(
            p_lines_df.drop_duplicates(subset=["Stat"]),
            f"{p_display} — Today's Locked Pre-Game Lines & Position-Adjusted Edges",
            "",
            book=st.session_state["selected_book"]
        )

    sc_c1, sc_c2 = st.columns([1.2, 2.8], vertical_alignment="bottom")
    with sc_c1:
        stat_choice = st.selectbox("Inspect Stat Trend", ["PTS", "REB", "AST", "FG3M", "PRA"])
    default_line = round(float(p_hist.tail(15)[stat_choice].median()) - 0.5, 1)
    if not p_lines_df.empty and stat_choice in set(p_lines_df["Stat"]):
        default_line = float(p_lines_df[p_lines_df["Stat"] == stat_choice]["RawLine"].iloc[0])
    with sc_c2:
        custom_line = st.number_input(f"Target {stat_choice} Line", value=max(0.5, default_line), step=0.5)

    s_series = p_hist[stat_choice].dropna()
    hr5 = (s_series.tail(5) > custom_line).mean() * 100
    hr10 = (s_series.tail(10) > custom_line).mean() * 100
    hr20 = (s_series.tail(20) > custom_line).mean() * 100
    hr_all = (s_series.tail(60) > custom_line).mean() * 100

    hk1, hk2, hk3, hk4 = st.columns(4)
    hk1.metric(f"Last 5 Over {custom_line}", f"{hr5:.0f}%", f"{int((s_series.tail(5) > custom_line).sum())}/5 Games")
    hk2.metric(f"Last 10 Over {custom_line}", f"{hr10:.0f}%", f"{int((s_series.tail(10) > custom_line).sum())}/10 Games")
    hk3.metric(f"Last 20 Over {custom_line}", f"{hr20:.0f}%", f"{int((s_series.tail(20) > custom_line).sum())}/20 Games")
    hk4.metric(f"Sample Over {custom_line}", f"{hr_all:.0f}%", f"{int((s_series.tail(60) > custom_line).sum())}/{len(s_series.tail(60))} Games")

    st.plotly_chart(build_hit_rate_chart(p_hist, stat_choice, custom_line, p_display), use_container_width=True)

    chart_col, heat_col = st.columns([1.1, 1.05], gap="large")

    with chart_col:
        recent_gids = tuple(p_hist["GAME_ID"].drop_duplicates().tail(8).tolist())
        shots_df = fetch_player_shot_chart(ath_id, recent_gids)
        st.plotly_chart(build_shot_chart_figure(shots_df, p_display), use_container_width=True)
        if not shots_df.empty:
            z_cols = st.columns(3)
            for zi, z_name in enumerate(["Paint / Rim", "Mid-Range", "3-Point"]):
                z_sub = shots_df[shots_df["zone"] == z_name]
                fg_pct = (z_sub["made"].mean() * 100) if not z_sub.empty else 0.0
                share_pct = (len(z_sub) / len(shots_df)) * 100
                z_cols[zi].metric(z_name, f"{fg_pct:.0f}% FG", f"{share_pct:.0f}% of Shots ({int(z_sub['made'].sum())}/{len(z_sub)})")

    with heat_col:
        st.markdown(f"##### {opp_team} Defense vs. {pos_plural} (DvP Heat Map)")
        st.caption(f"Rank 1 = Toughest vs {pos_plural} · Rank 30 = Allows Most to {pos_plural}")
        opp_pos_rks = team_metrics["dvp_ranks"].get(opp_team, {}).get(pos_group, {})
        opp_pos_avgs = team_metrics["dvp_avg"].get(opp_team, {}).get(pos_group, {})
        opp_pos_mults = team_metrics["dvp_mult"].get(opp_team, {}).get(pos_group, {})

        heat_items = [
            ("PTS", f"PTS Allowed to {pos_plural}"),
            ("REB", f"REB Allowed to {pos_plural}"),
            ("AST", f"AST Allowed to {pos_plural}"),
            ("FG3M", f"3PM Allowed to {pos_plural}"),
            ("PRA", f"PRA Allowed to {pos_plural}"),
        ]
        tiles_html = []
        for s_code, s_lbl in heat_items:
            rk = opp_pos_rks.get(s_code, 15)
            avg_v = opp_pos_avgs.get(s_code, 0.0)
            mult_v = opp_pos_mults.get(s_code, 1.0)
            diff_pct = (mult_v - 1.0) * 100
            if rk >= 21:
                t_cls = "heat-smash"
                tag = "Smash"
            elif rk <= 10:
                t_cls = "heat-tough"
                tag = "Tough"
            else:
                t_cls = "heat-neutral"
                tag = "Neutral"
            tiles_html.append(
                f'<div class="heat-tile {t_cls}">'
                f'<div class="heat-label">{s_lbl}</div>'
                f'<div class="heat-rank">{ordinal(rk)} <span style="font-size:0.75rem; font-weight:600;">({tag})</span></div>'
                f'<div class="heat-sub">{avg_v:.1f}/gm ({diff_pct:+.1f}% vs Lg)</div>'
                f'</div>'
            )

        if "HOME_AWAY" in p_hist.columns and p_hist["HOME_AWAY"].astype(str).str.lower().isin(["home", "away"]).any():
            target_ha = "home" if loc_label == "Home" else "away"
            ha_sub = p_hist[p_hist["HOME_AWAY"].astype(str).str.lower() == target_ha].tail(25)
            ha_val = ha_sub[stat_choice].mean() if not ha_sub.empty else s_series.tail(25).mean()
            tiles_html.append(
                f'<div class="heat-tile heat-neutral">'
                f'<div class="heat-label">{loc_label} Split ({stat_choice})</div>'
                f'<div class="heat-rank">{ha_val:.1f} Avg</div>'
                f'<div class="heat-sub">Last {len(ha_sub)} {loc_label} Games</div>'
                f'</div>'
            )

        st.markdown(f'<div class="heatmap-grid">{"".join(tiles_html)}</div>', unsafe_allow_html=True)

        st.markdown(f"<div style='margin-top:1rem; font-weight:700; color:#F8FAFC; font-size:0.9rem;'>Recent Primary {pos_plural} vs {opp_team}</div>", unsafe_allow_html=True)
        opp_pos_pool = df[
            (df["OPP_NAME"] == opp_team) &
            (df["POS_GROUP"] == pos_group) &
            (df["MIN"] >= 30)
        ].copy()
        if not opp_pos_pool.empty:
            opp_pos_pool = (
                opp_pos_pool.sort_values(by=["GAME_DATE", "MIN"], ascending=[False, False])
                .drop_duplicates(subset=["GAME_ID"], keep="first")
                .head(2)
            )

        if opp_pos_pool.empty:
            st.caption(f"No recent primary {pos_plural.lower()} logs found against {opp_team}.")
        else:
            dvp_rows = []
            for _, dr in opp_pos_pool.iterrows():
                d_s = dr["GAME_DATE"].strftime("%b %d") if pd.notna(dr["GAME_DATE"]) else "-"
                d_norm = dr["NORM_NAME"]
                d_med = media_lookup.get(d_norm, {})
                d_badge = player_badge_html(
                    dr["PLAYER_NAME"],
                    d_med.get("headshot", DEFAULT_HEADSHOT),
                    get_team_logo(str(dr.get("TEAM_NAME", ""))),
                    dr.get("RAW_POS", pos_group),
                    book=st.session_state["selected_book"]
                )
                dvp_rows.append(
                    f'<tr>'
                    f'<td style="width:36%;">{d_badge}</td>'
                    f'<td style="width:12%;">{d_s}</td>'
                    f'<td style="width:10%;" class="col-right">{dr["MIN"]:.0f}m</td>'
                    f'<td style="width:10%;" class="proj-cell col-right">{dr["PTS"]:.0f}</td>'
                    f'<td style="width:10%;" class="col-right">{dr["REB"]:.0f}</td>'
                    f'<td style="width:10%;" class="col-right">{dr["AST"]:.0f}</td>'
                    f'<td style="width:12%;" class="line-cell col-right">{dr["PRA"]:.0f}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div class="board-card" style="margin-top:0.4rem;"><div class="table-scroll-wrap"><table class="prop-table">'
                f'<thead><tr><th style="width:36%;">Player</th><th style="width:12%;">Date</th>'
                f'<th style="width:10%;" class="col-right">MIN</th><th style="width:10%;" class="col-right">PTS</th>'
                f'<th style="width:10%;" class="col-right">REB</th><th style="width:10%;" class="col-right">AST</th>'
                f'<th style="width:12%;" class="col-right">PRA</th></tr></thead>'
                f'<tbody>{"".join(dvp_rows)}</tbody></table></div></div>',
                unsafe_allow_html=True
            )

    st.markdown(f"##### {p_display} — Recent Game Log (Last 15 Games)")
    gl_rows = []
    for _, gr in p_hist.tail(15).iloc[::-1].iterrows():
        g_dt = gr["GAME_DATE"].strftime("%b %d") if pd.notna(gr["GAME_DATE"]) else "-"
        opp_n = str(gr.get("OPP_NAME", ""))
        opp_lg = get_team_logo(opp_n)
        ha_tag = "vs" if str(gr.get("HOME_AWAY", "")).lower() == "home" else "@"
        fg_str = f"{int(gr['FGM'])}-{int(gr['FGA'])}" if pd.notna(gr.get("FGM")) and pd.notna(gr.get("FGA")) else "-"
        gl_rows.append(
            f'<tr>'
            f'<td style="width:11%;">{g_dt}</td>'
            f'<td style="width:23%;"><img class="team-inline-logo" src="{opp_lg}" />{ha_tag} {opp_n}</td>'
            f'<td style="width:9%;" class="col-right">{gr["MIN"]:.0f}</td>'
            f'<td style="width:9%;" class="proj-cell col-right">{gr["PTS"]:.0f}</td>'
            f'<td style="width:9%;" class="col-right">{gr["REB"]:.0f}</td>'
            f'<td style="width:9%;" class="col-right">{gr["AST"]:.0f}</td>'
            f'<td style="width:9%;" class="col-right">{gr["FG3M"]:.0f}</td>'
            f'<td style="width:11%;" class="line-cell col-right">{gr["PRA"]:.0f}</td>'
            f'<td style="width:10%;" class="col-right">{fg_str}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div class="board-card"><div class="table-scroll-wrap"><table class="prop-table">'
        f'<thead><tr>'
        f'<th style="width:11%;">Date</th><th style="width:23%;">Opponent</th>'
        f'<th style="width:9%;" class="col-right">MIN</th><th style="width:9%;" class="col-right">PTS</th>'
        f'<th style="width:9%;" class="col-right">REB</th><th style="width:9%;" class="col-right">AST</th>'
        f'<th style="width:9%;" class="col-right">3PM</th><th style="width:11%;" class="col-right">PRA</th>'
        f'<th style="width:10%;" class="col-right">FG</th>'
        f'</tr></thead><tbody>{"".join(gl_rows)}</tbody></table></div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="site-footer"><span><b>{BRAND_NAME}</b> · Quantitative NBA Analytics & Simulation Engine</span>'
        f'<span>For informational & entertainment purposes only. Please wager responsibly.</span></div>',
        unsafe_allow_html=True
    )
    st.stop()

# =========================================================
# SAME-TAB GAME DETAIL PAGE
# =========================================================
target_game = st.session_state.get("active_game_page")
if target_game and "@" in target_game:
    nav_c1, _ = st.columns([1.4, 4.6])
    with nav_c1:
        if st.button("← Back to Full Slate", icon=":material/arrow_back:", width="stretch"):
            st.session_state["active_game_page"] = None
            if "game" in st.query_params:
                del st.query_params["game"]
            st.rerun()

    parts = [t.strip() for t in target_game.split("@")]
    away_sel, home_sel = parts[0], parts[1]
    g_line = game_lines.get(target_game, {"spread_home": 0.0, "total": 228.0, "tip_str": ""})
    detail_sim = simulate_full_game(df, away_sel, home_sel, team_metrics, g_line, media_lookup, out_players_key=out_players_tuple, n_sims=15000)

    fav_short = home_sel.split()[-1] if detail_sim["proj_spread"] < 0 else away_sel.split()[-1]
    spread_num = abs(detail_sim["proj_spread"])
    book_spread_str = f"{home_sel.split()[-1]} {g_line['spread_home']:+.1f}" if g_line["spread_home"] != 0 else "N/A"
    away_logo = get_team_logo(away_sel)
    home_logo = get_team_logo(home_sel)

    st.markdown(
        f'<div class="game-hero"><div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">'
        f'<div><div style="font-size:0.78rem; color:#10B981; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">'
        f'Projected Matchup & Player Box Scores · {g_line.get("tip_str", slate_date_label)}</div>'
        f'<div style="font-size:1.55rem; font-weight:800; color:#F8FAFC; margin-top:0.35rem; display:flex; align-items:center; flex-wrap:wrap; gap:8px;">'
        f'<img src="{away_logo}" style="width:28px; height:28px; object-fit:contain;" /><span>{away_sel}</span>'
        f'<span style="color:#10B981;">{detail_sim["away_score"]}</span><span style="color:#64748B; font-size:1.1rem;">@</span>'
        f'<img src="{home_logo}" style="width:28px; height:28px; object-fit:contain;" /><span>{home_sel}</span>'
        f'<span style="color:#10B981;">{detail_sim["home_score"]}</span></div></div>'
        f'<div style="text-align:right; font-size:0.86rem; color:#CBD5E1; line-height:1.6;">'
        f'<div>Projected Line: <b>{fav_short} -{spread_num}</b> &nbsp;|&nbsp; Book Line: <b>{book_spread_str}</b></div>'
        f'<div>Projected Total: <b>{detail_sim["proj_total"]}</b> &nbsp;|&nbsp; Book Total: <b>{g_line["total"]}</b></div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    away_outs = team_out_lists.get(away_sel, [])
    home_outs = team_out_lists.get(home_sel, [])
    if away_outs or home_outs:
        out_parts = []
        if away_outs:
            out_parts.append(f"<b>{away_sel} OUT:</b> {', '.join(away_outs)}")
        if home_outs:
            out_parts.append(f"<b>{home_sel} OUT:</b> {', '.join(home_outs)}")
        st.markdown(f'<div class="inj-strip">🚫 <b>Injury Report (Excluded from Projections):</b> &nbsp;{" &nbsp;|&nbsp; ".join(out_parts)}</div>', unsafe_allow_html=True)

    st.markdown("##### Projected Active Player Box Scores (Click Player to Scout)")
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        render_box_score_table(f"{away_sel} (Away)", detail_sim["away_win_prob"], detail_sim["away_box"], injury_map, book=st.session_state["selected_book"], from_game=target_game)
    with b2:
        render_box_score_table(f"{home_sel} (Home)", detail_sim["home_win_prob"], detail_sim["home_box"], injury_map, book=st.session_state["selected_book"], from_game=target_game)

    st.markdown(f"##### Locked Pre-Game Player Lines & DvP Matchup Ratings — {away_sel} @ {home_sel}")
    game_props_df = edges_df[edges_df["FullMatchup"] == target_game] if not edges_df.empty else pd.DataFrame()
    if not game_props_df.empty:
        game_props_df = game_props_df.drop_duplicates(subset=["Player", "Stat"]).sort_values(by=["Stat", "Win Prob"], ascending=[True, False])
    render_clean_prop_board(
        game_props_df,
        f"{away_sel} @ {home_sel} — Pre-Game Lines, Projections & DvP Ratings",
        "No pre-game player props were posted for this matchup.",
        book=st.session_state["selected_book"],
        from_game=target_game
    )

    st.markdown(
        f'<div class="site-footer"><span><b>{BRAND_NAME}</b> · Quantitative NBA Analytics & Simulation Engine</span>'
        f'<span>For informational & entertainment purposes only. Please wager responsibly.</span></div>',
        unsafe_allow_html=True
    )
    st.stop()

# =========================================================
# MAIN DASHBOARD VIEW (SLATE-FIRST)
# =========================================================
ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.2, 2.6, 0.8], vertical_alignment="bottom")
with ctrl_c1:
    bookmaker = st.selectbox(
        "Sportsbook",
        ["draftkings", "fanduel", "betmgm", "caesars"],
        index=["draftkings", "fanduel", "betmgm", "caesars"].index(st.session_state["selected_book"]),
        format_func=lambda x: {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "caesars": "Caesars"}[x]
    )
    if bookmaker != st.session_state["selected_book"]:
        st.session_state["selected_book"] = bookmaker
        st.query_params["book"] = bookmaker
        st.rerun()
with ctrl_c2:
    search_player = st.text_input("Search", placeholder="Filter by player or team...")
with ctrl_c3:
    if st.button("Refresh", icon=":material/refresh:", width="stretch"):
        fetch_and_lock_propline_slate.clear()
        fetch_nba_injury_report.clear()
        st.rerun()

slate_teams = set()
for g_lbl in all_game_labels:
    for t in g_lbl.split("@"):
        slate_teams.add(t.strip())
slate_outs_count = sum(len(v) for k, v in team_out_lists.items() if k in slate_teams)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(f"Today's Slate ({slate_date_label})", f"{len(all_game_labels)} Games")
kpi2.metric("Active Pre-Game Props", f"{len(edges_df)} Lines")
kpi3.metric("Ruled Out (Filtered)", f"{slate_outs_count} Players")
kpi4.metric("Last Slate Sync", fetched_at)

st.write("")

# Streamlined 3-Tab Layout (No More Noisy Straight Bets Board)
tab_slate, tab_parlays, tab_custom = st.tabs([
    ":material/scoreboard: Slate & Matchup Projections",
    ":material/bolt: Auto-Parlay Builder",
    ":material/tune: Custom SGP Lab"
])

# =========================================================
# TAB 1: SLATE & MATCHUP PROJECTIONS (DEFAULT LANDING)
# =========================================================
with tab_slate:
    st.markdown(f"##### {slate_date_label} Projected NBA Scores & Matchups")
    st.caption("Injury-adjusted Monte Carlo simulations. Click any game card to open box scores, player props, and DvP matchups.")

    slate_list = all_game_labels if all_game_labels else [
        "Denver Nuggets @ Phoenix Suns",
        "Boston Celtics @ New York Knicks",
        "Dallas Mavericks @ Oklahoma City Thunder"
    ]

    grid_cols = st.columns(3)
    for idx, g_lbl in enumerate(slate_list):
        parts = [t.strip() for t in g_lbl.split("@")]
        if len(parts) != 2:
            continue
        away_t, home_t = parts[0], parts[1]
        
        # Apply search filter if active
        if search_player.strip() and not (search_player.strip().lower() in away_t.lower() or search_player.strip().lower() in home_t.lower()):
            continue

        g_line = game_lines.get(g_lbl, {"spread_home": 0.0, "total": 228.0, "tip_str": ""})
        sim_res = simulate_full_game(df, away_t, home_t, team_metrics, g_line, media_lookup, out_players_key=out_players_tuple, n_sims=15000)
        fav_short = home_t.split()[-1] if sim_res["proj_spread"] < 0 else away_t.split()[-1]
        spread_num = abs(sim_res["proj_spread"])
        tip_badge = g_line.get("tip_str", "")
        away_logo = get_team_logo(away_t)
        home_logo = get_team_logo(home_t)

        with grid_cols[idx % 3]:
            st.markdown(
                f'<div class="matchup-card">'
                f'<div class="matchup-team-row"><span class="matchup-team"><img class="team-inline-logo" src="{away_logo}" />{away_t} <span class="matchup-prob">({sim_res["away_win_prob"]}%)</span></span><span class="matchup-pts">{sim_res["away_score"]}</span></div>'
                f'<div class="matchup-team-row"><span class="matchup-team"><img class="team-inline-logo" src="{home_logo}" />{home_t} <span class="matchup-prob">({sim_res["home_win_prob"]}%)</span></span><span class="matchup-pts">{sim_res["home_score"]}</span></div>'
                f'<div class="matchup-footer"><span>{tip_badge} · Line: <b>{fav_short} -{spread_num}</b></span><span>Total: <b>{sim_res["proj_total"]}</b></span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            if st.button(
                f"View Box Score & Props · {away_t.split()[-1]} @ {home_t.split()[-1]}",
                key=f"open_game_{idx}",
                icon=":material/arrow_forward:",
                width="stretch"
            ):
                st.session_state["active_game_page"] = g_lbl
                st.query_params["game"] = g_lbl
                st.rerun()

# =========================================================
# TAB 2: AUTO-PARLAY BUILDER
# =========================================================
with tab_parlays:
    if not slate_props:
        st.info("Auto-parlays lock in automatically as soon as pre-game player props open for the slate.")
    else:
        col_multi, col_sgp = st.columns(2, gap="large")

        with col_multi:
            st.markdown("##### Cross-Game +EV Parlay")
            st.caption("Combines the highest win-probability active model props from each pre-game matchup.")
            num_parlay_legs = st.radio("Parlay Size", [2, 3, 4], horizontal=True)
            if not edges_df.empty:
                best_per_game = edges_df.drop_duplicates(subset=["FullMatchup"]).head(num_parlay_legs)
                if len(best_per_game) >= 2:
                    comb_dec = float(np.prod([american_to_decimal(o) for o in best_per_game["Odds"]]))
                    comb_mod_p = float(np.prod(best_per_game["Win Prob"] / 100.0))
                    parlay_odds = round((comb_dec - 1) * 100) if comb_dec >= 2.0 else round(-100 / (comb_dec - 1))

                    m1, m2 = st.columns(2)
                    m1.metric("Combined Odds", f"{parlay_odds:+d}")
                    m2.metric("Combined Win Prob", f"{comb_mod_p*100:.1f}%")

                    slip_lines = [f"{BRAND_NAME} ({num_parlay_legs}-Leg Parlay | {parlay_odds:+d})"]
                    for _, row in best_per_game.iterrows():
                        slip_lines.append(f"- {row['Player']} | {row['Line']} ({row['Odds']:+d})")
                        p_badge = player_badge_html(
                            row["Player"], row["Headshot"], row["TeamLogo"], row.get("Pos", ""),
                            book=st.session_state["selected_book"], inj_status=row.get("InjStatus", "")
                        )
                        st.markdown(
                            f'<div class="pick-card"><div class="pick-top">{p_badge}<span class="line-cell">{row["Odds"]:+d}</span></div>'
                            f'<div class="pick-line">{row["Line"]}</div>'
                            f'<div class="pick-details">Proj: <b>{row["Projection"]}</b> · Rating: <b>{row["Matchup Rating"]}</b> · Win %: <b>{row["Win Prob"]:.0f}%</b></div></div>',
                            unsafe_allow_html=True
                        )

                    with st.expander("Copy Parlay Slip", icon=":material/content_copy:"):
                        st.code("\n".join(slip_lines), language=None)
                else:
                    st.warning("At least 2 pre-game matchups with active props are needed to build a cross-game parlay.")

        with col_sgp:
            st.markdown("##### Correlated Same-Game Parlays")
            st.caption("Pairs active pre-game lines with strong historical game-log correlations.")
            sgp_game = st.selectbox("Select Matchup", list(slate_props.keys()))
            sgp_df = build_auto_sgps(df, sgp_game, slate_props[sgp_game], injury_map)
            if sgp_df.empty:
                st.warning("No high-correlation active prop pairs found for this matchup yet.")
            else:
                st.dataframe(
                    sgp_df.drop(columns=["Game"]),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Angle": st.column_config.TextColumn("Angle", width="small"),
                        "Leg 1": st.column_config.TextColumn("Leg 1", width="medium"),
                        "Leg 2": st.column_config.TextColumn("Leg 2", width="medium"),
                        "Correlation": st.column_config.NumberColumn("Corr (r)", format="%+.3f"),
                        "Book Naive": st.column_config.NumberColumn("Naive Odds", format="%+d"),
                        "True Fair": st.column_config.NumberColumn("True Fair", format="%+d"),
                    }
                )

# =========================================================
# TAB 3: CUSTOM SGP LAB
# =========================================================
with tab_custom:
    st.markdown("##### Custom Multi-Leg SGP Correlation Pricer")
    st.caption("Calculate true joint probability and fair odds for any custom player prop combination.")
    teams = sorted(df["TEAM_NAME"].dropna().unique())
    c1, c2 = st.columns(2)
    with c1:
        team_a = st.selectbox("Team A", teams, index=teams.index("Denver Nuggets") if "Denver Nuggets" in teams else 0)
    with c2:
        team_b = st.selectbox("Team B (Opponent)", teams, index=teams.index("Phoenix Suns") if "Phoenix Suns" in teams else 1)

    matchup_players = sorted(df[df["TEAM_NAME"].isin([team_a, team_b])]["PLAYER_NAME"].dropna().unique())
    stats = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]
    num_legs = st.slider("Number of SGP Legs", min_value=2, max_value=5, value=3)
    legs = []

    cols = st.columns(num_legs)
    for i in range(num_legs):
        with cols[i]:
            st.markdown(f"**Leg {i+1}**")
            p = st.selectbox("Player", matchup_players, index=min(i, len(matchup_players)-1), key=f"p{i}")
            side = st.radio("Side", ["Over", "Under"], horizontal=True, key=f"side{i}")
            s = st.selectbox("Stat", stats, index=2 if i == 0 else 0, key=f"s{i}")
            o = st.number_input("Selected Odds", value=-115, step=5, key=f"o{i}")
            opp_o = st.number_input("Other Side Odds", value=-105, step=5, key=f"opp{i}")
            legs.append({"player": p, "side": side, "stat": s, "odds": int(o), "opp_odds": int(opp_o)})

    book_odds = st.number_input("Sportsbook Offered SGP Odds (e.g. +550)", value=550, step=10)
    if st.button("Price Custom SGP", type="primary"):
        res = price_sgp(df, legs, int(book_odds))
        if "error" in res:
            st.error(res["error"])
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Shared Games", res["shared_games"])
            m2.metric("Naive Odds", f"{res['naive_odds']:+d}")
            m3.metric("TRUE Fair Odds", f"{res['fair_odds']:+d}")
            m4.metric("Fair Probability", f"{res['fair_prob']*100:.1f}%")
            st.dataframe(res["corr_df"], width="stretch")

st.markdown(
    f'<div class="site-footer"><span><b>{BRAND_NAME}</b> · Quantitative NBA Analytics & Simulation Engine</span>'
    f'<span>For informational & entertainment purposes only. Please wager responsibly.</span></div>',
    unsafe_allow_html=True
)