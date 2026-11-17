import re
import unicodedata
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm, gamma

# ==========================================
# BRAND CONFIGURATION
# ==========================================
BRAND_NAME = "Hector's Picks"
BRAND_SUBTITLE = "NBA Prop Projections · +EV Parlay Engine · Monte Carlo Matchup Simulator"

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
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1340px;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }

.brand-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.95rem 1.35rem;
    background: #111622;
    border: 1px solid #1E2638;
    border-radius: 12px;
    margin-bottom: 0.85rem;
    flex-wrap: wrap;
    gap: 0.75rem;
}
.brand-left { display: flex; align-items: center; gap: 0.8rem; }
.brand-logo {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.05rem;
    color: #06100C;
    letter-spacing: -0.5px;
}
.brand-name {
    font-size: 1.38rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.4px;
    margin: 0;
    line-height: 1.15;
}
.brand-sub {
    font-size: 0.8rem;
    color: #8B97AD;
    margin: 0.1rem 0 0 0;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.28);
    color: #34D399;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 11px;
    border-radius: 999px;
}
.status-dot {
    width: 7px;
    height: 7px;
    background: #10B981;
    border-radius: 50%;
}

.player-flex {
    display: flex;
    align-items: center;
    gap: 9px;
    overflow: hidden;
}
.headshot-wrap {
    position: relative;
    width: 32px;
    height: 32px;
    flex-shrink: 0;
}
.headshot-img {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    object-fit: cover;
    background: #1E2638;
    border: 1px solid #2D3952;
}
.team-mini-logo {
    position: absolute;
    bottom: -2px;
    right: -3px;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: #0F1420;
    padding: 1px;
    object-fit: contain;
    border: 1px solid #263147;
}
.team-inline-logo {
    width: 20px;
    height: 20px;
    object-fit: contain;
    vertical-align: middle;
    margin-right: 6px;
}
.player-name-txt {
    font-weight: 600;
    color: #F8FAFC;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.board-card {
    background: #111622;
    border: 1px solid #1E2638;
    border-radius: 11px;
    margin-bottom: 1.1rem;
    overflow: hidden;
}
.board-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 1rem;
    background: #151C2C;
    border-bottom: 1px solid #1E2638;
    border-left: 3px solid #10B981;
}
.board-title {
    font-size: 0.96rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.2px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.board-count {
    font-size: 0.73rem;
    font-weight: 600;
    color: #94A3B8;
    background: #1E2638;
    padding: 2px 9px;
    border-radius: 999px;
}
table.prop-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    table-layout: fixed;
}
table.prop-table th {
    text-align: left;
    padding: 0.52rem 0.65rem;
    color: #8B97AD;
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    border-bottom: 1px solid #1E2638;
    background: #111622;
}
table.prop-table td {
    padding: 0.46rem 0.65rem;
    color: #E2E8F0;
    border-bottom: 1px solid #182032;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    vertical-align: middle;
}
table.prop-table tr:last-child td { border-bottom: none; }
table.prop-table tr:hover td { background: #161E30; }
.col-right { text-align: right !important; }
.line-cell { font-weight: 600; color: #38BDF8; }
.proj-cell { font-weight: 700; color: #F8FAFC; }
.ev-pos {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 1px 7px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.75rem;
}
.ev-neg {
    color: #94A3B8;
    font-weight: 500;
    font-size: 0.77rem;
}
.res-won {
    display: inline-block;
    background: rgba(16, 185, 129, 0.18);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 1px 7px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.72rem;
}
.res-lost {
    display: inline-block;
    background: rgba(239, 68, 68, 0.16);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.38);
    padding: 1px 7px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.72rem;
}
.res-live {
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 1px 7px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.72rem;
}
.res-pre {
    color: #94A3B8;
    font-size: 0.74rem;
    font-weight: 600;
}
.rating-good { color: #34D399; font-weight: 600; }
.rating-bad { color: #F87171; font-weight: 600; }
.rating-neutral { color: #94A3B8; }

.pick-card {
    background: #111622;
    border: 1px solid #1E2638;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}
.pick-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
}
.pick-line { font-size: 0.92rem; font-weight: 600; color: #10B981; margin-bottom: 0.25rem; }
.pick-details { font-size: 0.79rem; color: #8B97AD; line-height: 1.4; }

.matchup-card {
    background: #111622;
    border: 1px solid #1E2638;
    border-radius: 11px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.5rem;
}
.matchup-team-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.28rem 0;
}
.matchup-team {
    font-size: 0.96rem;
    font-weight: 600;
    color: #F1F5F9;
    display: flex;
    align-items: center;
}
.matchup-prob { font-size: 0.78rem; color: #64748B; margin-left: 5px; }
.matchup-pts { font-size: 1.25rem; font-weight: 800; color: #10B981; }
.matchup-footer {
    display: flex;
    justify-content: space-between;
    margin-top: 0.55rem;
    padding-top: 0.5rem;
    border-top: 1px solid #1A2234;
    font-size: 0.78rem;
    color: #8B97AD;
}

.game-hero {
    background: linear-gradient(135deg, #111622 0%, #161F33 100%);
    border: 1px solid #26324B;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}

.site-footer {
    margin-top: 2.5rem;
    padding-top: 1.1rem;
    border-top: 1px solid #1E2638;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    font-size: 0.76rem;
    color: #64748B;
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

DEFAULT_HEADSHOT = "https://a.espncdn.com/combiner/i?img=/i/headshots/nophoto.png&w=96&h=70"

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

def player_badge_html(player_name: str, headshot_url: str, team_logo_url: str) -> str:
    return (
        f'<div class="player-flex">'
        f'<div class="headshot-wrap">'
        f'<img class="headshot-img" src="{headshot_url}" onerror="this.src=\'{DEFAULT_HEADSHOT}\'" />'
        f'<img class="team-mini-logo" src="{team_logo_url}" />'
        f'</div>'
        f'<span class="player-name-txt">{player_name}</span>'
        f'</div>'
    )

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

# Persistent Daily Pre-Game Snapshot Vault (Survives Tip-Offs All Night Long)
@st.cache_resource
def get_daily_pregame_vault() -> dict:
    return {}

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
        "turnovers": "TOV"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "ATHLETE_ID" not in df.columns:
        df["ATHLETE_ID"] = np.nan
    if "HEADSHOT_HREF" not in df.columns:
        df["HEADSHOT_HREF"] = ""
    if "TEAM_LOGO" not in df.columns:
        df["TEAM_LOGO"] = ""

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
        h_url = build_headshot_url(r.get("ATHLETE_ID"), str(r.get("HEADSHOT_HREF", "")))
        media[r["NORM_NAME"]] = {
            "headshot": h_url,
            "team_logo": t_logo,
            "team_name": t_name,
        }
    return media

@st.cache_data(ttl=21600)
def compute_advanced_team_metrics(df: pd.DataFrame) -> dict:
    stats = ["PTS", "REB", "AST", "FG3M", "PRA"]
    if "OPP_NAME" not in df.columns:
        return {"def_mult": {}, "team_pace": {}, "last_game": {}}

    max_season = df["SEASON"].max()
    recent_df = df[df["SEASON"] >= max_season - 1].copy()

    game_allowed = recent_df.groupby(["OPP_NAME", "GAME_ID", "SEASON"])[stats].sum().reset_index()
    game_allowed["w"] = np.where(game_allowed["SEASON"] == max_season, 2.0, 1.0)

    def weighted_avg(grp):
        w = grp["w"].values
        return pd.Series({s: np.average(grp[s].values, weights=w) for s in stats})

    team_allowed = game_allowed.groupby("OPP_NAME").apply(weighted_avg, include_groups=False)
    league_avg = team_allowed.mean()
    def_mult = (team_allowed / league_avg).fillna(1.0).to_dict(orient="index")

    game_scored = recent_df.groupby(["TEAM_NAME", "GAME_ID", "SEASON"])["PTS"].sum().reset_index()
    game_scored["w"] = np.where(game_scored["SEASON"] == max_season, 2.0, 1.0)
    team_scored = game_scored.groupby("TEAM_NAME").apply(
        lambda g: np.average(g["PTS"].values, weights=g["w"].values), include_groups=False
    )
    team_pace = {}
    lg_pts = float(league_avg["PTS"])
    for team in team_scored.index:
        off_factor = float(team_scored[team]) / lg_pts
        def_factor = def_mult.get(team, {}).get("PTS", 1.0)
        team_pace[team] = 0.5 * (off_factor + def_factor)

    last_game = {}
    if df["GAME_DATE"].notna().any():
        last_game = df.dropna(subset=["GAME_DATE"]).groupby("TEAM_NAME")["GAME_DATE"].max().to_dict()

    return {"def_mult": def_mult, "team_pace": team_pace, "last_game": last_game}

def format_matchup_rating(mult: float) -> tuple[str, str]:
    score = round(mult * 100, 1)
    if score >= 102.0:
        return f"{score} · Smash", "rating-good"
    elif score >= 100.5:
        return f"{score} · Favorable", "rating-good"
    elif score <= 98.0:
        return f"{score} · Tough", "rating-bad"
    elif score <= 99.5:
        return f"{score} · Below Avg", "rating-bad"
    return f"{score} · Neutral", "rating-neutral"

@st.cache_data(ttl=120)
def fetch_espn_live_box_stats(date_yyyymmdd: str) -> dict:
    """Fetches live/final player stats from ESPN's scoreboard & summary endpoints to grade locked pre-game bets."""
    player_stats = {}
    try:
        sb_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_yyyymmdd}"
        sb = requests.get(sb_url, timeout=8).json()
        for ev in sb.get("events", []):
            ev_id = ev.get("id")
            status_state = ev.get("status", {}).get("type", {}).get("state", "pre")  # pre, in, post
            if status_state == "pre":
                continue
            sum_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={ev_id}"
            s_data = requests.get(sum_url, timeout=8).json()
            for team_blk in s_data.get("boxscore", {}).get("players", []):
                for stat_grp in team_blk.get("statistics", []):
                    labels = stat_grp.get("labels", [])
                    if not labels:
                        continue
                    idx_map = {lbl: i for i, lbl in enumerate(labels)}
                    for ath in stat_grp.get("athletes", []):
                        p_name = ath.get("athlete", {}).get("displayName", "")
                        vals = ath.get("stats", [])
                        if not p_name or not vals:
                            continue
                        def parse_num(lbl_name):
                            i = idx_map.get(lbl_name)
                            if i is None or i >= len(vals):
                                return 0.0
                            raw = str(vals[i]).split("-")[0]
                            try:
                                return float(raw)
                            except Exception:
                                return 0.0
                        pts = parse_num("PTS")
                        reb = parse_num("REB")
                        ast = parse_num("AST")
                        fg3m = parse_num("3PT")
                        player_stats[normalize_name(p_name)] = {
                            "state": status_state,  # "in" or "post"
                            "PTS": pts,
                            "REB": reb,
                            "AST": ast,
                            "FG3M": fg3m,
                            "PRA": pts + reb + ast,
                        }
    except Exception:
        pass
    return player_stats

@st.cache_data(ttl=600)
def fetch_and_lock_pregame_slate(api_key: str, bookmaker: str = "draftkings", max_games: int = 10):
    """Captures pre-game lines before tip-off and locks them in the daily vault all night long."""
    et_tz = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(et_tz)
    today_date_et = now_et.date()
    date_key = f"{today_date_et.isoformat()}_{bookmaker}"
    cutoff_utc = now_utc + timedelta(minutes=3)

    vault = get_daily_pregame_vault()
    day_bucket = vault.setdefault(date_key, {"games": {}, "props": {}})

    odds_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    game_res = requests.get(
        odds_url,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "spreads,totals",
            "oddsFormat": "american",
            "bookmakers": bookmaker,
        },
        timeout=10
    )
    game_res.raise_for_status()
    raw_games = game_res.json()

    pregame_to_scan = []
    for g in raw_games:
        c_str = g.get("commence_time", "")
        if not c_str:
            continue
        tip_utc = datetime.fromisoformat(c_str.replace("Z", "+00:00"))
        tip_et = tip_utc.astimezone(et_tz)
        if tip_et.date() != today_date_et:
            continue

        lbl = f"{g['away_team']} @ {g['home_team']}"
        tip_str = tip_et.strftime("%I:%M %p ET").lstrip("0")

        # Save/update game metadata in today's persistent bucket
        if lbl not in day_bucket["games"]:
            spread_home = 0.0
            total_line = 228.0
            for b in g.get("bookmakers", []):
                for m in b.get("markets", []):
                    if m["key"] == "spreads":
                        for o in m.get("outcomes", []):
                            if o.get("name") == g["home_team"] and o.get("point") is not None:
                                spread_home = float(o["point"])
                    elif m["key"] == "totals":
                        for o in m.get("outcomes", []):
                            if o.get("name") == "Over" and o.get("point") is not None:
                                total_line = float(o["point"])
            day_bucket["games"][lbl] = {
                "spread_home": spread_home,
                "total": total_line,
                "id": g["id"],
                "tip_str": tip_str,
                "tip_utc": tip_utc.isoformat(),
            }

        # Only query player props API if the game has NOT tipped off yet
        if tip_utc > cutoff_utc:
            g["_tip_et"] = tip_et
            pregame_to_scan.append(g)

    pregame_to_scan.sort(key=lambda x: x["_tip_et"])

    market_map = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_threes": "FG3M",
        "player_points_rebounds_assists": "PRA",
    }

    for g in pregame_to_scan[:max_games]:
        lbl = f"{g['away_team']} @ {g['home_team']}"
        url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{g['id']}/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": ",".join(market_map.keys()),
            "oddsFormat": "american",
            "bookmakers": bookmaker
        }
        try:
            res = requests.get(url, params=params, timeout=12)
            res.raise_for_status()
            data = res.json()
        except Exception:
            continue

        candidates = {}
        for book in data.get("bookmakers", []):
            for market in book.get("markets", []):
                stat_code = market_map.get(market["key"])
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

        props = {}
        for norm_p, p_obj in candidates.items():
            clean_stats = {}
            for stat_code, lines_dict in p_obj["stats"].items():
                two_way = [
                    (ln, sides) for ln, sides in lines_dict.items()
                    if "Over" in sides and "Under" in sides
                ]
                if not two_way:
                    continue
                best_ln, best_sides = min(
                    two_way,
                    key=lambda item: abs(item[1]["Over"] + 110) + abs(item[1]["Under"] + 110)
                )
                clean_stats[stat_code] = {
                    "line": best_ln,
                    "Over": best_sides["Over"],
                    "Under": best_sides["Under"],
                }
            if clean_stats:
                props[norm_p] = {"display_name": p_obj["display_name"], "stats": clean_stats}

        # Lock this game's pre-game props in the vault so they remain visible after tip-off
        if props:
            day_bucket["props"][lbl] = props

    all_game_labels = sorted(
        day_bucket["games"].keys(),
        key=lambda k: day_bucket["games"][k].get("tip_utc", "")
    )
    slate_date_label = today_date_et.strftime("%A, %b %d")
    fetched_at = now_et.strftime("%I:%M %p ET")
    return all_game_labels, day_bucket["games"], day_bucket["props"], fetched_at, slate_date_label, today_date_et.strftime("%Y%m%d")

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

def grade_pick_result(norm_p: str, stat: str, side: str, line: float, live_box: dict) -> tuple[str, str]:
    """Grades a locked pre-game pick against live or final box score stats."""
    p_live = live_box.get(norm_p)
    if not p_live:
        return "Pre-Game", "res-pre"
    actual = p_live.get(stat, 0.0)
    state = p_live.get("state", "in")
    hit = (actual > line) if side == "Over" else (actual < line)

    if state == "post":
        if hit:
            return f"WON ({actual:.0f})", "res-won"
        return f"LOST ({actual:.0f})", "res-lost"
    return f"LIVE · {actual:.0f}", "res-live"

def evaluate_straight_edges(df: pd.DataFrame, slate_props: dict, game_lines: dict, team_metrics: dict, media_lookup: dict, live_box: dict) -> pd.DataFrame:
    rows = []
    grouped = {k: v for k, v in df.groupby("NORM_NAME")}
    def_mults_all = team_metrics["def_mult"]
    last_games = team_metrics["last_game"]
    today_ts = pd.Timestamp.now().normalize()

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
            if norm_p not in grouped:
                continue
            p_df = grouped[norm_p]
            if len(p_df) < 12:
                continue

            player_team = p_df["TEAM_NAME"].iloc[-1]
            is_home = (player_team == home_team)
            opp_team = away_team if is_home else home_team
            opp_mults = def_mults_all.get(opp_team, {})

            p_media = media_lookup.get(norm_p, {})
            headshot_url = p_media.get("headshot", DEFAULT_HEADSHOT)
            team_logo_url = get_team_logo(player_team)

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

                raw_def = opp_mults.get(stat, 1.0)
                damp_def = 1.0 + 0.55 * (raw_def - 1.0)
                total_env_mult = damp_def * vegas_pace_mult * loc_mult * b2b_mult
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
                    ev_pct = (mod_p * dec - 1) * 100
                    edge_pct = (mod_p - nv_p) * 100
                    rating_mult = total_env_mult if side == "Over" else (2.0 - total_env_mult)
                    rating_txt, rating_cls = format_matchup_rating(rating_mult)
                    res_txt, res_cls = grade_pick_result(norm_p, stat, side, line, live_box)

                    rows.append({
                        "Player": p_data["display_name"],
                        "NORM_NAME": norm_p,
                        "Headshot": headshot_url,
                        "TeamLogo": team_logo_url,
                        "Line": f"{side} {line} {stat}",
                        "Side": side,
                        "Stat": stat,
                        "Projection": round(float(proj), 1),
                        "Matchup": short_matchup(game_label),
                        "FullMatchup": game_label,
                        "Matchup Rating": rating_txt,
                        "RatingClass": rating_cls,
                        "Result": res_txt,
                        "ResultClass": res_cls,
                        "Odds": int(odds),
                        "Win Prob": round(float(mod_p * 100), 1),
                        "Edge": round(float(edge_pct), 1),
                        "EV": round(float(ev_pct), 1),
                    })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="EV", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=1800)
def simulate_full_game(df: pd.DataFrame, away_team: str, home_team: str, team_metrics: dict, game_line: dict, media_lookup: dict, n_sims: int = 15000):
    def_mults_all = team_metrics["def_mult"]
    last_games = team_metrics["last_game"]
    today_ts = pd.Timestamp.now().normalize()
    vegas_total = game_line.get("total", 228.0)
    vegas_pace_mult = 1.0 + 0.45 * ((vegas_total - 228.0) / 228.0)
    max_season = df["SEASON"].max()

    def get_rotation(team: str, opp: str, is_home: bool):
        t_df = df[df["TEAM_NAME"] == team].copy()
        if t_df.empty:
            return np.full(n_sims, 110.0), pd.DataFrame()

        curr_season_df = t_df[t_df["SEASON"] == max_season]
        active_pool = curr_season_df if not curr_season_df.empty else t_df

        recent_games = active_pool["GAME_ID"].drop_duplicates().tail(10)
        recent_df = active_pool[active_pool["GAME_ID"].isin(recent_games)]

        rot_players = (
            recent_df.groupby("PLAYER_NAME")["MIN"]
            .agg(["count", "mean"])
            .query("count >= 2")
            .sort_values(by="mean", ascending=False)
            .head(10)
            .index.tolist()
        )
        if not rot_players:
            rot_players = recent_df["PLAYER_NAME"].unique().tolist()[:10]

        sub = t_df[t_df["PLAYER_NAME"].isin(rot_players)]
        opp_mults = def_mults_all.get(opp, {})
        loc_mult = 1.012 if is_home else 0.988
        b2b_mult = 0.975 if (team in last_games and (today_ts - last_games[team]).days == 1) else 1.0

        player_params = []
        for p in rot_players:
            p_sub = sub[sub["PLAYER_NAME"] == p].tail(35)
            mins = exp_weighted_mean(p_sub["MIN"].values, decay=0.90)
            stats_proj = {}
            for st_col in ["PTS", "REB", "AST", "FG3M"]:
                rates = (p_sub[st_col] / p_sub["MIN"]).values
                base = exp_weighted_mean(rates, decay=0.94) * mins
                mult = (1.0 + 0.55 * (opp_mults.get(st_col, 1.0) - 1.0)) * vegas_pace_mult * loc_mult * b2b_mult
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
                "Headshot": p_media.get("headshot", DEFAULT_HEADSHOT),
                "TeamLogo": team_logo_url,
                "MIN": round(p_info["min"], 1),
                "PTS": round(float(np.mean(sim_p)), 1),
                "REB": round(float(np.mean(sim_r)), 1),
                "AST": round(float(np.mean(sim_a)), 1),
                "3PM": round(float(np.mean(sim_3)), 1),
                "PRA": round(float(np.mean(sim_p + sim_r + sim_a)), 1),
            })

        team_totals = sim_pts_matrix.sum(axis=1)
        return team_totals, pd.DataFrame(box_rows)

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

def build_auto_sgps(df: pd.DataFrame, game_label: str, game_props: dict, min_games: int = 15) -> pd.DataFrame:
    active_norms = [np_name for np_name in game_props.keys() if np_name in set(df["NORM_NAME"])]
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

# Zero-Indentation HTML Renderers
def render_clean_prop_board(source_df: pd.DataFrame, section_title: str, empty_msg: str):
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
        ev_val = r["EV"]
        ev_html = f'<span class="ev-pos">{ev_val:+.1f}%</span>' if ev_val > 0 else f'<span class="ev-neg">{ev_val:+.1f}%</span>'
        p_cell = player_badge_html(r["Player"], r["Headshot"], r["TeamLogo"])
        res_html = f'<span class="{r["ResultClass"]}">{r["Result"]}</span>'
        rows_html.append(
            f'<tr>'
            f'<td style="width:21%;">{p_cell}</td>'
            f'<td style="width:13%;" class="line-cell">{r["Line"]}</td>'
            f'<td style="width:7%;" class="proj-cell col-right">{r["Projection"]:.1f}</td>'
            f'<td style="width:12%;">{r["Matchup"]}</td>'
            f'<td style="width:14%;" class="{r["RatingClass"]}">{r["Matchup Rating"]}</td>'
            f'<td style="width:7%;" class="col-right">{r["Odds"]:+d}</td>'
            f'<td style="width:7%;" class="col-right">{r["Win Prob"]:.0f}%</td>'
            f'<td style="width:8%;" class="col-right">{ev_html}</td>'
            f'<td style="width:11%;" class="col-right">{res_html}</td>'
            f'</tr>'
        )

    table_html = (
        f'<div class="board-card">'
        f'<div class="board-header"><span class="board-title">{section_title}</span><span class="board-count">{len(source_df)} Plays</span></div>'
        f'<table class="prop-table">'
        f'<thead><tr>'
        f'<th style="width:21%;">Player</th>'
        f'<th style="width:13%;">Pre-Game Line</th>'
        f'<th style="width:7%;" class="col-right">Proj</th>'
        f'<th style="width:12%;">Matchup</th>'
        f'<th style="width:14%;">Matchup Rating</th>'
        f'<th style="width:7%;" class="col-right">Odds</th>'
        f'<th style="width:7%;" class="col-right">Win %</th>'
        f'<th style="width:8%;" class="col-right">EV %</th>'
        f'<th style="width:11%;" class="col-right">Status / Result</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

def render_box_score_table(team_name: str, win_prob: float, box_df: pd.DataFrame):
    t_logo = get_team_logo(team_name)
    rows_html = []
    for _, r in box_df.iterrows():
        p_cell = player_badge_html(r["Player"], r["Headshot"], r["TeamLogo"])
        rows_html.append(
            f'<tr>'
            f'<td style="width:36%;">{p_cell}</td>'
            f'<td style="width:10%;" class="col-right">{r["MIN"]:.1f}</td>'
            f'<td style="width:11%;" class="proj-cell col-right">{r["PTS"]:.1f}</td>'
            f'<td style="width:10%;" class="col-right">{r["REB"]:.1f}</td>'
            f'<td style="width:10%;" class="col-right">{r["AST"]:.1f}</td>'
            f'<td style="width:11%;" class="col-right">{r["3PM"]:.1f}</td>'
            f'<td style="width:12%;" class="line-cell col-right">{r["PRA"]:.1f}</td>'
            f'</tr>'
        )

    html = (
        f'<div class="board-card">'
        f'<div class="board-header"><span class="board-title"><img class="team-inline-logo" src="{t_logo}" />{team_name}</span><span class="board-count">Win Prob: {win_prob:.1f}%</span></div>'
        f'<table class="prop-table">'
        f'<thead><tr>'
        f'<th style="width:36%;">Player</th>'
        f'<th style="width:10%;" class="col-right">MIN</th>'
        f'<th style="width:11%;" class="col-right">PTS</th>'
        f'<th style="width:10%;" class="col-right">REB</th>'
        f'<th style="width:10%;" class="col-right">AST</th>'
        f'<th style="width:11%;" class="col-right">3PM</th>'
        f'<th style="width:12%;" class="col-right">PRA</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# Initialize same-tab navigation state
if "active_game_page" not in st.session_state:
    st.session_state["active_game_page"] = None
if "selected_book" not in st.session_state:
    st.session_state["selected_book"] = "draftkings"

# Load multi-season database, media lookup & compute advanced team metrics
df = load_nba_data()
media_lookup = build_player_media_lookup(df)
team_metrics = compute_advanced_team_metrics(df)

saved_key = ""
try:
    saved_key = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    saved_key = ""

# Top Brand Header
st.markdown(
    f'<div class="brand-bar"><div class="brand-left"><div class="brand-logo">HP</div>'
    f'<div><p class="brand-name">{BRAND_NAME}</p><p class="brand-sub">{BRAND_SUBTITLE}</p></div></div>'
    f'<span class="status-badge"><span class="status-dot"></span> DAILY PRE-GAME VAULT & LIVE GRADER</span></div>',
    unsafe_allow_html=True
)

if not saved_key:
    saved_key = st.text_input("Enter Odds API Key (or set ODDS_API_KEY in Streamlit Secrets)", type="password")

all_game_labels = []
game_lines = {}
slate_props = {}
fetched_at = "Just now"
slate_date_label = ""
espn_date_str = ""

if saved_key:
    try:
        all_game_labels, game_lines, slate_props, fetched_at, slate_date_label, espn_date_str = fetch_and_lock_pregame_slate(
            saved_key, bookmaker=st.session_state["selected_book"], max_games=10
        )
    except Exception as e:
        st.error(f"Live odds feed unavailable: {e}")

live_box = fetch_espn_live_box_stats(espn_date_str) if espn_date_str else {}
edges_df = evaluate_straight_edges(df, slate_props, game_lines, team_metrics, media_lookup, live_box) if slate_props else pd.DataFrame()

# =========================================================
# SAME-TAB GAME DETAIL PAGE (When a game is clicked)
# =========================================================
target_game = st.session_state.get("active_game_page")
if target_game and "@" in target_game:
    nav_c1, _ = st.columns([1.4, 4.6])
    with nav_c1:
        if st.button("← Back to Full Slate", icon=":material/arrow_back:", width="stretch"):
            st.session_state["active_game_page"] = None
            st.rerun()

    parts = [t.strip() for t in target_game.split("@")]
    away_sel, home_sel = parts[0], parts[1]
    g_line = game_lines.get(target_game, {"spread_home": 0.0, "total": 228.0, "tip_str": ""})
    detail_sim = simulate_full_game(df, away_sel, home_sel, team_metrics, g_line, media_lookup, n_sims=15000)

    fav_short = home_sel.split()[-1] if detail_sim["proj_spread"] < 0 else away_sel.split()[-1]
    spread_num = abs(detail_sim["proj_spread"])
    book_spread_str = f"{home_sel.split()[-1]} {g_line['spread_home']:+.1f}" if g_line["spread_home"] != 0 else "N/A"
    away_logo = get_team_logo(away_sel)
    home_logo = get_team_logo(home_sel)

    st.markdown(
        f'<div class="game-hero"><div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">'
        f'<div><div style="font-size:0.78rem; color:#10B981; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">'
        f'Matchup Simulation & Player Projections · {g_line.get("tip_str", slate_date_label)}</div>'
        f'<div style="font-size:1.55rem; font-weight:800; color:#F8FAFC; margin-top:0.35rem; display:flex; align-items:center; flex-wrap:wrap; gap:8px;">'
        f'<img src="{away_logo}" style="width:28px; height:28px; object-fit:contain;" /><span>{away_sel}</span>'
        f'<span style="color:#10B981;">{detail_sim["away_score"]}</span><span style="color:#64748B; font-size:1.1rem;">@</span>'
        f'<img src="{home_logo}" style="width:28px; height:28px; object-fit:contain;" /><span>{home_sel}</span>'
        f'<span style="color:#10B981;">{detail_sim["home_score"]}</span></div></div>'
        f'<div style="text-align:right; font-size:0.86rem; color:#CBD5E1; line-height:1.6;">'
        f'<div>Model Line: <b>{fav_short} -{spread_num}</b> &nbsp;|&nbsp; Book Line: <b>{book_spread_str}</b></div>'
        f'<div>Model Total: <b>{detail_sim["proj_total"]}</b> &nbsp;|&nbsp; Book Total: <b>{g_line["total"]}</b></div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    st.markdown("##### Projected Player Box Scores (15,000 Simulations)")
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        render_box_score_table(f"{away_sel} (Away)", detail_sim["away_win_prob"], detail_sim["away_box"])
    with b2:
        render_box_score_table(f"{home_sel} (Home)", detail_sim["home_win_prob"], detail_sim["home_box"])

    st.markdown(f"##### Locked Pre-Game Player Lines & Live Grading — {away_sel} @ {home_sel}")
    game_props_df = edges_df[edges_df["FullMatchup"] == target_game] if not edges_df.empty else pd.DataFrame()
    if not game_props_df.empty:
        game_props_df = game_props_df.drop_duplicates(subset=["Player", "Stat"]).sort_values(by=["Stat", "EV"], ascending=[True, False])
    render_clean_prop_board(
        game_props_df,
        f"{away_sel} @ {home_sel} — Pre-Game Lines, Projections & Results",
        "No pre-game player props were captured before tip-off for this game."
    )

    st.markdown(
        f'<div class="site-footer"><span><b>{BRAND_NAME}</b> · Quantitative NBA Analytics & Simulation Engine</span>'
        f'<span>For informational & entertainment purposes only. Please wager responsibly.</span></div>',
        unsafe_allow_html=True
    )
    st.stop()

# =========================================================
# MAIN DASHBOARD VIEW
# =========================================================
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.1, 1.5, 1.8, 0.7], vertical_alignment="bottom")
with ctrl_c1:
    bookmaker = st.selectbox(
        "Sportsbook",
        ["draftkings", "fanduel", "betmgm", "caesars"],
        index=["draftkings", "fanduel", "betmgm", "caesars"].index(st.session_state["selected_book"]),
        format_func=lambda x: {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "caesars": "Caesars"}[x]
    )
    if bookmaker != st.session_state["selected_book"]:
        st.session_state["selected_book"] = bookmaker
        st.rerun()
with ctrl_c2:
    min_ev = st.slider("Minimum EV %", min_value=-15.0, max_value=15.0, value=-10.0, step=0.5)
with ctrl_c3:
    search_player = st.text_input("Search", placeholder="Filter by player or team...")
with ctrl_c4:
    if st.button("Refresh", icon=":material/refresh:", width="stretch"):
        fetch_and_lock_pregame_slate.clear()
        fetch_espn_live_box_stats.clear()
        st.rerun()

filtered = edges_df[edges_df["EV"] >= min_ev] if not edges_df.empty else pd.DataFrame()
if not filtered.empty and search_player.strip():
    q = search_player.strip()
    filtered = filtered[
        filtered["Player"].str.contains(q, case=False, na=False) |
        filtered["FullMatchup"].str.contains(q, case=False, na=False)
    ]

# Top KPI Summary Strip (Includes Won/Lost Tracker for Locked Pre-Game Plays)
won_cnt = int(filtered["Result"].str.startswith("WON").sum()) if not filtered.empty else 0
lost_cnt = int(filtered["Result"].str.startswith("LOST").sum()) if not filtered.empty else 0
record_str = f"{won_cnt}-{lost_cnt}" if (won_cnt + lost_cnt) > 0 else "In Progress"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(f"Today's Slate ({slate_date_label})", f"{len(all_game_labels)} Games")
kpi2.metric("Locked Pre-Game Lines", f"{len(filtered)} Props")
kpi3.metric("Graded Record (W-L)", record_str)
kpi4.metric("Last Box Score Sync", fetched_at)

st.write("")

# Navigation Tabs
tab_straight, tab_slate, tab_parlays, tab_custom = st.tabs([
    ":material/local_fire_department: Top Straight Bets",
    ":material/scoreboard: Slate & Projected Box Scores",
    ":material/bolt: Auto-Parlay Builder",
    ":material/tune: Custom SGP Lab"
])

# =========================================================
# TAB 1: TOP STRAIGHT BETS
# =========================================================
with tab_straight:
    if not slate_props:
        st.info(
            f"All games on {slate_date_label} tipped off before today's pre-game snapshot vault was initialized, "
            f"so their pre-tip sportsbook lines are no longer open on the live feed. Starting at midnight ET tonight, "
            f"every pre-game line will automatically lock into the daily vault before tip-off and stay visible with live WON/LOST grading all night."
        )
    elif filtered.empty:
        st.warning(f"No lines match your current filter ({min_ev:+.1f}% EV). Lower the Minimum EV slider above to view all posted props.")
    else:
        render_clean_prop_board(filtered[filtered["Stat"] == "PTS"].head(10), "Points Props (PTS) — Top 10", "No PTS pre-game lines matching your filter.")
        render_clean_prop_board(filtered[filtered["Stat"] == "REB"].head(10), "Rebounds Props (REB) — Top 10", "No REB pre-game lines matching your filter.")
        render_clean_prop_board(filtered[filtered["Stat"] == "AST"].head(10), "Assists Props (AST) — Top 10", "No AST pre-game lines matching your filter.")
        render_clean_prop_board(filtered[filtered["Stat"] == "FG3M"].head(10), "3-Pointers Made Props (3PM) — Top 10", "No 3PM pre-game lines matching your filter.")
        render_clean_prop_board(filtered[filtered["Stat"] == "PRA"].head(10), "Points + Rebounds + Assists (PRA) — Top 10", "No PRA pre-game lines matching your filter.")

# =========================================================
# TAB 2: SLATE & SAME-TAB GAME PAGE LAUNCHER
# =========================================================
with tab_slate:
    st.markdown(f"##### {slate_date_label} Projected NBA Scores")
    st.caption("Click any matchup below to open its full projected player box scores, locked pre-game prop lines, projections, and live results.")

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
        g_line = game_lines.get(g_lbl, {"spread_home": 0.0, "total": 228.0, "tip_str": ""})
        sim_res = simulate_full_game(df, away_t, home_t, team_metrics, g_line, media_lookup, n_sims=15000)
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
                st.rerun()

# =========================================================
# TAB 3: AUTO-PARLAY BUILDER
# =========================================================
with tab_parlays:
    if not slate_props:
        st.info("Auto-parlays lock in automatically as soon as pre-game player props open for the slate.")
    else:
        col_multi, col_sgp = st.columns(2, gap="large")

        with col_multi:
            st.markdown("##### Cross-Game +EV Parlay")
            st.caption("Combines the highest independent model edge from each pre-game matchup.")
            num_parlay_legs = st.radio("Parlay Size", [2, 3, 4], horizontal=True)
            if not edges_df.empty:
                best_per_game = edges_df.drop_duplicates(subset=["FullMatchup"]).head(num_parlay_legs)
                if len(best_per_game) >= 2:
                    comb_dec = float(np.prod([american_to_decimal(o) for o in best_per_game["Odds"]]))
                    comb_mod_p = float(np.prod(best_per_game["Win Prob"] / 100.0))
                    parlay_ev = (comb_mod_p * comb_dec - 1) * 100
                    parlay_odds = round((comb_dec - 1) * 100) if comb_dec >= 2.0 else round(-100 / (comb_dec - 1))

                    m1, m2 = st.columns(2)
                    m1.metric("Combined Odds", f"{parlay_odds:+d}")
                    m2.metric("Parlay EV", f"{parlay_ev:+.1f}%")

                    slip_lines = [f"{BRAND_NAME} ({num_parlay_legs}-Leg +EV Parlay | {parlay_odds:+d})"]
                    for _, row in best_per_game.iterrows():
                        slip_lines.append(f"- {row['Player']} | {row['Line']} ({row['Odds']:+d}) [{row['Result']}]")
                        p_badge = player_badge_html(row["Player"], row["Headshot"], row["TeamLogo"])
                        st.markdown(
                            f'<div class="pick-card"><div class="pick-top">{p_badge}<span class="ev-pos">{row["Odds"]:+d}</span></div>'
                            f'<div class="pick-line">{row["Line"]} · <span class="{row["ResultClass"]}">{row["Result"]}</span></div>'
                            f'<div class="pick-details">Proj: <b>{row["Projection"]}</b> · Rating: <b>{row["Matchup Rating"]}</b> · EV: <b>{row["EV"]:+.1f}%</b></div></div>',
                            unsafe_allow_html=True
                        )

                    with st.expander("Copy Parlay Slip", icon=":material/content_copy:"):
                        st.code("\n".join(slip_lines), language=None)
                else:
                    st.warning("At least 2 pre-game matchups with props are needed to build a cross-game parlay.")

        with col_sgp:
            st.markdown("##### Correlated Same-Game Parlays")
            st.caption("Pairs locked pre-game lines with strong historical game-log correlations.")
            sgp_game = st.selectbox("Select Matchup", list(slate_props.keys()))
            sgp_df = build_auto_sgps(df, sgp_game, slate_props[sgp_game])
            if sgp_df.empty:
                st.warning("No high-correlation prop pairs found for this matchup yet.")
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
# TAB 4: CUSTOM SGP LAB
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
            m4.metric("Expected Value (EV)", f"{res['ev_pct']:+.2f}%")
            st.dataframe(res["corr_df"], width="stretch")

# Clean Bespoke Footer
st.markdown(
    f'<div class="site-footer"><span><b>{BRAND_NAME}</b> · Quantitative NBA Analytics & Simulation Engine</span>'
    f'<span>For informational & entertainment purposes only. Please wager responsibly.</span></div>',
    unsafe_allow_html=True
)