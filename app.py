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
BRAND_SUBTITLE = "NBA & NFL Prop Projections · Current-Season Edge Engine · Analytics Terminal"
PROPLINE_API_KEY = "b2858a5f3784220e7da1dee8d9916ab7"
pl_client = PropLine(PROPLINE_API_KEY)

st.set_page_config(
    page_title=f"{BRAND_NAME} — NBA & NFL Prop Analytics",
    page_icon=":material/analytics:",
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
.team-sub-txt {
    font-size: 0.72rem;
    color: #8B97AD;
    margin-left: 4px;
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
.alt-pill {
    font-size: 0.62rem;
    font-weight: 700;
    color: #38BDF8;
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 1px 5px;
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

# ==========================================
# TEAM LOGOS & MAPPINGS (NBA + NFL)
# ==========================================
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

NFL_TEAM_LOGOS = {
    "Arizona Cardinals": "https://a.espncdn.com/i/teamlogos/nfl/500/ari.png",
    "Atlanta Falcons": "https://a.espncdn.com/i/teamlogos/nfl/500/atl.png",
    "Baltimore Ravens": "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png",
    "Buffalo Bills": "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png",
    "Carolina Panthers": "https://a.espncdn.com/i/teamlogos/nfl/500/car.png",
    "Chicago Bears": "https://a.espncdn.com/i/teamlogos/nfl/500/chi.png",
    "Cincinnati Bengals": "https://a.espncdn.com/i/teamlogos/nfl/500/cin.png",
    "Cleveland Browns": "https://a.espncdn.com/i/teamlogos/nfl/500/cle.png",
    "Dallas Cowboys": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    "Denver Broncos": "https://a.espncdn.com/i/teamlogos/nfl/500/den.png",
    "Detroit Lions": "https://a.espncdn.com/i/teamlogos/nfl/500/det.png",
    "Green Bay Packers": "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "Houston Texans": "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png",
    "Indianapolis Colts": "https://a.espncdn.com/i/teamlogos/nfl/500/ind.png",
    "Jacksonville Jaguars": "https://a.espncdn.com/i/teamlogos/nfl/500/jax.png",
    "Kansas City Chiefs": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "Las Vegas Raiders": "https://a.espncdn.com/i/teamlogos/nfl/500/lv.png",
    "Los Angeles Chargers": "https://a.espncdn.com/i/teamlogos/nfl/500/lac.png",
    "Los Angeles Rams": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "Miami Dolphins": "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png",
    "Minnesota Vikings": "https://a.espncdn.com/i/teamlogos/nfl/500/min.png",
    "New England Patriots": "https://a.espncdn.com/i/teamlogos/nfl/500/ne.png",
    "New Orleans Saints": "https://a.espncdn.com/i/teamlogos/nfl/500/no.png",
    "New York Giants": "https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png",
    "New York Jets": "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png",
    "Philadelphia Eagles": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "Pittsburgh Steelers": "https://a.espncdn.com/i/teamlogos/nfl/500/pit.png",
    "San Francisco 49ers": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "Seattle Seahawks": "https://a.espncdn.com/i/teamlogos/nfl/500/sea.png",
    "Tampa Bay Buccaneers": "https://a.espncdn.com/i/teamlogos/nfl/500/tb.png",
    "Tennessee Titans": "https://a.espncdn.com/i/teamlogos/nfl/500/ten.png",
    "Washington Commanders": "https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png",
}

NFL_ABBR_TO_FULL = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "OAK": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "SD": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "STL": "Los Angeles Rams",
    "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
    "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders", "WSH": "Washington Commanders"
}

ESPN_TO_ODDS_TEAM = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Rams": "Los Angeles Rams",
    "LA Chargers": "Los Angeles Chargers",
}

SPORT_CONFIG = {
    "NBA": {
        "sport_key": "basketball_nba",
        "espn_sport": "basketball/nba",
        "active_season": 2027,
        "seasons": (2027, 2026),
        "default_total": 228.0,
        "stats": ["PTS", "REB", "AST", "FG3M", "PRA"],
        "stat_labels": {"PTS": "Points", "REB": "Rebounds", "AST": "Assists", "FG3M": "3PM", "PRA": "PRA"},
        "pos_groups": ["G", "F", "C"],
        "pos_plural": {"G": "Guards", "F": "Forwards", "C": "Centers"},
        "markets": [
            "player_points", "player_rebounds", "player_assists", "player_threes",
            "player_points_rebounds_assists", "player_points_alternate",
            "player_rebounds_alternate", "player_assists_alternate", "spreads", "totals"
        ],
        "market_map": {
            "player_points": "PTS", "player_points_alternate": "PTS",
            "player_rebounds": "REB", "player_rebounds_alternate": "REB",
            "player_assists": "AST", "player_assists_alternate": "AST",
            "player_threes": "FG3M",
            "player_points_rebounds_assists": "PRA"
        }
    },
    "NFL": {
        "sport_key": "americanfootball_nfl",
        "espn_sport": "football/nfl",
        "active_season": 2026,
        "seasons": (2026, 2025),
        "default_total": 45.5,
        "stats": ["PASS_YDS", "RUSH_YDS", "REC_YDS", "REC", "PASS_TD", "RUSH_REC_YDS"],
        "stat_labels": {
            "PASS_YDS": "Pass Yds", "RUSH_YDS": "Rush Yds", "REC_YDS": "Rec Yds",
            "REC": "Receptions", "PASS_TD": "Pass TDs", "RUSH_REC_YDS": "Rush+Rec Yds"
        },
        "pos_groups": ["QB", "RB", "WR", "TE"],
        "pos_plural": {"QB": "Quarterbacks", "RB": "Running Backs", "WR": "Wide Receivers", "TE": "Tight Ends"},
        "markets": [
            "player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions",
            "player_pass_tds", "player_rush_reception_yds",
            "player_pass_yds_alternate", "player_rush_yds_alternate", "player_reception_yds_alternate",
            "spreads", "totals"
        ],
        "market_map": {
            "player_pass_yds": "PASS_YDS", "player_pass_yds_alternate": "PASS_YDS",
            "player_rush_yds": "RUSH_YDS", "player_rush_yds_alternate": "RUSH_YDS",
            "player_reception_yds": "REC_YDS", "player_reception_yds_alternate": "REC_YDS",
            "player_receptions": "REC",
            "player_pass_tds": "PASS_TD",
            "player_rush_reception_yds": "RUSH_REC_YDS"
        }
    }
}

DEFAULT_HEADSHOT = "https://a.espncdn.com/combiner/i?img=/i/headshots/nophoto.png&w=96&h=70"

def resolve_team_name(raw_team: str, league: str) -> str:
    t = str(raw_team or "").strip()
    if not t:
        return ""
    if league == "NFL":
        up = t.upper()
        if up in NFL_ABBR_TO_FULL:
            return NFL_ABBR_TO_FULL[up]
    return ESPN_TO_ODDS_TEAM.get(t, t)

def map_position_group(raw_pos: str, league: str) -> str:
    p = str(raw_pos).strip().upper()
    if league == "NBA":
        if p in ("C", "CENTER"):
            return "C"
        if p in ("PG", "SG", "G", "GUARD"):
            return "G"
        return "F"
    else:
        if p in ("QB",):
            return "QB"
        if p in ("RB", "FB", "HB"):
            return "RB"
        if p in ("TE",):
            return "TE"
        return "WR"

def get_team_logo(team_name: str, league: str) -> str:
    resolved = resolve_team_name(team_name, league)
    if league == "NBA":
        return NBA_TEAM_LOGOS.get(resolved, "https://a.espncdn.com/i/teamlogos/leagues/500/nba.png")
    return NFL_TEAM_LOGOS.get(resolved, "https://a.espncdn.com/i/teamlogos/leagues/500/nfl.png")

def build_headshot_url(athlete_id, href: str = "", league: str = "NBA") -> str:
    if isinstance(href, str) and href.startswith("http"):
        return href
    if pd.notna(athlete_id):
        try:
            aid = int(float(athlete_id))
            if aid > 0:
                sport_path = "nba" if league == "NBA" else "nfl"
                return f"https://a.espncdn.com/combiner/i?img=/i/headshots/{sport_path}/players/full/{aid}.png&w=96&h=70"
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

def player_badge_html(player_name: str, headshot_url: str, team_logo_url: str, pos_label: str = "", team_short: str = "", league: str = "NBA", book: str = "draftkings", from_game: str = "", inj_status: str = "") -> str:
    href = f"?league={league}&player={quote_plus(player_name)}&book={book}"
    if from_game:
        href += f"&game={quote_plus(from_game)}"
    pos_html = f'<span class="pos-pill">{pos_label}</span>' if pos_label else ""
    tm_html = f'<span class="team-sub-txt">{team_short}</span>' if team_short else ""
    inj_html = ""
    if inj_status and any(k in inj_status.lower() for k in ("questionable", "day-to-day", "probable", "gtd")):
        inj_html = '<span class="inj-q-pill">GTD</span>'
    return (
        f'<a class="player-link" href="{href}" target="_self" title="Open {player_name} Scouting Profile">'
        f'<div class="headshot-wrap">'
        f'<img class="headshot-img" src="{headshot_url}" onerror="this.src=\'{DEFAULT_HEADSHOT}\'" />'
        f'<img class="team-mini-logo" src="{team_logo_url}" />'
        f'</div>'
        f'<span class="player-name-txt">{player_name}</span>{tm_html}'
        f'{pos_html}{inj_html}'
        f'</a>'
    )

@st.cache_resource
def get_daily_pregame_vault() -> dict:
    return {}

@st.cache_data(ttl=21600)
def fetch_live_espn_rosters(league: str) -> dict:
    espn_path = SPORT_CONFIG[league]["espn_sport"]
    roster_map = {}
    try:
        teams_url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/teams?limit=40"
        t_data = requests.get(teams_url, timeout=8).json()
        for sp in t_data.get("sports", []):
            for lg in sp.get("leagues", []):
                for t_item in lg.get("teams", []):
                    tm = t_item.get("team", {})
                    tid = tm.get("id")
                    tname = resolve_team_name(tm.get("displayName", ""), league)
                    if not tid or not tname:
                        continue
                    r_url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/teams/{tid}/roster"
                    r_data = requests.get(r_url, timeout=6).json()
                    for entry in r_data.get("athletes", []):
                        items = entry.get("items", [entry]) if isinstance(entry, dict) else []
                        for ath in items:
                            full_n = ath.get("displayName") or ath.get("fullName", "")
                            if not full_n:
                                continue
                            norm_n = normalize_name(full_n)
                            pos_abbr = ath.get("position", {}).get("abbreviation", "")
                            aid = ath.get("id")
                            hs = ath.get("headshot", {}).get("href", "")
                            roster_map[norm_n] = {
                                "team_name": tname,
                                "raw_pos": pos_abbr,
                                "athlete_id": int(aid) if str(aid).isdigit() else 0,
                                "headshot": hs,
                            }
    except Exception:
        pass
    return roster_map

@st.cache_data(ttl=900)
def fetch_injury_report(league: str) -> tuple[dict, dict]:
    player_injuries = {}
    team_out_lists = {}
    espn_path = SPORT_CONFIG[league]["espn_sport"]
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/injuries"
        data = requests.get(url, timeout=8).json()
        for team_blk in data.get("injuries", []):
            raw_team = team_blk.get("displayName", "")
            team_name = resolve_team_name(raw_team, league)
            for item in team_blk.get("injuries", []):
                ath = item.get("athlete", {})
                p_name = ath.get("displayName", "")
                status = str(item.get("status", "")).strip()
                short_comment = str(item.get("shortComment", "")).strip()
                if not p_name or not status:
                    continue
                norm_p = normalize_name(p_name)
                st_low = status.lower()
                is_out = any(k in st_low for k in ("out", "suspension", "suspended", "doubtful", "inactive", "injured reserve", "ir"))
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
def load_sports_data(league: str):
    seasons = SPORT_CONFIG[league]["seasons"]
    if league == "NBA":
        frames = []
        for yr in seasons:
            url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{yr}.parquet"
            try:
                part = pd.read_parquet(url)
                if not part.empty:
                    part["SEASON"] = yr
                    frames.append(part)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df = df[df["minutes"] >= 8].copy()
        df["game_id"] = df["game_id"].astype(str)
        df["GAME_DATE"] = pd.to_datetime(df["game_date"], errors="coerce") if "game_date" in df.columns else pd.NaT
        df = df.sort_values(["SEASON", "GAME_DATE"]).drop_duplicates(subset=["game_id", "athlete_display_name"])

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
            "minutes": "USAGE",
            "points": "PTS",
            "rebounds": "REB",
            "assists": "AST",
            "three_point_field_goals_made": "FG3M",
            "field_goals_made": "FGM",
            "field_goals_attempted": "FGA",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        for col in ["ATHLETE_ID", "FGM", "FGA"]:
            if col not in df.columns:
                df[col] = np.nan
        for col in ["HEADSHOT_HREF", "TEAM_LOGO", "HOME_AWAY", "RAW_POS"]:
            if col not in df.columns:
                df[col] = ""

        df["TEAM_NAME"] = df["TEAM_NAME"].apply(lambda x: resolve_team_name(x, "NBA"))
        df["OPP_NAME"] = df["OPP_NAME"].apply(lambda x: resolve_team_name(x, "NBA"))

        valid_pos = df[df["RAW_POS"].astype(str).str.strip() != ""].drop_duplicates(subset=["PLAYER_NAME"], keep="last")
        pos_map = dict(zip(valid_pos["PLAYER_NAME"], valid_pos["RAW_POS"]))
        df["RAW_POS"] = df["PLAYER_NAME"].map(pos_map).fillna(df["RAW_POS"]).fillna("F")
        df["POS_GROUP"] = df["RAW_POS"].apply(lambda x: map_position_group(x, "NBA"))
        df["NORM_NAME"] = df["PLAYER_NAME"].apply(normalize_name)
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        return df.sort_values(["SEASON", "GAME_DATE"]).reset_index(drop=True)

    else:
        frames = []
        for yr in seasons:
            urls = [
                f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{yr}.parquet",
                f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{yr}.parquet",
            ]
            for url in urls:
                try:
                    part = pd.read_parquet(url)
                    if not part.empty:
                        part["SEASON"] = yr
                        frames.append(part)
                        break
                except Exception:
                    continue

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        if "season_type" in df.columns:
            df = df[df["season_type"].isin(["REG", "POST"])].copy()

        name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
        df["PLAYER_NAME"] = df[name_col].fillna("").astype(str)
        df = df[df["PLAYER_NAME"].str.strip() != ""].copy()

        pos_col = "position" if "position" in df.columns else "position_group"
        df["RAW_POS"] = df[pos_col].fillna("WR").astype(str)
        df = df[df["RAW_POS"].isin(["QB", "RB", "WR", "TE", "FB", "HB"])].copy()

        team_col = "recent_team" if "recent_team" in df.columns else ("team" if "team" in df.columns else None)
        opp_col = "opponent_team" if "opponent_team" in df.columns else ("opp" if "opp" in df.columns else None)
        raw_t = df[team_col].fillna("") if team_col else pd.Series("", index=df.index)
        raw_o = df[opp_col].fillna("") if opp_col else pd.Series("", index=df.index)
        df["TEAM_NAME"] = raw_t.apply(lambda x: resolve_team_name(x, "NFL"))
        df["OPP_NAME"] = raw_o.apply(lambda x: resolve_team_name(x, "NFL"))

        wk = pd.to_numeric(df.get("week", 1), errors="coerce").fillna(1).astype(int)
        df["WEEK_NUM"] = wk
        df["GAME_ID"] = df["SEASON"].astype(str) + "_W" + wk.astype(str) + "_" + df["TEAM_NAME"].astype(str)
        df["GAME_DATE"] = pd.to_datetime(df["SEASON"].astype(str) + "-09-01") + pd.to_timedelta((wk - 1) * 7, unit="D")

        df = df.sort_values(["SEASON", "WEEK_NUM"]).drop_duplicates(subset=["GAME_ID", "PLAYER_NAME"])

        df["PASS_YDS"] = pd.to_numeric(df.get("passing_yards", 0), errors="coerce").fillna(0.0)
        df["RUSH_YDS"] = pd.to_numeric(df.get("rushing_yards", 0), errors="coerce").fillna(0.0)
        df["REC_YDS"] = pd.to_numeric(df.get("receiving_yards", 0), errors="coerce").fillna(0.0)
        df["REC"] = pd.to_numeric(df.get("receptions", 0), errors="coerce").fillna(0.0)
        df["PASS_TD"] = pd.to_numeric(df.get("passing_tds", 0), errors="coerce").fillna(0.0)
        df["RUSH_TD"] = pd.to_numeric(df.get("rushing_tds", 0), errors="coerce").fillna(0.0)
        df["REC_TD"] = pd.to_numeric(df.get("receiving_tds", 0), errors="coerce").fillna(0.0)
        df["RUSH_REC_YDS"] = df["RUSH_YDS"] + df["REC_YDS"]
        df["TOTAL_PTS"] = (df["RUSH_TD"] + df["REC_TD"]) * 6.0 + df["PASS_TD"] * 4.0

        att = pd.to_numeric(df.get("attempts", 0), errors="coerce").fillna(0.0)
        car = pd.to_numeric(df.get("carries", 0), errors="coerce").fillna(0.0)
        tgt = pd.to_numeric(df.get("targets", 0), errors="coerce").fillna(0.0)
        df["USAGE"] = np.where(df["RAW_POS"] == "QB", att + car, car + tgt)

        df["HEADSHOT_HREF"] = df.get("headshot_url", pd.Series("", index=df.index)).fillna("")
        df["ATHLETE_ID"] = np.nan
        df["TEAM_LOGO"] = df["TEAM_NAME"].apply(lambda x: get_team_logo(x, "NFL"))
        df["HOME_AWAY"] = ""
        df["POS_GROUP"] = df["RAW_POS"].apply(lambda x: map_position_group(x, "NFL"))
        df["NORM_NAME"] = df["PLAYER_NAME"].apply(normalize_name)
        return df.sort_values(["SEASON", "WEEK_NUM"]).reset_index(drop=True)

@st.cache_data(ttl=21600)
def build_player_media_lookup(df: pd.DataFrame, league: str) -> dict:
    live_rosters = fetch_live_espn_rosters(league)
    media = {}

    if not df.empty:
        valid_teams = df[df["TEAM_NAME"].astype(str).str.strip() != ""].sort_values("GAME_DATE").drop_duplicates(subset=["NORM_NAME"], keep="last")
        latest = df.sort_values("GAME_DATE").drop_duplicates(subset=["NORM_NAME"], keep="last")
        team_by_norm = dict(zip(valid_teams["NORM_NAME"], valid_teams["TEAM_NAME"]))

        for _, r in latest.iterrows():
            n_key = r["NORM_NAME"]
            live_info = live_rosters.get(n_key, {})
            t_name = resolve_team_name(live_info.get("team_name") or team_by_norm.get(n_key) or r.get("TEAM_NAME", ""), league)
            t_logo = get_team_logo(t_name, league)
            aid = live_info.get("athlete_id") or r.get("ATHLETE_ID")
            raw_hs = live_info.get("headshot") or str(r.get("HEADSHOT_HREF", ""))
            h_url = build_headshot_url(aid, raw_hs, league)
            raw_p = live_info.get("raw_pos") or str(r.get("RAW_POS", "")).strip() or r.get("POS_GROUP", "F")
            media[n_key] = {
                "athlete_id": int(float(aid)) if pd.notna(aid) else 0,
                "headshot": h_url,
                "team_logo": t_logo,
                "team_name": t_name,
                "team_short": t_name.split()[-1] if t_name else "",
                "raw_pos": raw_p,
                "pos_group": map_position_group(raw_p, league),
            }

    for n_key, live_info in live_rosters.items():
        if n_key not in media:
            t_name = live_info["team_name"]
            raw_p = live_info.get("raw_pos", "F" if league == "NBA" else "WR")
            media[n_key] = {
                "athlete_id": live_info.get("athlete_id", 0),
                "headshot": build_headshot_url(live_info.get("athlete_id"), live_info.get("headshot", ""), league),
                "team_logo": get_team_logo(t_name, league),
                "team_name": t_name,
                "team_short": t_name.split()[-1] if t_name else "",
                "raw_pos": raw_p,
                "pos_group": map_position_group(raw_p, league),
            }
    return media

@st.cache_data(ttl=21600)
def compute_advanced_team_metrics(df: pd.DataFrame, league: str) -> dict:
    stats = SPORT_CONFIG[league]["stats"]
    pos_groups = SPORT_CONFIG[league]["pos_groups"]
    active_season = SPORT_CONFIG[league]["active_season"]
    if df.empty or "OPP_NAME" not in df.columns:
        return {"dvp_mult": {}, "dvp_ranks": {}, "dvp_avg": {}, "last_game": {}}

    curr_df = df[df["SEASON"] == active_season].copy()
    if curr_df["GAME_ID"].nunique() >= 25:
        use_df = curr_df.copy()
        use_df["w"] = 1.0
    else:
        use_df = df[df["SEASON"] >= active_season - 1].copy()
        use_df["w"] = np.where(use_df["SEASON"] == active_season, 5.5, 1.0)

    pos_game_allowed = use_df.groupby(["OPP_NAME", "POS_GROUP", "GAME_ID", "w"])[stats].sum().reset_index()

    def weighted_avg(grp):
        w = grp["w"].values
        return pd.Series({s: np.average(grp[s].values, weights=w) for s in stats})

    dvp_allowed = pos_game_allowed.groupby(["OPP_NAME", "POS_GROUP"]).apply(weighted_avg, include_groups=False)

    dvp_mult = {}
    dvp_ranks = {}
    dvp_avg = {}

    for pos_g in pos_groups:
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

def build_hit_rate_chart(p_df: pd.DataFrame, stat_col: str, line_val: float, player_name: str, league: str) -> go.Figure:
    recent = p_df.tail(15 if league == "NFL" else 20).copy()
    if league == "NFL" and "WEEK_NUM" in recent.columns:
        dates = [f"{int(s)} W{int(w)}" for s, w in zip(recent["SEASON"], recent["WEEK_NUM"])]
    else:
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
        hovertemplate="<b>%{x}</b><br>" + stat_col + ": %{y:.1f}<extra></extra>"
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
    max_y = max(float(np.max(vals)) * 1.22, line_val * 1.35, 3.0)
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
def fetch_and_lock_propline_slate(league: str, bookmaker: str = "draftkings", max_games: int = 16):
    et_tz = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(et_tz)
    today_date_et = now_et.date()
    date_key = f"{league}_{today_date_et.isoformat()}_{bookmaker}"

    cfg = SPORT_CONFIG[league]
    sport_key = cfg["sport_key"]
    default_total = cfg["default_total"]
    market_list = cfg["markets"]
    stat_map = cfg["market_map"]

    vault = get_daily_pregame_vault()
    day_bucket = vault.setdefault(date_key, {"games": {}, "props": {}})

    try:
        events = pl_client.get_events(sport_key)
        pregame_events = []

        horizon = timedelta(days=1) if league == "NBA" else timedelta(days=7)

        for ev in events:
            c_str = ev.get("commence_time", "")
            if not c_str:
                continue
            tip_utc = datetime.fromisoformat(c_str.replace("Z", "+00:00"))
            tip_et = tip_utc.astimezone(et_tz)
            if league == "NBA" and tip_et.date() != today_date_et:
                continue
            if league == "NFL" and not (now_utc - timedelta(hours=6) <= tip_utc <= now_utc + horizon):
                continue

            away = resolve_team_name(ev.get("away_team", ""), league)
            home = resolve_team_name(ev.get("home_team", ""), league)
            lbl = f"{away} @ {home}"
            tip_str = tip_et.strftime("%a %I:%M %p ET").lstrip("0") if league == "NFL" else tip_et.strftime("%I:%M %p ET").lstrip("0")
            event_id = ev.get("id")

            day_bucket["games"][lbl] = {
                "spread_home": 0.0,
                "total": default_total,
                "id": event_id,
                "tip_str": tip_str,
                "tip_utc": tip_utc.isoformat(),
            }
            if tip_utc > (now_utc - timedelta(hours=4)):
                ev["_tip_et"] = tip_et
                pregame_events.append(ev)

        pregame_events.sort(key=lambda x: x["_tip_et"])

        for ev in pregame_events[:max_games]:
            away = resolve_team_name(ev.get("away_team", ""), league)
            home = resolve_team_name(ev.get("home_team", ""), league)
            lbl = f"{away} @ {home}"
            event_id = ev.get("id")

            odds_data = pl_client.get_odds(sport_key, event_id=event_id, markets=market_list)

            spread_home = 0.0
            total_line = default_total
            candidates = {}

            for book in odds_data.get("bookmakers", []):
                for market in book.get("markets", []):
                    m_key = market.get("key")
                    if m_key == "spreads":
                        for o in market.get("outcomes", []):
                            if resolve_team_name(o.get("name", ""), league) == home and o.get("point") is not None:
                                spread_home = float(o["point"])
                    elif m_key == "totals":
                        for o in market.get("outcomes", []):
                            if o.get("name") == "Over" and o.get("point") is not None:
                                total_line = float(o["point"])
                    else:
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
                        "all_lines": lines_dict
                    }
                if clean_stats:
                    props[norm_p] = {"display_name": p_obj["display_name"], "stats": clean_stats}

            if props:
                day_bucket["props"][lbl] = props

    except Exception:
        pass

    all_game_labels = sorted(day_bucket["games"].keys(), key=lambda k: day_bucket["games"][k].get("tip_utc", ""))
    slate_date_label = today_date_et.strftime("%A, %b %d") if league == "NBA" else f"Week Slate ({today_date_et.strftime('%b %d')})"
    fetched_at = now_et.strftime("%I:%M %p ET")
    return all_game_labels, day_bucket["games"], day_bucket["props"], fetched_at, slate_date_label

def american_to_prob(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds: int) -> float:
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

def decimal_to_american(dec: float) -> int:
    if dec <= 1.001:
        return -10000
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100))
    return int(round(-100.0 / (dec - 1.0)))

def prob_to_american(p: float) -> int:
    p = np.clip(p, 0.0001, 0.9999)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)

def get_no_vig_prob(odds_selected: int, odds_opposing: int = 0) -> float:
    p_sel = american_to_prob(odds_selected)
    if odds_opposing != 0:
        p_opp = american_to_prob(odds_opposing)
        return p_sel / (p_sel + p_opp)
    return p_sel * 0.955

def exp_weighted_mean(values: np.ndarray, decay: float = 0.90) -> float:
    n = len(values)
    weights = decay ** np.arange(n - 1, -1, -1)
    return float(np.sum(values * weights) / np.sum(weights))

def exp_weighted_std(values: np.ndarray, mean_val: float, decay: float = 0.90) -> float:
    n = len(values)
    weights = decay ** np.arange(n - 1, -1, -1)
    var = np.sum(weights * (values - mean_val) ** 2) / np.sum(weights)
    return float(np.sqrt(max(var, 0.45)))

def get_current_season_slice(p_df: pd.DataFrame, league: str) -> pd.DataFrame:
    active_season = SPORT_CONFIG[league]["active_season"]
    curr = p_df[p_df["SEASON"] == active_season].copy()
    if len(curr) >= 4:
        return curr
    return p_df.tail(8).copy()

def is_meaningful_rotation_prop(row: pd.Series, league: str) -> bool:
    """Filters out low-minute/low-usage bench depth and tiny prop lines from Top Straight Bets."""
    avg_usg = float(row.get("AvgUsage", 0.0))
    line_val = float(row.get("RawLine", 0.0))
    stat = str(row.get("Stat", ""))
    pos_g = str(row.get("PosGroup", ""))

    if league == "NBA":
        if avg_usg < 21.0:
            return False
        min_lines = {"PTS": 8.5, "REB": 4.5, "AST": 3.5, "FG3M": 1.5, "PRA": 14.5}
        return line_val >= min_lines.get(stat, 4.5)
    else:
        if pos_g == "QB":
            return avg_usg >= 18.0 and line_val >= 149.5
        if avg_usg < 5.5:
            return False
        min_lines = {"RUSH_YDS": 24.5, "REC_YDS": 24.5, "REC": 2.5, "RUSH_REC_YDS": 39.5, "PASS_TD": 1.0}
        return line_val >= min_lines.get(stat, 24.5)

def skewed_prob_over(line: float, proj_mean: float, proj_std: float, emp_series: pd.Series) -> float:
    emp_recent = (emp_series.tail(5) > line).mean()
    emp_season = (emp_series.tail(15) > line).mean()
    emp_blend = 0.65 * emp_recent + 0.35 * emp_season

    if proj_mean <= 0.05:
        return 0.01

    variance = max(proj_std ** 2, proj_mean * 1.05)
    shape_k = (proj_mean ** 2) / variance
    scale_theta = variance / proj_mean
    dist_over = float(1.0 - gamma.cdf(line, a=shape_k, scale=scale_theta))

    w_dist = min(0.75, max(0.55, len(emp_series) / 20.0))
    return float(np.clip(w_dist * dist_over + (1.0 - w_dist) * emp_blend, 0.02, 0.98))

def compute_player_stat_projection(p_df: pd.DataFrame, stat: str, opp_pos_mults: dict, vegas_total: float, default_total: float, is_home: bool, spread_home: float, usage_bump: float, league: str) -> tuple[float, float]:
    active_season = SPORT_CONFIG[league]["active_season"]
    curr_df = p_df[p_df["SEASON"] == active_season]
    prior_df = p_df[p_df["SEASON"] < active_season]

    curr_vals = curr_df[stat].dropna().values
    prior_vals = prior_df[stat].dropna().tail(10).values

    if len(curr_vals) >= 6:
        recent_form = exp_weighted_mean(curr_vals[-6:], decay=0.88)
        season_avg = float(np.mean(curr_vals))
        base_proj = 0.50 * recent_form + 0.50 * season_avg
        std_vals = curr_vals
    elif len(curr_vals) >= 1:
        w_curr = len(curr_vals) / 6.0
        curr_avg = float(np.mean(curr_vals))
        prior_avg = float(np.mean(prior_vals)) if len(prior_vals) > 0 else curr_avg
        base_proj = w_curr * curr_avg + (1.0 - w_curr) * prior_avg
        std_vals = np.concatenate([prior_vals[-6:], curr_vals])
    elif len(prior_vals) > 0:
        base_proj = exp_weighted_mean(prior_vals, decay=0.90)
        std_vals = prior_vals
    else:
        return 0.0, 1.0

    total_factor = 1.0 + 0.22 * ((vegas_total - default_total) / max(default_total, 1.0))
    raw_dvp = float(opp_pos_mults.get(stat, 1.0))
    dvp_factor = 1.0 + 0.40 * (raw_dvp - 1.0)
    loc_factor = 1.012 if is_home else 0.988

    script_factor = 1.0
    if league == "NFL":
        pos_g = p_df["POS_GROUP"].iloc[-1] if "POS_GROUP" in p_df.columns else ""
        team_spread = spread_home if is_home else -spread_home
        if team_spread <= -6.5 and pos_g == "RB" and stat in ("RUSH_YDS", "RUSH_REC_YDS"):
            script_factor = 1.04
        elif team_spread >= 6.5 and pos_g in ("QB", "WR", "TE") and stat in ("PASS_YDS", "REC_YDS", "REC", "PASS_TD"):
            script_factor = 1.03

    proj = max(0.1, base_proj * total_factor * dvp_factor * loc_factor * script_factor * usage_bump)
    std = exp_weighted_std(std_vals, proj, decay=0.90)
    return float(proj), float(std)

def evaluate_straight_edges(df: pd.DataFrame, slate_props: dict, game_lines: dict, team_metrics: dict, media_lookup: dict, injury_map: dict, league: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    grouped = {k: v for k, v in df.groupby("NORM_NAME")}
    dvp_mults_all = team_metrics["dvp_mult"]
    default_total = SPORT_CONFIG[league]["default_total"]
    min_games_req = 4 if league == "NBA" else 3

    team_usage_bump = {}
    for norm_inj, inj_info in injury_map.items():
        if inj_info.get("is_out") and norm_inj in grouped:
            inj_slice = get_current_season_slice(grouped[norm_inj], league)
            inj_usage = inj_slice["USAGE"].mean()
            threshold = 24.0 if league == "NBA" else 6.0
            if inj_usage >= threshold:
                tm = resolve_team_name(inj_info.get("team", ""), league)
                team_usage_bump[tm] = min(1.08, team_usage_bump.get(tm, 1.0) + 0.03)

    for game_label, game_props in slate_props.items():
        teams_in_game = [resolve_team_name(t.strip(), league) for t in game_label.split("@")]
        if len(teams_in_game) != 2:
            continue
        away_team, home_team = teams_in_game[0], teams_in_game[1]
        g_line_info = game_lines.get(game_label, {"spread_home": 0.0, "total": default_total})
        vegas_total = g_line_info["total"]
        spread_home = g_line_info["spread_home"]

        for norm_p, p_data in game_props.items():
            inj_rec = injury_map.get(norm_p, {})
            if inj_rec.get("is_out"):
                continue

            if norm_p not in grouped:
                continue
            p_df = grouped[norm_p].copy()
            if len(p_df) < min_games_req:
                continue

            if league == "NBA" and p_df["USAGE"].median() >= 24.0:
                clean_p_df = p_df[p_df["USAGE"] >= 18.0]
                if len(clean_p_df) >= 4:
                    p_df = clean_p_df

            p_media = media_lookup.get(norm_p, {})
            player_team = resolve_team_name(p_media.get("team_name") or p_df["TEAM_NAME"].iloc[-1], league)
            if player_team == away_team:
                is_home = False
                opp_team = home_team
            elif player_team == home_team:
                is_home = True
                opp_team = away_team
            else:
                is_home = False
                opp_team = home_team if player_team else away_team

            pos_group = p_media.get("pos_group", p_df["POS_GROUP"].iloc[-1])
            raw_pos = p_media.get("raw_pos", pos_group)
            headshot_url = p_media.get("headshot", DEFAULT_HEADSHOT)
            team_logo_url = get_team_logo(player_team, league)
            team_short = player_team.split()[-1] if player_team else ""
            opp_pos_mults = dvp_mults_all.get(opp_team, {}).get(pos_group, {})
            usage_bump = team_usage_bump.get(player_team, 1.0)

            curr_slice = get_current_season_slice(p_df, league)
            proj_usage = exp_weighted_mean(curr_slice["USAGE"].dropna().values, decay=0.90)
            star_tier_weight = 1.0 + (min(proj_usage, 36.0) / 36.0) * 0.25 if league == "NBA" else 1.0 + (min(proj_usage, 35.0) / 35.0) * 0.25

            for stat, info in p_data["stats"].items():
                if stat not in p_df.columns or "Over" not in info or "Under" not in info:
                    continue

                valid_emp = curr_slice[stat].dropna().tail(12)
                if len(valid_emp) < min_games_req:
                    continue

                proj, std = compute_player_stat_projection(
                    p_df, stat, opp_pos_mults, vegas_total, default_total, is_home, spread_home, usage_bump, league
                )

                main_line = info["line"]
                all_lines_dict = info.get("all_lines", {main_line: {"Over": info["Over"], "Under": info["Under"]}})

                for line_val, sides in all_lines_dict.items():
                    is_alt = not np.isclose(line_val, main_line, atol=0.01)
                    for side in ("Over", "Under"):
                        if is_alt and side == "Under":
                            continue
                        if side not in sides:
                            continue
                        odds = int(sides[side])
                        mod_p = skewed_prob_over(line_val, proj, std, valid_emp) if side == "Over" else (1.0 - skewed_prob_over(line_val, proj, std, valid_emp))

                        opp_side = "Under" if side == "Over" else "Over"
                        opp_odds = int(sides.get(opp_side, -110))
                        nv_p = get_no_vig_prob(odds, opp_odds)
                        edge_pct = (mod_p - nv_p) * 100

                        raw_mult = float(opp_pos_mults.get(stat, 1.0))
                        rating_mult = raw_mult if side == "Over" else (2.0 - raw_mult)
                        rating_txt, rating_cls = format_matchup_rating(rating_mult, pos_group)

                        matchup_fit = 1.0
                        if "Smash" in rating_txt or "Favorable" in rating_txt:
                            matchup_fit = 1.30 if side == "Over" else 0.50
                        elif "Tough" in rating_txt:
                            matchup_fit = 1.30 if side == "Under" else 0.50

                        blended_score = mod_p * matchup_fit * star_tier_weight

                        rows.append({
                            "Player": p_data["display_name"],
                            "NORM_NAME": norm_p,
                            "Team": player_team,
                            "TeamShort": team_short,
                            "Pos": raw_pos,
                            "PosGroup": pos_group,
                            "AvgUsage": round(float(proj_usage), 1),
                            "InjStatus": inj_rec.get("status", ""),
                            "Headshot": headshot_url,
                            "TeamLogo": team_logo_url,
                            "Line": f"{side} {line_val} {stat}",
                            "RawLine": float(line_val),
                            "Side": side,
                            "Stat": stat,
                            "IsAlt": is_alt,
                            "Projection": round(float(proj), 1),
                            "Matchup": short_matchup(game_label),
                            "FullMatchup": game_label,
                            "OppTeam": opp_team,
                            "Matchup Rating": rating_txt,
                            "RatingClass": rating_cls,
                            "Odds": odds,
                            "Win Prob": round(float(mod_p * 100), 1),
                            "Edge": round(float(edge_pct), 1),
                            "BlendedScore": blended_score,
                        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="BlendedScore", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=1800)
def simulate_full_game(df: pd.DataFrame, away_team: str, home_team: str, team_metrics: dict, game_line: dict, media_lookup: dict, league: str, out_players_key: tuple = (), n_sims: int = 15000):
    away_team = resolve_team_name(away_team, league)
    home_team = resolve_team_name(home_team, league)
    dvp_mults_all = team_metrics["dvp_mult"]
    default_total = SPORT_CONFIG[league]["default_total"]
    active_season = SPORT_CONFIG[league]["active_season"]
    vegas_total = game_line.get("total", default_total)
    vegas_spread_home = game_line.get("spread_home", 0.0)
    if df.empty:
        return {
            "away_score": round(default_total / 2.0, 1),
            "home_score": round(default_total / 2.0, 1),
            "away_win_prob": 50.0,
            "home_win_prob": 50.0,
            "proj_spread": 0.0,
            "proj_total": default_total,
            "away_box": pd.DataFrame(),
            "home_box": pd.DataFrame(),
        }

    out_set = set(out_players_key)

    pts_col = "PTS" if league == "NBA" else "TOTAL_PTS"
    curr_season_df = df[df["SEASON"] == active_season]
    pts_source = curr_season_df if curr_season_df["GAME_ID"].nunique() >= 20 else df
    team_game_pts = pts_source.groupby(["TEAM_NAME", "GAME_ID"])[pts_col].sum().groupby("TEAM_NAME").mean().to_dict()
    lg_avg_pts = float(np.mean(list(team_game_pts.values()))) if team_game_pts else (default_total / 2.0)

    def get_rotation(team: str, opp: str, is_home: bool):
        t_df = df[df["TEAM_NAME"] == team].copy()
        if t_df.empty:
            return np.full(n_sims, default_total / 2.0), pd.DataFrame()

        active_pool = t_df[t_df["SEASON"] == active_season].copy()
        if active_pool["GAME_ID"].nunique() < 3:
            active_pool = t_df.copy()
        active_pool = active_pool[~active_pool["NORM_NAME"].isin(out_set)]

        recent_games = active_pool["GAME_ID"].drop_duplicates().tail(5 if league == "NFL" else 8)
        recent_df = active_pool[active_pool["GAME_ID"].isin(recent_games)]

        if league == "NFL":
            qb_pool = recent_df[recent_df["POS_GROUP"] == "QB"]
            top_qb = (
                qb_pool.groupby("PLAYER_NAME")["USAGE"]
                .mean()
                .sort_values(ascending=False)
                .head(1)
                .index.tolist()
            )
            skill_pool = recent_df[recent_df["POS_GROUP"].isin(["RB", "WR", "TE"])]
            top_skill = (
                skill_pool.groupby("PLAYER_NAME")["USAGE"]
                .agg(["count", "mean"])
                .query("count >= 1")
                .sort_values(by="mean", ascending=False)
                .head(7)
                .index.tolist()
            )
            rot_players = top_qb + top_skill
        else:
            rot_players = (
                recent_df.groupby("PLAYER_NAME")["USAGE"]
                .agg(["count", "mean"])
                .query("count >= 1")
                .sort_values(by="mean", ascending=False)
                .head(9)
                .index.tolist()
            )

        if not rot_players:
            rot_players = recent_df["PLAYER_NAME"].unique().tolist()[:8]

        sub = t_df[t_df["PLAYER_NAME"].isin(rot_players)]
        sim_cols = ["PTS", "REB", "AST", "FG3M"] if league == "NBA" else ["PASS_YDS", "RUSH_YDS", "REC_YDS", "REC"]

        player_params = []
        for p in rot_players:
            p_sub = sub[sub["PLAYER_NAME"] == p].copy()
            if league == "NBA" and p_sub["USAGE"].median() >= 24.0:
                filtered_sub = p_sub[p_sub["USAGE"] >= 18.0]
                p_sub = filtered_sub if len(filtered_sub) >= 3 else p_sub

            norm_p = normalize_name(p)
            p_media = media_lookup.get(norm_p, {})
            pos_g = p_media.get("pos_group", p_sub["POS_GROUP"].iloc[-1])
            opp_pos_mults = dvp_mults_all.get(opp, {}).get(pos_g, {})
            curr_u_slice = get_current_season_slice(p_sub, league)
            proj_u = exp_weighted_mean(curr_u_slice["USAGE"].values, decay=0.90)

            stats_proj = {}
            for st_col in sim_cols:
                m_val, s_val = compute_player_stat_projection(
                    p_sub, st_col, opp_pos_mults, vegas_total, default_total, is_home, vegas_spread_home, 1.0, league
                )
                stats_proj[st_col] = (m_val, s_val)
            player_params.append({"player": p, "usage": proj_u, "stats": stats_proj})

        rng = np.random.default_rng(42)
        box_rows = []
        team_logo_url = get_team_logo(team, league)
        team_short = team.split()[-1] if team else ""

        def sim_gamma_stat(m_val, s_val):
            if m_val <= 0.1:
                return np.zeros(n_sims)
            v = max(s_val ** 2, m_val * 1.05)
            return rng.gamma(shape=(m_val ** 2) / v, scale=v / m_val, size=n_sims)

        for p_info in player_params:
            norm_p = normalize_name(p_info["player"])
            p_media = media_lookup.get(norm_p, {})
            if league == "NBA":
                s_pts = sim_gamma_stat(*p_info["stats"]["PTS"])
                s_reb = sim_gamma_stat(*p_info["stats"]["REB"])
                s_ast = sim_gamma_stat(*p_info["stats"]["AST"])
                s_3pm = sim_gamma_stat(*p_info["stats"]["FG3M"])
                box_rows.append({
                    "Player": p_info["player"],
                    "TeamShort": team_short,
                    "Pos": p_media.get("raw_pos", "F"),
                    "Headshot": p_media.get("headshot", DEFAULT_HEADSHOT),
                    "TeamLogo": team_logo_url,
                    "USG": round(p_info["usage"], 1),
                    "S1": float(np.mean(s_pts)),
                    "S2": float(np.mean(s_reb)),
                    "S3": float(np.mean(s_ast)),
                    "S4": float(np.mean(s_3pm)),
                    "COMBO": float(np.mean(s_pts + s_reb + s_ast)),
                })
            else:
                s_pass = sim_gamma_stat(*p_info["stats"]["PASS_YDS"])
                s_rush = sim_gamma_stat(*p_info["stats"]["RUSH_YDS"])
                s_recy = sim_gamma_stat(*p_info["stats"]["REC_YDS"])
                s_rec = sim_gamma_stat(*p_info["stats"]["REC"])
                box_rows.append({
                    "Player": p_info["player"],
                    "TeamShort": team_short,
                    "Pos": p_media.get("raw_pos", "WR"),
                    "Headshot": p_media.get("headshot", DEFAULT_HEADSHOT),
                    "TeamLogo": team_logo_url,
                    "USG": round(p_info["usage"], 1),
                    "S1": float(np.mean(s_pass)),
                    "S2": float(np.mean(s_rush)),
                    "S3": float(np.mean(s_recy)),
                    "S4": float(np.mean(s_rec)),
                    "COMBO": float(np.mean(s_rush + s_recy)),
                })

        vegas_implied = (vegas_total / 2.0) - (vegas_spread_home / 2.0) if is_home else (vegas_total / 2.0) + (vegas_spread_home / 2.0)
        hist_team_pts = team_game_pts.get(team, lg_avg_pts)
        if league == "NFL":
            hist_team_pts = max(16.0, min(32.0, hist_team_pts + 6.0))
        expected_team_pts = 0.35 * hist_team_pts + 0.65 * vegas_implied

        box_df = pd.DataFrame(box_rows)
        if league == "NBA" and not box_df.empty and box_df["S1"].sum() > 0:
            target_rot_pts = expected_team_pts * 0.94
            pts_scale = target_rot_pts / box_df["S1"].sum()
            box_df["S1"] = (box_df["S1"] * pts_scale).round(1)
            box_df["S2"] = (box_df["S2"] * np.sqrt(pts_scale)).round(1)
            box_df["S3"] = (box_df["S3"] * np.sqrt(pts_scale)).round(1)
            box_df["S4"] = box_df["S4"].round(1)
            box_df["COMBO"] = (box_df["S1"] + box_df["S2"] + box_df["S3"]).round(1)
        elif not box_df.empty:
            for c in ["S1", "S2", "S3", "S4", "COMBO"]:
                box_df[c] = box_df[c].round(1)

        sim_scores = rng.normal(expected_team_pts, 8.5 if league == "NFL" else 11.0, size=n_sims)
        sim_scores = np.maximum(3.0 if league == "NFL" else 88.0, sim_scores)
        return sim_scores, box_df

    away_scores, away_box = get_rotation(away_team, home_team, is_home=False)
    home_scores, home_box = get_rotation(home_team, away_team, is_home=True)

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

def build_auto_sgps(df: pd.DataFrame, game_label: str, game_props: dict, injury_map: dict, league: str, min_games: int = 6) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    active_season = SPORT_CONFIG[league]["active_season"]
    active_norms = [
        np_name for np_name in game_props.keys()
        if np_name in set(df["NORM_NAME"]) and not injury_map.get(np_name, {}).get("is_out")
    ]
    curr_sub = df[(df["NORM_NAME"].isin(active_norms)) & (df["SEASON"] == active_season)].copy()
    sub = curr_sub if curr_sub["GAME_ID"].nunique() >= min_games else df[df["NORM_NAME"].isin(active_norms)].copy()
    if sub.empty:
        return pd.DataFrame()

    stats = ["PTS", "REB", "AST", "FG3M"] if league == "NBA" else ["PASS_YDS", "RUSH_YDS", "REC_YDS", "REC"]
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

    if not sgp_rows:
        return pd.DataFrame()
    res_df = pd.DataFrame(sgp_rows)
    return res_df.reindex(res_df["Correlation"].abs().sort_values(ascending=False).index).head(15).reset_index(drop=True)

def optimize_parlay_legs(edges_df: pd.DataFrame, target_american_odds: int, locked_labels: list, exact_legs: int = 4, use_alts: bool = True, exclude_norms: list = None):
    if edges_df.empty:
        return pd.DataFrame(), 0, 0.0, "No active props available."

    if exclude_norms is None:
        exclude_norms = []

    target_dec = american_to_decimal(target_american_odds) if target_american_odds > 0 else 2.0
    pool = edges_df.copy()
    if not use_alts:
        pool = pool[~pool["IsAlt"]]

    pool = pool[~((pool["IsAlt"]) & (pool["Side"] == "Under"))]
    pool["PickLabel"] = pool["Player"] + " — " + pool["Line"] + " (" + pool["Odds"].apply(lambda x: f"{x:+d}") + ")"

    selected_rows = []
    used_players = set()
    under_players = set()
    current_dec = 1.0

    for lbl in locked_labels:
        match = pool[pool["PickLabel"] == lbl]
        if not match.empty:
            row = match.iloc[0]
            selected_rows.append(row)
            used_players.add(row["NORM_NAME"])
            if row["Side"] == "Under":
                under_players.add(row["NORM_NAME"])
            current_dec *= american_to_decimal(int(row["Odds"]))

    while len(selected_rows) < exact_legs:
        rem_legs = exact_legs - len(selected_rows)
        needed_dec = max(1.01, target_dec / current_dec)
        req_avg_dec = needed_dec ** (1.0 / rem_legs)

        avail = pool[
            (~pool["NORM_NAME"].isin(used_players)) &
            (~pool["NORM_NAME"].isin(exclude_norms)) &
            ~((pool["Side"] == "Under") & (pool["NORM_NAME"].isin(under_players)))
        ].copy()
        if avail.empty:
            break

        avail["DecOdds"] = avail["Odds"].apply(american_to_decimal)
        pace_cands = avail[avail["DecOdds"] >= req_avg_dec * 0.90]
        if pace_cands.empty:
            pace_cands = avail.sort_values(by=["DecOdds", "BlendedScore"], ascending=[False, False])
        else:
            pace_cands = pace_cands.sort_values(by=["BlendedScore", "Win Prob"], ascending=[False, False])

        used_games = {r["FullMatchup"] for r in selected_rows}
        diff_game_cands = pace_cands[~pace_cands["FullMatchup"].isin(used_games)]
        chosen_pool = diff_game_cands if not diff_game_cands.empty else pace_cands

        best_cand = chosen_pool.iloc[0]
        selected_rows.append(best_cand)
        used_players.add(best_cand["NORM_NAME"])
        if best_cand["Side"] == "Under":
            under_players.add(best_cand["NORM_NAME"])
        current_dec *= american_to_decimal(int(best_cand["Odds"]))

    if not selected_rows:
        return pd.DataFrame(), 0, 0.0, "Could not build a parlay with current filters."

    res_df = pd.DataFrame(selected_rows)
    comb_prob = float(np.prod(res_df["Win Prob"] / 100.0)) * 100.0
    final_american = decimal_to_american(current_dec)

    status_msg = ""
    if current_dec < target_dec:
        status_msg = f"Max available slate payout reached: {final_american:+d} across {len(res_df)} legs."

    return res_df, final_american, round(comb_prob, 2), status_msg

def render_clean_prop_board(source_df: pd.DataFrame, section_title: str, empty_msg: str, league: str = "NBA", book: str = "draftkings", from_game: str = ""):
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
        t_short = r.get("TeamShort") or (str(r.get("Team", "")).split()[-1] if r.get("Team") else "")
        p_cell = player_badge_html(
            r["Player"], r["Headshot"], r["TeamLogo"], r.get("Pos", ""),
            team_short=t_short, league=league, book=book, from_game=from_game, inj_status=r.get("InjStatus", "")
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

def render_top_projections_board(source_df: pd.DataFrame, stat_col: str, section_title: str, league: str):
    sub = source_df.sort_values(by=stat_col, ascending=False).head(5)
    if sub.empty:
        return
    rows_html = []
    for _, r in sub.iterrows():
        t_name = resolve_team_name(r.get("TEAM_NAME", ""), league)
        t_short = t_name.split()[-1] if t_name else ""
        p_cell = player_badge_html(
            r["PLAYER_NAME"], r["HEADSHOT_HREF"], r["TEAM_LOGO"], r.get("RAW_POS", ""),
            team_short=t_short, league=league
        )
        val = r[stat_col]
        rows_html.append(
            f'<tr>'
            f'<td style="width:45%;">{p_cell}</td>'
            f'<td style="width:25%;">{t_name or "—"}</td>'
            f'<td style="width:30%;" class="proj-cell col-right">{val:.1f} {stat_col}</td>'
            f'</tr>'
        )
    table_html = (
        f'<div class="board-card">'
        f'<div class="board-header"><span class="board-title">{section_title}</span><span class="board-count">Top 5</span></div>'
        f'<div class="table-scroll-wrap"><table class="prop-table">'
        f'<thead><tr>'
        f'<th style="width:45%;">Player</th>'
        f'<th style="width:25%;">Team</th>'
        f'<th style="width:30%;" class="col-right">Projection</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

def render_box_score_table(team_name: str, win_prob: float, box_df: pd.DataFrame, injury_map: dict, league: str, book: str = "draftkings", from_game: str = ""):
    t_logo = get_team_logo(team_name, league)
    headers = ["MIN", "PTS", "REB", "AST", "3PM", "PRA"] if league == "NBA" else ["OPP/TGT", "PASS", "RUSH", "REC YD", "REC", "RUSH+REC"]
    rows_html = []
    for _, r in box_df.iterrows():
        norm_p = normalize_name(r["Player"])
        inj_st = injury_map.get(norm_p, {}).get("status", "")
        p_cell = player_badge_html(
            r["Player"], r["Headshot"], r["TeamLogo"], r.get("Pos", ""),
            team_short=r.get("TeamShort", ""), league=league, book=book, from_game=from_game, inj_status=inj_st
        )
        rows_html.append(
            f'<tr>'
            f'<td style="width:38%;">{p_cell}</td>'
            f'<td style="width:10%;" class="col-right">{r["USG"]:.1f}</td>'
            f'<td style="width:10%;" class="proj-cell col-right">{r["S1"]:.1f}</td>'
            f'<td style="width:10%;" class="col-right">{r["S2"]:.1f}</td>'
            f'<td style="width:10%;" class="col-right">{r["S3"]:.1f}</td>'
            f'<td style="width:10%;" class="col-right">{r["S4"]:.1f}</td>'
            f'<td style="width:12%;" class="line-cell col-right">{r["COMBO"]:.1f}</td>'
            f'</tr>'
        )

    html = (
        f'<div class="board-card">'
        f'<div class="board-header"><span class="board-title"><img class="team-inline-logo" src="{t_logo}" />{team_name}</span><span class="board-count">Win Prob: {win_prob:.1f}%</span></div>'
        f'<div class="table-scroll-wrap"><table class="prop-table">'
        f'<thead><tr>'
        f'<th style="width:38%;">Active Roster (Click Player)</th>'
        f'<th style="width:10%;" class="col-right">{headers[0]}</th>'
        f'<th style="width:10%;" class="col-right">{headers[1]}</th>'
        f'<th style="width:10%;" class="col-right">{headers[2]}</th>'
        f'<th style="width:10%;" class="col-right">{headers[3]}</th>'
        f'<th style="width:10%;" class="col-right">{headers[4]}</th>'
        f'<th style="width:12%;" class="col-right">{headers[5]}</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# ROUTING & STATE INITIALIZATION
# ==========================================
qp_league = st.query_params.get("league", "NBA").upper()
if qp_league not in ("NBA", "NFL"):
    qp_league = "NBA"
qp_player = st.query_params.get("player", "")
qp_game = st.query_params.get("game", "")
qp_book = st.query_params.get("book", "draftkings")
if qp_book not in ["draftkings", "fanduel", "betmgm", "caesars"]:
    qp_book = "draftkings"

if "selected_league" not in st.session_state:
    st.session_state["selected_league"] = qp_league
if "selected_book" not in st.session_state:
    st.session_state["selected_book"] = qp_book
if qp_player:
    st.session_state["active_player_profile"] = qp_player
if qp_game:
    st.session_state["active_game_page"] = qp_game

active_league = st.session_state["selected_league"]
active_season_yr = SPORT_CONFIG[active_league]["active_season"]

st.markdown(
    f'<div class="brand-bar"><div class="brand-left"><div class="brand-logo">HP</div>'
    f'<div><p class="brand-name">{BRAND_NAME}</p><p class="brand-sub">{BRAND_SUBTITLE}</p></div></div>'
    f'<span class="status-badge"><span class="status-dot"></span> {active_league} {active_season_yr} PROPLINE + DvP ENGINE</span></div>',
    unsafe_allow_html=True
)

df = load_sports_data(active_league)
injury_map, team_out_lists = fetch_injury_report(active_league)
out_players_tuple = tuple(sorted([k for k, v in injury_map.items() if v.get("is_out")]))
media_lookup = build_player_media_lookup(df, active_league)
team_metrics = compute_advanced_team_metrics(df, active_league)

all_game_labels, game_lines, slate_props, fetched_at, slate_date_label = fetch_and_lock_propline_slate(
    league=active_league, bookmaker=st.session_state["selected_book"], max_games=16
)

edges_df = evaluate_straight_edges(df, slate_props, game_lines, team_metrics, media_lookup, injury_map, active_league) if slate_props else pd.DataFrame()
main_edges_df = edges_df[~edges_df["IsAlt"]].copy() if not edges_df.empty else pd.DataFrame()

team_today_opp = {}
for g_lbl in all_game_labels:
    pts_g = [resolve_team_name(t.strip(), active_league) for t in g_lbl.split("@")]
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
        back_lbl = "← Back to Matchup Box Score" if st.session_state.get("active_game_page") else f"← Back to {active_league} Slate"
        if st.button(back_lbl, icon=":material/arrow_back:", width="stretch"):
            st.session_state["active_player_profile"] = None
            if "player" in st.query_params:
                del st.query_params["player"]
            st.rerun()

    norm_scout = normalize_name(scout_player)
    p_hist = df[df["NORM_NAME"] == norm_scout].copy() if not df.empty else pd.DataFrame()
    if p_hist.empty:
        st.warning(f"No historical game logs found for {scout_player} in {active_league}.")
        st.stop()

    curr_p_hist = get_current_season_slice(p_hist, active_league)
    p_display = p_hist["PLAYER_NAME"].iloc[-1]
    p_media = media_lookup.get(norm_scout, {})
    p_team = resolve_team_name(p_media.get("team_name") or p_hist["TEAM_NAME"].iloc[-1], active_league)
    h_url = p_media.get("headshot", DEFAULT_HEADSHOT)
    t_logo = get_team_logo(p_team, active_league)
    pos_group = p_media.get("pos_group", p_hist["POS_GROUP"].iloc[-1])
    raw_pos = p_media.get("raw_pos", pos_group)
    pos_plural = SPORT_CONFIG[active_league]["pos_plural"].get(pos_group, "Players")
    p_inj = injury_map.get(norm_scout, {})

    if p_team in team_today_opp:
        opp_team, loc_label, matchup_lbl = team_today_opp[p_team]
    else:
        opp_team = resolve_team_name(str(p_hist["OPP_NAME"].iloc[-1]), active_league)
        loc_label = "Matchup"

    opp_logo = get_team_logo(opp_team, active_league)
    p_lines_df = main_edges_df[main_edges_df["NORM_NAME"] == norm_scout] if not main_edges_df.empty else pd.DataFrame()

    inj_banner_tag = ""
    if p_inj:
        cls_inj = "inj-out-pill" if p_inj.get("is_out") else "inj-q-pill"
        inj_banner_tag = f' &nbsp;<span class="{cls_inj}">INJURY REPORT: {p_inj["status"].upper()}</span>'

    if active_league == "NBA":
        avg_line_1 = f"{curr_p_hist['PTS'].mean():.1f} PTS · {curr_p_hist['REB'].mean():.1f} REB · {curr_p_hist['AST'].mean():.1f} AST"
        avg_line_2 = f"{curr_p_hist.tail(5)['PTS'].mean():.1f} PTS · {curr_p_hist.tail(5)['REB'].mean():.1f} REB · {curr_p_hist.tail(5)['AST'].mean():.1f} AST"
    else:
        avg_line_1 = f"{curr_p_hist['PASS_YDS'].mean():.1f} PASS · {curr_p_hist['RUSH_YDS'].mean():.1f} RUSH · {curr_p_hist['REC_YDS'].mean():.1f} REC YD"
        avg_line_2 = f"{curr_p_hist.tail(5)['PASS_YDS'].mean():.1f} PASS · {curr_p_hist.tail(5)['RUSH_YDS'].mean():.1f} RUSH · {curr_p_hist.tail(5)['REC_YDS'].mean():.1f} REC YD"

    st.markdown(
        f'<div class="game-hero"><div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">'
        f'<div style="display:flex; align-items:center; gap:14px;">'
        f'<img src="{h_url}" onerror="this.src=\'{DEFAULT_HEADSHOT}\'" style="width:68px; height:68px; border-radius:50%; object-fit:cover; background:#1E2638; border:2px solid #10B981;" />'
        f'<div><div style="font-size:0.78rem; color:#10B981; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">{active_league} {active_season_yr} Scouting · Position: {raw_pos} ({pos_plural}){inj_banner_tag}</div>'
        f'<div style="font-size:1.65rem; font-weight:800; color:#F8FAFC; line-height:1.2;">{p_display}</div>'
        f'<div style="font-size:0.86rem; color:#94A3B8; margin-top:3px; display:flex; align-items:center; gap:6px;">'
        f'<img src="{t_logo}" style="width:18px; height:18px;" /> <b>{p_team or "Active Roster"}</b> &nbsp;·&nbsp; {loc_label} vs '
        f'<img src="{opp_logo}" style="width:18px; height:18px;" /> <b>{opp_team or "TBD"}</b></div></div></div>'
        f'<div style="text-align:right; font-size:0.84rem; color:#CBD5E1;">'
        f'<div>{active_season_yr} Avg ({len(curr_p_hist)} GP): <b>{avg_line_1}</b></div>'
        f'<div style="margin-top:3px;">Last 5 Form: <b>{avg_line_2}</b></div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    if not p_lines_df.empty:
        render_clean_prop_board(
            p_lines_df.drop_duplicates(subset=["Stat"]),
            f"{p_display} — Locked Pre-Game Lines & Position-Adjusted Edges",
            "",
            league=active_league,
            book=st.session_state["selected_book"]
        )

    avail_stats = SPORT_CONFIG[active_league]["stats"]
    sc_c1, sc_c2 = st.columns([1.2, 2.8], vertical_alignment="bottom")
    with sc_c1:
        stat_choice = st.selectbox("Inspect Stat Trend", avail_stats)
    default_line = round(float(curr_p_hist[stat_choice].median()) - 0.5, 1)
    if not p_lines_df.empty and stat_choice in set(p_lines_df["Stat"]):
        default_line = float(p_lines_df[p_lines_df["Stat"] == stat_choice]["RawLine"].iloc[0])
    with sc_c2:
        custom_line = st.number_input(f"Target {stat_choice} Line", value=max(0.5, default_line), step=0.5)

    s_series = curr_p_hist[stat_choice].dropna()
    hr5 = (s_series.tail(5) > custom_line).mean() * 100
    hr10 = (s_series.tail(10) > custom_line).mean() * 100
    hr_season = (s_series > custom_line).mean() * 100
    all_series = p_hist[stat_choice].dropna()
    hr_all = (all_series.tail(30) > custom_line).mean() * 100

    hk1, hk2, hk3, hk4 = st.columns(4)
    hk1.metric(f"Last 5 Over {custom_line}", f"{hr5:.0f}%", f"{int((s_series.tail(5) > custom_line).sum())}/5 Games")
    hk2.metric(f"Last 10 Over {custom_line}", f"{hr10:.0f}%", f"{int((s_series.tail(10) > custom_line).sum())}/{min(10, len(s_series))} Games")
    hk3.metric(f"{active_season_yr} Season Over {custom_line}", f"{hr_season:.0f}%", f"{int((s_series > custom_line).sum())}/{len(s_series)} Games")
    hk4.metric(f"Multi-Year Over {custom_line}", f"{hr_all:.0f}%", f"{int((all_series.tail(30) > custom_line).sum())}/{min(30, len(all_series))} Games")

    st.plotly_chart(build_hit_rate_chart(curr_p_hist, stat_choice, custom_line, p_display, active_league), use_container_width=True)

    st.markdown(f"##### {opp_team} {active_season_yr} Defense vs. {pos_plural} (DvP Heat Map)")
    st.caption(f"Rank 1 = Toughest vs {pos_plural} · Rank {30 if active_league == 'NBA' else 32} = Allows Most to {pos_plural}")
    opp_pos_rks = team_metrics["dvp_ranks"].get(opp_team, {}).get(pos_group, {})
    opp_pos_avgs = team_metrics["dvp_avg"].get(opp_team, {}).get(pos_group, {})
    opp_pos_mults = team_metrics["dvp_mult"].get(opp_team, {}).get(pos_group, {})

    tiles_html = []
    for s_code in avail_stats:
        s_lbl = f"{SPORT_CONFIG[active_league]['stat_labels'][s_code]} Allowed"
        rk = opp_pos_rks.get(s_code, 16)
        avg_v = opp_pos_avgs.get(s_code, 0.0)
        mult_v = opp_pos_mults.get(s_code, 1.0)
        diff_pct = (mult_v - 1.0) * 100
        if rk >= 21:
            t_cls, tag = "heat-smash", "Smash"
        elif rk <= 10:
            t_cls, tag = "heat-tough", "Tough"
        else:
            t_cls, tag = "heat-neutral", "Neutral"
        tiles_html.append(
            f'<div class="heat-tile {t_cls}">'
            f'<div class="heat-label">{s_lbl}</div>'
            f'<div class="heat-rank">{ordinal(rk)} <span style="font-size:0.75rem; font-weight:600;">({tag})</span></div>'
            f'<div class="heat-sub">{avg_v:.1f}/gm ({diff_pct:+.1f}% vs Lg)</div>'
            f'</div>'
        )
    st.markdown(f'<div class="heatmap-grid">{"".join(tiles_html)}</div>', unsafe_allow_html=True)

    st.markdown(f"##### {p_display} — {active_season_yr} Game Log")
    gl_rows = []
    gl_headers = ["Date", "Opponent", "MIN", "PTS", "REB", "AST", "3PM", "PRA"] if active_league == "NBA" else ["Season/Wk", "Opponent", "OPP/TGT", "PASS", "RUSH", "REC YD", "REC", "RUSH+REC"]
    for _, gr in curr_p_hist.tail(15).iloc[::-1].iterrows():
        g_dt = f"{int(gr['SEASON'])} W{int(gr['WEEK_NUM'])}" if active_league == "NFL" and "WEEK_NUM" in gr else (gr["GAME_DATE"].strftime("%b %d") if pd.notna(gr["GAME_DATE"]) else "-")
        opp_n = resolve_team_name(str(gr.get("OPP_NAME", "")), active_league)
        opp_lg = get_team_logo(opp_n, active_league)
        ha_tag = "vs" if str(gr.get("HOME_AWAY", "")).lower() == "home" else "@"
        if active_league == "NBA":
            c1, c2, c3, c4, c5, c6 = f"{gr['USAGE']:.0f}", f"{gr['PTS']:.0f}", f"{gr['REB']:.0f}", f"{gr['AST']:.0f}", f"{gr['FG3M']:.0f}", f"{gr['PRA']:.0f}"
        else:
            c1, c2, c3, c4, c5, c6 = f"{gr['USAGE']:.0f}", f"{gr['PASS_YDS']:.0f}", f"{gr['RUSH_YDS']:.0f}", f"{gr['REC_YDS']:.0f}", f"{gr['REC']:.0f}", f"{gr['RUSH_REC_YDS']:.0f}"

        gl_rows.append(
            f'<tr>'
            f'<td style="width:13%;">{g_dt}</td>'
            f'<td style="width:27%;"><img class="team-inline-logo" src="{opp_lg}" />{ha_tag} {opp_n or "—"}</td>'
            f'<td style="width:10%;" class="col-right">{c1}</td>'
            f'<td style="width:10%;" class="proj-cell col-right">{c2}</td>'
            f'<td style="width:10%;" class="col-right">{c3}</td>'
            f'<td style="width:10%;" class="col-right">{c4}</td>'
            f'<td style="width:10%;" class="col-right">{c5}</td>'
            f'<td style="width:10%;" class="line-cell col-right">{c6}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div class="board-card"><div class="table-scroll-wrap"><table class="prop-table">'
        f'<thead><tr>'
        f'<th style="width:13%;">{gl_headers[0]}</th><th style="width:27%;">{gl_headers[1]}</th>'
        f'<th style="width:10%;" class="col-right">{gl_headers[2]}</th><th style="width:10%;" class="col-right">{gl_headers[3]}</th>'
        f'<th style="width:10%;" class="col-right">{gl_headers[4]}</th><th style="width:10%;" class="col-right">{gl_headers[5]}</th>'
        f'<th style="width:10%;" class="col-right">{gl_headers[6]}</th><th style="width:10%;" class="col-right">{gl_headers[7]}</th>'
        f'</tr></thead><tbody>{"".join(gl_rows)}</tbody></table></div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="site-footer"><span><b>{BRAND_NAME}</b> · Quantitative NBA & NFL Analytics Engine</span>'
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
        if st.button(f"← Back to {active_league} Slate", icon=":material/arrow_back:", width="stretch"):
            st.session_state["active_game_page"] = None
            if "game" in st.query_params:
                del st.query_params["game"]
            st.rerun()

    parts = [resolve_team_name(t.strip(), active_league) for t in target_game.split("@")]
    away_sel, home_sel = parts[0], parts[1]
    g_line = game_lines.get(target_game, {"spread_home": 0.0, "total": SPORT_CONFIG[active_league]["default_total"], "tip_str": ""})
    detail_sim = simulate_full_game(df, away_sel, home_sel, team_metrics, g_line, media_lookup, active_league, out_players_key=out_players_tuple, n_sims=15000)

    fav_short = home_sel.split()[-1] if detail_sim["proj_spread"] < 0 else away_sel.split()[-1]
    spread_num = abs(detail_sim["proj_spread"])
    book_spread_str = f"{home_sel.split()[-1]} {g_line['spread_home']:+.1f}" if g_line["spread_home"] != 0 else "N/A"
    away_logo = get_team_logo(away_sel, active_league)
    home_logo = get_team_logo(home_sel, active_league)

    st.markdown(
        f'<div class="game-hero"><div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">'
        f'<div><div style="font-size:0.78rem; color:#10B981; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">'
        f'{active_league} {active_season_yr} Projected Matchup & Box Scores · {g_line.get("tip_str", slate_date_label)}</div>'
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

    st.markdown("##### Projected Active Player Projections (Click Player to Scout)")
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        render_box_score_table(f"{away_sel} (Away)", detail_sim["away_win_prob"], detail_sim["away_box"], injury_map, active_league, book=st.session_state["selected_book"], from_game=target_game)
    with b2:
        render_box_score_table(f"{home_sel} (Home)", detail_sim["home_win_prob"], detail_sim["home_box"], injury_map, active_league, book=st.session_state["selected_book"], from_game=target_game)

    st.markdown(f"##### Locked Pre-Game Player Lines & DvP Matchup Ratings — {away_sel} @ {home_sel}")
    game_props_df = main_edges_df[main_edges_df["FullMatchup"] == target_game] if not main_edges_df.empty else pd.DataFrame()
    if not game_props_df.empty:
        game_props_df = game_props_df.drop_duplicates(subset=["Player", "Stat"]).sort_values(by=["Stat", "Win Prob"], ascending=[True, False])
    render_clean_prop_board(
        game_props_df,
        f"{away_sel} @ {home_sel} — Pre-Game Lines, Projections & DvP Ratings",
        "No pre-game player props were posted for this matchup yet.",
        league=active_league,
        book=st.session_state["selected_book"],
        from_game=target_game
    )

    st.markdown(
        f'<div class="site-footer"><span><b>{BRAND_NAME}</b> · Quantitative NBA & NFL Analytics Engine</span>'
        f'<span>For informational & entertainment purposes only. Please wager responsibly.</span></div>',
        unsafe_allow_html=True
    )
    st.stop()

# =========================================================
# MAIN DASHBOARD VIEW (LEAGUE SWITCHER + CONTROLS)
# =========================================================
ctrl_c0, ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([0.9, 1.1, 2.3, 0.7], vertical_alignment="bottom")
with ctrl_c0:
    league_choice = st.selectbox(
        "League",
        ["NBA", "NFL"],
        index=["NBA", "NFL"].index(active_league),
        format_func=lambda x: "🏀 NBA" if x == "NBA" else "🏈 NFL"
    )
    if league_choice != active_league:
        st.session_state["selected_league"] = league_choice
        st.session_state["parlay_swapped_norms"] = []
        st.query_params["league"] = league_choice
        st.rerun()

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
    search_player = st.text_input("Search", placeholder=f"Filter {active_league} players or teams...")
with ctrl_c3:
    if st.button("Refresh", icon=":material/refresh:", width="stretch"):
        fetch_and_lock_propline_slate.clear()
        fetch_injury_report.clear()
        fetch_live_espn_rosters.clear()
        st.rerun()

slate_teams = set()
for g_lbl in all_game_labels:
    for t in g_lbl.split("@"):
        resolved_t = resolve_team_name(t.strip(), active_league)
        if resolved_t:
            slate_teams.add(resolved_t)
slate_outs_count = sum(len(v) for k, v in team_out_lists.items() if k in slate_teams)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(f"{active_league} Slate ({slate_date_label})", f"{len(all_game_labels)} Games")
kpi2.metric("Active Main Lines", f"{len(main_edges_df)} Lines")
kpi3.metric("Ruled Out (Filtered)", f"{slate_outs_count} Players")
kpi4.metric("Last Slate Sync", fetched_at)

st.write("")

tab_straight, tab_slate, tab_parlays = st.tabs([
    ":material/local_fire_department: Top Straight Bets",
    ":material/scoreboard: Slate & Matchup Projections",
    ":material/bolt: Target-Odds Parlay Optimizer"
])

# =========================================================
# TAB 1: TOP STRAIGHT BETS (ROTATION & LINE-FLOOR FILTERED)
# =========================================================
with tab_straight:
    if not slate_props:
        st.info(f"{active_league} straight bets lock in automatically as soon as pre-game player props open for the slate.")
    elif main_edges_df.empty:
        st.warning(f"No {active_league} pre-game props available for evaluation yet.")
    else:
        bk = st.session_state["selected_book"]
        rotation_main_df = main_edges_df[main_edges_df.apply(lambda r: is_meaningful_rotation_prop(r, active_league), axis=1)].copy()
        if rotation_main_df.empty:
            rotation_main_df = main_edges_df.copy()

        sub_proj, sub_overs, sub_unders = st.tabs([
            "🔥 Top Projections (High Ceiling)",
            "📈 Top Overs",
            "📉 Top Unders"
        ])

        with sub_proj:
            st.caption(f"Highest {active_season_yr} statistical projections for active {active_league} slate players.")
            active_slate_norms = set()
            for g_lbl, g_props in slate_props.items():
                for np_k in g_props.keys():
                    if not injury_map.get(np_k, {}).get("is_out"):
                        active_slate_norms.add(np_k)

            slate_p_rows = []
            grouped_df = {k: v for k, v in df.groupby("NORM_NAME")}
            stat_keys = SPORT_CONFIG[active_league]["stats"][:4]
            for np_k in active_slate_norms:
                if np_k not in grouped_df:
                    continue
                p_df = get_current_season_slice(grouped_df[np_k], active_league)
                if active_league == "NBA" and p_df["USAGE"].median() >= 24.0:
                    clean_p = p_df[p_df["USAGE"] >= 18.0]
                    if len(clean_p) >= 4:
                        p_df = clean_p
                if len(p_df) < 2:
                    continue
                p_med = media_lookup.get(np_k, {})
                t_name = resolve_team_name(p_med.get("team_name") or p_df["TEAM_NAME"].iloc[-1], active_league)
                row_dict = {
                    "PLAYER_NAME": p_df["PLAYER_NAME"].iloc[-1],
                    "TEAM_NAME": t_name,
                    "HEADSHOT_HREF": p_med.get("headshot", DEFAULT_HEADSHOT),
                    "TEAM_LOGO": get_team_logo(t_name, active_league),
                    "RAW_POS": p_med.get("raw_pos", ""),
                }
                for sk in stat_keys:
                    vals = p_df[sk].dropna().values
                    row_dict[sk] = float(0.55 * exp_weighted_mean(vals[-6:], decay=0.88) + 0.45 * np.mean(vals)) if len(vals) > 0 else 0.0
                slate_p_rows.append(row_dict)

            if slate_p_rows:
                sp_df = pd.DataFrame(slate_p_rows)
                pc1, pc2 = st.columns(2, gap="medium")
                labels_map = SPORT_CONFIG[active_league]["stat_labels"]
                with pc1:
                    render_top_projections_board(sp_df, stat_keys[0], f"Top Projected {labels_map[stat_keys[0]]}", active_league)
                    render_top_projections_board(sp_df, stat_keys[2], f"Top Projected {labels_map[stat_keys[2]]}", active_league)
                with pc2:
                    render_top_projections_board(sp_df, stat_keys[1], f"Top Projected {labels_map[stat_keys[1]]}", active_league)
                    render_top_projections_board(sp_df, stat_keys[3], f"Top Projected {labels_map[stat_keys[3]]}", active_league)

        with sub_overs:
            st.caption("High-volume rotation & starter Over plays filtered to remove low-minute bench depth and micro-lines.")
            overs_df = rotation_main_df[rotation_main_df["Side"] == "Over"].sort_values(by="BlendedScore", ascending=False)
            render_clean_prop_board(overs_df.head(15), f"Top {active_league} Over Plays — Rotation & Star Lines", "No qualifying rotation Over plays found.", league=active_league, book=bk)

        with sub_unders:
            st.caption("High-volume rotation & starter Under plays backed by tough defensive matchups.")
            unders_df = rotation_main_df[rotation_main_df["Side"] == "Under"].sort_values(by="BlendedScore", ascending=False)
            render_clean_prop_board(unders_df.head(15), f"Top {active_league} Under Plays — Rotation & Star Lines", "No qualifying rotation Under plays found.", league=active_league, book=bk)

# =========================================================
# TAB 2: SLATE & MATCHUP PROJECTIONS
# =========================================================
with tab_slate:
    st.markdown(f"##### {active_league} {slate_date_label} Projected Scores & Matchups")
    st.caption("Injury-adjusted Monte Carlo simulations. Click any game card to open projected box scores and player props.")

    if not all_game_labels:
        st.info(f"No upcoming {active_league} games posted on the schedule right now.")
    else:
        grid_cols = st.columns(3)
        for idx, g_lbl in enumerate(all_game_labels):
            parts = [resolve_team_name(t.strip(), active_league) for t in g_lbl.split("@")]
            if len(parts) != 2:
                continue
            away_t, home_t = parts[0], parts[1]

            if search_player.strip() and not (search_player.strip().lower() in away_t.lower() or search_player.strip().lower() in home_t.lower()):
                continue

            g_line = game_lines.get(g_lbl, {"spread_home": 0.0, "total": SPORT_CONFIG[active_league]["default_total"], "tip_str": ""})
            sim_res = simulate_full_game(df, away_t, home_t, team_metrics, g_line, media_lookup, active_league, out_players_key=out_players_tuple, n_sims=15000)
            fav_short = home_t.split()[-1] if sim_res["proj_spread"] < 0 else away_t.split()[-1]
            spread_num = abs(sim_res["proj_spread"])
            tip_badge = g_line.get("tip_str", "")
            away_logo = get_team_logo(away_t, active_league)
            home_logo = get_team_logo(home_t, active_league)

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
                    f"View Projections & Props · {away_t.split()[-1]} @ {home_t.split()[-1]}",
                    key=f"open_game_{active_league}_{idx}",
                    icon=":material/arrow_forward:",
                    width="stretch"
                ):
                    st.session_state["active_game_page"] = g_lbl
                    st.query_params["game"] = g_lbl
                    st.rerun()

# =========================================================
# TAB 3: PARLAY OPTIMIZER & SGP FINDER
# =========================================================
with tab_parlays:
    if not slate_props or edges_df.empty:
        st.info(f"{active_league} Parlay Optimizer unlocks as soon as pre-game player props open.")
    else:
        col_multi, col_sgp = st.columns([1.25, 1.0], gap="large")

        with col_multi:
            st.markdown(f"##### 🎯 {active_league} Target Odds & Leg-Lock Parlay Builder")
            st.caption("Specify your target American odds and leg count. Alternate Over milestones engage automatically to hit high payouts.")

            opt_c1, opt_c2, opt_c3 = st.columns([1.0, 1.0, 0.9], vertical_alignment="bottom")
            with opt_c1:
                target_odds = st.number_input("Target Odds (+)", min_value=100, max_value=100000, value=400, step=100, key=f"parlay_target_odds_{active_league}")
            with opt_c2:
                exact_legs = st.slider("Parlay Legs", min_value=2, max_value=15, value=4, key=f"parlay_exact_legs_{active_league}")
            with opt_c3:
                allow_alts = st.checkbox("Use Alt Overs", value=True, key=f"parlay_use_alts_{active_league}")

            valid_lock_df = (edges_df if allow_alts else main_edges_df)
            valid_lock_df = valid_lock_df[~((valid_lock_df["IsAlt"]) & (valid_lock_df["Side"] == "Under"))].sort_values(by="BlendedScore", ascending=False).copy()
            valid_lock_df["PickLabel"] = valid_lock_df["Player"] + " — " + valid_lock_df["Line"] + " (" + valid_lock_df["Odds"].apply(lambda x: f"{x:+d}") + ")"
            lock_options = valid_lock_df["PickLabel"].unique().tolist()

            locked_plays = st.multiselect(
                "🔒 Lock In Play(s) (Optional)",
                options=lock_options,
                placeholder=f"Search & lock any {active_league} player prop...",
                key=f"parlay_locked_{active_league}"
            )

            if "parlay_swapped_norms" not in st.session_state:
                st.session_state["parlay_swapped_norms"] = []

            opt_df, final_odds, comb_win_prob, warn_msg = optimize_parlay_legs(
                edges_df,
                target_american_odds=int(target_odds),
                locked_labels=locked_plays,
                exact_legs=int(exact_legs),
                use_alts=allow_alts,
                exclude_norms=st.session_state["parlay_swapped_norms"]
            )

            if warn_msg:
                st.caption(f"ℹ️ {warn_msg}")

            if not opt_df.empty:
                m1, m2, m3 = st.columns(3)
                m1.metric("Final Parlay Odds", f"{final_odds:+d}", f"Target: +{target_odds:,}")
                m2.metric("Parlay Size", f"{len(opt_df)} Legs", f"{len(locked_plays)} Locked")
                m3.metric("Model Hit Prob", f"{comb_win_prob:.2f}%", f"Implied: {american_to_prob(final_odds)*100:.2f}%")

                slip_lines = [f"{BRAND_NAME} {active_league} ({len(opt_df)}-Leg Parlay | {final_odds:+d})"]
                for _, row in opt_df.iterrows():
                    is_locked = row["PickLabel"] in locked_plays
                    lock_tag = '<span class="inj-q-pill" style="background:rgba(16,185,129,0.18); color:#34D399; border-color:rgba(16,185,129,0.4);">🔒 LOCKED</span>' if is_locked else '<span class="pos-pill">✨ AI PICK</span>'
                    alt_tag = '<span class="alt-pill">ALT</span>' if row.get("IsAlt") else ''
                    slip_lines.append(f"- {'[LOCKED] ' if is_locked else ''}{row['Player']} | {row['Line']} ({row['Odds']:+d})")

                    p_badge = player_badge_html(
                        row["Player"], row["Headshot"], row["TeamLogo"], row.get("Pos", ""),
                        team_short=row.get("TeamShort", ""), league=active_league, book=st.session_state["selected_book"], inj_status=row.get("InjStatus", "")
                    )

                    leg_col_1, leg_col_2 = st.columns([4.2, 1.2], vertical_alignment="center")
                    with leg_col_1:
                        st.markdown(
                            f'<div class="pick-card" style="margin-bottom:0.2rem;">'
                            f'<div class="pick-top">{p_badge}<div style="display:flex; align-items:center; gap:6px;">{alt_tag}{lock_tag}<span class="line-cell">{row["Odds"]:+d}</span></div></div>'
                            f'<div class="pick-line">{row["Line"]}</div>'
                            f'<div class="pick-details">Proj: <b>{row["Projection"]}</b> · Matchup: <b>{row["Matchup Rating"]}</b> · Win Prob: <b>{row["Win Prob"]:.0f}%</b></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with leg_col_2:
                        if not is_locked:
                            if st.button("🔄 Swap", key=f"swap_{active_league}_{row['NORM_NAME']}_{row['Stat']}_{row['RawLine']}", width="stretch"):
                                st.session_state["parlay_swapped_norms"].append(row["NORM_NAME"])
                                st.rerun()

                if st.button("Clear Swaps & Reset Rerolls", key=f"reset_swaps_{active_league}"):
                    st.session_state["parlay_swapped_norms"] = []
                    st.rerun()

                with st.expander("📋 Copy Parlay Slip", icon=":material/content_copy:"):
                    st.code("\n".join(slip_lines), language=None)

        with col_sgp:
            st.markdown(f"##### ⚡ Correlated {active_league} Same-Game Parlays")
            st.caption("Pairs active pre-game lines with strong historical game-log correlations (e.g., QB Pass Yds + WR Rec Yds).")
            sgp_game = st.selectbox("Select Matchup", list(slate_props.keys()), key=f"sgp_matchup_{active_league}")
            sgp_df = build_auto_sgps(df, sgp_game, slate_props[sgp_game], injury_map, active_league)
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

st.markdown(
    f'<div class="site-footer"><span><b>{BRAND_NAME}</b> · Quantitative NBA & NFL Analytics Engine</span>'
    f'<span>For informational & entertainment purposes only. Please wager responsibly.</span></div>',
    unsafe_allow_html=True
)