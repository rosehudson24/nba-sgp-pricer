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
        max-width: 1380px;
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

    .prop-section-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0.85rem;
        background: #111622;
        border: 1px solid #1E2638;
        border-left: 3px solid #10B981;
        border-radius: 8px;
        margin: 0.75rem 0 0.45rem 0;
    }
    .prop-section-title span.title-text {
        font-size: 0.95rem;
        font-weight: 700;
        color: #F1F5F9;
        letter-spacing: -0.2px;
    }
    .prop-section-title span.count-pill {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94A3B8;
        background: #1A2234;
        padding: 2px 8px;
        border-radius: 999px;
    }

    .pick-card {
        background: #111622;
        border: 1px solid #1E2638;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.18s ease;
    }
    .pick-card:hover { border-color: #2E3A54; }
    .pick-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.25rem;
    }
    .pick-player {
        font-size: 0.98rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .ev-pill {
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.32);
        font-size: 0.74rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 999px;
    }
    .pick-line {
        font-size: 0.92rem;
        font-weight: 600;
        color: #10B981;
        margin-bottom: 0.25rem;
    }
    .pick-details {
        font-size: 0.79rem;
        color: #8B97AD;
        line-height: 1.4;
    }
    .pick-matchup {
        font-size: 0.74rem;
        color: #64748B;
        margin-top: 0.35rem;
        padding-top: 0.35rem;
        border-top: 1px solid #1A2234;
    }

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
        padding: 0.25rem 0;
    }
    .matchup-team {
        font-size: 0.98rem;
        font-weight: 600;
        color: #F1F5F9;
    }
    .matchup-prob {
        font-size: 0.78rem;
        color: #64748B;
        margin-left: 4px;
    }
    .matchup-pts {
        font-size: 1.25rem;
        font-weight: 800;
        color: #10B981;
    }
    .matchup-footer {
        display: flex;
        justify-content: space-between;
        margin-top: 0.55rem;
        padding-top: 0.5rem;
        border-top: 1px solid #1A2234;
        font-size: 0.78rem;
        color: #8B97AD;
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
        "athlete_display_name": "PLAYER_NAME",
        "team_display_name": "TEAM_NAME",
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
    df["NORM_NAME"] = df["PLAYER_NAME"].apply(normalize_name)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df

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

def format_matchup_rating(mult: float) -> str:
    score = round(mult * 100, 1)
    if score >= 102.0:
        return f"{score} · Smash"
    elif score >= 100.5:
        return f"{score} · Favorable"
    elif score <= 98.0:
        return f"{score} · Tough"
    elif score <= 99.5:
        return f"{score} · Below Avg"
    return f"{score} · Neutral"

@st.cache_data(ttl=900)
def fetch_slate_by_day(api_key: str, bookmaker: str = "draftkings", day_offset: int = 0, max_games: int = 10):
    """Fetches strictly the NBA slate matching day_offset (0 = Today ET, 1 = Tomorrow ET)."""
    et_tz = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(et_tz)
    target_date_et = (now_et + timedelta(days=day_offset)).date()

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

    day_games = []
    for g in raw_games:
        c_str = g.get("commence_time", "")
        if not c_str:
            continue
        tip_utc = datetime.fromisoformat(c_str.replace("Z", "+00:00"))
        tip_et = tip_utc.astimezone(et_tz)
        if tip_et.date() == target_date_et:
            g["_is_pregame"] = tip_utc > now_utc
            g["_tip_str"] = tip_et.strftime("%I:%M %p ET").lstrip("0")
            day_games.append(g)

    day_games.sort(key=lambda x: (0 if x["_is_pregame"] else 1, x.get("commence_time", "")))

    game_lines = {}
    all_game_labels = []
    for g in day_games:
        lbl = f"{g['away_team']} @ {g['home_team']}"
        all_game_labels.append(lbl)
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
        game_lines[lbl] = {
            "spread_home": spread_home,
            "total": total_line,
            "id": g["id"],
            "is_pregame": g["_is_pregame"],
            "tip_str": g["_tip_str"],
        }

    market_map = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_threes": "FG3M",
        "player_points_rebounds_assists": "PRA",
    }
    slate_props = {}

    for g in day_games[:max_games]:
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

        props = {}
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
                    if not player or side not in ("Over", "Under") or line is None:
                        continue
                    norm_p = normalize_name(player)
                    entry = props.setdefault(norm_p, {"display_name": player, "stats": {}})
                    stat_entry = entry["stats"].setdefault(stat_code, {"line": float(line)})
                    stat_entry[side] = int(price)
        if props:
            slate_props[lbl] = props

    fetched_at = now_et.strftime("%b %d · %I:%M %p ET")
    slate_date_label = target_date_et.strftime("%A, %b %d")
    return all_game_labels, game_lines, slate_props, fetched_at, slate_date_label

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

def skewed_prob_over(line: float, proj_mean: float, proj_std: float, stat: str, emp_series: pd.Series) -> float:
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

def evaluate_straight_edges(df: pd.DataFrame, slate_props: dict, game_lines: dict, team_metrics: dict) -> pd.DataFrame:
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
        g_line_info = game_lines.get(game_label, {"spread_home": 0.0, "total": 228.0, "is_pregame": True})
        vegas_total = g_line_info["total"]
        abs_spread = abs(g_line_info["spread_home"])
        status_tag = "Pre-Game" if g_line_info.get("is_pregame", True) else "Live"

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
                model_over = skewed_prob_over(line, proj, std, stat, valid_df[stat])
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

                    rows.append({
                        "Player": p_data["display_name"],
                        "NORM_NAME": norm_p,
                        "Line": f"{side} {line} {stat}",
                        "Side": side,
                        "Stat": stat,
                        "Status": status_tag,
                        "Projection": round(float(proj), 1),
                        "Matchup": short_matchup(game_label),
                        "FullMatchup": game_label,
                        "Matchup Rating": format_matchup_rating(rating_mult),
                        "Odds": int(odds),
                        "Win Prob": round(float(mod_p * 100), 1),
                        "Edge": round(float(edge_pct), 1),
                        "EV": round(float(ev_pct), 1),
                    })

    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows)
    df_out["_pre_sort"] = np.where(df_out["Status"] == "Pre-Game", 0, 1)
    df_out = df_out.sort_values(by=["_pre_sort", "EV"], ascending=[True, False]).drop(columns=["_pre_sort"]).reset_index(drop=True)
    return df_out

@st.cache_data(ttl=1800)
def simulate_full_game(df: pd.DataFrame, away_team: str, home_team: str, team_metrics: dict, game_line: dict, n_sims: int = 15000):
    def_mults_all = team_metrics["def_mult"]
    last_games = team_metrics["last_game"]
    today_ts = pd.Timestamp.now().normalize()
    vegas_total = game_line.get("total", 228.0)
    vegas_pace_mult = 1.0 + 0.45 * ((vegas_total - 228.0) / 228.0)

    def get_rotation(team: str, opp: str, is_home: bool):
        t_df = df[df["TEAM_NAME"] == team].copy()
        if t_df.empty:
            return np.full(n_sims, 110.0), pd.DataFrame()

        recent_games = t_df["GAME_ID"].drop_duplicates().tail(15)
        recent_df = t_df[t_df["GAME_ID"].isin(recent_games)]

        rot_players = (
            recent_df.groupby("PLAYER_NAME")["MIN"]
            .agg(["count", "mean"])
            .query("count >= 3")
            .sort_values(by="mean", ascending=False)
            .head(9)
            .index.tolist()
        )
        if not rot_players:
            rot_players = recent_df["PLAYER_NAME"].unique().tolist()[:9]

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

            box_rows.append({
                "Player": p_info["player"],
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

# Load multi-season database & compute advanced team metrics
df = load_nba_data()
team_metrics = compute_advanced_team_metrics(df)

saved_key = ""
try:
    saved_key = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    saved_key = ""

# Top Brand Header
st.markdown(f"""
<div class="brand-bar">
    <div class="brand-left">
        <div class="brand-logo">HP</div>
        <div>
            <p class="brand-name">{BRAND_NAME}</p>
            <p class="brand-sub">{BRAND_SUBTITLE}</p>
        </div>
    </div>
    <span class="status-badge"><span class="status-dot"></span> TODAY'S NBA SLATE</span>
</div>
""", unsafe_allow_html=True)

# Inline Control Strip
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5 = st.columns([1.1, 1.1, 1.4, 1.5, 0.7], vertical_alignment="bottom")
with ctrl_c1:
    slate_day = st.selectbox("Slate Day", ["Today's Slate", "Tomorrow's Slate"])
with ctrl_c2:
    bookmaker = st.selectbox(
        "Sportsbook",
        ["draftkings", "fanduel", "betmgm", "caesars"],
        format_func=lambda x: {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "caesars": "Caesars"}[x]
    )
with ctrl_c3:
    min_ev = st.slider("Minimum EV %", min_value=-15.0, max_value=15.0, value=-10.0, step=0.5)
with ctrl_c4:
    search_player = st.text_input("Search", placeholder="Player or team...")
with ctrl_c5:
    if st.button("Refresh", icon=":material/refresh:", width="stretch"):
        fetch_slate_by_day.clear()
        st.rerun()

if not saved_key:
    saved_key = st.text_input("Enter Odds API Key (or set ODDS_API_KEY in Streamlit Secrets)", type="password")

day_offset = 0 if slate_day == "Today's Slate" else 1
all_game_labels = []
game_lines = {}
slate_props = {}
fetched_at = "Just now"
slate_date_label = ""

if saved_key:
    try:
        all_game_labels, game_lines, slate_props, fetched_at, slate_date_label = fetch_slate_by_day(
            saved_key, bookmaker=bookmaker, day_offset=day_offset, max_games=10
        )
        if not slate_props:
            fetch_slate_by_day.clear()
    except Exception as e:
        st.error(f"Live odds feed unavailable: {e}")

edges_df = evaluate_straight_edges(df, slate_props, game_lines, team_metrics) if slate_props else pd.DataFrame()
filtered = edges_df[edges_df["EV"] >= min_ev] if not edges_df.empty else pd.DataFrame()

if not filtered.empty and search_player.strip():
    q = search_player.strip()
    filtered = filtered[
        filtered["Player"].str.contains(q, case=False, na=False) |
        filtered["FullMatchup"].str.contains(q, case=False, na=False)
    ]

# Top KPI Summary Strip
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(slate_date_label or "Active Slate", f"{len(all_game_labels)} Games")
kpi2.metric("Lines Evaluated", f"{len(filtered)} Props")
top_pick_str = f"{filtered.iloc[0]['Player']} ({filtered.iloc[0]['EV']:+.1f}%)" if not filtered.empty else "Slate Finalized"
kpi3.metric("Top Model Edge", top_pick_str)
kpi4.metric("Last Board Sync", fetched_at)

st.write("")

# Navigation Tabs
tab_straight, tab_slate, tab_parlays, tab_custom = st.tabs([
    ":material/local_fire_department: Top Straight Bets",
    ":material/scoreboard: Tonight's Slate & Box Scores",
    ":material/bolt: Auto-Parlay Builder",
    ":material/tune: Custom SGP Lab"
])

# =========================================================
# TAB 1: TOP STRAIGHT BETS (Zero-Scroll Top 10 Per Prop Grid)
# =========================================================
def render_prop_board(stat_code: str, section_title: str, top_n: int = 10):
    stat_df = filtered[filtered["Stat"] == stat_code].head(top_n)
    st.markdown(f"""
    <div class="prop-section-title">
        <span class="title-text">{section_title}</span>
        <span class="count-pill">Top {len(stat_df)}</span>
    </div>
    """, unsafe_allow_html=True)
    if stat_df.empty:
        st.caption(f"No {stat_code} props posted on this slate right now.")
        return

    display_cols = ["Player", "Line", "Projection", "Matchup", "Matchup Rating", "Odds", "Win Prob", "EV"]
    col_cfg = {
        "Player": st.column_config.TextColumn("Player", width="medium"),
        "Line": st.column_config.TextColumn("Line", width="small"),
        "Projection": st.column_config.NumberColumn("Proj", format="%.1f", width="small"),
        "Matchup": st.column_config.TextColumn("Game", width="small"),
        "Matchup Rating": st.column_config.TextColumn("Matchup Rating", width="medium"),
        "Odds": st.column_config.NumberColumn("Odds", format="%+d", width="small"),
        "Win Prob": st.column_config.ProgressColumn("Win %", min_value=0, max_value=100, format="%.0f%%", width="small"),
        "EV": st.column_config.NumberColumn("EV %", format="%+.1f%%", width="small"),
    }
    # Exact pixel height so all rows render without any internal scrollbars
    exact_height = (len(stat_df) + 1) * 35 + 3
    st.dataframe(
        stat_df[display_cols],
        width="stretch",
        hide_index=True,
        height=exact_height,
        column_config=col_cfg
    )

with tab_straight:
    if not slate_props:
        st.info(f"All games on {slate_date_label or 'today’s slate'} have concluded or closed prop markets. Tomorrow's slate will automatically take over after midnight ET (or switch to **Tomorrow's Slate** in the top-left selector to preview early lines).")
    elif filtered.empty:
        st.warning(f"No lines match your current filter ({min_ev:+.1f}% EV). Lower the Minimum EV slider above to view all posted props.")
    else:
        # 2x2 Grid + Full-Width PRA Row so all Top-10 Prop Boards are visible together without scrolling
        row1_c1, row1_c2 = st.columns(2, gap="medium")
        with row1_c1:
            render_prop_board("PTS", "Points Props (PTS)", top_n=10)
        with row1_c2:
            render_prop_board("REB", "Rebounds Props (REB)", top_n=10)

        row2_c1, row2_c2 = st.columns(2, gap="medium")
        with row2_c1:
            render_prop_board("AST", "Assists Props (AST)", top_n=10)
        with row2_c2:
            render_prop_board("FG3M", "3-Pointers Made Props (3PM)", top_n=10)

        render_prop_board("PRA", "Points + Rebounds + Assists (PRA)", top_n=10)

# =========================================================
# TAB 2: TONIGHT'S SLATE & CLICK-IN PROJECTED BOX SCORES
# =========================================================
with tab_slate:
    st.markdown(f"##### {slate_date_label or 'Slate'} Projected NBA Scores & Box Scores")
    st.caption("Simulated 15,000 times using per-minute rate models, opponent defensive ratings, Vegas pace environments, and rest/location splits.")

    slate_list = all_game_labels if all_game_labels else [
        "Denver Nuggets @ Phoenix Suns",
        "Boston Celtics @ New York Knicks",
        "Dallas Mavericks @ Oklahoma City Thunder"
    ]

    if "selected_box_game" not in st.session_state or st.session_state["selected_box_game"] not in slate_list:
        st.session_state["selected_box_game"] = slate_list[0]

    grid_cols = st.columns(3)
    for idx, g_lbl in enumerate(slate_list):
        parts = [t.strip() for t in g_lbl.split("@")]
        if len(parts) != 2:
            continue
        away_t, home_t = parts[0], parts[1]
        g_line = game_lines.get(g_lbl, {"spread_home": 0.0, "total": 228.0, "tip_str": ""})
        sim_res = simulate_full_game(df, away_t, home_t, team_metrics, g_line, n_sims=15000)
        fav_short = home_t.split()[-1] if sim_res["proj_spread"] < 0 else away_t.split()[-1]
        spread_num = abs(sim_res["proj_spread"])
        tip_badge = g_line.get("tip_str", "")

        with grid_cols[idx % 3]:
            st.markdown(f"""
            <div class="matchup-card">
                <div class="matchup-team-row">
                    <span class="matchup-team">{away_t} <span class="matchup-prob">({sim_res['away_win_prob']}%)</span></span>
                    <span class="matchup-pts">{sim_res['away_score']}</span>
                </div>
                <div class="matchup-team-row">
                    <span class="matchup-team">{home_t} <span class="matchup-prob">({sim_res['home_win_prob']}%)</span></span>
                    <span class="matchup-pts">{sim_res['home_score']}</span>
                </div>
                <div class="matchup-footer">
                    <span>{tip_badge} · Line: <b>{fav_short} -{spread_num}</b></span>
                    <span>Total: <b>{sim_res['proj_total']}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"View Box Score · {away_t.split()[-1]} @ {home_t.split()[-1]}", key=f"btn_box_{idx}", width="stretch"):
                st.session_state["selected_box_game"] = g_lbl

    st.divider()

    active_game = st.selectbox(
        "Projected Box Score Matchup",
        slate_list,
        index=slate_list.index(st.session_state["selected_box_game"])
    )
    st.session_state["selected_box_game"] = active_game

    away_sel, home_sel = [t.strip() for t in active_game.split("@")]
    detail_sim = simulate_full_game(df, away_sel, home_sel, team_metrics, game_lines.get(active_game, {"spread_home": 0.0, "total": 228.0}), n_sims=15000)

    st.markdown(f"##### Projected Box Score: {away_sel} ({detail_sim['away_score']}) @ {home_sel} ({detail_sim['home_score']})")
    box_c1, box_c2 = st.columns(2, gap="large")
    box_cfg = {
        "Player": st.column_config.TextColumn("Player", width="medium"),
        "MIN": st.column_config.NumberColumn("MIN", format="%.1f"),
        "PTS": st.column_config.NumberColumn("PTS", format="%.1f"),
        "REB": st.column_config.NumberColumn("REB", format="%.1f"),
        "AST": st.column_config.NumberColumn("AST", format="%.1f"),
        "3PM": st.column_config.NumberColumn("3PM", format="%.1f"),
        "PRA": st.column_config.NumberColumn("PRA", format="%.1f"),
    }
    with box_c1:
        st.caption(f"{away_sel} (Away) · Win Probability: {detail_sim['away_win_prob']}%")
        st.dataframe(detail_sim["away_box"], width="stretch", hide_index=True, column_config=box_cfg)
    with box_c2:
        st.caption(f"{home_sel} (Home) · Win Probability: {detail_sim['home_win_prob']}%")
        st.dataframe(detail_sim["home_box"], width="stretch", hide_index=True, column_config=box_cfg)

# =========================================================
# TAB 3: AUTO-PARLAY BUILDER
# =========================================================
with tab_parlays:
    if not slate_props:
        st.info("Auto-parlays will populate automatically when player props are active on the selected slate.")
    else:
        col_multi, col_sgp = st.columns(2, gap="large")

        with col_multi:
            st.markdown("##### Cross-Game +EV Parlay")
            st.caption("Combines the highest independent model edge from each matchup on the slate.")
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
                        slip_lines.append(f"- {row['Player']} | {row['Line']} ({row['Odds']:+d})")
                        st.markdown(f"""
                        <div class="pick-card">
                            <div class="pick-top">
                                <span class="pick-player">{row['Player']}</span>
                                <span class="ev-pill">{row['Odds']:+d}</span>
                            </div>
                            <div class="pick-line">{row['Line']}</div>
                            <div class="pick-details">Proj: <b>{row['Projection']}</b> · Rating: <b>{row['Matchup Rating']}</b> · EV: <b>{row['EV']:+.1f}%</b></div>
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander("Copy Parlay Slip", icon=":material/content_copy:"):
                        st.code("\n".join(slip_lines), language=None)
                else:
                    st.warning("At least 2 matchups with props are needed to build a cross-game parlay.")

        with col_sgp:
            st.markdown("##### Correlated Same-Game Parlays")
            st.caption("Pairs live lines with strong historical game-log correlations.")
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
st.markdown(f"""
<div class="site-footer">
    <span><b>{BRAND_NAME}</b> · Quantitative NBA Analytics & Simulation Engine</span>
    <span>For informational & entertainment purposes only. Please wager responsibly.</span>
</div>
""", unsafe_allow_html=True)