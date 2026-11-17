import re
import unicodedata
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm

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

# Custom Bespoke Dark Theme & Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    section[data-testid="stSidebar"] { display: none; }
    div[data-testid="stDecoration"] { display: none; }

    /* Top Navigation & Brand Bar */
    .brand-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.1rem 1.4rem;
        background: #111622;
        border: 1px solid #1E2638;
        border-radius: 12px;
        margin-bottom: 1rem;
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .brand-left { display: flex; align-items: center; gap: 0.85rem; }
    .brand-logo {
        width: 38px;
        height: 38px;
        border-radius: 9px;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.1rem;
        color: #06100C;
        letter-spacing: -0.5px;
    }
    .brand-name {
        font-size: 1.45rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.4px;
        margin: 0;
        line-height: 1.15;
    }
    .brand-sub {
        font-size: 0.83rem;
        color: #8B97AD;
        margin: 0.15rem 0 0 0;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.28);
        color: #34D399;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 999px;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background: #10B981;
        border-radius: 50%;
    }

    /* Section Headers */
    .prop-section-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.65rem 0.95rem;
        background: #111622;
        border: 1px solid #1E2638;
        border-left: 3px solid #10B981;
        border-radius: 8px;
        margin: 1.35rem 0 0.65rem 0;
    }
    .prop-section-title span.title-text {
        font-size: 1rem;
        font-weight: 700;
        color: #F1F5F9;
        letter-spacing: -0.2px;
    }
    .prop-section-title span.count-pill {
        font-size: 0.76rem;
        font-weight: 600;
        color: #94A3B8;
        background: #1A2234;
        padding: 2px 9px;
        border-radius: 999px;
    }

    /* Bet Cards */
    .pick-card {
        background: #111622;
        border: 1px solid #1E2638;
        border-radius: 11px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.65rem;
        transition: border-color 0.18s ease, transform 0.18s ease;
    }
    .pick-card:hover {
        border-color: #2E3A54;
    }
    .pick-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }
    .pick-player {
        font-size: 1.02rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .ev-pill {
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.32);
        font-size: 0.76rem;
        font-weight: 700;
        padding: 2px 9px;
        border-radius: 999px;
    }
    .pick-line {
        font-size: 0.96rem;
        font-weight: 600;
        color: #10B981;
        margin-bottom: 0.3rem;
    }
    .pick-details {
        font-size: 0.82rem;
        color: #8B97AD;
        line-height: 1.45;
    }
    .pick-matchup {
        font-size: 0.76rem;
        color: #64748B;
        margin-top: 0.45rem;
        padding-top: 0.4rem;
        border-top: 1px solid #1A2234;
    }

    /* Game Scoreboard Cards */
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

    /* Footer */
    .site-footer {
        margin-top: 3rem;
        padding-top: 1.25rem;
        border-top: 1px solid #1E2638;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem;
        font-size: 0.78rem;
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

@st.cache_data
def load_nba_data(seasons=(2024, 2025, 2026)):
    frames = []
    for yr in seasons:
        url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{yr}.parquet"
        try:
            frames.append(pd.read_parquet(url))
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True)
    df = df[df["minutes"] >= 12].copy()
    df["game_id"] = df["game_id"].astype(str)
    if "game_date" in df.columns:
        df = df.sort_values("game_date")
    df = df.drop_duplicates(subset=["game_id", "athlete_display_name"])
    rename_map = {
        "game_id": "GAME_ID",
        "athlete_display_name": "PLAYER_NAME",
        "team_display_name": "TEAM_NAME",
        "opponent_team_display_name": "OPP_NAME",
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
    return df

@st.cache_data
def compute_matchup_ratings(df: pd.DataFrame) -> dict:
    if "OPP_NAME" not in df.columns:
        return {}
    stats = ["PTS", "REB", "AST", "FG3M"]
    game_allowed = df.groupby(["OPP_NAME", "GAME_ID"])[stats].sum().reset_index()
    team_avg = game_allowed.groupby("OPP_NAME")[stats].mean()
    league_avg = team_avg.mean()
    multipliers = (team_avg / league_avg).fillna(1.0)
    return multipliers.to_dict(orient="index")

def format_matchup_rating(mult: float) -> str:
    score = round(mult * 100, 1)
    diff = round((mult - 1.0) * 100, 1)
    if score >= 102.0:
        return f"{score} · Smash ({diff:+.1f}%)"
    elif score >= 100.5:
        return f"{score} · Favorable ({diff:+.1f}%)"
    elif score <= 98.0:
        return f"{score} · Tough ({diff:+.1f}%)"
    elif score <= 99.5:
        return f"{score} · Below Avg ({diff:+.1f}%)"
    return f"{score} · Neutral ({diff:+.1f}%)"

@st.cache_data(ttl=1800)
def fetch_full_slate_props(api_key: str, bookmaker: str = "draftkings", max_games: int = 6):
    events_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    ev_res = requests.get(events_url, params={"apiKey": api_key}, timeout=10)
    ev_res.raise_for_status()
    games = ev_res.json()

    market_map = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_threes": "FG3M",
    }
    slate_props = {}
    all_game_labels = [f"{g['away_team']} @ {g['home_team']}" for g in games]

    for g in games[:max_games]:
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

    fetched_at = datetime.now().strftime("%I:%M %p")
    return all_game_labels, slate_props, fetched_at

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

def quarter_kelly_units(win_prob: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    if b <= 0:
        return 0.0
    full_kelly = (win_prob * (b + 1.0) - 1.0) / b
    return round(max(full_kelly * 0.25 * 100, 0.0), 2)

def evaluate_straight_edges(df: pd.DataFrame, slate_props: dict, def_ratings: dict) -> pd.DataFrame:
    rows = []
    grouped = {k: v for k, v in df.groupby("NORM_NAME")}

    for game_label, game_props in slate_props.items():
        teams_in_game = [t.strip() for t in game_label.split("@")]
        for norm_p, p_data in game_props.items():
            if norm_p not in grouped:
                continue
            p_df = grouped[norm_p]
            if len(p_df) < 12:
                continue

            player_team = p_df["TEAM_NAME"].iloc[-1]
            opp_team = next((t for t in teams_in_game if t != player_team), "")
            opp_mults = def_ratings.get(opp_team, {})

            for stat, info in p_data["stats"].items():
                if stat not in p_df.columns or "Over" not in info or "Under" not in info:
                    continue
                line = info["line"]
                series = p_df[stat].dropna()
                if len(series) < 12:
                    continue

                recent = series.tail(10)
                base_proj = 0.60 * recent.mean() + 0.40 * series.mean()
                raw_mult = opp_mults.get(stat, 1.0)
                adj_mult = 1.0 + 0.50 * (raw_mult - 1.0)
                proj = base_proj * adj_mult
                std = max(series.std(), 1.25)

                emp_over = (series.tail(15) > line).mean()
                norm_over = 1 - norm.cdf(line, loc=proj, scale=std)
                model_over = 0.45 * emp_over + 0.55 * norm_over
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
                    units = quarter_kelly_units(mod_p, dec)
                    rating_mult = adj_mult if side == "Over" else (2.0 - adj_mult)
                    rows.append({
                        "Player": p_data["display_name"],
                        "NORM_NAME": norm_p,
                        "Line": f"{side} {line} {stat}",
                        "Side": side,
                        "Stat": stat,
                        "Projection": round(float(proj), 1),
                        "Matchup": game_label,
                        "Matchup Rating": format_matchup_rating(rating_mult),
                        "Odds": int(odds),
                        "Win Prob": round(float(mod_p * 100), 1),
                        "Edge": round(float(edge_pct), 1),
                        "EV": round(float(ev_pct), 1),
                        "Units": round(float(units), 2),
                    })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="EV", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=1800)
def simulate_full_game(df: pd.DataFrame, away_team: str, home_team: str, def_ratings: dict, n_sims: int = 10000):
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
        opp_mults = def_ratings.get(opp, {})
        home_factor = 1.015 if is_home else 0.985

        player_params = []
        for p in rot_players:
            p_sub = sub[sub["PLAYER_NAME"] == p]
            rec = p_sub.tail(10)
            mins = float(rec["MIN"].mean())
            stats_proj = {}
            for st_col in ["PTS", "REB", "AST", "FG3M"]:
                base = 0.65 * rec[st_col].mean() + 0.35 * p_sub[st_col].mean()
                mult = 1.0 + 0.50 * (opp_mults.get(st_col, 1.0) - 1.0)
                mean_val = float(base * mult * (home_factor if st_col == "PTS" else 1.0))
                std_val = float(max(p_sub[st_col].std(), 0.8))
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
        sim_pts_matrix = np.zeros((n_sims, len(rot_players)))
        box_rows = []

        for idx, p_info in enumerate(player_params):
            pts_m, pts_s = p_info["stats"]["PTS"]
            sim_p = np.clip(pts_m + z[:, idx] * pts_s, 0, None)
            sim_pts_matrix[:, idx] = sim_p

            reb_m, reb_s = p_info["stats"]["REB"]
            ast_m, ast_s = p_info["stats"]["AST"]
            fg3_m, fg3_s = p_info["stats"]["FG3M"]

            sim_r = np.clip(rng.normal(reb_m, reb_s, size=n_sims), 0, None)
            sim_a = np.clip(rng.normal(ast_m, ast_s, size=n_sims), 0, None)
            sim_3 = np.clip(rng.normal(fg3_m, fg3_s, size=n_sims), 0, None)

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

# Load core dataset & matchup ratings
df = load_nba_data()
def_ratings = compute_matchup_ratings(df)

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
    <span class="status-badge"><span class="status-dot"></span> LIVE MODEL & ODDS FEED</span>
</div>
""", unsafe_allow_html=True)

# Inline Top Control Strip (No Sidebar Needed)
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.2, 1.8, 1.8, 0.8], vertical_alignment="bottom")
with ctrl_c1:
    bookmaker = st.selectbox(
        "Sportsbook",
        ["draftkings", "fanduel", "betmgm", "caesars"],
        format_func=lambda x: {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "caesars": "Caesars"}[x]
    )
with ctrl_c2:
    min_ev = st.slider("Minimum Edge (EV %)", min_value=-5.0, max_value=15.0, value=1.5, step=0.5)
with ctrl_c3:
    search_player = st.text_input("Filter by Player or Team", placeholder="Search player or matchup...")
with ctrl_c4:
    if st.button("Refresh", icon=":material/refresh:", width="stretch"):
        fetch_full_slate_props.clear()
        st.rerun()

if not saved_key:
    saved_key = st.text_input("Enter Odds API Key (or set ODDS_API_KEY in Streamlit Secrets)", type="password")

all_game_labels = []
slate_props = {}
fetched_at = "Just now"

if saved_key:
    try:
        all_game_labels, slate_props, fetched_at = fetch_full_slate_props(saved_key, bookmaker=bookmaker, max_games=6)
    except Exception as e:
        st.error(f"Live odds feed unavailable: {e}")

edges_df = evaluate_straight_edges(df, slate_props, def_ratings) if slate_props else pd.DataFrame()
filtered = edges_df[edges_df["EV"] >= min_ev] if not edges_df.empty else pd.DataFrame()

if not filtered.empty and search_player.strip():
    q = search_player.strip()
    filtered = filtered[
        filtered["Player"].str.contains(q, case=False, na=False) |
        filtered["Matchup"].str.contains(q, case=False, na=False)
    ]

# Top KPI Summary Strip
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active Slate", f"{len(slate_props)} Games")
kpi2.metric("+EV Straight Plays", f"{len(filtered)} Edges")
top_pick_str = f"{filtered.iloc[0]['Player']} ({filtered.iloc[0]['EV']:+.1f}%)" if not filtered.empty else "Awaiting Lines"
kpi3.metric("Top Model Edge", top_pick_str)
kpi4.metric("Last Board Sync", fetched_at)

st.write("")

# Clean Material-Icon Tabs
tab_straight, tab_slate, tab_parlays, tab_custom = st.tabs([
    ":material/local_fire_department: Top Straight Bets",
    ":material/scoreboard: Tonight's Slate & Box Scores",
    ":material/bolt: Auto-Parlay Builder",
    ":material/tune: Custom SGP Lab"
])

# =========================================================
# TAB 1: TOP STRAIGHT BETS (Sectioned by Prop Category)
# =========================================================
with tab_straight:
    if not slate_props:
        st.info("No live NBA player props are currently posted on the board (games may already be underway or lines haven't opened yet).")
    elif filtered.empty:
        st.warning(f"No straight plays currently meet the {min_ev:+.1f}% EV threshold. Lower the EV slider above to view more lines.")
    else:
        st.markdown("##### Featured Highest-EV Plays")
        top_cols = st.columns(3)
        for idx, (_, row) in enumerate(filtered.head(3).iterrows()):
            with top_cols[idx]:
                st.markdown(f"""
                <div class="pick-card">
                    <div class="pick-top">
                        <span class="pick-player">{row['Player']}</span>
                        <span class="ev-pill">{row['EV']:+.1f}% EV</span>
                    </div>
                    <div class="pick-line">{row['Line']} ({row['Odds']:+d})</div>
                    <div class="pick-details">
                        Projection: <b>{row['Projection']}</b> · Win Prob: <b>{row['Win Prob']}%</b><br>
                        Matchup Rating: <b>{row['Matchup Rating']}</b>
                    </div>
                    <div class="pick-matchup">{row['Matchup']} · Rec. Size: {row['Units']:.2f}u</div>
                </div>
                """, unsafe_allow_html=True)

        prop_sections = [
            ("PTS", "Points Props (PTS)"),
            ("REB", "Rebounds Props (REB)"),
            ("AST", "Assists Props (AST)"),
            ("FG3M", "3-Pointers Made Props (3PM)"),
        ]
        display_cols = ["Player", "Line", "Projection", "Matchup", "Matchup Rating", "Odds", "Win Prob", "EV", "Units"]
        col_cfg = {
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Line": st.column_config.TextColumn("Line", width="medium"),
            "Projection": st.column_config.NumberColumn("Projection", format="%.1f"),
            "Matchup": st.column_config.TextColumn("Matchup", width="medium"),
            "Matchup Rating": st.column_config.TextColumn("Matchup Rating", width="medium"),
            "Odds": st.column_config.NumberColumn("Book Odds", format="%+d"),
            "Win Prob": st.column_config.ProgressColumn("Win Prob", min_value=0, max_value=100, format="%.1f%%"),
            "EV": st.column_config.NumberColumn("EV %", format="%+.1f%%"),
            "Units": st.column_config.NumberColumn("Rec. Size", format="%.2fu"),
        }

        for stat_code, section_title in prop_sections:
            stat_df = filtered[filtered["Stat"] == stat_code]
            st.markdown(f"""
            <div class="prop-section-title">
                <span class="title-text">{section_title}</span>
                <span class="count-pill">{len(stat_df)} Plays</span>
            </div>
            """, unsafe_allow_html=True)
            if stat_df.empty:
                st.caption(f"No {stat_code} props meeting the {min_ev:+.1f}% EV filter right now.")
            else:
                st.dataframe(
                    stat_df[display_cols],
                    width="stretch",
                    hide_index=True,
                    column_config=col_cfg
                )

# =========================================================
# TAB 2: TONIGHT'S SLATE & CLICK-IN PROJECTED BOX SCORES
# =========================================================
with tab_slate:
    st.markdown("##### Tonight's Projected NBA Scores")
    st.caption("Simulated 10,000 times using player rotation baselines and opponent defensive matchup ratings. Select any matchup to inspect its projected player box score.")

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
        sim_res = simulate_full_game(df, away_t, home_t, def_ratings, n_sims=10000)
        fav_short = home_t.split()[-1] if sim_res["proj_spread"] < 0 else away_t.split()[-1]
        spread_num = abs(sim_res["proj_spread"])

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
                    <span>Proj Line: <b>{fav_short} -{spread_num}</b></span>
                    <span>Proj Total: <b>{sim_res['proj_total']}</b></span>
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
    detail_sim = simulate_full_game(df, away_sel, home_sel, def_ratings, n_sims=10000)

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
        st.info("Auto-parlays will populate automatically as soon as tonight's player props open on the sportsbook feed.")
    else:
        col_multi, col_sgp = st.columns(2, gap="large")

        with col_multi:
            st.markdown("##### Cross-Game +EV Parlay")
            st.caption("Combines the highest independent model edge from each matchup on tonight's slate.")
            num_parlay_legs = st.radio("Parlay Size", [2, 3, 4], horizontal=True)
            if not edges_df.empty:
                best_per_game = edges_df.drop_duplicates(subset=["Matchup"]).head(num_parlay_legs)
                if len(best_per_game) >= 2:
                    comb_dec = float(np.prod([american_to_decimal(o) for o in best_per_game["Odds"]]))
                    comb_mod_p = float(np.prod(best_per_game["Win Prob"] / 100.0))
                    parlay_ev = (comb_mod_p * comb_dec - 1) * 100
                    parlay_odds = round((comb_dec - 1) * 100) if comb_dec >= 2.0 else round(-100 / (comb_dec - 1))
                    parlay_units = quarter_kelly_units(comb_mod_p, comb_dec)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Combined Odds", f"{parlay_odds:+d}")
                    m2.metric("Parlay EV", f"{parlay_ev:+.1f}%")
                    m3.metric("Rec. Size", f"{parlay_units:.2f}u")

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
                    st.warning("At least 2 active games with props are needed to build a cross-game parlay.")

        with col_sgp:
            st.markdown("##### Correlated Same-Game Parlays")
            st.caption("Pairs live lines tonight with strong historical game-log correlations.")
            sgp_game = st.selectbox("Select Matchup", list(slate_props.keys()))
            sgp_df = build_auto_sgps(df, sgp_game, slate_props[sgp_game])
            if sgp_df.empty:
                st.warning("No high-correlation live prop pairs found for this matchup yet.")
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