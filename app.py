import re
import unicodedata
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="EdgeLab NBA | Model & SGP Engine", layout="wide")
st.title("🏀 EdgeLab NBA — Model Projections, Straight Bets & Auto-Parlays")

def normalize_name(name: str) -> str:
    """Normalizes player names so sportsbook props match ESPN box scores cleanly."""
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
    df = df[df["minutes"] >= 15].copy()
    df["game_id"] = df["game_id"].astype(str)
    if "game_date" in df.columns:
        df = df.sort_values("game_date")
    df = df.drop_duplicates(subset=["game_id", "athlete_display_name"])
    df = df.rename(columns={
        "game_id": "GAME_ID",
        "athlete_display_name": "PLAYER_NAME",
        "team_display_name": "TEAM_NAME",
        "points": "PTS",
        "rebounds": "REB",
        "assists": "AST",
        "three_point_field_goals_made": "FG3M",
        "steals": "STL",
        "blocks": "BLK",
        "turnovers": "TOV"
    })
    df["NORM_NAME"] = df["PLAYER_NAME"].apply(normalize_name)
    return df

@st.cache_data(ttl=600)
def fetch_tonights_nba_games(api_key: str) -> list:
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    res = requests.get(url, params={"apiKey": api_key}, timeout=10)
    res.raise_for_status()
    return res.json()

@st.cache_data(ttl=300)
def fetch_game_player_props(api_key: str, event_id: str, bookmaker: str = "draftkings") -> dict:
    market_map = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_threes": "FG3M",
    }
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": ",".join(market_map.keys()),
        "oddsFormat": "american",
        "bookmakers": bookmaker
    }
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    data = res.json()

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
    return props

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

def evaluate_straight_edges(df: pd.DataFrame, slate_props: dict) -> pd.DataFrame:
    """Compares weighted historical player projections + empirical hit rates vs. live book lines."""
    rows = []
    grouped = {k: v for k, v in df.groupby("NORM_NAME")}

    for game_label, game_props in slate_props.items():
        for norm_p, p_data in game_props.items():
            if norm_p not in grouped:
                continue
            p_df = grouped[norm_p]
            if len(p_df) < 12:
                continue

            for stat, info in p_data["stats"].items():
                if stat not in p_df.columns or "Over" not in info or "Under" not in info:
                    continue
                line = info["line"]
                series = p_df[stat].dropna()
                if len(series) < 12:
                    continue

                # Weighted projection: 60% Last 10 games + 40% Full Sample
                recent = series.tail(10)
                proj = 0.60 * recent.mean() + 0.40 * series.mean()
                std = max(series.std(), 1.25)

                # Blend empirical hit rate (Last 15) with Normal distribution probability
                emp_over = (series.tail(15) > line).mean()
                norm_over = 1 - norm.cdf(line, loc=proj, scale=std)
                model_over = 0.50 * emp_over + 0.50 * norm_over
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
                    rows.append({
                        "Game": game_label,
                        "Player": p_data["display_name"],
                        "NORM_NAME": norm_p,
                        "Pick": f"{side} {line} {stat}",
                        "Side": side,
                        "Stat": stat,
                        "Line": line,
                        "Model Proj": round(proj, 1),
                        "Book Odds": odds,
                        "Model Win %": round(mod_p * 100, 1),
                        "No-Vig %": round(nv_p * 100, 1),
                        "Edge %": round(edge_pct, 1),
                        "EV %": round(ev_pct, 1),
                    })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="EV %", ascending=False).reset_index(drop=True)

def build_auto_sgps(df: pd.DataFrame, game_label: str, game_props: dict, min_games: int = 15) -> pd.DataFrame:
    """Finds top correlated pairs in a single game where both players have live lines tonight."""
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
            # Positive stack: Over + Over
            p_a = get_no_vig_prob(st1["Over"], st1.get("Under", 0))
            p_b = get_no_vig_prob(st2["Over"], st2.get("Under", 0))
            z = np.random.multivariate_normal([0, 0], [[1, c_val], [c_val, 1]], size=40000)
            u = norm.cdf(z)
            joint_p = float(np.mean((u[:, 0] < p_a) & (u[:, 1] < p_b)))
            sgp_rows.append({
                "Game": game_label,
                "Type": "📈 Positive Stack",
                "Leg 1": f"{game_props[p1]['display_name']} Over {st1['line']} {s1} ({st1['Over']:+d})",
                "Leg 2": f"{game_props[p2]['display_name']} Over {st2['line']} {s2} ({st2['Over']:+d})",
                "Correlation": round(c_val, 3),
                "Naive Odds": f"{prob_to_american(p_a * p_b):+d}",
                "True Fair Odds": f"{prob_to_american(joint_p):+d}",
            })
        elif c_val <= -0.22:
            # Negative fade: Leg 1 Over + Leg 2 Under
            p_a = get_no_vig_prob(st1["Over"], st1.get("Under", 0))
            p_b = get_no_vig_prob(st2["Under"], st2.get("Over", 0))
            eff_corr = abs(c_val)
            z = np.random.multivariate_normal([0, 0], [[1, eff_corr], [eff_corr, 1]], size=40000)
            u = norm.cdf(z)
            joint_p = float(np.mean((u[:, 0] < p_a) & (u[:, 1] < p_b)))
            sgp_rows.append({
                "Game": game_label,
                "Type": "📉 Negative Fade",
                "Leg 1": f"{game_props[p1]['display_name']} Over {st1['line']} {s1} ({st1['Over']:+d})",
                "Leg 2": f"{game_props[p2]['display_name']} Under {st2['line']} {s2} ({st2['Under']:+d})",
                "Correlation": round(c_val, 3),
                "Naive Odds": f"{prob_to_american(p_a * p_b):+d}",
                "True Fair Odds": f"{prob_to_american(joint_p):+d}",
            })

    if not sgp_rows:
        return pd.DataFrame()
    return pd.DataFrame(sgp_rows).reindex(
        pd.DataFrame(sgp_rows)["Correlation"].abs().sort_values(ascending=False).index
    ).head(15).reset_index(drop=True)

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

    corr_matrix = merged[col_names].corr().values
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

with st.spinner("Loading multi-season NBA box scores..."):
    df = load_nba_data()

# Sidebar: Slate Scanner Controls
saved_key = ""
try:
    saved_key = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    saved_key = ""

st.sidebar.header("📡 Live Slate & Model Feed")
api_key = saved_key if saved_key else st.sidebar.text_input("The Odds API Key", type="password")
bookmaker = st.sidebar.selectbox("Sportsbook", ["draftkings", "fanduel", "betmgm", "caesars"])

if api_key:
    try:
        games = fetch_tonights_nba_games(api_key)
        if games:
            game_labels = {f"{g['away_team']} @ {g['home_team']}": g for g in games}
            selected_games = st.sidebar.multiselect(
                "Games to Scan (Pick 1–4 to save API credits)",
                list(game_labels.keys()),
                default=list(game_labels.keys())[:2]
            )
            if st.sidebar.button("🚀 Scan Live Props & Build Picks", type="primary"):
                slate_data = {}
                with st.spinner("Pulling live sportsbook lines & running projection model..."):
                    for lbl in selected_games:
                        g_obj = game_labels[lbl]
                        slate_data[lbl] = fetch_game_player_props(api_key, g_obj["id"], bookmaker)
                st.session_state["slate_props"] = slate_data
                st.sidebar.success(f"Scanned {len(slate_data)} game(s)!")
        else:
            st.sidebar.info("No upcoming NBA games posted right now.")
    except Exception as e:
        st.sidebar.error(f"API Error: {e}")

slate_props = st.session_state.get("slate_props", {})

tab_straight, tab_parlays, tab_custom = st.tabs([
    "🔥 Best Straight Bets (Model)",
    "⚡ Auto-Parlay Generator (Multi-Game & SGP)",
    "🎯 Custom SGP Pricer"
])

# TAB 1: BEST STRAIGHT BETS
with tab_straight:
    st.subheader("Top Model Projections vs. Sportsbook Lines")
    if not slate_props:
        st.info("Select games in the left sidebar and click **🚀 Scan Live Props & Build Picks** to generate tonight's straight bets.")
    else:
        edges_df = evaluate_straight_edges(df, slate_props)
        if edges_df.empty:
            st.warning("No matching player props found for the selected games yet.")
        else:
            min_ev = st.slider("Minimum Model EV % Filter", min_value=-5.0, max_value=15.0, value=2.0, step=0.5)
            filtered = edges_df[edges_df["EV %"] >= min_ev].drop(columns=["NORM_NAME", "Side", "Stat", "Line"])
            st.dataframe(filtered, use_container_width=True, hide_index=True)

# TAB 2: AUTO-PARLAY GENERATOR (CROSS-GAME + SGP)
with tab_parlays:
    if not slate_props:
        st.info("Click **🚀 Scan Live Props & Build Picks** in the sidebar first to generate automatic parlays.")
    else:
        col_multi, col_sgp = st.columns(2)

        with col_multi:
            st.subheader("🌐 Auto Multi-Game +EV Parlay")
            st.caption("Combines your top independent model edges across different games (1 leg per game max).")
            num_parlay_legs = st.radio("Parlay Size", [2, 3, 4], horizontal=True)
            edges_df = evaluate_straight_edges(df, slate_props)
            if not edges_df.empty:
                # Pick top edge per distinct game so legs are truly independent
                best_per_game = edges_df.drop_duplicates(subset=["Game"]).head(num_parlay_legs)
                if len(best_per_game) >= 2:
                    comb_dec = float(np.prod([american_to_decimal(o) for o in best_per_game["Book Odds"]]))
                    comb_mod_p = float(np.prod(best_per_game["Model Win %"] / 100.0))
                    parlay_ev = (comb_mod_p * comb_dec - 1) * 100
                    parlay_odds = round((comb_dec - 1) * 100) if comb_dec >= 2.0 else round(-100 / (comb_dec - 1))

                    m1, m2 = st.columns(2)
                    m1.metric("Combined Sportsbook Odds", f"{parlay_odds:+d}")
                    m2.metric("Projected Parlay EV", f"{parlay_ev:+.1f}%")

                    for idx, row in best_per_game.iterrows():
                        st.success(f"**{row['Player']} — {row['Pick']}** ({row['Book Odds']:+d}) | Model Proj: **{row['Model Proj']}** | EV: **{row['EV %']:+.1f}%** *({row['Game']})*")
                else:
                    st.warning("Select at least 2 games in the sidebar to build a multi-game parlay.")

        with col_sgp:
            st.subheader("🔗 Auto Correlated Same-Game Parlays")
            st.caption("Pairs live props tonight with strong historical correlations (boosted true hit rate).")
            sgp_game = st.selectbox("Select Game for Auto-SGP", list(slate_props.keys()))
            sgp_df = build_auto_sgps(df, sgp_game, slate_props[sgp_game])
            if sgp_df.empty:
                st.warning("No high-correlation live prop pairs found for this game yet.")
            else:
                st.dataframe(sgp_df.drop(columns=["Game"]), use_container_width=True, hide_index=True)

# TAB 3: CUSTOM SGP PRICER
with tab_custom:
    st.subheader("Manual Multi-Leg SGP Correlation Pricer")
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
    if st.button("Calculate Fair Odds & Edge", type="primary"):
        res = price_sgp(df, legs, int(book_odds))
        if "error" in res:
            st.error(res["error"])
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Shared Games", res["shared_games"])
            m2.metric("Naive Odds", f"{res['naive_odds']:+d}")
            m3.metric("TRUE Fair Odds", f"{res['fair_odds']:+d}")
            m4.metric("Expected Value (EV)", f"{res['ev_pct']:+.2f}%")
            st.dataframe(res["corr_df"], use_container_width=True)