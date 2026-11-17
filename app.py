import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="SGP Fair-Odds Engine", layout="wide")
st.title("🏀 NBA SGP Correlation & Fair-Odds Pricer")

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
    df = df.drop_duplicates(subset=["game_id", "athlete_display_name"])
    return df.rename(columns={
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
        "player_steals": "STL",
        "player_blocks": "BLK"
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
                if not player or side not in ("Over", "Under"):
                    continue
                props.setdefault(player, {}).setdefault(stat_code, {"line": line})
                props[player][stat_code][side] = int(price)
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

def scan_matchup_correlations(df: pd.DataFrame, team_a: str, team_b: str, min_games: int = 20):
    sub = df[df["TEAM_NAME"].isin([team_a, team_b])].copy()
    counts = sub["PLAYER_NAME"].value_counts()
    active_players = counts[counts >= min_games].index
    sub = sub[sub["PLAYER_NAME"].isin(active_players)]

    stats = ["PTS", "REB", "AST", "FG3M"]
    pivoted = sub.pivot(index="GAME_ID", columns="PLAYER_NAME", values=stats)
    pivoted.columns = [f"{player} ({stat})" for stat, player in pivoted.columns]

    corr = pivoted.corr(min_periods=min_games)
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).unstack().dropna().reset_index()
    pairs.columns = ["Leg B", "Leg A", "Correlation"]
    pairs = pairs[["Leg A", "Leg B", "Correlation"]]

    pairs = pairs[pairs["Leg A"].str.split(" \\(").str[0] != pairs["Leg B"].str.split(" \\(").str[0]]
    pairs["Correlation"] = pairs["Correlation"].round(3)

    top_pos = pairs.sort_values(by="Correlation", ascending=False).head(15).reset_index(drop=True)
    top_neg = pairs.sort_values(by="Correlation", ascending=True).head(15).reset_index(drop=True)
    return top_pos, top_neg

with st.spinner("Loading multi-season NBA box scores..."):
    df = load_nba_data()

saved_key = ""
try:
    saved_key = st.secrets.get("ODDS_API_KEY", "")
except Exception:
    saved_key = ""

st.sidebar.header("📡 Live Sportsbook Lines")
if saved_key:
    st.sidebar.success("API Key connected via Streamlit Secrets")
    api_key = saved_key
else:
    api_key = st.sidebar.text_input("The Odds API Key", type="password", help="Free key at the-odds-api.com")

bookmaker = st.sidebar.selectbox("Sportsbook", ["draftkings", "fanduel", "betmgm", "caesars"])

live_props = {}
default_team_a, default_team_b = "Denver Nuggets", "Phoenix Suns"

if api_key:
    try:
        games = fetch_tonights_nba_games(api_key)
        if games:
            game_labels = {f"{g['away_team']} @ {g['home_team']}": g for g in games}
            selected_label = st.sidebar.selectbox("Tonight's NBA Slate", list(game_labels.keys()))
            selected_game = game_labels[selected_label]
            default_team_a = selected_game["away_team"]
            default_team_b = selected_game["home_team"]

            if st.sidebar.button("Pull Live Player Props", type="primary"):
                st.session_state["live_props"] = fetch_game_player_props(api_key, selected_game["id"], bookmaker)
                st.sidebar.success(f"Loaded props for {len(st.session_state['live_props'])} players!")
            live_props = st.session_state.get("live_props", {})
        else:
            st.sidebar.info("No upcoming NBA games posted right now.")
    except Exception as e:
        st.sidebar.error(f"API Error: {e}")

teams = sorted(df["TEAM_NAME"].dropna().unique())
c1, c2 = st.columns(2)
with c1:
    idx_a = teams.index(default_team_a) if default_team_a in teams else 0
    team_a = st.selectbox("Team A", teams, index=idx_a)
with c2:
    idx_b = teams.index(default_team_b) if default_team_b in teams else 1
    team_b = st.selectbox("Team B (Opponent)", teams, index=idx_b)

tab_pricer, tab_scanner = st.tabs(["🎯 Fair-Odds & EV Pricer", "🔍 Top Correlation Scanner"])

with tab_pricer:
    matchup_players = sorted(df[df["TEAM_NAME"].isin([team_a, team_b])]["PLAYER_NAME"].dropna().unique())
    if live_props:
        active_book_players = [p for p in matchup_players if p in live_props]
        if active_book_players:
            matchup_players = active_book_players

    stats = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]
    num_legs = st.slider("Number of SGP Legs", min_value=2, max_value=5, value=3)
    legs = []

    cols = st.columns(num_legs)
    for i in range(num_legs):
        with cols[i]:
            st.subheader(f"Leg {i+1}")
            p = st.selectbox("Player", matchup_players, index=min(i, len(matchup_players)-1), key=f"p{i}")
            side = st.radio("Side", ["Over", "Under"], horizontal=True, key=f"side{i}")
            s = st.selectbox("Stat", stats, index=2 if i == 0 else 0, key=f"s{i}")

            prop_info = live_props.get(p, {}).get(s, {})
            if prop_info:
                st.info(f"Live {bookmaker.title()} Line: **{prop_info.get('line')}**")
                default_sel = prop_info.get(side, -115)
                opp_side = "Under" if side == "Over" else "Over"
                default_opp = prop_info.get(opp_side, -105)
            else:
                default_sel, default_opp = -115, -105

            o = st.number_input("Selected Side Odds", value=int(default_sel), step=5, key=f"o{i}")
            opp_o = st.number_input("Other Side Odds (0 = auto-est)", value=int(default_opp), step=5, key=f"opp{i}")
            legs.append({"player": p, "side": side, "stat": s, "odds": int(o), "opp_odds": int(opp_o)})

    st.divider()
    book_odds = st.number_input("Sportsbook Offered SGP Odds (e.g. +550)", value=550, step=10)

    if st.button("Calculate Fair Odds & Edge", type="primary"):
        res = price_sgp(df, legs, int(book_odds))
        if "error" in res:
            st.error(res["error"])
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Shared Games Sample", res["shared_games"])
            m2.metric("Naive Independent Odds", f"{res['naive_odds']:+d}")
            m3.metric("TRUE Fair Odds", f"{res['fair_odds']:+d}")
            m4.metric("Expected Value (EV)", f"{res['ev_pct']:+.2f}%")

            st.caption(f"True No-Vig Win Probabilities per Leg: {', '.join(f'{p}%' for p in res['fair_leg_probs'])}")

            if res["ev_pct"] > 0:
                st.success(f"+EV PLAY DETECTED: Sportsbook is offering {book_odds:+d} vs. fair price of {res['fair_odds']:+d} ({res['ev_pct']:+.2f}% edge).")
            else:
                st.warning(f"NEGATIVE EV: Sportsbook correlation tax is costing you {abs(res['ev_pct']):.2f}% in edge.")

            st.subheader("Historical Stat Correlation Matrix")
            st.dataframe(res["corr_df"], use_container_width=True)

with tab_scanner:
    st.subheader(f"Strongest Historical Correlations: {team_a} vs. {team_b}")
    min_g = st.slider("Minimum Shared Games Sample", min_value=10, max_value=50, value=20)
    if st.button("Scan Matchup Pairs", type="primary"):
        pos_df, neg_df = scan_matchup_correlations(df, team_a, team_b, min_games=min_g)
        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("**📈 Top Positive Stacks (Pair Over + Over or Under + Under)**")
            st.dataframe(pos_df, use_container_width=True)
        with col_neg:
            st.markdown("**📉 Top Negative Fades (Pair Over + Under)**")
            st.dataframe(neg_df, use_container_width=True)