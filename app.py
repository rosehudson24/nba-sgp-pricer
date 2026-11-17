import streamlit as st
from engine import (
    load_nba_data,
    price_sgp,
    scan_matchup_correlations,
    fetch_tonights_nba_games,
    fetch_game_player_props,
)

st.set_page_config(page_title="SGP Fair-Odds Engine", layout="wide")
st.title("🏀 NBA SGP Correlation & Fair-Odds Pricer")

@st.cache_data
def get_data():
    return load_nba_data()

@st.cache_data(ttl=600)
def cached_games(api_key: str):
    return fetch_tonights_nba_games(api_key)

@st.cache_data(ttl=300)
def cached_props(api_key: str, event_id: str, bookmaker: str):
    return fetch_game_player_props(api_key, event_id, bookmaker)

with st.spinner("Loading multi-season NBA box scores..."):
    df = get_data()

# Auto-load API key from Streamlit Cloud Secrets if configured
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
        games = cached_games(api_key)
        if games:
            game_labels = {f"{g['away_team']} @ {g['home_team']}": g for g in games}
            selected_label = st.sidebar.selectbox("Tonight's NBA Slate", list(game_labels.keys()))
            selected_game = game_labels[selected_label]
            default_team_a = selected_game["away_team"]
            default_team_b = selected_game["home_team"]

            if st.sidebar.button("Pull Live Player Props", type="primary"):
                st.session_state["live_props"] = cached_props(api_key, selected_game["id"], bookmaker)
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