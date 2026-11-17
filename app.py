import streamlit as st
from engine import load_nba_data, price_sgp, scan_matchup_correlations

st.set_page_config(page_title="SGP Fair-Odds Engine", layout="wide")
st.title("🏀 NBA SGP Correlation & Fair-Odds Pricer")

@st.cache_data
def get_data():
    return load_nba_data()

with st.spinner("Loading multi-season NBA box scores..."):
    df = get_data()

teams = sorted(df["TEAM_NAME"].dropna().unique())
c1, c2 = st.columns(2)
with c1:
    team_a = st.selectbox("Team A", teams, index=teams.index("Denver Nuggets") if "Denver Nuggets" in teams else 0)
with c2:
    team_b = st.selectbox("Team B (Opponent)", teams, index=teams.index("Phoenix Suns") if "Phoenix Suns" in teams else 1)

tab_pricer, tab_scanner = st.tabs(["🎯 Fair-Odds & EV Pricer", "🔍 Top Correlation Scanner"])

with tab_pricer:
    matchup_players = sorted(df[df["TEAM_NAME"].isin([team_a, team_b])]["PLAYER_NAME"].dropna().unique())
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
            o = st.number_input("Selected Side Odds", value=-115, step=5, key=f"o{i}")
            opp_o = st.number_input("Other Side Odds (0 = auto-est)", value=-105, step=5, key=f"opp{i}")
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