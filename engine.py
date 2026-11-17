import numpy as np
import pandas as pd
from scipy.stats import norm

def load_nba_data(seasons=(2024, 2025, 2026)):
    """Loads multi-season NBA player box scores from sportsdataverse."""
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

def american_to_prob(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds: int) -> float:
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

def prob_to_american(p: float) -> int:
    p = np.clip(p, 0.0001, 0.9999)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)

def get_no_vig_prob(odds_selected: int, odds_opposing: int = 0) -> float:
    """Calculates exact 2-way no-vig probability if opposing odds are given, else estimates 4.5% hold."""
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
    """Finds top positive and negative player-stat correlations across two teams."""
    sub = df[df["TEAM_NAME"].isin([team_a, team_b])].copy()
    # Keep active rotation players (at least min_games played)
    counts = sub["PLAYER_NAME"].value_counts()
    active_players = counts[counts >= min_games].index
    sub = sub[sub["PLAYER_NAME"].isin(active_players)]

    stats = ["PTS", "REB", "AST", "FG3M"]
    pivoted = sub.pivot(index="GAME_ID", columns="PLAYER_NAME", values=stats)
    pivoted.columns = [f"{player} ({stat})" for stat, player in pivoted.columns]

    corr = pivoted.corr(min_periods=min_games)
    # Keep upper triangle only to avoid duplicates and self-correlations
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).unstack().dropna().reset_index()
    pairs.columns = ["Leg B", "Leg A", "Correlation"]
    pairs = pairs[["Leg A", "Leg B", "Correlation"]]

    # Exclude same-player stat pairs (e.g., Jokic PTS + Jokic FG3M)
    pairs = pairs[pairs["Leg A"].str.split(" \\(").str[0] != pairs["Leg B"].str.split(" \\(").str[0]]
    pairs["Correlation"] = pairs["Correlation"].round(3)

    top_pos = pairs.sort_values(by="Correlation", ascending=False).head(15).reset_index(drop=True)
    top_neg = pairs.sort_values(by="Correlation", ascending=True).head(15).reset_index(drop=True)
    return top_pos, top_neg