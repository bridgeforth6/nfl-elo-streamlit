import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import nfl_data_py as nfl
import openpyxl  # noqa: F401  (needed by pandas Excel writer)

# -----------------------------
# Streamlit app configuration
# -----------------------------
st.set_page_config(page_title="NFL Elo Rating & Win Prob Forecaster", layout="wide")
st.title("NFL Elo Rating and Win Probability Forecaster")

# -----------------------------
# Sidebar controls
# -----------------------------
initial_elo = st.sidebar.number_input(
    "Initial Elo Rating", value=1500, min_value=1000, max_value=2000, step=50
)
reversion_factor = st.sidebar.slider(
    "Reversion Factor at End of Season", min_value=0.0, max_value=1.0, value=0.33, step=0.01
)
k_value = st.sidebar.number_input("K Value for Elo Update", value=20, min_value=1, max_value=100, step=1)
home_advantage = st.sidebar.number_input("Home Advantage Elo Boost", value=48, min_value=0, max_value=100, step=1)

seed = st.sidebar.number_input("Random Seed (for reproducibility)", value=42, min_value=0, max_value=10_000, step=1)
random.seed(int(seed))
np.random.seed(int(seed))

# Allow user to choose first season to include
first_season = st.sidebar.number_input("First Season to Include", value=2010, min_value=1970, max_value=datetime.now().year)

# -----------------------------
# Helper: get seasons to fetch
# -----------------------------
def get_available_seasons(start_year: int) -> list[int]:
    """
    Build a season list from `start_year` through the *current* year.
    After fetching schedules, we also clamp to the actual seasons present in the data,
    in case the current-season schedule isn't published yet by nfl_data_py.
    """
    this_year = datetime.now().year
    candidate_years = list(range(start_year, this_year + 1))
    # Fetch schedules to confirm what's really available
    sched = nfl.import_schedules(candidate_years)
    if "season" not in sched.columns or sched.empty:
        return []
    return sorted(sched["season"].unique().tolist())

# -----------------------------
# Main run
# -----------------------------
if st.button("Run Calculations"):
    years = get_available_seasons(first_season)
    if not years:
        st.error("No schedule data available. Try lowering the first season or check your nfl_data_py installation.")
        st.stop()

    # Fetch confirmed seasons
    schedule_data = nfl.import_schedules(years).copy()

    # Normalize some expected columns (older years can be quirky)
    expected_cols = [
        "season", "week", "game_type", "home_team", "away_team",
        "home_score", "away_score", "div_game", "away_rest",
        "home_qb_name", "away_qb_name"
    ]
    missing = [c for c in expected_cols if c not in schedule_data.columns]
    if missing:
        st.warning(f"Some expected columns are missing in schedule data: {missing}")

    # Show what we actually have (collapsed by default)
    with st.expander("Available columns in schedule data"):
        st.write(sorted(schedule_data.columns.tolist()))

    # Initialize Elo ratings
    elo_ratings: dict[str, float] = {}
    teams = pd.concat([schedule_data["home_team"], schedule_data["away_team"]]).dropna().unique()
    for t in teams:
        elo_ratings[t] = float(initial_elo)

    # --- Uncertainty helpers ---
    def add_uncertainty_to_win_prob(win_prob: float) -> float:
        noise = random.uniform(-0.05, 0.05)  # ±5%
        return min(max(win_prob + noise, 0.0), 1.0)

    def add_uncertainty_to_home_advantage() -> float:
        return float(home_advantage) + random.uniform(-10, 10)

    # --- Elo math helpers ---
    def update_elo(team_elo: float, opp_elo: float, result: float, point_diff: float, k: float = float(k_value)) -> float:
        expected_score = 1.0 / (1.0 + 10.0 ** ((opp_elo - team_elo) / 400.0))
        # MOV multiplier from FiveThirtyEight-style approach
        elo_diff = abs(team_elo - opp_elo)
        mov_multiplier = (math.log(point_diff + 1.0) * 2.2) / ((elo_diff * 0.001) + 2.2)
        return team_elo + k * mov_multiplier * (result - expected_score)

    def calculate_win_probability(home_elo: float, away_elo: float, is_playoffs: bool = False) -> float:
        adjusted_home = add_uncertainty_to_home_advantage()
        elo_diff = home_elo + adjusted_home - away_elo
        if is_playoffs:
            elo_diff *= 1.2  # weight playoffs slightly higher
        win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        return add_uncertainty_to_win_prob(round(float(win_prob), 2))

    # -----------------------------
    # Process each season
    # -----------------------------
    all_matchup_rows: list[dict] = []

    for year in years:
        yearly = schedule_data[schedule_data["season"] == year].sort_values(["week"]).copy()
        matchup_rows: list[dict] = []

        for _, game in yearly.iterrows():
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            home_points = game.get("home_score")
            away_points = game.get("away_score")
            div_game = game.get("div_game", False)
            away_rest = game.get("away_rest", np.nan)
            home_qb_name = game.get("home_qb_name", None)
            away_qb_name = game.get("away_qb_name", None)
            game_type = str(game.get("game_type", "")).upper()

            # Results (1=home win, 0.5=tie, 0=away win) only if both scores present
            have_scores = (pd.notna(home_points) and pd.notna(away_points))
            if have_scores:
                if home_points > away_points:
                    home_result = 1.0
                elif home_points < away_points:
                    home_result = 0.0
                else:
                    home_result = 0.5
                point_diff = abs(float(home_points) - float(away_points))
            else:
                home_result = None
                point_diff = 0.0

            home_elo = float(elo_ratings.get(home_team, initial_elo))
            away_elo = float(elo_ratings.get(away_team, initial_elo))

            is_playoffs = (game_type == "POST")
            win_probability = calculate_win_probability(home_elo, away_elo, is_playoffs=is_playoffs)

            # Save matchup row
            row = {
                "Season": year,
                "Week": game.get("week"),
                "Game Type": game_type,
                "Home Team": home_team,
                "Away Team": away_team,
                "Home Score": home_points,
                "Away Score": away_points,
                "Home Win Probability": f"{int(win_probability * 100)}%",
                "Elo Difference": round(home_elo - away_elo),
                "Divisional Game": div_game,
                "Away Rest": away_rest,
                "Home QB": home_qb_name,
                "Away QB": away_qb_name,
                "Result": (
                    "Home Win" if (home_result == 1.0) else
                    ("Away Win" if (home_result == 0.0) else ("Tie" if (home_result == 0.5) else "Upcoming"))
                ),
            }
            matchup_rows.append(row)

            # Update Elo if we have a finished game
            if have_scores:
                elo_ratings[home_team] = update_elo(home_elo, away_elo, home_result, point_diff)
                elo_ratings[away_team] = update_elo(away_elo, home_elo, 1.0 - home_result, point_diff)

        # Revert to mean at end of season
        if elo_ratings:
            elo_mean = sum(elo_ratings.values()) / len(elo_ratings)
            for t in elo_ratings:
                elo_ratings[t] = elo_ratings[t] - ((elo_ratings[t] - elo_mean) * float(reversion_factor))

        # Persist season outputs (optional but kept from your original)
        season_df = pd.DataFrame({
            "Team": list(elo_ratings.keys()),
            "Elo Rating": [round(r) for r in elo_ratings.values()],
            "Season": year
        })
        try:
            season_df.to_excel(f"elo_ratings_{year}.xlsx", index=False, engine="openpyxl")
        except Exception as e:
            st.warning(f"Could not write elo_ratings_{year}.xlsx: {e}")

        matchup_df = pd.DataFrame(matchup_rows)
        matchup_df.index += 1
        try:
            matchup_df.to_excel(f"matchups_{year}.xlsx", index=False, engine="openpyxl")
        except Exception as e:
            st.warning(f"Could not write matchups_{year}.xlsx: {e}")

        all_matchup_rows.extend(matchup_rows)

    # -----------------------------
    # Display current Elo ratings
    # -----------------------------
    sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    elo_df = pd.DataFrame(sorted_elo, columns=["Team", "Elo Rating"])
    # Filter franchise codes that no longer exist in schedules
    inactive_codes = {"STL", "OAK", "SD"}  # can expand if you see others
    elo_df = elo_df[~elo_df["Team"].isin(inactive_codes)].copy()
    elo_df["Elo Rating"] = elo_df["Elo Rating"].round()
    elo_df = elo_df.reset_index(drop=True)
    elo_df.index += 1

    st.subheader("Current Elo Ratings")
    st.dataframe(elo_df, use_container_width=True)

    # -----------------------------
    # Display current season matchups (played + upcoming)
    # -----------------------------
    current_season = int(schedule_data["season"].max())
    this_season = schedule_data[schedule_data["season"] == current_season].copy()

    # latest completed week if any
    played_mask = (this_season["home_score"].notna() & this_season["away_score"].notna())
    latest_week_played = this_season.loc[played_mask, "week"].max()

    # If no games have been played yet (e.g., preseason done but Week 1 pending),
    # show Week 1; otherwise show up to next week after the last completed one.
    if pd.isna(latest_week_played):
        show_up_to_week = this_season["week"].min()
    else:
        show_up_to_week = int(latest_week_played) + 1

    to_show = this_season[this_season["week"] <= show_up_to_week].copy().sort_values(["week"])

    current_rows = []
    for _, game in to_show.iterrows():
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        home_points = game.get("home_score")
        away_points = game.get("away_score")
        div_game = game.get("div_game", False)
        away_rest = game.get("away_rest", np.nan)
        home_qb_name = game.get("home_qb_name", None)
        away_qb_name = game.get("away_qb_name", None)
        game_type = str(game.get("game_type", "")).upper()

        home_elo = float(elo_ratings.get(home_team, initial_elo))
        away_elo = float(elo_ratings.get(away_team, initial_elo))
        is_playoffs = (game_type == "POST")
        win_probability = calculate_win_probability(home_elo, away_elo, is_playoffs=is_playoffs)

        have_scores = (pd.notna(home_points) and pd.notna(away_points))
        if have_scores:
            if home_points > away_points:
                result = "Home Win"
            elif home_points < away_points:
                result = "Away Win"
            else:
                result = "Tie"
        else:
            result = "Upcoming"

        current_rows.append({
            "Season": current_season,
            "Week": game.get("week"),
            "Game Type": game_type,
            "Home Team": home_team,
            "Away Team": away_team,
            "Home Win Probability": f"{int(win_probability * 100)}%",
            "Elo Difference": round(home_elo - away_elo),
            "Divisional Game": div_game,
            "Away Rest": away_rest,
            "Home QB": home_qb_name,
            "Away QB": away_qb_name,
            "Result": result
        })

    current_matchup_df = pd.DataFrame(current_rows)
    current_matchup_df.index += 1

    st.subheader(f"Current Season ({current_season}) Matchups & Win Probabilities")
    st.dataframe(
        current_matchup_df[["Week", "Game Type", "Home Team", "Away Team", "Home Win Probability", "Result"]],
        use_container_width=True
    )

    st.caption(
        "Tip: If you still see last year’s season at the top, try clearing the cache or restarting the app. "
        "This version automatically targets the most recent season available in nfl_data_py."
    )
