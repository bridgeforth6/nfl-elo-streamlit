import pandas as pd
import nfl_data_py as nfl
import openpyxl
import numpy as np
import math
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report
import statsmodels.api as sm

# Set years for analysis
years = list(range(2010, 2025))

# Streamlit app configuration
st.title('NFL Elo Rating and Win Probability Forecaster')

# User inputs for model parameters
initial_elo = st.sidebar.number_input('Initial Elo Rating', value=1500, min_value=1000, max_value=2000, step=50)
reversion_factor = st.sidebar.slider('Reversion Factor at End of Season', min_value=0.0, max_value=1.0, value=0.33, step=0.01)
k_value = st.sidebar.number_input('K Value for Elo Update', value=20, min_value=1, max_value=100, step=1)
home_advantage = st.sidebar.number_input('Home Advantage Elo Boost', value=48, min_value=0, max_value=100, step=1)

# Initialize Elo ratings for each team at the start
elo_ratings = {}

# Fetch schedule data for the specific years
schedule_data = nfl.import_schedules(years)

# Print all available columns in the dataset
print("Available columns in schedule data:", schedule_data.columns)

# Initialize Elo ratings for all teams
teams = pd.concat([schedule_data['home_team'], schedule_data['away_team']]).unique()
for team in teams:
    elo_ratings[team] = initial_elo

# Remove inactive teams from Elo ratings
inactive_teams = ['STL', 'OAK', 'SD']
for team in inactive_teams:
    if team in elo_ratings:
        del elo_ratings[team]

# Function to update Elo ratings based on match result and margin of victory multiplier
def update_elo(team_elo, opponent_elo, result, point_diff, k=k_value):
    expected_score = 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))
    elo_diff = abs(team_elo - opponent_elo)
    mov_multiplier = (math.log(point_diff + 1) * 2.2) / ((elo_diff * 0.001) + 2.2)
    new_elo = team_elo + k * mov_multiplier * (result - expected_score)
    return new_elo

# Function to calculate win probability
def calculate_win_probability(home_elo, away_elo, is_playoffs=False):
    elo_diff = home_elo + home_advantage - away_elo  # Add home_advantage Elo points to home team for home-field advantage
    if is_playoffs:
        elo_diff *= 1.2  # Increase weight for playoff games
    win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
    return round(win_prob, 2)

# Iterate over each season to calculate Elo ratings and win probabilities
all_matchup_data = []
for year in years:
    # Filter data for the current year
    yearly_data = schedule_data[schedule_data['season'] == year]

    # Create a list to store matchup data for the season
    matchup_data = []

    # Iterate over each game in the season
    for _, game in yearly_data.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        home_points = game['home_score']
        away_points = game['away_score']
        div_game = game['div_game']
        away_rest = game['away_rest']
        home_qb_name = game['home_qb_name']
        away_qb_name = game['away_qb_name']

        # Determine result: 1 if home team wins, 0.5 if tie, 0 if away team wins
        if home_points > away_points:
            home_result = 1
        elif home_points < away_points:
            home_result = 0
        else:
            home_result = 0.5

        # Update Elo ratings for both teams
        home_elo = elo_ratings.get(home_team, initial_elo)
        away_elo = elo_ratings.get(away_team, initial_elo)
        point_diff = abs(home_points - away_points)

        # Check if the game is a playoff game (week 19 or later)
        is_playoffs = game['week'] >= 19

        # Calculate win probability for the home team
        win_probability = calculate_win_probability(home_elo, away_elo, is_playoffs=is_playoffs)

        # Store matchup data
        matchup_data.append({
            'Season': year,
            'Week': game['week'],
            'Home Team': home_team,
            'Away Team': away_team,
            'Home Score': home_points,
            'Away Score': away_points,
            'Home Win Probability': win_probability,
            'Elo Difference': round(home_elo - away_elo),
            'Divisional Game': div_game,
            'Away Rest': away_rest,
            'Home QB': home_qb_name,
            'Away QB': away_qb_name,
            'Result': 'Home Win' if home_result == 1 else ('Away Win' if home_result == 0 else 'Tie')
        })

        # Update Elo ratings
        if not np.isnan(home_points) and not np.isnan(away_points):
            elo_ratings[home_team] = update_elo(home_elo, away_elo, home_result, point_diff)
            elo_ratings[away_team] = update_elo(away_elo, home_elo, 1 - home_result, point_diff)

    # Calculate the mean Elo rating for the end of the season
    elo_mean = sum(elo_ratings.values()) / len(elo_ratings)

    # Revert Elo ratings towards the mean at the end of the season
    for team in elo_ratings:
        elo_ratings[team] = elo_ratings[team] - ((elo_ratings[team] - elo_mean) * reversion_factor)

    # Save the final Elo ratings for the season to an Excel file
    season_data = pd.DataFrame({'Team': list(elo_ratings.keys()), 'Elo Rating': [round(rating) for rating in elo_ratings.values()], 'Season': year})
    season_data.to_excel(f'elo_ratings_{year}.xlsx', index=False, engine='openpyxl')

    # Save the matchup data for the season to an Excel file
    matchup_df = pd.DataFrame(matchup_data)
    matchup_df.index += 1  # Start index at 1 instead of 0
    matchup_df.to_excel(f'matchups_{year}.xlsx', index=False, engine='openpyxl')

    # Append to all matchup data for accuracy analysis
    all_matchup_data.extend(matchup_data)

# Streamlit: Display current Elo ratings
sorted_elo_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
elo_df = pd.DataFrame(sorted_elo_ratings, columns=['Team', 'Elo Rating'])
elo_df.index += 1  # Start index at 1 instead of 0
st.write("### Current Elo Ratings")
st.dataframe(elo_df)

# Streamlit: Display current matchups and win probabilities
current_year = max(years)
current_year_matchup_data = schedule_data[schedule_data['season'] == current_year]
latest_week_played = current_year_matchup_data.loc[~current_year_matchup_data['home_score'].isna(), 'week'].max()
current_year_matchup_data = current_year_matchup_data[current_year_matchup_data['week'] <= latest_week_played + 1]

current_matchup_data = []
for _, game in current_year_matchup_data.iterrows():
    home_team = game['home_team']
    away_team = game['away_team']
    home_points = game['home_score']
    away_points = game['away_score']
    div_game = game['div_game']
    away_rest = game['away_rest']
    home_qb_name = game['home_qb_name']
    away_qb_name = game['away_qb_name']

    # Determine result: 1 if home team wins, 0.5 if tie, 0 if away team wins
    if not np.isnan(home_points) and not np.isnan(away_points):
        if home_points > away_points:
            home_result = 1
        elif home_points < away_points:
            home_result = 0
        else:
            home_result = 0.5

        # Calculate win probability for the home team
        home_elo = elo_ratings.get(home_team, initial_elo)
        away_elo = elo_ratings.get(away_team, initial_elo)
        win_probability = calculate_win_probability(home_elo, away_elo)

        # Store matchup data
        current_matchup_data.append({
            'Season': current_year,
            'Week': game['week'],
            'Home Team': home_team,
            'Away Team': away_team,
            'Home Score': home_points,
            'Away Score': away_points,
            'Home Win Probability': win_probability,
            'Elo Difference': round(home_elo - away_elo),
            'Divisional Game': div_game,
            'Away Rest': away_rest,
            'Home QB': home_qb_name,
            'Away QB': away_qb_name,
            'Result': 'Home Win' if home_result == 1 else ('Away Win' if home_result == 0 else 'Tie')
        })

current_matchup_df = pd.DataFrame(current_matchup_data)
current_matchup_df.index += 1  # Start index at 1 instead of 0
st.write("### Current Matchups and Win Probabilities")
st.dataframe(current_matchup_df[['Week', 'Home Team', 'Away Team', 'Home Win Probability', 'Result']])
