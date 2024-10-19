import streamlit as st
import pandas as pd
import numpy as np

# Set up initial ratings and constants
INITIAL_RATING = st.sidebar.number_input("Initial Elo Rating", min_value=1000, max_value=2000, value=1500, step=50)
K = st.sidebar.slider("Elo K-Factor", min_value=10, max_value=40, value=20, step=1)  # Adjust this value for more or less responsiveness
DECAY_RATE = st.sidebar.slider("Season Elo Decay Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)  # 20% decay towards mean rating between seasons
INITIAL_YEAR = st.sidebar.number_input("Initial Year", min_value=1900, max_value=2025, value=2000, step=1)  # Start of Elo rating initialization

# Function to calculate Elo ratings
def calculate_elo(rating_a, rating_b, result_a, K=K):
    # Expected result
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    # New rating
    new_rating_a = rating_a + K * (result_a - expected_a)
    return new_rating_a

# Function to apply Elo decay towards the mean rating
def apply_season_decay(team_data, decay_rate=DECAY_RATE):
    mean_rating = team_data['Rating'].mean()
    team_data['Rating'] = team_data['Rating'].apply(lambda x: x + decay_rate * (mean_rating - x))
    return team_data

# Initialize data for teams
def initialize_teams():
    teams = [
        "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
        "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
        "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
        "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
        "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
        "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
        "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
        "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
    ]
    return pd.DataFrame({
        'Team': teams,
        'Rating': [INITIAL_RATING] * len(teams),
        'Year': [INITIAL_YEAR] * len(teams)
    })

# Initialize or load team data in session state
if 'team_data' not in st.session_state:
    st.session_state['team_data'] = initialize_teams()

# Streamlit UI
st.title("NFL Team Elo Ratings")

# Display current Elo ratings
st.subheader("Current Elo Ratings")
st.table(st.session_state['team_data'])

# Load historical game data (you can replace this with your actual data source)
def load_historical_data():
    # Placeholder for loading real historical data, replace with actual file path or data retrieval
    data = {
        'team': ["Buffalo Bills", "Buffalo Bills", "Kansas City Chiefs", "Kansas City Chiefs"],
        'opp': ["New England Patriots", "Miami Dolphins", "Denver Broncos", "Las Vegas Raiders"],
        'home_team': [True, False, True, False],
        'points_for': [21, 14, 35, 24],
        'points_allowed': [17, 20, 28, 24],
        'year': [2022, 2022, 2022, 2022]
    }
    return pd.DataFrame(data)

historical_data = load_historical_data()

# Remove duplicate matchups
historical_data['matchup'] = historical_data.apply(lambda row: tuple(sorted([row['team'], row['opp']])), axis=1)
historical_data = historical_data.drop_duplicates(subset=['year', 'matchup'])

# Calculate Elo ratings based on historical data
st.subheader("Update Elo Ratings from Historical Data")
if st.button("Calculate Elo Ratings from Historical Data"):
    for index, game in historical_data.iterrows():
        team_a = game['team']
        team_b = game['opp']
        rating_a = st.session_state['team_data'].loc[st.session_state['team_data']['Team'] == team_a, 'Rating'].values[0]
        rating_b = st.session_state['team_data'].loc[st.session_state['team_data']['Team'] == team_b, 'Rating'].values[0]

        # Determine the result based on points scored
        if game['points_for'] > game['points_allowed']:
            result_a = 1  # Team A wins
        elif game['points_for'] < game['points_allowed']:
            result_a = 0  # Team B wins
        else:
            result_a = 0.5  # Draw

        # Calculate new ratings
        new_rating_a = calculate_elo(rating_a, rating_b, result_a)
        new_rating_b = calculate_elo(rating_b, rating_a, 1 - result_a)

        # Update the ratings in the session state dataframe
        st.session_state['team_data'].loc[st.session_state['team_data']['Team'] == team_a, 'Rating'] = new_rating_a
        st.session_state['team_data'].loc[st.session_state['team_data']['Team'] == team_b, 'Rating'] = new_rating_b

    st.success("Ratings Updated from Historical Data!")
    st.table(st.session_state['team_data'])

# Apply Elo rating decay at the start of a new season
st.subheader("Start a New Season")
if st.button("Apply Elo Decay for New Season"):
    st.session_state['team_data'] = apply_season_decay(st.session_state['team_data'])
    st.success("Elo ratings have been adjusted for the new season!")
    st.table(st.session_state['team_data'])

# Save the team data (if you wish to persist data across sessions)
# You could also use session state, a database, or a file
