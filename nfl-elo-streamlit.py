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
@st.cache
def initialize_teams():
    teams = [
        "Kansas City Chiefs", "Green Bay Packers", "Buffalo Bills",
        "New England Patriots", "Dallas Cowboys", "San Francisco 49ers",
        "Los Angeles Rams", "Miami Dolphins"
    ]
    return pd.DataFrame({
        'Team': teams,
        'Rating': [INITIAL_RATING] * len(teams),
        'Year': [INITIAL_YEAR] * len(teams)
    })

# Load team data
team_data = initialize_teams()

# Streamlit UI
st.title("NFL Team Elo Ratings")

# Display current Elo ratings
st.subheader("Current Elo Ratings")
st.table(team_data)

# Simulate game results
st.subheader("Enter Game Results")

team_a = st.selectbox("Select Team A", team_data['Team'])
team_b = st.selectbox("Select Team B", team_data['Team'])
result = st.selectbox("Select Result", ["Team A Wins", "Team B Wins", "Draw"])

# Calculate and update ratings
if st.button("Calculate New Elo Ratings"):
    rating_a = team_data.loc[team_data['Team'] == team_a, 'Rating'].values[0]
    rating_b = team_data.loc[team_data['Team'] == team_b, 'Rating'].values[0]
    
    if result == "Team A Wins":
        new_rating_a = calculate_elo(rating_a, rating_b, 1)
        new_rating_b = calculate_elo(rating_b, rating_a, 0)
    elif result == "Team B Wins":
        new_rating_a = calculate_elo(rating_a, rating_b, 0)
        new_rating_b = calculate_elo(rating_b, rating_a, 1)
    else:  # Draw
        new_rating_a = calculate_elo(rating_a, rating_b, 0.5)
        new_rating_b = calculate_elo(rating_b, rating_a, 0.5)

    # Update the ratings in the dataframe
    team_data.loc[team_data['Team'] == team_a, 'Rating'] = new_rating_a
    team_data.loc[team_data['Team'] == team_b, 'Rating'] = new_rating_b

    st.success("Ratings Updated!")
    st.table(team_data)

# Apply Elo rating decay at the start of a new season
st.subheader("Start a New Season")
if st.button("Apply Elo Decay for New Season"):
    team_data = apply_season_decay(team_data)
    st.success("Elo ratings have been adjusted for the new season!")
    st.table(team_data)

# Save the team data (if you wish to persist data across sessions)
# You could also use session state, a database, or a file