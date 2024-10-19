import streamlit as st
import pandas as pd
import numpy as np
from pro_football_reference_web_scraper import team_game_log as t

# Set up initial ratings and constants
INITIAL_RATING = st.sidebar.number_input("Initial Elo Rating", min_value=1000, max_value=2000, value=1500, step=50)
K = st.sidebar.slider("Elo K-Factor", min_value=10, max_value=40, value=20, step=1)  # Adjust this value for more or less responsiveness
DECAY_RATE = st.sidebar.slider("Season Elo Decay Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)  # 20% decay towards mean rating between seasons
INITIAL_YEAR = 2010  # Fixed initial year to start calculations from

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
@st.cache_data
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

# Load team data
if 'team_data' not in st.session_state:
    st.session_state['team_data'] = initialize_teams()
team_data = st.session_state['team_data']

# Streamlit UI
st.title("NFL Team Elo Ratings")

# Adjustable inputs for Elo calculation range and current season
selected_end_year = st.sidebar.number_input("End Year for Historical Data Calculation", min_value=2010, max_value=2023, value=2023, step=1)
selected_week = st.sidebar.slider("Select Week for Current Season", min_value=1, max_value=22, value=1, step=1)

# Load historical game data using NFL web scraper
@st.cache_data
def load_historical_data(start_year, end_year):
    # Fetch data using the web scraper for all seasons in the range
    historical_data = pd.DataFrame()
    matchup_tracker = set()  # To track unique matchups

    for year in range(start_year, end_year + 1):
        for team in team_data['Team']:
            try:
                game_log = t.get_team_game_log(team=team, season=year)
                if game_log.empty:
                    st.warning(f"No data available for {team} in {year}. Skipping...")
                    continue
                for _, game in game_log.iterrows():
                    # Check if required fields exist before proceeding
                    if 'opp' not in game or 'points_for' not in game or 'points_allowed' not in game or 'week' not in game:
                        continue
                    
                    team_a = team
                    team_b = game['opp']
                    points_for = game['points_for']
                    points_allowed = game['points_allowed']
                    home_team = game.get('home_team', False)
                    week = game['week']

                    # Create a sorted matchup tuple to track unique games
                    matchup = tuple(sorted([team_a, team_b, year, week]))
                    if matchup not in matchup_tracker:
                        matchup_tracker.add(matchup)
                        historical_data = pd.concat([historical_data, pd.DataFrame([{
                            'team_a': team_a,
                            'team_b': team_b,
                            'points_for': points_for,
                            'points_allowed': points_allowed,
                            'home_team': home_team,
                            'year': year,
                            'week': week
                        }])], ignore_index=True)
            except Exception as e:
                st.warning(f"Error fetching data for {team} in {year}: {e}. Skipping...")
    return historical_data

# Get historical data for the selected range
if 'historical_data' not in st.session_state or st.session_state['historical_data_range'] != (INITIAL_YEAR, selected_end_year):
    st.session_state['historical_data'] = load_historical_data(INITIAL_YEAR, selected_end_year)
    st.session_state['historical_data_range'] = (INITIAL_YEAR, selected_end_year)
historical_data = st.session_state['historical_data']

# Display Elo ratings at the end of historical range (End of 2023)
st.subheader("Elo Ratings at the End of Historical Data ({}-{})".format(INITIAL_YEAR, selected_end_year))
st.table(team_data)

# Calculate Elo ratings based on historical data
if st.button("Calculate Elo Ratings from Historical Data"):
    for index, game in historical_data.iterrows():
        team_a = game['team_a']
        team_b = game['team_b']
        try:
            rating_a = team_data.loc[team_data['Team'] == team_a, 'Rating'].values[0]
            rating_b = team_data.loc[team_data['Team'] == team_b, 'Rating'].values[0]
        except IndexError:
            st.warning(f"Missing ratings for one of the teams: {team_a} or {team_b}. Skipping this game.")
            continue

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

        # Update the ratings in the dataframe
        st.session_state['team_data'].loc[st.session_state['team_data']['Team'] == team_a, 'Rating'] = new_rating_a
        st.session_state['team_data'].loc[st.session_state['team_data']['Team'] == team_b, 'Rating'] = new_rating_b

    st.success("Ratings Updated from Historical Data!")
    st.table(st.session_state['team_data'])

# Load current season data
@st.cache_data
def load_current_season_data(season_year, week):
    current_season_data = pd.DataFrame()
    matchup_tracker = set()  # Track matchups to avoid duplicates

    for team in team_data['Team']:
        try:
            game_log = t.get_team_game_log(team=team, season=season_year)
            if game_log.empty:
                st.warning(f"No data available for {team} in {season_year}. Skipping...")
                continue
            for _, game in game_log.iterrows():
                if game['week'] <= week:
                    # Check if required fields exist before proceeding
                    if 'opp' not in game or 'points_for' not in game or 'points_allowed' not in game or 'week' not in game:
                        continue

                    team_a = team
                    team_b = game['opp']
                    matchup = tuple(sorted([team_a, team_b, season_year, game['week']]))
                    if matchup not in matchup_tracker:
                        matchup_tracker.add(matchup)
                        current_season_data = pd.concat([current_season_data, pd.DataFrame([{
                            'team_a': team_a,
                            'team_b': team_b,
                            'points_for': game['points_for'],
                            'points_allowed': game['points_allowed'],
                            'home_team': game.get('home_team', False),
                            'week': game['week'],
                            'year': season_year
                        }])], ignore_index=True)
        except Exception as e:
            st.warning(f"Error fetching data for {team} in {season_year}, week {week}: {e}. Skipping...")
    return current_season_data

# Load current season data based on user-selected week
if 'current_season_data' not in st.session_state or st.session_state['current_season_week'] != selected_week:
    st.session_state['current_season_data'] = load_current_season_data(2023, selected_week)
    st.session_state['current_season_week'] = selected_week
current_season_data = st.session_state['current_season_data']

# Display Elo ratings for the current season based on user-selected week
st.subheader("Elo Ratings for the Current Season (Week {})".format(selected_week))
if st.button("Calculate Elo Ratings for Current Season"):
    for index, game in current_season_data.iterrows():
        team_a = game['team_a']
        team_b = game['team_b']
        try:
            rating_a = team_data.loc[team_data['Team'] == team_a, 'Rating'].values[0]
            rating_b = team_data.loc[team_data['Team'] == team_b, 'Rating'].values[0]
        except IndexError:
            st.warning(f"Missing ratings for one of the teams: {team_a} or {team_b}. Skipping this game.")
            continue

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

        # Update the ratings in the dataframe
        st.session_state['team_data'].loc[st.session_state['team_data']['Team'] == team_a, 'Rating'] = new_rating_a
        st.session_state['team_data'].loc[st.session_state['team_data']['Team'] == team_b, 'Rating'] = new_rating_b

    st.success("Ratings Updated for Current Season!")
    st.table(st.session_state['team_data'])
