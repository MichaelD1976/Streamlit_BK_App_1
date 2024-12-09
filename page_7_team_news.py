import streamlit as st # type: ignore
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import time
from datetime import date, datetime, timedelta
from mymodule.functions import get_fixtures


# -----------------------------------------------------------------------------

FROM_DATE = date.today()
TO_DATE = date.today() + timedelta(days=7)
API_SEASON = '2024'  

# ------------------------------------------------------------------------------

# Function to get injuries for a fixture
def get_injuries(fixture_id):
    url = "https://api-football-v1.p.rapidapi.com/v3/injuries"

    if not st.secrets:
        load_dotenv()
        API_KEY = os.getenv("API_KEY_FOOTBALL_API")

    else:
        # Use Streamlit secrets in production
        API_KEY = st.secrets["rapidapi"]["API_KEY_FOOTBALL_API"]

    headers = {
        "X-RapidAPI-Key": API_KEY,  # Replace with your actual API key
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    
    querystring = {"fixture": fixture_id}
    
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    injury_list = []
    if 'response' in data and data['response']:
        for injury in data['response']:
            player = injury.get('player', {})
            team = injury.get('team', {})
            
            injury_details = {
                'Player Name': player.get('name', 'N/A'),
                'Injury Type': player.get('type', 'N/A'),
                'Injury Reason': player.get('reason', 'N/A'),
                'Player Photo': player.get('photo', 'N/A'),
                'Team Name': team.get('name', 'N/A')
            }
            
            injury_list.append(injury_details)

    return pd.DataFrame(injury_list) if injury_list else pd.DataFrame()
    
# ------------------------------------------------------------------------

# Function to get injuries for all fixtures in a DataFrame
def get_all_injuries_for_fixtures(df_fixtures):
    # List to store all injury data
    all_injuries = []

    # Iterate through each fixture ID in the DataFrame
    for fixture_id in df_fixtures['Fixture ID']:
        # Call get_injuries for each fixture
        injuries_df = get_injuries(fixture_id)
        
        # If injuries are found, append to the list
        if not injuries_df.empty:
            injuries_df['Fixture ID'] = fixture_id  # Add fixture ID for reference
            all_injuries.append(injuries_df)
    
    # Concatenate all injury data into a single DataFrame
    return pd.concat(all_injuries, ignore_index=True) if all_injuries else pd.DataFrame()

# ------------------------------------------------------------------

# Define selection options
league_options = {
    # 'All_leagues': 'ALL',  # Uncomment for future development
    'Premier League': 'England Premier',
    'La Liga': 'Spain La Liga',
    'Bundesliga': 'Germany Bundesliga',
    'Ligue 1': 'France Ligue 1',
    'Serie A': 'Italy Serie A',
    # 'Premiership': 'Scotland Premier',
    # 'Eredivisie': 'Netherlands Eredivisie',
    # 'Jupiler Pro League': 'Belgium Jupiler',
    # 'Primeira Liga': 'Portugal Liga I',
    # 'Championship': 'England Championship',
    # '2. Bundesliga': 'Germany 2 Bundesliga',
    # 'League One': 'England League One',
    # 'League Two': 'England League Two',
}

    # Dictionary to map league names to their IDs
leagues_dict = {
    "England Premier": '39',
    "Spain La Liga": '140',
    "Germany Bundesliga": '78',
    "Italy Serie A": '135',
    "France Ligue 1": '61',
    # 'England Championship': '40',
    # 'England League One': '41',
    # 'England League Two': '42',
    # "Germany 2 Bundesliga": '79',
    # "Netherlands Eredivisie": "88",
    # "Belgium Jupiler": "144",
    # "Portugal Liga I": '94',
    # "Scotland Premier": '179'
}


# -------------------------------------------------------------------


def main():

    selected_league = st.sidebar.selectbox('Select League', options=list(league_options.values()), label_visibility = 'visible')
    league_id = leagues_dict[selected_league ]

    # Display
    st.header('Team News', divider='blue')
    st.subheader(f'{selected_league} team news for upcoming fixtures')
    st.write("")

    df_fixtures = get_fixtures(league_id, FROM_DATE, TO_DATE, API_SEASON)
    if df_fixtures.empty:
        st.write('No fixtures available for the coming 7 days')
        st.stop()
    # st.write(df_fixtures)

    injuries_df = get_all_injuries_for_fixtures(df_fixtures)
    if injuries_df.empty:
        st.write('No injury news currently available')
        st.stop()
    # st.write(injuries_df)

    injuries_df = injuries_df[['Team Name', 'Player Name',  'Injury Type', 'Injury Reason']]
    st.write(injuries_df)


if __name__ == '__main__':
    main()