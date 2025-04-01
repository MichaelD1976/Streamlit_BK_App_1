
import streamlit as st # type: ignore
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import time
from datetime import date, datetime, timedelta
from mymodule.functions import get_fixtures, calculate_true_from_true_raw
import gc


API_SEASON = '2024'
MARKET_IDS = ['1', '5']             # WDW & Ov/Un
BOOKMAKERS = ['8']                  # Pinnacle = 4, 365 = 8

# Define selection options
league_options = {
    # 'All_leagues': 'ALL',  # Uncomment for future development
    'Premier League': 'England Premier',
    'La Liga': 'Spain La Liga',
    'Bundesliga': 'Germany Bundesliga',
    'Ligue 1': 'France Ligue 1',
    'Serie A': 'Italy Serie A',
    'Premiership': 'Scotland Premier',
    'Eredivisie': 'Netherlands Eredivisie',
    'Jupiler Pro League': 'Belgium Jupiler',
    'Primeira Liga': 'Portugal Liga I',
    'Championship': 'England Championship',
    '2. Bundesliga': 'Germany 2 Bundesliga',
    'League One': 'England League One',
    'League Two': 'England League Two',
}

    # Dictionary to map league names to their IDs
leagues_dict = {
    "England Premier": '39',
    "Spain La Liga": '140',
    "Germany Bundesliga": '78',
    "Italy Serie A": '135',
    "France Ligue 1": '61',
    'England Championship': '40',
    'England League One': '41',
    'England League Two': '42',
    "Germany 2 Bundesliga": '79',
    "Netherlands Eredivisie": "88",
    "Belgium Jupiler": "144",
    "Portugal Liga I": '94',
    "Scotland Premier": '179'
}
# -------------------------------------------



def main():

    st.header('Popular Matches Predictor')

    today = datetime.now()
    week_from_today = today + timedelta(days=3)


    from_date = today
    to_date = week_from_today

    # Sidebar for 'match_date' input  # WIDGET
    from_date = st.sidebar.date_input( "Select Date", from_date, label_visibility = 'visible')

    # Sidebar for 'to_date' input  # WIDGET
    to_date = from_date
    

    df_list = []

    # Ensure leagues_dict is iterable
    for league_id in leagues_dict.values():
        df_fixtures = get_fixtures(league_id, from_date, to_date, API_SEASON)
        
        if isinstance(df_fixtures, pd.DataFrame):  # Ensure it is a DataFrame before appending
            df_list.append(df_fixtures)

    # Concatenate all DataFrames
    if df_list:  # Check if df_list is not empty to avoid errors
        df_fixts = pd.concat(df_list, ignore_index=True)
    else:
        df_fixts = pd.DataFrame()  # Return an empty DataFrame if nothing was collected

    # st.write(df_fixts)


    fixt_id_list = list(df_fixts['Fixture ID'].unique())

    if not st.secrets:
        load_dotenv()
        API_KEY = os.getenv("API_KEY_FOOTBALL_API")

    else:
        # Use Streamlit secrets in production
        API_KEY = st.secrets["rapidapi"]["API_KEY_FOOTBALL_API"]

    @st.cache_data
    def get_odds(fixture_id, market_id, bookmakers):
        url = "https://api-football-v1.p.rapidapi.com/v3/odds"
        headers = {
            "X-RapidAPI-Key": API_KEY,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
        querystring = {
            "fixture": fixture_id,
            "bet": market_id,
            "timezone": "Europe/London"
        }

        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()

        if 'response' in data and data['response']:
            odds_dict = {
                'Fixture ID': fixture_id,
                'Home Win': None,
                'Draw': None,
                'Away Win': None,
                # 'Over 2.5': None,                 # goals not needed in ml model for corners. v.big odds on might not have 2.5
                # 'Under 2.5': None,                # line which will cause the whole match to be dropped in later script
            }

            # Loop through bookmakers
            for bookmaker_data in data['response'][0].get('bookmakers', []):
                if str(bookmaker_data['id']) in bookmakers:
                    # Loop through each market (bet) offered by the bookmaker
                    for bet_data in bookmaker_data['bets']:
                        if bet_data['id'] == int(market_id):  # Ensure it's the selected market
                            # Extract the outcomes (selections) and their corresponding odds
                            for value in bet_data['values']:
                                selection = value['value']
                                odd = value['odd']
                                
                                # Assign the odds based on the selection type
                                if selection == 'Home':
                                    odds_dict['Home Win'] = odd
                                elif selection == 'Draw':
                                    odds_dict['Draw'] = odd
                                elif selection == 'Away':
                                    odds_dict['Away Win'] = odd
                                # elif selection == 'Over 2.5':     
                                #     odds_dict['Over 2.5'] = odd     
                                # elif selection == 'Under 2.5':
                                #     odds_dict['Under 2.5'] = odd


            # Create a DataFrame with a single row containing all the odds
            odds_df = pd.DataFrame([odds_dict])
            return odds_df
            
        # Return empty DataFrame if no data is found
        return pd.DataFrame()
        

    # Collect odds for all fixtures
    all_odds_df = pd.DataFrame()  # DataFrame to collect all odds
    
    # Iterate through each fixture ID and get odds
    for fixture_id in fixt_id_list:
        for market_id in MARKET_IDS:
            odds_df = get_odds(fixture_id, market_id, BOOKMAKERS)
            if not odds_df.empty:
                all_odds_df = pd.concat([all_odds_df, odds_df], ignore_index=True)

    # st.write(all_odds_df)

    # Use groupby and fillna to collapse rows and remove None values
    df_collapsed = all_odds_df.groupby('Fixture ID').first().combine_first(
        all_odds_df.groupby('Fixture ID').last()).reset_index()

    # st.write(df_collapsed)
    # st.write(df_fixts)


    # Merge odds df_fixts with df_collapsed
    df = df_fixts.merge(df_collapsed, on='Fixture ID')
    # df = df.dropna()

    del df_collapsed
    gc.collect()
    df = df[['Date', 'League Name', 'Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win']]
    st.write(df)


    ## TODO
    ## add more leagues, dict for leagie_id's
    ## have a ranking value (RV_1) for each league
    ## have a ranking value (RV_2) for each match odds range category eg 1.5 = 1, 2.5 = 5
    ## algo to multiply RV_1 with RV_2 to to create overall ranking value (OVR) for the each match
    ## order by OVR, display top 25 matches with OVR Rating



if __name__ == "__main__":
    main()
