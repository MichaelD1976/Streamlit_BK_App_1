import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from unidecode import unidecode
import os 
from mymodule.functions import generate_team_fst, team_names_api_to_t1x2_dict, get_fixtures

# NordicBet 27
# 10Bet 1
# William Hill 7
# Bet365 8
# Marathonbet 2
# Unibet 16
# Betfair 3
# Betsson 26
# Fonbet 33
# Pinnacle 4
# SBO 5
# 1xBet 11
# Betano 32
# Betway 24
# Tipico 22
# Betcris 20
# Dafabet 9


# Matchwinner 1
# Asian Hcap 4
# Goals Over/Under 5
# Handicap 9
# DC 12
# HT/FT 7
# CS 10
# BTTS 8
# Win Both Halves 32
# DNB 182
# Market: Corners 1x2, ID: 55
# Market: Corners Over Under, ID: 45
# F Goalscorer 93
# Anytime GS 92
# Market: Cards Over/Under, ID: 80
# Market: Cards Asian Handicap, ID: 81
# Market: Total ShotOnGoal, ID: 87
# Market: Home Total ShotOnGoal, ID: 88
# Market: Away Total ShotOnGoal, ID: 89
# Market: Penalty Awarded, ID: 163
# Market: Offsides Total, ID: 164
# Market: Offsides 1x2, ID: 165
# win both halves 32
# win either half 39



def get_reversed_team_name_dict():
    return {v: k for k, v in team_names_api_to_t1x2_dict.items()}


# Load environment variables (API key)
load_dotenv()
API_KEY = os.getenv('API_KEY_FOOTBALL_API')
CURRENT_SEASON = '2025-26' # to fetch team FST's
API_SEASON = '2025' # current season as used by api-foot

# Custom CSS 
st.markdown(
    '''
    <style>
    .streamlit-expanderHeader {
        background-color: white;
        color: black; # Adjust this for expander header color
    }
    .streamlit-expanderContent {
        background-color: white;
        color: black; # Expander content color
    }
    </style>
    ''',
    unsafe_allow_html=True
)


# ----------------------------------------------------------------------------------

# Function to fetch injuries for a specific fixture
@st.cache_data(ttl=600)  # every 10 mins
def get_injuries(fixture_id):
    url = "https://api-football-v1.p.rapidapi.com/v3/injuries"
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    
    querystring = {"fixture": fixture_id}
    
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    # st.write(API_KEY)

    injury_list = []
    if 'response' in data and data['response']:
        for injury in data['response']:
            player = injury['player']
            team = injury['team']
            # st.write 
            
            injury_details = {
                'Player Name': player['name'],
                'Injury Type': player['type'],
                'Injury Reason': player['reason'],
                'Player Photo': player['photo'],
                'Team Name': team['name'],
                'Player_id': player['id']
            }
            
            injury_list.append(injury_details)

    return pd.DataFrame(injury_list) if injury_list else pd.DataFrame()

# ---------------------------------------------------------
@st.cache_data(ttl=600)  # every 10 mins
def get_lineups(fixture_id):
    # Base URL for the API call
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups"
    
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }

    # Prepare an empty list to hold all player data
    all_players_data = []

    # Loop over each fixture ID

    querystring = {"fixture": fixture_id}
    
    # Make the API call
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    
    # # Extract the 'response' field which contains the teams' data
    # lineups = data['response']

    # Extract the 'response' field which contains the teams' data
    lineups = data.get('response', [])

    if not lineups:
        st.error(f"Awaiting lineups")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found
        
    
    # Loop through teams (home and away) in the lineup
    for team_data in lineups:
        team_name = team_data['team']['name']
        
        # Extract starting XI information
        starting_xi = team_data.get('startXI', None)

        if starting_xi is None:
            # st.warning('Awaiting lineups') # st.warning(f"Starting XI not available for {team_name}")
            continue  # Skip this team if no starting XI is available
        
        # For each player in the starting XI, extract relevant details
        for player_data in starting_xi:
            player = player_data.get('player', {})  # Ensure we safely access 'player'
            player_info = {
                'team': team_name,
                'id': player.get('id'),
                'name': player.get('name'),
                'number': player.get('number'),
                'pos': player.get('pos'),
                'Player_id': player.get('id')
            }
            
            # Append this player's data to the overall list
            all_players_data.append(player_info)

    # Convert the list of player data to a pandas DataFrame
    df = pd.DataFrame(all_players_data) if all_players_data else pd.DataFrame()

    # convert team name to 1x2 standardized
    if not df.empty:
        # # 1. remove non latin syntax unidecode         ***********   OK TO DELETE THIS SECTION  *********
        # # remove foreign lettering - Home, store as new column
        # new_team_name = []
        # for name in df['team']:
        #     name = unidecode.unidecode(name)
        #     new_team_name.append(name)
        # df['team'] = new_team_name

        # # Manually replace Umlauts as a fallback
        # df['team'] = df['team'].apply(lambda x: unidecode.unidecode(x)
        #                                         .replace('ö', 'o')
        #                                         .replace('ü', 'u')
        #                                         .replace('ä', 'a')
        #                                         .replace('Ö', 'O')
        #                                         .replace('Ü', 'U')
        #                                         .replace('Ä', 'A'))
        
        df['team'] = df['team'].map(team_names_api_to_t1x2_dict).fillna(df['team'])

    # Convert the list of player data to a pandas DataFrame
    return df
   

# ----------------------------------------------------------

# Function to fetch ODDS for a specific fixture and market
@st.cache_resource
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
        odds_dict = {}
        selections = []  # Use a list to maintain order

        # Loop through bookmakers
        for bookmaker_data in data['response'][0].get('bookmakers', []):
            if str(bookmaker_data['id']) in bookmakers:
                bookmaker_name = bookmaker_data['name']
                odds_dict[bookmaker_name] = {}
                
                # Loop through each market (bet) offered by the bookmaker
                for bet_data in bookmaker_data['bets']:
                    if bet_data['id'] == int(market_id):  # Ensure it's the selected market
                        # Extract the outcomes (selections) and their corresponding odds
                        for value in bet_data['values']:
                            selection = value['value']  # Extract outcome
                            odd = value['odd']          # Extract the corresponding odd
                            selections.append(selection)  # Append to maintain order
                            odds_dict[bookmaker_name][selection] = odd

        # Construct the DataFrame
        if selections:
            # Use a list of unique selections to preserve order
            unique_selections = list(dict.fromkeys(selections))  # Remove duplicates while maintaining order
            odds_df = pd.DataFrame(index=unique_selections)  # Set the index to unique selections
            for bookmaker, odds in odds_dict.items():
                odds_df[bookmaker] = [odds.get(selection, None) for selection in odds_df.index]  # Fill odds
            return odds_df
    return pd.DataFrame()  # Return empty DataFrame if no data is found

# ------------------------------------------------------------

# Load the CSV file
@st.cache_data
def load_data():
    time.sleep(0.5)  # Simulate a delay for loading
    df = pd.read_csv('data/outputs_processed/players/processed_f_api_combined_all_seasons_per90.csv')
    df_past_fixts = pd.read_csv('data/outputs_processed/teams/football-data_master_4.csv') # for showing past matches/stats
    df = df[df['Season'] == CURRENT_SEASON]
    return df, df_past_fixts
# ------------------------------------------------------------

# Main function to run the Streamlit app
def main():

    st.subheader("Fixtures and Team News page in development")

    # with st.spinner('Loading Data...'):
    #     df, df_past_fixts = load_data()


    # # Custom CSS 
    # st.markdown(
    #     '''
    #     <style>
    #     .streamlit-expanderHeader {
    #         background-color: white;
    #         color: black; # Adjust this for expander header color
    #     }
    #     .streamlit-expanderContent {
    #         background-color: white;
    #         color: black; # Expander content color
    #     }
    #     </style>
    #     ''',
    #     unsafe_allow_html=True
    # )
        
    # # Initialize default values
    # default_league = "England Premier"
    # today = datetime.now()
    # week_from_today = today + timedelta(days=3)

    # # Dictionary to map league names to their IDs
    # leagues_dict = {
    #     "England Premier": '39',
    #     "Germany Bundesliga": '78',
    #     "Spain La Liga": '140',
    #     "Italy Serie A": '135',
    #     "France Ligue 1": '61',
    #     # "Netherlands Eredivisie": "88",
    #     # "Belgium Jupiler": "144",
    #     # "Portugal Liga I": '94',
    #     # 'Scotland Premier': '179',
    #     # 'England Championship': '40',
    #     # 'England League One': '41',
    #     # 'England League Two': '42',
    #     # "Germany 2 Bundesliga": '79',
    #     # 'Italy Serie B': '136',
    #     # 'Spain La Liga 2': '141',
    #     # 'France Ligue 2': '62',
    #     # 'Turkey Super Lig': '203',
    #     # 'Greece Super League': '197'
    # }

    # # Define selection options

    # # league_options = {
    # #     'E0': 'England Premier',
    # #     'D1': 'Germany Bundesliga',
    # #     'SP1': 'Spain La Liga',
    # #     'I1': 'Italy Serie A',
    # #     'F1': 'France Ligue 1',
    # #     'N1': 'Netherlands Eredivisie',
    # #     'B1': 'Belgium Jupiler',
    # #     'P1': 'Portugal Liga I',
    # #     'SC0': 'Scotland Premier',
    # #     'E1': 'England Championship',
    # #     'E2': 'England League One',
    # #     'E3': 'England League Two',
    # #     'D2': 'Germany 2 Bundesliga',
    # #     'SP2': 'Spain La Liga 2',
    # #     'I2': 'Italy Serie B',
    # #     'F2': 'France Ligue 2',
    # # }

    # non_player_stat_leagues = ['Germany 2 Bundesliga', 'Italy Serie B', 'Spain La Liga 2', 'France Ligue 2', 'Turkey Super Lig', 'Greece Super League' ]

    # # Market and bookmaker settings
    # markets_dict = {
    #     "Match Winner": '1',
    #     "Asian Handicap": '4',
    #     "Goals Over/Under": '5',
    #     "Correct Score": '10',
    #     'Double Chance': '12',
    #     "Both Teams to Score": '8',
    #     'Half-time/Full-time': '7',
    #     "First Goalscorer": '93',
    #     'Anytime Goalscorer': '92',
    #     'Corners 1X2': '55',
    #     'Corners Total': '45',
    #     'Win Both Halves': '32',
    #     'Win Either Half': '39'
    # }

    # bookmaker_list = ['4','3','8', '11', '16', '7', '3', '24', '1', '5', '27', '2', '26', '32', '22', '9'] 

    # # --------------------------------------------------------------

    # # PUT SELECTIONS IN SESSION STATE SO WHEN PAGE REFRESHED, SELECTIONS DONT NEED TO BE PICKED AGAIN   
    # # Initialize default values if they don't exist in session state
    # if 'league_name' not in st.session_state:
    #     st.session_state.league_name = default_league
    # if 'from_date' not in st.session_state:
    #     st.session_state.from_date = today
    # if 'to_date' not in st.session_state:
    #     st.session_state.to_date = week_from_today


    # # Sidebar for league selection (WIDGET)
    # league_name = st.sidebar.selectbox(
    #     "Select a League",
    #     list(leagues_dict.keys()),
    #     index=list(leagues_dict.keys()).index(st.session_state.league_name),
    #     label_visibility='visible',
    # )

    # # Check if the league selection has changed
    # if league_name != st.session_state.league_name:
    #     # Update the session state to reflect the new league name
    #     st.session_state.league_name = league_name
        
    #     # Trigger a page refresh (only if the selection has changed)
    #     st.rerun()


    # # Sidebar for 'from_date' input  # WIDGET
    # from_date = st.sidebar.date_input(
    #     "From Date",
    #     st.session_state.from_date,
    #     label_visibility = 'visible'
    # )

    # # Check if the from_date has changed
    # if from_date != st.session_state.from_date:
    #     # Update the session state to reflect the new league name
    #     st.session_state.from_date = from_date        
    #     # Trigger a page refresh (only if the selection has changed)
    #     st.rerun()


    # # Calculate max 'to_date' based on 'from_date'
    # max_to_date = from_date + timedelta(days=30)

    # # Sidebar for 'to_date' input  # WIDGET
    # to_date = st.sidebar.date_input(
    #     "To Date",
    #     st.session_state.to_date,
    #     max_value=max_to_date,
    #     label_visibility = 'visible'
    # )

    # # Check if the to_date has changed
    # if to_date != st.session_state.to_date:
    #     # Update the session state to reflect the new league name
    #     st.session_state.to_date = to_date        
    #     # Trigger a page refresh (only if the selection has changed)
    #     st.rerun()



    # # Update from_date_str and to_date_str after confirming changes
    # from_date_str = st.session_state.from_date.strftime('%Y-%m-%d')
    # to_date_str = st.session_state.to_date.strftime('%Y-%m-%d')


    # # --------------------  MAIN PAGE  ---------------------------
    # st.header(f"{st.session_state.league_name} Fixtures")

    # st.caption('Downgrades from Best XI indicate the cumulative expected team weakness denoted as 100ths of a goal from the best possible starting XI based on current season data')
    # st.write("---")


    # # Automatically fetch fixtures on page load
    # league_id = leagues_dict[st.session_state.league_name]
    # df_fixtures = get_fixtures(league_id, from_date_str, to_date_str, API_SEASON)

    # # Display the fixtures and injuries in a structured layout
    # if not df_fixtures.empty:
    #     df_fixtures['Date'] = pd.to_datetime(df_fixtures['Date'], format='%d-%m-%y %H:%M')  # Convert to datetime
    #     df_fixtures.sort_values(by='Date', inplace=True)  # Sort by date

    #     for index, row in df_fixtures.iterrows():
    #         fixture_id = row['Fixture ID']
    #         home_team_name = row['Home Team']
    #         away_team_name = row['Away Team']

    #         # Create a three-column layout: Fixture | Injuries Home | Injuries Away
    #         col1, col2, col3, col4, col5, _ = st.columns([3, 0.5, 3.5, 0.5, 3.5, 0.2])

    #         # Left column: Display fixture details
    #         with col1:
    #             st.subheader(f"{home_team_name} vs {away_team_name}")
    #             st.write(f"Date: {row['Date']}")
    #             st.write(f"Venue: {row['Venue']} ({row['City']})")
    #             st.write(f"Status: {row['Status']}")
    #             if row['Status'] == 'Match Finished':
    #                 st.write(f"FT: {int(row['Fulltime Score Home'])} - {int(row['Fulltime Score Away'])}")
    #             elif row['Status'] == 'Halftime':
    #                 st.write(f"HT: {int(row['Halftime Score Home'])} - {int(row['Halftime Score Away'])}")
    #             elif row['Status'] == 'Match Postponed':
    #                 pass
    #             elif row['Status'] != 'Not Started':
    #                 st.write(f"Score: {int(row['Goals Home'])} - {int(row['Goals Away'])}")


    #         with col2:
    #          st.write("")
    #          st.image(row['Home Team Logo'], width=40)

    #         with col4:
    #          st.write("")
    #          st.image(row['Away Team Logo'], width=40)    


    #         with col3:
    #             st.write("")
                
    #             # ----------------- show fst -----------
    #             st.write("")
    #             # WIDGET
    #             with st.expander(f"Show {home_team_name} Best XI", expanded=False):
    #                 if league_name not in non_player_stat_leagues:   
    #                     df_filtered = df[df['Team'] == home_team_name]
    #                     team_df, _ = generate_team_fst(df_filtered, CURRENT_SEASON)  # returns 2 dfs, fst and squad ratings, not utilising squad here so '_'
    #                     st.dataframe(team_df, height=420)
    #                 else:
    #                     st.write(f'Best XI unavailable for {league_name}')

    #             # ----------------- team info -----------
    #             # WIDGET
    #             with st.expander(f"{home_team_name} team info", expanded=False):
    #                 df_lineups = get_lineups(fixture_id)

    #                 if 'team' in df_lineups.columns:
    #                     df_lineups['team'] = df_lineups['team'].apply(lambda x: unidecode(str(x)))
    #                     df_lineups['team'] = df_lineups['team'].map(team_names_api_to_t1x2_dict).fillna(df_lineups['team'])
    #                     df_home_lineup = df_lineups[df_lineups['team'] == home_team_name]
    #                     if not df_home_lineup.empty:
    #                         st.write('**Starting XI**')

    #                     # Initialize variable to track total downgrade sum for the starting XI
    #                     total_downgrade_sum = 0    
    #                     # Iterate through the home team lineup to calculate the sum of downgrades for present players

                        
    #                     for i, lineup_row in df_home_lineup.iterrows():
    #                         st.markdown(f"<div style='display: flex; align-items: center;'>"
    #                                     f"{lineup_row['number']} &nbsp;&nbsp;<strong>{lineup_row['name']}</strong></div>", unsafe_allow_html=True)

    #                         # Check if Player_id exists in team_df and get corresponding Downgrade
    #                         if league_name not in non_player_stat_leagues:
    #                             player_id = lineup_row['Player_id']
    #                             if player_id in team_df['Player_id'].values:
    #                                 downgrade_value = team_df.loc[team_df['Player_id'] == player_id, 'Downgrade'].values[0]
    #                                 total_downgrade_sum += abs(downgrade_value)  # Add the absolute value of Downgrade
    #                         else:
    #                             pass

    #                     if league_name not in non_player_stat_leagues:       
    #                         # Calculate the total absolute downgrades for all players in team_df
    #                         total_team_downgrade_sum = team_df['Downgrade'].abs().sum()
                        
    #                         # Display the total downgrades for the starting XI
    #                         # st.write('Line-up downgrade from FST:', total_downgrade_sum)
    #                         st.write("---")
    #                         # Calculate the difference between total downgrades in the line-up and the total downgrades in the team_df
    #                         downgrade_difference = total_team_downgrade_sum - total_downgrade_sum
    #                         st.write("Downgrade starting XI from Best XI:", -downgrade_difference) 
    #                     else:
    #                         pass



    #                 else:
    #                     df_injuries = get_injuries(fixture_id)  # Call within the expander
                        
    #                     if 'Team Name' in df_injuries.columns:
    #                         df_injuries['Team Name'] = df_injuries['Team Name'].apply(lambda x: unidecode(str(x)))
    #                         df_injuries['Team Name'] = df_injuries['Team Name'].map(team_names_api_to_t1x2_dict).fillna(df_injuries['Team Name'])
    #                         df_home_injuries = df_injuries[df_injuries['Team Name'] == home_team_name]
    #                         # Check if the DataFrame is empty (indicating no match)
    #                         if df_home_injuries.empty:
    #                             # Perform reverse lookup in case of team name change
    #                             reversed_team_name_dict = get_reversed_team_name_dict()
    #                             home_team_name_amended = reversed_team_name_dict.get(home_team_name)
    #                             # Retry filtering with updated team name if the reverse lookup is successful
    #                             if home_team_name_amended:
    #                                 df_home_injuries = df_injuries[df_injuries['Team Name'] == home_team_name_amended]

    #                         # display injuries if found       
    #                         if not df_home_injuries.empty:
    #                             for i, injury_row in df_home_injuries.iterrows():
    #                                 st.markdown(f"<div style='display: flex; align-items: center; font-size: 14px;'>"
    #                                             f"<img src='{injury_row['Player Photo']}' width='30' style='margin-right: 5px;'>"
    #                                             f"<strong>{injury_row['Player Name']}</strong> - {injury_row['Injury Type']} "
    #                                             f"({injury_row['Injury Reason']})</div>", unsafe_allow_html=True)
                                
    #                             if league_name not in non_player_stat_leagues:                                   
    #                                 # Extract total missing players downgrade values and return the sum
    #                                 merged_fst_inj_df_h = pd.merge(df_home_injuries, team_df[['Player_id', 'Downgrade']], on='Player_id', how='left')
    #                                 # Adjust the 'Downgrade' values based on 'Injury Type'
    #                                 merged_fst_inj_df_h['Adjusted_Downgrade'] = merged_fst_inj_df_h.apply(
    #                                 lambda row: row['Downgrade'] if row['Injury Type'] == 'Missing Fixture' else row['Downgrade'] * 0.5 if row['Injury Type'] == 'Questionable' else 0,
    #                                 axis=1)  

    #                                 # sum the adjusted downgrades             
    #                                 downgrade_sum = merged_fst_inj_df_h['Adjusted_Downgrade'].sum()
    #                                 st.write("---")
    #                                 st.write('Estimated line-up downgrade from Best XI:', downgrade_sum)
    #                             else:
    #                                 pass

                                    
    #                         else:
    #                             # case where no injuries found
    #                             st.write("No news")
    #                     else:
    #                         # case where 'Team Name' column missing
    #                         st.write("No news")



    #             st.write('')
    #             st.write('')

    #         # ------------- Show Odds ------------------
    #         # WIDGET
    #         show_odds = st.checkbox(f"Show odds", key=f"{fixture_id}_odds", label_visibility = 'visible')

    #         # Store the market selection in a session state to maintain it between interactions
    #         if 'selected_market' not in st.session_state:
    #             st.session_state.selected_market = None


    #         if show_odds:
    #             # WIDGET
    #             selected_market = st.selectbox("Select market", options=list(markets_dict.keys()), key=f"market_{fixture_id}", label_visibility = 'visible')
    #             st.session_state.selected_market = markets_dict[selected_market]
                

    #         # Placeholder for the odds DataFrame, displayed outside the columns
    #         if show_odds and st.session_state.selected_market:
    #             market_id = st.session_state.selected_market
    #             odds_df = get_odds(fixture_id, market_id, bookmaker_list)
                
    #             if not odds_df.empty:
    #                 st.write("")
    #                 st.dataframe(odds_df)
    #                 st.caption('Odds data updates every 3 hours. Only pre-match odds currently available.')
    #             else:
    #                 st.write("No odds available for this market.")


    #         # -----------  Show past matches --------------
    #         # WIDGET
    #         show_past_matches = st.checkbox('Show past matches/stats', key=f"{fixture_id}_past_matches", label_visibility = 'visible')  # New checkbox for past matches

    #         if show_past_matches:
    #             # Filter past matches for the selected teams
    #             past_matches_df = df_past_fixts[
    #                 ((df_past_fixts['HomeTeam'] == home_team_name) & 
    #                 (df_past_fixts['AwayTeam'] == away_team_name)) |
    #                 ((df_past_fixts['HomeTeam'] == away_team_name) & 
    #                 (df_past_fixts['AwayTeam'] == home_team_name))
    #             ]
                

    #             past_matches_df = past_matches_df[['Date', 'HomeTeam', 'AwayTeam', 'HG', 'AG', 'TG', 'HS', 'AS', 'TS', 'HST', 'AST', 'TST', 'HF', 'AF', 'TF', 'HC', 'AC', 'TC', 'HY', 'AY', 'TY', 'HR', 'AR', 'TR', 'Pin_H_Close', 'Pin_D_Close', 'Pin_A_Close']] 
    #             past_matches_df.reset_index(drop=True, inplace=True)
    #             avg_gls = round(past_matches_df['TG'].mean(), 2)
    #             avg_shots = round(past_matches_df['TS'].mean(), 2)
    #             avg_sot = round(past_matches_df['TST'].mean(), 2)
    #             avg_corn = round(past_matches_df['TC'].mean(), 2)
    #             avg_fls = round(past_matches_df['TF'].mean(), 2)
    #             avg_yel = round(past_matches_df['TY'].mean(), 2)

    #             # TODO HERE - filter selected league from df_past fixts, calculate average for each metric to then compare with specific match metric
    #             # - if big variance, tell user
    #             # st.write(df_past_fixts)
    #             # lg_avg_gls = df_past_fixts['Div']  # reference league_options to get league eg 'E0' from selected_league eg 'England Premier' to filter df_past_fixts


    #             # Display past matches if any
    #             if not past_matches_df.empty:
    #                 st.write("Past Matches:")
    #                 st.dataframe(past_matches_df)  # Display the past matches in a table format
    #                 st.write('**Past Match Stats:**')
    #                 st.write('Average Goals:', avg_gls)
    #                 st.write('Average Total Shots:', avg_shots)
    #                 st.write('Average Shots on Target:', avg_sot)
    #                 st.write('Average Corners:', avg_corn)
    #                 st.write('Average Fouls:', avg_fls)
    #                 st.write('Average Yellows:', avg_yel)

    #             else:
    #                 st.write("No past matches found.")  # Message if no matches are found
    #         # -----------------------------------------------------------------------------


    #         # Right column: Display injuries for away team in an expander
    #         with col5:
    #             st.write("")

    #             # ----------------- show fst -----------
    #             st.write("")
    #             # WIDGET
    #             with st.expander(f"Show {away_team_name} Best XI", expanded=False): 
    #                 if league_name not in non_player_stat_leagues:   
    #                     df_filtered = df[df['Team'] == away_team_name]
    #                     team_df, _ = generate_team_fst(df_filtered, CURRENT_SEASON)
    #                     st.dataframe(team_df, height=420)
    #                 else:
    #                     st.write(f'Best XI unavailable for {league_name}')

    #             # WIDGET
    #             with st.expander(f"{away_team_name} team info", expanded=False):                    
    #                 df_lineups = get_lineups(fixture_id)
    #                 #st.write(df_lineups)
    #                 if 'team' in df_lineups.columns:
    #                     df_away_lineup = df_lineups[df_lineups['team'] == away_team_name]
    #                     if not df_away_lineup.empty:
    #                         st.write('**Starting XI**')
    #                         # Initialize variable to track total downgrade sum for the starting XI
    #                         total_downgrade_sum = 0    

    #                         # Iterate through the home team lineup to calculate the sum of downgrades for present players
    #                         for i, lineup_row in df_away_lineup.iterrows():
    #                             st.markdown(f"<div style='display: flex; align-items: center;'>"
    #                                         f"{lineup_row['number']} &nbsp;&nbsp;<strong>{lineup_row['name']}</strong></div>", unsafe_allow_html=True)

    #                             # Check if Player_id exists in team_df and get corresponding Downgrade
    #                             if league_name not in non_player_stat_leagues:
    #                                 player_id = lineup_row['Player_id']
    #                                 if player_id in team_df['Player_id'].values:
    #                                     downgrade_value = team_df.loc[team_df['Player_id'] == player_id, 'Downgrade'].values[0]
    #                                     total_downgrade_sum += abs(downgrade_value)  # Add the absolute value of Downgrade
    #                             else:
    #                                 pass

    #                         if league_name not in non_player_stat_leagues:
    #                             # Calculate the total absolute downgrades for all players in team_df
    #                             total_team_downgrade_sum = team_df['Downgrade'].abs().sum()
                                
    #                             # Display the total downgrades for the starting XI
    #                             # st.write('Line-up downgrade from FST:', total_downgrade_sum)
    #                             st.write("---")
    #                             # Calculate the difference between total downgrades in the line-up and the total downgrades in the team_df
    #                             downgrade_difference = total_team_downgrade_sum - total_downgrade_sum
    #                             st.write("Downgrade starting XI from Best XI:", -downgrade_difference)
    #                         else:
    #                             pass

    #                 else:
    #                     df_injuries = get_injuries(fixture_id)  # Call within the expander
    #                     if 'Team Name' in df_injuries.columns:

    #                         df_away_injuries = df_injuries[df_injuries['Team Name'] == away_team_name]
    #                         # Check if the DataFrame is empty (indicating no match)
    #                         if df_away_injuries.empty:
    #                             # Perform reverse lookup in case of team name change
    #                             reversed_team_name_dict = get_reversed_team_name_dict()
    #                             away_team_name_amended = reversed_team_name_dict.get(away_team_name)
    #                             # Retry filtering with updated team name if the reverse lookup is successful
    #                             if away_team_name_amended:
    #                                 df_away_injuries = df_injuries[df_injuries['Team Name'] == away_team_name_amended]


    #                         if not df_away_injuries.empty:
    #                             for i, injury_row in df_away_injuries.iterrows():
    #                                 st.markdown(f"<div style='display: flex; align-items: center; font-size: 14px;'>"
    #                                             f"<img src='{injury_row['Player Photo']}' width='30' style='margin-right: 5px;'>"
    #                                             f"<strong>{injury_row['Player Name']}</strong> - {injury_row['Injury Type']} "
    #                                             f"({injury_row['Injury Reason']})</div>", unsafe_allow_html=True)
                                
    #                             if league_name not in non_player_stat_leagues:
    #                                 # Extract total missing players downgrade values and return the sum
    #                                 merged_fst_inj_df_a = pd.merge(df_away_injuries, team_df[['Player_id', 'Downgrade']], on='Player_id', how='left')
    #                                 # Adjust the 'Downgrade' values based on 'Injury Type'
    #                                 merged_fst_inj_df_a['Adjusted_Downgrade'] = merged_fst_inj_df_a.apply(
    #                                 lambda row: row['Downgrade'] if row['Injury Type'] == 'Missing Fixture' else row['Downgrade'] * 0.5 if row['Injury Type'] == 'Questionable' else 0,
    #                                 axis=1)

    #                                 # sum the adjusted downgrades               
    #                                 downgrade_sum = merged_fst_inj_df_a['Adjusted_Downgrade'].sum()
    #                                 st.write("---")
    #                                 st.write('Estimated line-up downgrade from Best XI:', downgrade_sum)
    #                             else:
    #                                 pass

    #                         else:
    #                             # case where no injuries found
    #                             st.write("No news")
    #                     else:
    #                         # case where 'Team Name' column is empty
    #                         st.write("No news")
                
        
    #         st.write("---")
    # else:
    #     st.write("No fixtures found for the selected date range and league.")



# Ensure the main function is called when the script is run directly
if __name__ == "__main__":
    main()