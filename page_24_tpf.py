import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os 
import requests
from unidecode import unidecode
from datetime import datetime, timedelta
from mymodule.functions import team_names_api_to_t1x2_dict, get_fixtures, get_table



# Load environment variables (API key)
load_dotenv()
API_KEY = os.getenv('API_KEY_FOOTBALL-API')
CURRENT_SEASON = '2025-26' # to fetch team FST's
API_SEASON = '2025' # current season as used by api-foot

default_league = 'England Premier'
today = datetime.now()
week_from_today = today + timedelta(days=7)

# Dictionary to map league names to their IDs
leagues_dict = {
    "England Premier": '39',
    "Germany Bundesliga": '78',
    "Spain La Liga": '140',
    "Italy Serie A": '135',
    "France Ligue 1": '61',
    "Netherlands Eredivisie": "88",
    # "Belgium Jupiler": "144",  # not included as too complicated 
    "Portugal Liga I": '94',
    'Scotland Premier': '179',
    'England Championship': '40',
    'England League One': '41',
    'England League Two': '42',
    "Germany 2 Bundesliga": '79',
    'Italy Serie B': '136',
    'Spain La Liga 2': '141',
    'France Ligue 2': '62',
    'Turkey Super Lig': '203',

}

# dictionary showing total MP each team, top positions TPF (eg 7 means top 7 positions achieve something) and bottom positions TPF (eg 18 means position 18 and below are relegation or rel play-offs) 
valid_positions_dict = {
    'England Premier': [38,7,18],
    'Germany Bundesliga': [34,7,16],
    'Spain La Liga': [38,6,18],
    'Italy Serie A': [38,6,18],
    'France Ligue 1': [34,6,16],
    'Netherlands Eredivisie': [34,8,16],
    'Portugal Liga I': [34,4,16],
    'Scotland Premier': [38,4,2],
    'England Championship': [46,8,22],
    'England League One': [46,6,22],    
    'England League Two': [46,6,22],
    'Germany 2 Bundesliga': [34,3,16],
    'Italy Serie B': [38,8,16],
    'Spain La Liga 2': [42,6,18],
    'France Ligue 2': [34,5,16],
    'Turkey Super Lig': [34,4,16],
}



def get_reversed_team_name_dict():
    return {v: k for k, v in team_names_api_to_t1x2_dict.items()}

def main():

    # PUT SELECTIONS IN SESSION STATE SO WHEN PAGE REFRESHED, SELECTIONS DONT NEED TO BE PICKED AGAIN   
    # Initialize default values if they don't exist in session state
    if 'league_name' not in st.session_state:
        st.session_state.league_name = default_league
    if 'from_date' not in st.session_state:
        st.session_state.from_date = today
    if 'to_date' not in st.session_state:
        st.session_state.to_date = week_from_today


    # Sidebar for league selection (WIDGET)
    league_name = st.sidebar.selectbox(
        "Select a League",
        list(leagues_dict.keys()),
        index=list(leagues_dict.keys()).index(st.session_state.league_name),
        label_visibility='visible',
    )

    # Check if the league selection has changed
    if league_name != st.session_state.league_name:
        # Update the session state to reflect the new league name
        st.session_state.league_name = league_name
        
        # Trigger a page refresh (only if the selection has changed)
        st.rerun()


    # Sidebar for 'from_date' input  # WIDGET
    from_date = st.sidebar.date_input(
        "From Date",
        st.session_state.from_date,
        label_visibility = 'visible'
    )

    # Check if the from_date has changed
    if from_date != st.session_state.from_date:
        # Update the session state to reflect the new league name
        st.session_state.from_date = from_date        
        # Trigger a page refresh (only if the selection has changed)
        st.rerun()


    # Calculate max 'to_date' based on 'from_date'
    max_to_date = from_date + timedelta(days=30)

    # Sidebar for 'to_date' input  # WIDGET
    to_date = st.sidebar.date_input(
        "To Date",
        st.session_state.to_date,
        max_value=max_to_date,
        label_visibility = 'visible'
    )

    # Check if the to_date has changed
    if to_date != st.session_state.to_date:
        # Update the session state to reflect the new league name
        st.session_state.to_date = to_date        
        # Trigger a page refresh (only if the selection has changed)
        st.rerun()



    # Update from_date_str and to_date_str after confirming changes
    from_date_str = st.session_state.from_date.strftime('%Y-%m-%d')
    to_date_str = st.session_state.to_date.strftime('%Y-%m-%d')



    # --------------------  MAIN PAGE  ---------------------------

    st.header(f"{st.session_state.league_name} - End of Season TPF", divider="blue")


    # Automatically fetch fixtures on page load

    league_id = leagues_dict[st.session_state.league_name]
    df_fixtures = get_fixtures(league_id, from_date_str, to_date_str, API_SEASON)
    # st.write('149', df_fixtures)
    if not df_fixtures.empty:
        df_fixtures['Date'] = pd.to_datetime(df_fixtures['Date'], format='%d-%m-%y %H:%M')  # Convert to datetime
        df_fixtures.sort_values(by='Date', inplace=True)  # Sort by date
        df_fixtures = df_fixtures[['Date', 'Home Team', 'Away Team']]  # Reorder columns

        # st.write('155', df_fixtures)

    else:
        st.write("No fixtures found for the selected date range and league.")
        st.stop()  # Stop execution if no fixtures are found

    
    # --------  Pull in league table ------------

    # Is it a split table league?
    table_chosen = 'main' #set all leagues as this default, but if it is a split league, then user can choose which table to use for the TPF classification (eg in Scotland, Championship or Relegation table)
    split_leagues = ['Scotland Premier', 'Denmark Superliga']  # example split leagues, need to confirm which leagues split and how to identify in API
    if st.session_state.league_name in split_leagues:
        st.error(f"{st.session_state.league_name} has a split league format - caution with outputs.")
        st.write("---")
        table_chosen = st.selectbox("Select which part of the split league table to use for TPF classification", ['Championship', 'Relegation'])
        if table_chosen == 'Championship':
            api_split_param = 1
        else:
            api_split_param = 2


  
    # function to access the api for the split leagues only
    def get_split_league_table(league, year):

        # Load environment variables (API key)
        load_dotenv()
        API_KEY = os.getenv('API_KEY_FOOTBALL_API')

        url = "https://api-football-v1.p.rapidapi.com/v3/standings"

        querystring = {
            "league": league,
            "season": year
        }

        headers = {
            "x-rapidapi-key": API_KEY,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code != 200:
            st.error("API request failed.")
            return pd.DataFrame()  # or handle it as needed
        response_data = response.json()

        # Extract standings
        standings = response_data['response'][0]['league']['standings'][api_split_param]    # api_split_param

        # Transform into DataFrame with all relevant columns
        data = []
        for team_info in standings:
            team = team_info['team']['name']
        # rank = team_info['rank']
        #  logo = team_info['team']['logo']
            points = team_info['points']
            goals_for = team_info['all']['goals']['for']
            goals_against = team_info['all']['goals']['against']
            played = team_info['all']['played']
            wins = team_info['all']['win']
            draws = team_info['all']['draw']
            losses = team_info['all']['lose']
            goals_diff = team_info['goalsDiff']
            form = team_info['form']

            data.append({
            #   'Badge': logo,
                'Team': team,
                'P': played,
                'W': wins,
                'D': draws,
                'L': losses,
                'GF': goals_for,
                'GA': goals_against,
                'GD': goals_diff,
                'Pts': points,
                'Form': form,
            })

        # Create DataFrame
        standings_df = pd.DataFrame(data)

        # Reset index starting from 1 to look like a proper league table
        standings_df.index = standings_df.index + 1
        # convert team name to 1x2 standardized

        if not standings_df.empty:
            standings_df['Team'] = standings_df['Team'].apply(unidecode)
            standings_df['Team'] = standings_df['Team'].map(team_names_api_to_t1x2_dict).fillna(standings_df['Team'])

        return standings_df
    

    # if it is a split league then call tkat api call function, else call normal get_table function
    
    if st.session_state.league_name in split_leagues:
        league_table_df = get_split_league_table(leagues_dict[st.session_state.league_name], API_SEASON[:4])
    else:
        league_table_df = get_table(leagues_dict[st.session_state.league_name], API_SEASON[:4])

    # st.write(league_table_df)

    league_table_df = league_table_df[['Team', 'P', 'GD','Pts']]  # Select relevant columns
    league_table_df['Total Matches'] = valid_positions_dict[st.session_state.league_name][0]  # Add total matches column based on league
    league_table_df['To Play'] =   league_table_df['Total Matches'] - league_table_df['P']  # Placeholder for TPF/NTPF classification
    league_table_df['max_points_to_win'] = league_table_df['To Play'] * 3  # Placeholder for max points available
    league_table_df['max_points_total'] = league_table_df.Pts + league_table_df['max_points_to_win']  # Placeholder for max total points possible
    league_table_df['highest_position'] = league_table_df['max_points_total'].rank(method='min', ascending=False)  


    # --- Calculate MINIMUM end of season finishing position ---
    pts = league_table_df['Pts'].values
    max_pts = league_table_df['max_points_total'].values

    lowest_positions = []

    for i in range(len(pts)):
        hypothetical = max_pts.copy()
        hypothetical[i] = pts[i]

        # WORST CASE ranking (ties pushed DOWN)
        temp_rank = pd.Series(hypothetical).rank(
            ascending=False,
            method='max'   
        )

        lowest_positions.append(int(temp_rank[i]))

    league_table_df['lowest_position'] = lowest_positions


    # --- Calculate MAXIMUM end of season finishing position ---

    pts = league_table_df['Pts'].values
    max_pts = league_table_df['max_points_total'].values

    max_positions = []

    for i in range(len(pts)):
        # scenario: this team maximises, others stay at current points
        hypothetical = pts.copy()
        
        # this team reaches max possible
        hypothetical[i] = max_pts[i]
        
        # rank
        rank = 1 + np.sum(hypothetical > max_pts[i])
        
        max_positions.append(rank)

    league_table_df['highest_position'] = max_positions

    # set value defaults
    bottom_valid_positions = valid_positions_dict[st.session_state.league_name][2]
    top_valid_positions = valid_positions_dict[st.session_state.league_name][1]

    # ---- Create 'Positions that matter' column based on valid_positions_dict for the league

    if league_name is 'Scotland Premier':  
        if table_chosen == 'Championship':
            top_valid_positions = 4
            bottom_valid_positions = 11 # only 6 teams
        elif table_chosen == 'Relegation': # relegation group
            top_valid_positions = 4 # all teams 10 and above are safe, only position 11 is relegated       
            bottom_valid_positions = 5 # 11 rel play-off, 12 relegated

    # elif league_name is 'Denmark Superliga':
    #     if table_chosen == 'Championship':
    #         top_valid_positions = 3
    #         bottom_valid_positions = 7   # only 6 teams
    #     else: # relegation group
    #         top_valid_positions = 7 # 7th      
    #         bottom_valid_positions = 12 # 11 rel play-off, 12 relegated

            
    if st.session_state.league_name not in split_leagues:
        st.markdown(
            f"""
            Positions that matter for {st.session_state.league_name}: 
            <span style='color:#00BFFF; font-weight:bold;'>Top {top_valid_positions}</span> 
            and 
            <span style='color:#FF4B4B; font-weight:bold;'>{bottom_valid_positions} or below.</span>
            """,
            unsafe_allow_html=True
        )

    # facility for user to change inputs - disabled for split leagues as max-value param causes probs
    if league_name not in split_leagues:
        change_positions = st.checkbox(f"Tick to change 'positions that matter' values for {st.session_state.league_name}", key='show_classification_columns')

   
        if change_positions:
            with st.container(border=True):
                top_valid_positions = st.number_input("Top valid positions (eg 6 means top 6 positions achieve something)", min_value=1, max_value=league_table_df.shape[0], value=top_valid_positions, key='top_valid_positions_input')
                bottom_valid_positions = st.number_input("Bottom valid positions (eg 18 means position 18 and below are relegation or rel play-offs)", min_value=1, max_value=league_table_df.shape[0], value=bottom_valid_positions, key='bottom_valid_positions_input')   

    st.write("")


    # --- Create 'Midtable NTPF' column ---
    if table_chosen != 'Championship' and table_chosen != 'Relegation':  # in top half of split leagues, midtable ntpf is only valid for teams that cannot break into the top valid positions 
        league_table_df['Midtable NTPF'] = league_table_df.apply(lambda row: 'NTPF' if (row['highest_position'] > top_valid_positions and row['lowest_position'] < bottom_valid_positions) else 'TPF', axis=1)
    elif table_chosen == 'Championship':
        league_table_df['Midtable NTPF'] = league_table_df.apply(lambda row: 'NTPF' if row['highest_position'] > top_valid_positions else 'TPF', axis=1)
    elif table_chosen == 'Relegation':
        league_table_df['test'] = league_table_df['lowest_position'] - bottom_valid_positions
        league_table_df['Midtable NTPF'] = league_table_df.apply(lambda row: 'NTPF' if row['lowest_position'] < bottom_valid_positions else 'TPF', axis=1)
        st.write(bottom_valid_positions)

    # st.write(bottom_valid_positions)
    # st.write(top_valid_positions)
    # st.write(table_chosen)

    # --- Create 'Rel TPF' column (is highest possible position still in the relegation zone) ---

    if table_chosen != 'Championship':  # in the split league 'Championship' half, there is no relegation, therefore the bottom teams cannot be stuck as already relegated if adrift thus keep everyone as tpf
        league_table_df['Rel NTPF'] = league_table_df.apply(lambda row: 'NTPF' if (row['highest_position'] >= bottom_valid_positions) else 'TPF', axis=1)
    else:
        league_table_df['Rel NTPF'] = 'TPF' 


    # --- Create 'Can Move Position' column (if highest_position is the same as current position, then team cannot move position) ---

    league_table_df["Cant move position"] = league_table_df.apply(lambda row: 'NTPF' if (row['highest_position'] == row['lowest_position']) else 'TPF', axis=1)


    # If any of the columns are NTPF then that team is NTPF, else TPF
    league_table_df['Final Classification'] = league_table_df.apply(lambda row: 'NTPF' if (row['Midtable NTPF'] == 'NTPF' or row['Rel NTPF'] == 'NTPF' or row['Cant move position'] == 'NTPF') else 'TPF', axis=1)


    st.subheader("League Table with TPF Classification")
    st.write(league_table_df)
    st.caption("" \
    "'Midtable NTPF' is if the team cannot reach a position that matters at the top, and cannot fall to a position that matters at the bottom. " \
    "'Rel NTPF' is if the team is already relegated. 'Can't Move Position NTPF' is if the team cannot move to a different position (eg. already won league). "   \
        "'Final Classification' is if any of the criteria are NTPF, then the team is NTPF. ")

    # st.write('388', df_fixtures)

    # Now merge Final Classification of each team in league_table_df with the Home Team and Away Team columns of df_fixtures
    classification_map = league_table_df.set_index('Team')['Final Classification']
    df_fixtures['Home tpf'] = df_fixtures['Home Team'].map(classification_map)
    df_fixtures['Away tpf'] = df_fixtures['Away Team'].map(classification_map)

    # st.write('394', df_fixtures)


    # If either both columns Home tpf and Away tpf are NTPF then that match is a NTPF match, else TPF match
    df_fixtures['MATCH TPF'] = df_fixtures.apply(lambda row: 'NTPF' if (row['Home tpf'] == 'NTPF' and row['Away tpf'] == 'NTPF') else 'ok', axis=1)

    st.subheader("Fixtures with TPF Classification")
    # if any rows are None, remove them from the df
    df_fixtures = df_fixtures.dropna(subset=['Home tpf', 'Away tpf'])

    if league_table_df['To Play'].mean() == 0:
        st.write("No more matches to play in regular season")
    else:
        st.write(df_fixtures)
        st.caption("Classification is based on mathematical certainty - goal difference is not factored in, so if a team can still mathematically reach a position that matters, they are classified as TPF, even if realistically they may not be able to catch up on GD.")



if __name__ == '__main__':
    main()