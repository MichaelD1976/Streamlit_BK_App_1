import streamlit as st
import pandas as pd
# import altair as alt
import numpy as np
from datetime import datetime, timedelta
import time
from scipy.optimize import minimize_scalar
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom
# from sklearn.preprocessing import PolynomialFeatures
import joblib
from mymodule.functions import get_fixtures,  calculate_home_away_lines_and_odds, calculate_totals_lines_and_odds, poisson_probabilities, calculate_true_from_true_raw, team_names_t1x2_to_BK_dict, generate_marginated_odds_with_fav_lock, calculate_expected_team_goals_from_1x2_refined, calculate_probability_grid_h_exp_vs_a_exp
import requests
import os
from dotenv import load_dotenv
import gc


dict_api_to_bk_league_names = {
     'England Premier':'England Premier League',
     'Spain La Liga' : 'Spain LaLiga',
 }

CURRENT_SEASON = '2026-27'
LAST_SEASON = '2025-26'
# TOTALS_BOOST = 1.02 # increase daily totals by this    DELETE

shots_model_h = joblib.load('models/shots/hs_linear_model.joblib')
shots_model_a = joblib.load('models/shots/as_linear_model.joblib')

# key = current gw, value is perc of last season
game_week_decay_dict = {
    1: 1,
    2: 0.95,
    3: 0.88,
    4: 0.78,
    5: 0.68,
    6: 0.58,
    7: 0.48,
    8: 0.40,
    9: 0.33,
    10: 0.27,
    11: 0.22,
    12: 0.18,
    13: 0.15,
    14: 0.13,
    15: 0.11,
    16: 0.09,
    17: 0.07,
    18: 0.06,
    19: 0.05,
    20: 0.04,
    21: 0.03,
    22: 0.02,
    23: 0.02
}

# ------------- Load the CSV file -----------------
@st.cache_data
def load_data():
    time.sleep(2)
    df = pd.read_csv('data/outputs_processed/teams/api-football_master_teams.csv')
    df_prom_rel = pd.read_csv('data/prom_rel.csv')
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    return df, df_prom_rel


# -------------------------------------------

def main():
    with st.spinner('Loading Data...'):
        df, df_prom_rel = load_data()

    if df.empty:
        st.write("No data available to display.")
        return

    # Sidebar for user input
    st.sidebar.title('Select Data Filters')

    # Define selection options
    league_options = {
        # 'All_leagues': 'ALL',  # Uncomment for future development
        'Premier League': 'England Premier',
        'Bundesliga': 'Germany Bundesliga',
        'La Liga': 'Spain La Liga',
        'Serie A': 'Italy Serie A',
        'Ligue 1': 'France Ligue 1',
        'Eredivisie': 'Netherlands Eredivisie',
        'Jupiler Pro League': 'Belgium Jupiler',
        'Primeira Liga': 'Portugal Liga I',
        'Premiership': 'Scotland Premier',
        'Championship': 'England Championship',
        'League One': 'England League One',
        'League Two': 'England League Two',
        '2. Bundesliga': 'Germany 2 Bundesliga',
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

    metric_options = {
        'Shots': ['HS', 'AS', 'TS'],
    }


    # Capture user selections
    # WIDGET
    selected_league = st.sidebar.selectbox('Select League', options=list(league_options.values()), label_visibility = 'visible')
    # selected_metric = st.sidebar.selectbox('Select Metric', options=list(metric_options.keys()))
    selected_metric = 'Shots'

    df = df[df['League'] == [key for key, value in league_options.items() if value == selected_league][0]]           


    this_df = df[(df['Season'] == CURRENT_SEASON)]  # remove all matches that are not current season
    last_df = df[(df['Season'] == LAST_SEASON)] 

    del df
    gc.collect()


    # -----------------------------------------------------------------------

    st.header(f'{selected_metric} Model - {selected_league}', divider='blue')

    # WIDGET
    show_model_info = st.checkbox('Model Information', label_visibility = 'visible')
    if show_model_info:
        st.caption('''
                 Shots ML model is trained on a pre-processed dataset of > 60k European domestic matches.
                 It is a standard linear regression model with tail-end data-derived calibration adjustments to capture the non-linear relationships. 
                 All outputs are generalised estimates only and are not league, country or team specific. R2 is approx 0.22.
                 ''')

       # get fixtures
    league_id = leagues_dict.get(selected_league)

    # st.write(this_df)
    ssn_avg_this = round(this_df['TS'].mean(), 2)
    ssn_avg_last = round(last_df['TS'].mean(), 2)


    #  -----------  create df with just teams, MP and metric options - CURRENT SEASON  ------------------------

    unique_teams_this = pd.concat([this_df['HomeTeam'], this_df['AwayTeam']]).unique()
    this_options_df= pd.DataFrame(unique_teams_this, columns=['Team'])
    metric_columns = metric_options[selected_metric]  
    MP = []
    H_f = []   # Home For: Average of metric_options[0] when team is HomeTeam
    H_ag = []  # Home Against: Average of metric_options[1] when team is HomeTeam
    A_f = []   # Away For: Average of metric_options[1] when team is AwayTeam
    A_ag = []  # Away Against: Average of metric_options[0] when team is AwayTeam
    # Calculate averages for each team and store in respective lists
    for team in unique_teams_this:
        # Filter rows for each team as HomeTeam and AwayTeam
        home_matches = this_df[this_df['HomeTeam'] == team]
        away_matches = this_df[this_df['AwayTeam'] == team]
        matches_played = len(home_matches) + len(away_matches)
        MP.append(matches_played)

        # Calculate averages for each metric based on team position (Home/Away)
        H_f_avg = home_matches[metric_columns[0]].mean()  # Home For (metric_options[0] as HomeTeam)
        H_ag_avg = home_matches[metric_columns[1]].mean() # Home Against (metric_options[1] as HomeTeam)
        A_f_avg = away_matches[metric_columns[1]].mean()  # Away For (metric_options[1] as AwayTeam)
        A_ag_avg = away_matches[metric_columns[0]].mean() # Away Against (metric_options[0] as AwayTeam)

        # Append the results to the lists
        H_f.append(H_f_avg)
        H_ag.append(H_ag_avg)
        A_f.append(A_f_avg)
        A_ag.append(A_ag_avg)

    this_options_df['MP'] = MP
    this_options_df['H_for'] = H_f
    this_options_df['H_ag'] = H_ag
    this_options_df['A_for'] = A_f
    this_options_df['A_ag'] = A_ag

    # if df is empty or less than 2 matches played in current season stop script
    if this_options_df.empty or this_options_df['MP'].mean() < 2:
        st.write(f"{selected_league} currently unavailable")
        st.stop()

    # Display the resulting DataFrame
    # WIDGET
    show_this_ssn_stats = st.checkbox(f'Show average current season {selected_metric} stats', label_visibility = 'visible')
    if show_this_ssn_stats:
        st.write(this_options_df)
        st.write('Current season avg per match:', ssn_avg_this)


    # ---- LAST SEASON ------------------

    unique_teams_last = pd.concat([last_df['HomeTeam'], last_df['AwayTeam']]).unique()
    last_options_df = pd.DataFrame(unique_teams_last, columns=['Team'])
    metric_columns = metric_options[selected_metric]  # Assuming metric_options is a dictionary as shown above
    MP = []
    H_f = []   # Home For: Average of metric_options[0] when team is HomeTeam
    H_ag = []  # Home Against: Average of metric_options[1] when team is HomeTeam
    A_f = []   # Away For: Average of metric_options[1] when team is AwayTeam
    A_ag = []  # Away Against: Average of metric_options[0] when team is AwayTeam
    # Calculate averages for each team and store in respective lists
    for team in unique_teams_last:
        # Filter rows for each team as HomeTeam and AwayTeam
        home_matches = last_df[last_df['HomeTeam'] == team]
        away_matches = last_df[last_df['AwayTeam'] == team]
        matches_played = len(home_matches) + len(away_matches)
        MP.append(matches_played)
        # Calculate averages for each metric based on team position (Home/Away)
        H_f_avg = home_matches[metric_columns[0]].mean()  # Home For (metric_options[0] as HomeTeam)
        H_ag_avg = home_matches[metric_columns[1]].mean() # Home Against (metric_options[1] as HomeTeam)
        A_f_avg = away_matches[metric_columns[1]].mean()  # Away For (metric_options[1] as AwayTeam)
        A_ag_avg = away_matches[metric_columns[0]].mean() # Away Against (metric_options[0] as AwayTeam)
        # Append the results to the lists
        H_f.append(H_f_avg)
        H_ag.append(H_ag_avg)
        A_f.append(A_f_avg)
        A_ag.append(A_ag_avg)
    # 6. Assign the calculated averages to new columns in df_mix
    last_options_df['MP'] = MP
    last_options_df['H_for'] = H_f
    last_options_df['H_ag'] = H_ag
    last_options_df['A_for'] = A_f
    last_options_df['A_ag'] = A_ag

    del last_df
    gc.collect()

    # Display last season DataFrame
    # WIDGET
    show_last_ssn_stats = st.checkbox(f'Show average last season {selected_metric} stats', label_visibility = 'visible')
    if show_last_ssn_stats:
        st.write(last_options_df)
        st.write('Last season avg per match:', ssn_avg_last)

    # ---------  Combine this and last based on current week in the season --------------------

    current_gw = int(this_options_df['MP'].mean())
    perc_last_ssn = game_week_decay_dict.get(current_gw, 0)
    perc_this_ssn = 1 - perc_last_ssn
    # st.write('perc_last_season:', perc_last_ssn)

    # -------- Identify new teams in the league, ascertain whether prom or rel in, generates upper or lower quantile average --------------

    # st.write(df_prom_rel)
    # st.write(df_prom_rel.dtypes)

    # Step 1: Identify missing teams in last season's table compared to this season's table
    missing_teams = this_options_df[~this_options_df['Team'].isin(last_options_df['Team'])]['Team'].unique()
    # teams_to_remove = last_options_df[~last_options_df['Team'].isin(this_options_df['Team'])]['Team'].unique()

    #st.write(teams_to_remove)

    # Convert columns to strings if they are not already strings, to handle any non-string entries gracefully
    df_prom_rel['promoted_in'] = df_prom_rel['promoted_in'].astype(str)
    df_prom_rel['relegated_in'] = df_prom_rel['relegated_in'].astype(str)

    # Step 2: Process df_prom_rel to get promoted and relegated teams
    # Split promoted and relegated teams from df_prom_rel into lists for each league row
    df_prom_rel['promoted_in'] = df_prom_rel['promoted_in'].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])
    df_prom_rel['relegated_in'] = df_prom_rel['relegated_in'].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])

    # Initialize lists to store results for promoted and relegated teams
    promoted_teams = []
    relegated_teams = []
    # st.write(df_prom_rel['promoted_in'])

    # Loop through each league's row in df_prom_rel to collect promoted and relegated teams
    for _, row in df_prom_rel.iterrows():
        # Directly use lists without re-splitting
        promoted_in = row['promoted_in']
        relegated_in = row['relegated_in']

        # # Add debugging output to confirm the structure
        # st.write(f"Promoted teams in row: {promoted_in}")
        # st.write(f"Relegated teams in row: {relegated_in}")

        # Append teams to the main lists if they are lists
        if isinstance(promoted_in, list):
            promoted_teams.extend(promoted_in)
        if isinstance(relegated_in, list):
            relegated_teams.extend(relegated_in)

    # Clean up team names by stripping any leading/trailing whitespace
    promoted_teams = [team.strip() for team in promoted_teams if team.strip()]
    relegated_teams = [team.strip() for team in relegated_teams if team.strip()]
    # st.write(promoted_teams)

    # Step 3: Calculate quantiles for relevant columns in this_options_df
    H_for_3rd_quantile = last_options_df['H_for'].quantile(0.25)
    A_for_3rd_quantile = last_options_df['A_for'].quantile(0.25)
    H_ag_1st_quantile = last_options_df['H_ag'].quantile(0.75)
    A_ag_1st_quantile = last_options_df['A_ag'].quantile(0.75)

    H_for_1st_quantile = last_options_df['H_for'].quantile(0.75)
    A_for_1st_quantile = last_options_df['A_for'].quantile(0.75)
    H_ag_3rd_quantile = last_options_df['H_ag'].quantile(0.25)
    A_ag_3rd_quantile = last_options_df['A_ag'].quantile(0.25)

    # Step 4: Define function to assign quantile values based on team status
    def get_team_row(team):
        if team in promoted_teams:
            return {
                'Team': team,
                'MP': last_options_df['MP'].max(),
                'H_for': H_for_3rd_quantile,
                'A_for': A_for_3rd_quantile,
                'H_ag': H_ag_1st_quantile,
                'A_ag': A_ag_1st_quantile
            }
        elif team in relegated_teams:
            return {
                'Team': team,
                'MP': last_options_df['MP'].max(),
                'H_for': H_for_1st_quantile,
                'A_for': A_for_1st_quantile,
                'H_ag': H_ag_3rd_quantile,
                'A_ag': A_ag_3rd_quantile
            }
        else:
            print(f"Warning: {team} not found in either promoted or relegated lists.")
            return None

    # Step 5: Generate new rows for missing teams based on promotion/relegation status
    new_rows = [get_team_row(team) for team in missing_teams]
    new_rows = [row for row in new_rows if row is not None]  # Remove any None values from teams not found

    # Convert new_rows to a DataFrame
    new_rows_df = pd.DataFrame(new_rows)

    # Step 6: Append new rows to last_options_df
    last_options_df_2 = pd.concat([last_options_df, new_rows_df], ignore_index=True)

    # Remove rows where the 'Team' column is in the list 'teams_to_remove'
    # last_options_df_3 = last_options_df_2[~last_options_df_2['Team'].isin(teams_to_remove)].reset_index(drop=True)

    # st.write(last_options_df_2)
    # st.write(last_options_df_3)
    # st.write(this_options_df)

   # --------------------------------------------------------------------------

    df_mix_1 = pd.merge(this_options_df, last_options_df_2, on=['Team'])
    df_mix_1['H_for'] = round((df_mix_1['H_for_x'] * perc_this_ssn) + (df_mix_1['H_for_y'] * perc_last_ssn), 2)
    df_mix_1['H_ag'] = round((df_mix_1['H_ag_x'] * perc_this_ssn) + (df_mix_1['H_ag_y'] * perc_last_ssn), 2)
    df_mix_1['A_for'] = round((df_mix_1['A_for_x'] * perc_this_ssn) + (df_mix_1['A_for_y'] * perc_last_ssn), 2)
    df_mix_1['A_ag'] = round((df_mix_1['A_ag_x'] * perc_this_ssn) + (df_mix_1['A_ag_y'] * perc_last_ssn), 2)

    df_mix = df_mix_1[['Team', 'H_for', 'H_ag', 'A_for', 'A_ag']]

    # WIDGET
    show_df_mix = st.checkbox(f'Show combined team {selected_metric} stats (weighted current & previous season)', label_visibility = 'visible')
    if show_df_mix:
        st.write(df_mix)
        st.caption('''
                 Current season and previous season statistics are merged based on a weighting of number of games through the current season.
                 Previous season data decays logarithmically from 100% at game 1 to 0 % by game 24. Teams new to a division are allocated
                 an initial defaulted previous season 1st or 3rd league quantile value (depending if promoted or relegated in), so predictions for those teams may be less reliable early season.
                 ''')


    # -------------------------------------------- CREATE ODDS FOR ALL UPCOMING FIXTURES --------------------------------------------------------------------


    # =============  REQUIREMENTS  ================================

    df_ou = pd.read_csv('data/over_under_exp_conversion.csv')
    df_dnb = pd.read_csv('data/dnb_sup_conversion.csv')
    df_dnb.drop(['dnb price'], axis=1, inplace=True)
    df_ou.drop(['Exp', 'Under', 'Over', 'Un2.5_%'], axis=1, inplace=True)
    df_ou.drop_duplicates(subset='Ov2.5_%', inplace=True)

    # Multiply initial modelled predictions (home_predictions_raw) by these factors for final home_prediction (SEE project 'shots_model_aug_26') 
    # KEY: initial prediction, VALUE: factor to multiply that initial prediction
    hs_calibration_dict = {  
        "<9": -0.01,  
        "9-11": 0.00,  
        "11-13": 0.01,  
        "13-15": 0.01,  
        "15-17": 0.01,   
        "17+": 0.015,  
    }  

    as_calibration_dict = {  
        "<9": 0.00,  
        "9-11": 0.00,  
        "11-13": 0.00,  
        "13-15": 0.00,  
        "15-17": 0.01,   
        "17+": 0.03,  
    } 

    def get_calibration_key(prediction, calibration_dict):
        if prediction < 9:
            key = "<9"
        elif prediction < 11:
            key = "9-11"
        elif prediction < 13:
            key = "11-13"
        elif prediction < 15:
            key = "13-15"
        elif prediction < 17:
            key = "15-17"
        else:
            key = "17+"

        return key

    # ==========================================================


    st.write("---")
    st.subheader(f'Generate odds for all upcoming {selected_league} matches (up to 7 days ahead)')

    column1,column2 = st.columns([1,2])

    with column1:
        margin_to_apply = st.number_input('Margin to apply:', step=0.01, value = 1.09, min_value=1.01, max_value=1.2, key='margin_to_apply')
        bias_to_apply = st.number_input('Overs bias to apply (reduce overs & increase unders odds by a set %):', step=0.01, value = 1.05, min_value=0.95, max_value=1.1, key='bias_to_apply')
        overs_boost = st.number_input('Overs boost to apply to initial home & away predictions:', step=0.01, value = 1.02, min_value=1.00, max_value=1.05, key='overs_boost')
        # is_bst = st.toggle('Set time outputs if BST(-1hr). Unselected = UTC', value=True)   # Use if FMH upload config is added

    with column2:
        # GET FIXTURES UP TO DATE
        today = datetime.now()
        max_up_to_date = today + timedelta(days=7)
        up_to_date = st.date_input(
            "To Date - return fixtures up to and including selected date (defaulted to 7 days from today)",
            max_value = max_up_to_date,
            value = max_up_to_date,
            label_visibility = 'visible'
        )

    generate_odds_all_matches = st.button(f'Click to generate')

    if generate_odds_all_matches:
        with st.spinner("Odds being compiled..."):
            try:

                # GET FIXTURES WEEK AHEAD
                from_date_str = today.strftime("%Y-%m-%d")
                to_date_str = up_to_date.strftime("%Y-%m-%d")
                MARKET_IDS = ['1', '5']             # WDW & Ov/Un
                BOOKMAKERS = ['4']                  # Pinnacle = 4, 365 = 8
                API_SEASON = CURRENT_SEASON[:4]


                df_fixtures = get_fixtures(league_id, from_date_str, to_date_str, API_SEASON)

                if df_fixtures.empty:
                    st.write("No data returned for the specified league and date range.")
                else:
                    # Proceed with the next steps if data is available
                    df_fixts = df_fixtures[['Fixture ID', 'Date', 'Home Team', 'Away Team']]
                    fixt_id_list = list(df_fixts['Fixture ID'].unique())
                    # st.write('Fixtures returned:', len(fixt_id_list)) 
                    # st.write(df_fixts)

                    if not st.secrets:
                        load_dotenv()
                        API_KEY = os.getenv('API_KEY_FOOTBALL_API')

                    else:
                        # Use Streamlit secrets in production
                        API_KEY = st.secrets["rapidapi"]["API_KEY_FOOTBALL_API"]

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
                            odds_dict = {
                                'Fixture ID': fixture_id,
                                'Home Win': None,
                                'Draw': None,
                                'Away Win': None,
                                'Over 2.5': None,
                                'Under 2.5': None,
                                'Over 3.5': None,
                                'Under 3.5': None
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
                                                elif selection == 'Over 2.5':
                                                    odds_dict['Over 2.5'] = odd
                                                elif selection == 'Under 2.5':
                                                    odds_dict['Under 2.5'] = odd
                                                elif selection == 'Over 3.5':
                                                    odds_dict['Over 3.5'] = odd
                                                elif selection == 'Under 3.5':
                                                    odds_dict['Under 3.5'] = odd

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
                            # st.write(odds_df) 
                            if not odds_df.empty:
                                all_odds_df = pd.concat([all_odds_df, odds_df], ignore_index=True)

                            del odds_df
                            gc.collect()

                    # Display the collected odds
                    # st.write('554',all_odds_df)

                    # Use groupby and fillna to collapse rows and remove None values
                    df_collapsed = all_odds_df.groupby('Fixture ID').first().combine_first(
                        all_odds_df.groupby('Fixture ID').last()).reset_index()

                    ########### FILL ANY NONE VALUE ROWS in Over/Under 2.5 Goals columns based on values in the O/U 3.5 columns ###########

                    # first make relevant columns numeric
                    for col in ["Over 3.5", "Under 3.5", "Over 2.5", "Under 2.5"]:
                        df_collapsed[col] = pd.to_numeric(df_collapsed[col], errors="coerce")

                    # function to generated implied ou2.5 FROM ou3.5    
                    def implied_ou_line(o_odds, u_odds, source_line=3.5, target_line=2.5):
                        """Infer O/U target_line odds given O/U source_line odds using a Poisson model."""
                        if pd.isna(o_odds) or pd.isna(u_odds):
                            return None, None

                        # Step 1: Convert odds to normalized probabilities
                        raw_probs = np.array([1/o_odds, 1/u_odds])
                        norm_probs = raw_probs / raw_probs.sum()
                        p_over_source = norm_probs[0]

                        # Step 2: Solve for lambda using the source line
                        def objective(lmbda):
                            p_model = 1 - poisson.cdf(int(source_line), lmbda)
                            return (p_model - p_over_source) ** 2

                        res = minimize_scalar(objective, bounds=(0.2, 6), method="bounded")
                        lam = res.x

                        # Step 3: Compute probabilities at target line
                        p_over_target = 1 - poisson.cdf(int(target_line), lam)
                        p_under_target = 1 - p_over_target

                        return 1/p_over_target, 1/p_under_target
                    
                    # Apply above function to each row which might be missing the ou2.5 values
                    def fill_missing_ou25(df):
                        for i, row in df.iterrows():
                            if pd.isna(row["Over 2.5"]) or pd.isna(row["Under 2.5"]):
                                o25, u25 = implied_ou_line(row["Over 3.5"], row["Under 3.5"],
                                                        source_line=3.5, target_line=2.5)
                                df.at[i, "Over 2.5"] = o25
                                df.at[i, "Under 2.5"] = u25
                        return df
                    
                    df_collapsed = fill_missing_ou25(df_collapsed)

                    ###########################################################################################

                    # st.write('605', df_collapsed)

                    # Merge odds df_fixts with df_collapsed
                    df = df_fixts.merge(df_collapsed, on='Fixture ID')

                    del df_collapsed
                    gc.collect()

                    #  ---------------  Create true wdw odds ---------------
                    # Convert columns to numeric (if they are strings or objects)
                    df['Home Win'] = pd.to_numeric(df['Home Win'], errors='coerce')
                    df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce')
                    df['Away Win'] = pd.to_numeric(df['Away Win'], errors='coerce')

                    df['O_2.5'] = pd.to_numeric(df['Over 2.5'], errors='coerce')
                    df['U_2.5'] = pd.to_numeric(df['Under 2.5'], errors='coerce')


                    df['margin_wdw'] = 1/df['Home Win'] + 1/df['Draw'] + 1/df['Away Win']
                    df['margin_ou'] = 1/df['O_2.5'] + 1/df['U_2.5']


                    df['h_pc_true_raw'] = (1 / df['Home Win']) / df['margin_wdw']
                    df['d_pc_true_raw'] = (1 / df['Draw']) / df['margin_wdw'] 
                    df['a_pc_true_raw'] = (1 / df['Away Win']) / df['margin_wdw'] 

                    df['ov_pc_true'] = round((1 / df['O_2.5']) / df['margin_ou'], 2)
                    df['un_pc_true'] = round((1 / df['U_2.5']) / df['margin_ou'], 2)

                    df[['h_pc_true', 'd_pc_true', 'a_pc_true']] = df.apply(
                        lambda row: calculate_true_from_true_raw(row['h_pc_true_raw'], row['d_pc_true_raw'], row['a_pc_true_raw'], row['margin_wdw']), 
                        axis=1, result_type='expand')
                    


                    # ------------------  Incorporate into the df stats from df_mix ------------------
                    # Merge for the Home Team
                    df = df.merge(df_mix[['Team', 'H_for', 'H_ag', 'A_for', 'A_ag']], 
                                left_on='Home Team', right_on='Team', 
                                how='left', suffixes=('', '_Home'))

                    # Merge for the Away Team
                    df = df.merge(df_mix[['Team', 'H_for', 'H_ag', 'A_for', 'A_ag']], 
                                left_on='Away Team', right_on='Team', 
                                how='left', suffixes=('', '_Away'))

                    # Drop the extra 'team' columns from both merges
                    df = df.drop(columns=['Team', 'Team_Away'])
                    df.rename(columns={'H_for':'H_h_for', 'H_ag':'H_h_ag', 'A_for':'H_a_for', 'A_ag': 'H_a_ag', 'H_for_Away': 'A_h_for', 'H_ag_Away':'A_h_ag', 'A_for_Away': 'A_a_for', 'A_ag_Away': 'A_a_ag'}, inplace=True)

                    # if any columns are None (ie havent played a home or away game yet)
                    cols_to_check= ['H_h_for', 'H_h_ag', 'H_a_for', 'H_a_ag', 'A_h_for', 'A_h_ag', 'A_a_for', 'A_a_ag']
                    for col in cols_to_check:
                        if df[col].isnull().any():  # check if there are any missing values
                            df[col] = df[col].fillna(df[col].mean())

                    # Function to add goal exp column to df
                    def get_gl_exp_value(row, df_ou):
                        # Extract the 'ov_pc_true' value from the current row in df
                        ov_pc_true = row['ov_pc_true']
                        
                        # Locate the row in df_ou where 'Ov2.5_%' matches the 'ov_pc_true' value
                        gl_exp_value = df_ou.loc[df_ou['Ov2.5_%'] == ov_pc_true, 'Exp1']
                        
                        # If there's no matching row, return NaN or a default value
                        return gl_exp_value.values[0] if not gl_exp_value.empty else None

                    # Apply the function to each row in df to create a new column 'gl_exp'
                    df['Gl_Exp'] = df.apply(lambda row: get_gl_exp_value(row, df_ou), axis=1)


                    # calculate home and away expected goals
                    df[["expected_home_goals", "expected_away_goals"]] = df.apply(
                        lambda row: calculate_expected_team_goals_from_1x2_refined(
                            row["Home Win"],
                            row["Draw"],
                            row["Away Win"],
                            row["Over 2.5"],
                            row["Under 2.5"]
                        ),
                        axis=1,
                        result_type="expand"
                    )


                    # ------------------------ APPLY ML MODELS ---------------------------------------

                    # st.write('692',df) 
                    
                    # -----HOME -----------

                    # Creating a 2D array where each row is a sample, and each column is a feature
                    ml_inputs_array_h = np.array([
                        df['expected_home_goals'], 
                        df['expected_away_goals'], 
                        df['H_h_for'],
                        df['H_h_ag'],
                        df['A_a_for'], 
                        df['A_a_ag'],
                    ]).T  # Transpose to make sure it's of shape (n_samples, n_features)

                    # Check for NaN values in ml_inputs_array_a
                    if np.any(np.isnan(ml_inputs_array_h)):
                        # If NaN values are found, handle them as per your requirement:
                        # Replace NaNs with a specific value (e.g., 0 or None), or drop rows with NaNs.
                        ml_inputs_array_h = np.nan_to_num(ml_inputs_array_h, nan=0)  # Replace NaNs with None

                    # Model Home
                    try:
                        # Predict using the model
                        shots_h_prediction_raw = shots_model_h.predict(
                            ml_inputs_array_h
                        ).ravel()

                        home_prediction_key = [
                            get_calibration_key(prediction, hs_calibration_dict)
                            for prediction in shots_h_prediction_raw
                        ]

                        home_prediction_factor = np.array([
                            hs_calibration_dict.get(key, 0)
                            for key in home_prediction_key
                        ])

                        shots_model_h_prediction = (
                            shots_h_prediction_raw * (1 + home_prediction_factor) * overs_boost
                        )

                        df['HS_Exp'] = np.round(shots_model_h_prediction, 2)


                        del ml_inputs_array_h
                        gc.collect()


                        # calculate_sot_lines_and_odds(prediction)
                        df[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = df.apply(
                            lambda row: calculate_home_away_lines_and_odds(row['HS_Exp'], selected_metric), 
                            axis=1, result_type='expand')
                        
                        df['h_main_un'] = round(1 / df['h_main_under_%'], 2)
                        df['h_main_ov'] = round(1 / df['h_main_over_%'], 2)
                        df['h_-1_un'] = round(1 / df['h_-1_under_%'], 2)
                        df['h_-1_ov'] = round(1 / df['h_-1_over_%'], 2)
                        df['h_+1_un'] = round(1 / df['h_+1_under_%'], 2)
                        df['h_+1_ov'] = round(1 / df['h_+1_over_%'], 2)

                    except Exception as e:
                        #st.write(f"An error occurred: {e}")
                        # If an error occurs, assign None to the predictions and related columns
                        df['HS_Exp'] = 0
                        df[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 
                            'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = 0


                    # ------ AWAY -----------

                    ml_inputs_array_a = np.array([
                        df['expected_home_goals'], 
                        df['expected_away_goals'], 
                        df['H_h_for'],
                        df['H_h_ag'],
                        df['A_a_for'], 
                        df['A_a_ag'],
                    ]).T  # Transpose to make sure it's of shape (n_samples, n_features)

                    # Check for NaN values in ml_inputs_array_a
                    if np.any(np.isnan(ml_inputs_array_a)):
                        # If NaN values are found, handle them as per your requirement:
                        # Replace NaNs with a specific value (e.g., 0 ), or drop rows with NaNs.
                        ml_inputs_array_a = np.nan_to_num(ml_inputs_array_a, nan=0)  # Replace NaNs with 0

                    # Model Away
                    try:

                       # Predict using the model
                        shots_a_prediction_raw = shots_model_a.predict(
                            ml_inputs_array_a
                        ).ravel()

                        away_prediction_key = [
                            get_calibration_key(prediction, as_calibration_dict)
                            for prediction in shots_a_prediction_raw
                        ]

                        away_prediction_factor = np.array([
                            as_calibration_dict.get(key, 0)
                            for key in away_prediction_key
                        ])

                        shots_model_a_prediction = (
                            shots_a_prediction_raw * (1 + away_prediction_factor) * overs_boost
                        )

                        df['AS_Exp'] = np.round(shots_model_a_prediction, 2)


                        del ml_inputs_array_a
                        gc.collect()


                        # calculate_corners_lines_and_odds(prediction)
                        df[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = df.apply(
                            lambda row: calculate_home_away_lines_and_odds(row['AS_Exp'], selected_metric), 
                            axis=1, result_type='expand')
                        
                        df['a_main_un'] = round(1 / df['a_main_under_%'], 2)
                        df['a_main_ov'] = round(1 / df['a_main_over_%'], 2)
                        df['a_-1_un'] = round(1 / df['a_-1_under_%'], 2)
                        df['a_-1_ov'] = round(1 / df['a_-1_over_%'], 2)
                        df['a_+1_un'] = round(1 / df['a_+1_under_%'], 2)
                        df['a_+1_ov'] = round(1 / df['a_+1_over_%'], 2)
                    
                    except Exception as e:
                        #st.write(f"An error occurred: {e}")
                        # If an error occurs, assign None to the predictions and related columns
                        df['AS_Exp'] = 0
                        df[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 
                            'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = 0
                    

                    # --------  TOTAL ---------------

                    df[['TS_Exp', 'T_main_line', 'T_-1_line', 'T_+1_line', 'T_-2_line', 'T_+2_line','T_main_under_%', 
                        'T_main_over_%', 'T_-1_under_%', 'T_-1_over_%', 'T_+1_under_%', 
                        'T_+1_over_%', 'T_-2_under_%', 'T_-2_over_%', 'T_+2_under_%', 
                        'T_+2_over_%',]] = df.apply(
                        lambda row: calculate_totals_lines_and_odds(
                            row['HS_Exp'], 
                            row['AS_Exp'], 
                            total_metrics_df=calculate_probability_grid_h_exp_vs_a_exp(row['HS_Exp'], row['AS_Exp'])[1]
                        ),
                        axis=1, 
                        result_type='expand'
                    )

                    df['T_main_un'] = round(1 / df['T_main_under_%'], 2)
                    df['T_main_ov'] = round(1 / df['T_main_over_%'], 2)
                    df['T_-1_un'] = round(1 / df['T_-1_under_%'], 2)
                    df['T_-1_ov'] = round(1 / df['T_-1_over_%'], 2)
                    df['T_+1_un'] = round(1 / df['T_+1_under_%'], 2)
                    df['T_+1_ov'] = round(1 / df['T_+1_over_%'], 2)
                    df['T_-2_un'] = round(1 / df['T_-2_under_%'], 2)
                    df['T_-2_ov'] = round(1 / df['T_-2_over_%'], 2)
                    df['T_+2_un'] = round(1 / df['T_+2_under_%'], 2)
                    df['T_+2_ov'] = round(1 / df['T_+2_over_%'], 2)


                    df[['H_most_%', 'Tie_%', 'A_most_%']] = df.apply(
                        lambda row: pd.Series(calculate_probability_grid_h_exp_vs_a_exp(row['HS_Exp'], row['AS_Exp'])[2:5]), 
                        axis=1, 
                        result_type='expand'
                    )


                    # -------------------------------------------------------

                    df['H_most'] = round(1 / df['H_most_%'], 2)
                    df['Tie'] = round(1 / df['Tie_%'], 2)
                    df['A_most'] = round(1 / df['A_most_%'], 2)

                    # Sub-select final columns
                    df_final = df[['Date', 'Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win', 'Gl_Exp',
                                'HS_Exp', 'h_main_line', 'h_main_un', 'h_main_ov', 
                                'h_-1_line', 'h_-1_un', 'h_-1_ov',
                                'h_+1_line', 'h_+1_un', 'h_+1_ov',
                                'AS_Exp', 'a_main_line', 'a_main_un', 'a_main_ov',
                                'a_-1_line', 'a_-1_un', 'a_-1_ov',
                                'a_+1_line', 'a_+1_un', 'a_+1_ov',
                                'TS_Exp', 'T_main_line', 'T_main_un', 'T_main_ov', 
                                'T_-1_line', 'T_-1_un', 'T_-1_ov',
                                'T_+1_line', 'T_+1_un', 'T_+1_ov',
                                'T_-2_line', 'T_-2_un', 'T_-2_ov',
                                'T_+2_line', 'T_+2_un', 'T_+2_ov',
                                'H_most', 'Tie', 'A_most'
                                ]].copy()

                    # st.write(df_final)

                    # select columns on which to apply margin
                    cols_to_add_margin = ['h_main_un', 'h_main_ov', 
                                'h_-1_un', 'h_-1_ov',
                                'h_+1_un', 'h_+1_ov',
                                'a_main_un', 'a_main_ov',
                                'a_-1_un', 'a_-1_ov',
                                'a_+1_un', 'a_+1_ov',
                                'T_main_un', 'T_main_ov', 
                                'T_-1_un', 'T_-1_ov',
                                'T_+1_un', 'T_+1_ov',
                                'T_-2_un', 'T_-2_ov',
                                'T_+2_un', 'T_+2_ov',
                                'H_most', 'Tie', 'A_most'
                    ]


                    # Apply margins and apply bias for '_un' and '_ov' columns 
                    for col in cols_to_add_margin:
                        if col.endswith('_ov'):  # For '_ov' columns, divide by margin_to_apply
                            df_final = df_final.assign(**{f'{col}_w.%': df_final[col].apply(lambda x: round(x / margin_to_apply / bias_to_apply, 2))})
                        elif col.endswith('_un'):  # For '_un' columns, multiply by bias_to_apply
                            df_final = df_final.assign(**{f'{col}_w.%': df_final[col].apply(lambda x: round(x / margin_to_apply * bias_to_apply, 2))})
                        else:
                            df_final = df_final.assign(**{f'{col}_w.%': df_final[col].apply(lambda x: round(x / margin_to_apply, 2))})  # covers the H_Most / A_most


                    # Rescale margins back to original 'margin_to_apply'
                    for base_col in set(c.rsplit('_', 1)[0] for c in cols_to_add_margin):
                        un_col = f"{base_col}_un_w.%"
                        ov_col = f"{base_col}_ov_w.%"
                        
                        if un_col in df_final.columns and ov_col in df_final.columns:
                            # Compute the inverse sum of both adjusted values
                            inverse_sum = (1 / df_final[un_col]) + (1 / df_final[ov_col])
                            
                            # Compute scaling factor to make inverse sum equal to margin_to_apply
                            scale_factor = margin_to_apply / inverse_sum

                            # Apply scaling factor to both columns
                            df_final[un_col] = (df_final[un_col] / scale_factor).round(2)
                            df_final[ov_col] = (df_final[ov_col] / scale_factor).round(2)

                            
                    # Create a copy of the DataFrame with the new columns added
                    df_final_wm = df_final.copy()

                    # Display the updated DataFrame
                    st.subheader('All Lines')
                    st.caption('Marginalised odds appended')
                    st.write(df_final_wm)

                    # Warning if not all match  retrieved from API call matches the final df
                    if len(df) != len(fixt_id_list):
                        st.warning('Odds for 1 or more matches not currently available - use single match pricing option above')

                    # ---------  show simplified odds - just main lines  ---------
                    df_simple = df_final_wm[['Date', 'Home Team', 'Away Team', 'T_main_line', 'T_main_un_w.%', 'T_main_ov_w.%','h_main_line', 'h_main_un_w.%', 'h_main_ov_w.%', 'a_main_line', 'a_main_un_w.%', 'a_main_ov_w.%']]
            
                    #  ----- Calculate Daily Total Shots and GOALS --------

                    # Convert to datetime
                    df_final_wm['Date'] = pd.to_datetime(df_final_wm['Date'], format="%d-%m-%y %H:%M", errors="coerce")

                    # Group by the day only (ignoring time)
                    df_final_wm['Day'] = df_final_wm['Date'].dt.date  # Extract just the date (day)

                    aggregated_shots = df_final_wm.groupby('Day').agg(
                        TS=('TS_Exp', 'sum'), 
                        Match_Count=('TS_Exp', 'size')
                    ).reset_index()

                    aggregated_gl = df_final_wm.groupby('Day').agg(
                        TG=('Gl_Exp', 'sum'), 
                        Match_Count=('Gl_Exp', 'size')
                    ).reset_index()

                    df_result_shots = aggregated_shots[aggregated_shots['Match_Count'] >= 2]
                    df_result_gl = aggregated_gl[aggregated_gl['Match_Count'] >= 2]

                    # ------- Get increment prior to calling poisson functions for Daily Totals  --------------------------------

                    def calculate_increment(main_line):
                        """Determine increment based on main_line value."""
                        if main_line > 35:
                            return 3
                        elif main_line > 14:
                            return 2
                        return 1

                    # -------  Display Simple DF and Daily Shots side by side  --------------

                    st.write("---")

                    st.subheader('Main Lines')
                    st.write("")
                    st.write(df_simple)
                    st.write("---")

                    col1, col2 = st.columns([1,1])

                    with col2:
                        st.subheader('Total Daily Goals')
                        st.caption(f"Total expected Goals for {selected_league} for specified day")
                        st.write("")
                        st.write(df_result_gl)

                        # Get poisson odds and lines for each day returned for Daly Goals
                        for _, row in df_result_gl.iterrows():
                            exp = row['TG']
                            day = row['Day']
                            main_line = np.floor(exp) + 0.5

                            increment = calculate_increment(main_line)

                            line_minus_1 = main_line - increment
                            line_minus_2 = main_line - increment * 2
                            line_plus_1 = main_line + increment
                            line_plus_2 = main_line + increment * 2

                            probabilities = poisson_probabilities(exp, main_line, line_minus_1, line_plus_1, line_minus_2, line_plus_2)
                
                            st.caption(f"{day} (100% Prices)")
                            st.write(f'(Line {line_plus_2}) - Over', round(1 / probabilities[f'over_plus_2 {line_plus_2}'], 2), f'Under', round(1 / probabilities[f'under_plus_2 {line_plus_2}'], 2))
                            st.write(f'(Line {line_plus_1}) - Over', round(1 / probabilities[f'over_plus_1 {line_plus_1}'], 2), f'Under', round(1 / probabilities[f'under_plus_1 {line_plus_1}'], 2))
                            st.write(f'**(Main Line {main_line}) - Over**', round(1 / probabilities[f'over_main {main_line}'], 2), f'**Under**', round(1 / probabilities[f'under_main {main_line}'], 2))
                            st.write(f'(Line {line_minus_1}) - Over', round(1 / probabilities[f'over_minus_1 {line_minus_1}'], 2), f'Under', round(1 / probabilities[f'under_minus_1 {line_minus_1}'], 2))
                            st.write(f'(Line {line_minus_2}) - Over', round(1 / probabilities[f'over_minus_2 {line_minus_2}'], 2), f'Under', round(1 / probabilities[f'under_minus_2 {line_minus_2}'], 2))
                            st.write("")

                    with col1:
                        st.subheader('Total Daily Shots')
                        st.caption(f"Total expected Shots for {selected_league} for specified day")
                        st.write("")
                        st.write(df_result_shots)

                        # Get poisson odds and lines for each day returned for Daily SOT
                        for _, row in df_result_shots.iterrows():
                            exp = row['TS']
                            day = row['Day']
                            main_line = np.floor(exp) + 0.5

                            increment = calculate_increment(main_line)

                            line_minus_1 = main_line - increment
                            line_minus_2 = main_line - increment * 2
                            line_plus_1 = main_line + increment
                            line_plus_2 = main_line + increment * 2

                            probabilities = poisson_probabilities(exp, main_line, line_minus_1, line_plus_1, line_minus_2, line_plus_2)

                            st.caption(f"{day} (100% Prices)")
                            st.write(f'(Line {line_plus_2}) - Over', round(1 / probabilities[f'over_plus_2 {line_plus_2}'], 2), f'Under', round(1 / probabilities[f'under_plus_2 {line_plus_2}'], 2))
                            st.write(f'(Line {line_plus_1}) - Over', round(1 / probabilities[f'over_plus_1 {line_plus_1}'], 2), f'Under', round(1 / probabilities[f'under_plus_1 {line_plus_1}'], 2))
                            st.write(f'**(Main Line {main_line}) - Over**', round(1 / probabilities[f'over_main {main_line}'], 2), f'**Under**', round(1 / probabilities[f'under_main {main_line}'], 2))
                            st.write(f'(Line {line_minus_1}) - Over', round(1 / probabilities[f'over_minus_1 {line_minus_1}'], 2), f'Under', round(1 / probabilities[f'under_minus_1 {line_minus_1}'], 2))
                            st.write(f'(Line {line_minus_2}) - Over', round(1 / probabilities[f'over_minus_2 {line_minus_2}'], 2), f'Under', round(1 / probabilities[f'under_minus_2 {line_minus_2}'], 2))
                            st.write("")            
                            
            except Exception as e:
                st.write(f'An error has occurred whilst compiling: {e}')





if __name__ == "__main__":
    main()