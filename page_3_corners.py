
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from scipy.stats import poisson, nbinom
from sklearn.preprocessing import PolynomialFeatures
import joblib
from mymodule.functions import get_fixtures,  calculate_home_away_lines_and_odds, calculate_true_from_true_raw, poisson_probabilities
import requests
import os
from dotenv import load_dotenv
import gc


dict_api_to_bk_league_names = {
     'England Premier':'England Premier League',
     'Spain La Liga' : 'Spain LaLiga',
 }

CURRENT_SEASON = '2025-26'
LAST_SEASON = '2024-25'
OVERS_BOOST = 1.02 # increase all overs expectations by this amount as a foundation. 10.5 > 10.71. Odds change outputs also dafaulted on front-end.
TOTALS_BOOST = 1.02

corners_model_h = joblib.load('models/corners/corners_home_neg_binom.pkl')
corners_model_a = joblib.load('models/corners/corners_away_neg_binom.pkl')

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
        'Ligue 1': 'France Ligue 1',
        'Serie A': 'Italy Serie A',
        'Premier Soccer League': 'South Africa Premier',
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
        "Scotland Premier": '179',
        "South Africa Premier": "288",
    }

    metric_options = {
        'Corners': ['HC', 'AC', 'TC'],
      #  'Fouls': ['HF', 'AF', 'TF'],
      #  'Shots on Target': ['HST', 'AST', 'TST'],
      #  'Shots': ['HS', 'AS', 'TS'],
    }


    # Capture user selections  # WIDGET
    selected_league = st.sidebar.selectbox('Select League', options=list(league_options.values()), label_visibility = 'visible')
    selected_metric = 'Corners'

 
    df = df[df['League'] == [key for key, value in league_options.items() if value == selected_league][0]]           


    this_df = df[(df['Season'] == CURRENT_SEASON)]  # remove all matches that are not current season
    last_df = df[(df['Season'] == LAST_SEASON)] 

    # delete df to free space
    del df
    gc.collect()

    team_options = sorted(this_df['HomeTeam'].unique().tolist())
    # selected_team_h = st.sidebar.selectbox("Select Home Team", options=team_options, index=0)
    # selected_team_a = st.sidebar.selectbox("Select Away Team", options=team_options, index=1)

    # selected_home_team = st.sidebar.selectbox('Select Home Team', options=sorted(list(df_ratings.Team)), index=0)

    # -----------------------------------------------------------------------

    st.header(f'{selected_metric} Model - {selected_league}', divider='blue')

    # show_model_info = st.checkbox('Model Info')
    # if show_model_info:
    #     st.caption('''
    #             Evaluation metrics show adequate modelling performance. Given strong market, use outputs as a reference to position
    #             our prices if market differences or for early marketing requirements.
    #              ''')

    # get fixtures
    league_id = leagues_dict.get(selected_league)

    # st.write(this_df)


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
    show_this_ssn_stats = st.checkbox(f'Show current season {selected_metric} stats', label_visibility = 'visible')  # WIDGET
    if show_this_ssn_stats:
        st.write(this_options_df)


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

    # this prevents leagues which have play-offs having those games allocated to last season's table
    last_options_df = last_options_df[last_options_df['MP'] > 10]

    # # Display last season DataFrame
    # show_last_ssn_stats = st.checkbox(f'Show last season {selected_metric} stats', label_visibility = 'visible')  # WIDGET
    # if show_last_ssn_stats:
    #     st.write(last_options_df)


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
    teams_to_remove = last_options_df[~last_options_df['Team'].isin(this_options_df['Team'])]['Team'].unique()

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
    last_options_df_3 = last_options_df_2[~last_options_df_2['Team'].isin(teams_to_remove)].reset_index(drop=True)

    # st.write(last_options_df_2)
    # st.write(last_options_df_3)
    # st.write(this_options_df)

    # --------------------------------------------------------------------------

    df_mix_1 = pd.merge(this_options_df, last_options_df_3, on=['Team'])
    df_mix_1['H_for'] = round((df_mix_1['H_for_x'] * perc_this_ssn) + (df_mix_1['H_for_y'] * perc_last_ssn), 2)
    df_mix_1['H_ag'] = round((df_mix_1['H_ag_x'] * perc_this_ssn) + (df_mix_1['H_ag_y'] * perc_last_ssn), 2)
    df_mix_1['A_for'] = round((df_mix_1['A_for_x'] * perc_this_ssn) + (df_mix_1['A_for_y'] * perc_last_ssn), 2)
    df_mix_1['A_ag'] = round((df_mix_1['A_ag_x'] * perc_this_ssn) + (df_mix_1['A_ag_y'] * perc_last_ssn), 2)


    df_mix = df_mix_1[['Team', 'H_for', 'H_ag', 'A_for', 'A_ag']]

    # # to calc dixon-coles
    # df_mix_hf_av = df_mix['H_for'].mean()
    # df_mix_hag_av = df_mix['H_ag'].mean()
    # df_mix_af_av = df_mix['A_for'].mean()
    # df_mix_aag_av = df_mix['A_ag'].mean()


    show_df_mix = st.checkbox(f'Show team {selected_metric} stats (weighted current & previous season)', label_visibility = 'visible')  # WIDGET
    if show_df_mix:
        st.write(df_mix)
        st.caption('''
                 Current season and previous season statistics are merged based on a weighting of number of games through the current season.
                 Previous season data decays logarithmically from 100% at game 1 to 0 % by game 24. Teams new to a division are allocated
                 an initial defaulted previous season 1st or 3rd league quantile value (depending if promoted or relegated in), so predictions for those teams may be less reliable early season.
                 ''')


    # Get metric columns for the selected metric
    metric_columns = metric_options[selected_metric][:2]  # First two columns only, HC and AC for 'Corners'

    # ht_h_for= round(df_mix.loc[df_mix['Team'] == selected_team_h, 'H_for'].values[0], 2)
    # ht_h_ag= round(df_mix.loc[df_mix['Team'] == selected_team_h, 'H_ag'].values[0], 2)
    # ht_a_for = round(df_mix.loc[df_mix['Team'] == selected_team_h, 'A_for'].values[0], 2)
    # ht_a_ag = round(df_mix.loc[df_mix['Team'] == selected_team_h, 'A_ag'].values[0], 2)

    # at_h_for = round(df_mix.loc[df_mix['Team'] == selected_team_a, 'H_for'].values[0], 2)
    # at_h_ag = round(df_mix.loc[df_mix['Team'] == selected_team_a, 'H_ag'].values[0], 2)
    # at_a_for = round(df_mix.loc[df_mix['Team'] == selected_team_a, 'A_for'].values[0], 2)
    # at_a_ag = round(df_mix.loc[df_mix['Team'] == selected_team_a, 'A_ag'].values[0], 2)

    # st.write("---")

    # # for dixon-coles model
    # ht_f_mult = ht_h_for / df_mix_hf_av
    # ht_ag_mult = ht_h_ag / df_mix_hag_av
    # at_f_mult = at_a_for / df_mix_af_av
    # at_ag_mult = at_a_ag / df_mix_aag_av
    # st.write(ht_f_mult, ht_ag_mult, at_f_mult, at_ag_mult)
    # at_pred = at_f_mult * ht_ag_mult * df_mix_af_av 
    # ht_pred = ht_f_mult * at_ag_mult * df_mix_hf_av 
    # st.write('ht_dc_pred:',ht_pred)
    # st.write('at_dc_pred:',at_pred)


    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # st.write('Enter Approximate Match Odds:')
    # c1,c2,c3,c4, c5 = st.columns([1,1,1,1,5])
    # with c1:
    #     try:
    #         h_odds = float(st.text_input('Home Odds', value = 2.10, label_visibility = 'visible'))  # WIDGET
    #     except ValueError:
    #             st.error("Please enter a valid number.")
    #             return
    # with c2:
    #     try:
    #         d_odds = float(st.text_input('Draw Odds', value = 3.30, label_visibility = 'visible'))  # WIDGET
    #     except ValueError:
    #             st.error("Please enter a valid number.")
    #             return
    # with c3:
    #     try:
    #         a_odds = float(st.text_input('Away Odds', value = 3.50, label_visibility = 'visible'))  # WIDGET
    #     except ValueError:
    #             st.error("Please enter a valid number.")
    #             return
    
    # with c4:
    #     st.write("")
    #     st.write("")
    #     margin = round(1/h_odds + 1/d_odds + 1/a_odds, 2)


    # h_pc_true_raw = 1/(h_odds * margin)
    # d_pc_true_raw = 1/(d_odds * margin)
    # a_pc_true_raw = 1/(a_odds * margin)

    # with c1:
    #  st.write('Margin:', margin)

    # # Error message if < 100 %
    # if margin < 1:
    #     st.warning('Odds must be > 1.00 !')

    # # --------------------------------------------------------
    # h_pc_true, d_pc_true, a_pc_true = calculate_true_from_true_raw(h_pc_true_raw, d_pc_true_raw, a_pc_true_raw, margin)

    # # with c5:
    # #     st.write('Home true probability is approx:', h_pc_true)
    # #     st.write('Draw true probability is approx:', d_pc_true)
    # #     st.write('Away true probability is approx:', a_pc_true)

    # # ---------------------------------------------------------

    # st.write("---")
    # st.header(f'{selected_metric} Model Outputs')
    # st.write(' - Model Feature Inputs - weighted averages current & previous season')

    # # st.write(metric_columns[2])

    h_lg_avg = round(this_df[metric_columns[0]].mean(), 2)  # HC or H_SOT
    a_lg_avg = round(this_df[metric_columns[1]].mean(), 2)     # AC or A_SOT

    # cl1,cl2, cl3 = st.columns([7,1,7])
    # with cl1:
    #     st.subheader('Home')
    #     st.caption(f'Home win probability: {h_pc_true}')
    #     st.caption(f'Home Team, home for: {ht_h_for}')
    #     st.caption(f'Home Team, away for: {ht_a_for}')
    #     st.caption(f'Away Team, away against: {at_a_ag}')
    #     st.caption(f'Away Team, home against: {at_h_ag}')
    #     st.caption(f'Home {selected_metric} League Average: {h_lg_avg}')

    # with cl3:
    #     st.subheader('Away')
    #     st.caption(f'Away win probability: {a_pc_true}')
    #     st.caption(f'Away Team, away for: {at_a_for}')
    #     st.caption(f'Away Team, home for: {at_h_for}')
    #     st.caption(f'HomeTeam, away against: {ht_a_ag}')
    #     st.caption(f'Home Team, home against: {ht_h_ag}')
    #     st.caption(f'Away {selected_metric} League Average: {a_lg_avg}')

    # # ------------  Enter own parameters -----------------

    # with cl1:
    #     show_manual_inputs_h = st.checkbox('Enter inputs manually (Home)', label_visibility = 'visible')  # WIDGET
    #     if show_manual_inputs_h:
    #         h_pc_true = st.number_input('Home Team, win probability', value=h_pc_true, label_visibility = 'visible') 
    #         ht_h_for = st.number_input('Home Team, home for', value = ht_h_for, label_visibility = 'visible')
    #         ht_a_for = st.number_input('Home Team, away for', value = ht_a_for, label_visibility = 'visible')
    #         at_a_ag = st.number_input('Away Team, away against', value = at_a_ag, label_visibility = 'visible')
    #         at_h_ag = st.number_input('Away Team, home against', value = at_h_ag, label_visibility = 'visible' )
    #         h_lg_avg = st.number_input('Home Competition Average', value = h_lg_avg, label_visibility = 'visible')

    # with cl3:
    #     show_manual_inputs_a = st.checkbox('Enter inputs manually (Away)', label_visibility = 'visible')  # WIDGET
    #     if show_manual_inputs_a:
    #         a_pc_true = st.number_input('Away Team, win probability', value=a_pc_true, label_visibility = 'visible')
    #         at_a_for = st.number_input('Away Team, away for', value = at_a_for, label_visibility = 'visible')
    #         at_h_for= st.number_input('Away Team, home for', value = at_h_for, label_visibility = 'visible')
    #         ht_a_ag = st.number_input('Home Team, away against', value = ht_a_ag, label_visibility = 'visible')
    #         ht_h_ag = st.number_input('Home Team, home against', value = ht_h_ag, label_visibility = 'visible')
    #         a_lg_avg = st.number_input('Away Competition Average', value = a_lg_avg, label_visibility = 'visible')



    # inputs_array_h = np.array([[h_pc_true, ht_h_for, ht_a_for, at_a_ag, at_h_ag, h_lg_avg]])
    # inputs_array_a = np.array([[a_pc_true, at_a_for, at_h_for, ht_a_ag, ht_h_ag, a_lg_avg]])


    # # Model Home
    # poly_h = PolynomialFeatures(degree=2, include_bias=True)
    # X_poly_input_h = poly_h.fit_transform(inputs_array_h)
    # corners_model_h_prediction = corners_model_h.predict(X_poly_input_h)
    # home_prediction = round(corners_model_h_prediction[0], 2)
    # with cl1:
    #     st.success(f'Home Prediction: {home_prediction}')

    # # Model Away
    # poly_a = PolynomialFeatures(degree=2, include_bias=True)
    # X_poly_input_a = poly_a.fit_transform(inputs_array_a)
    # corners_model_a_prediction = corners_model_a.predict(X_poly_input_a)   
    # away_prediction = round(corners_model_a_prediction[0], 2)
    # with cl3:
    #     st.success(f'Away Prediction: {away_prediction}')

    # # ---------------  CREATE OVER UNDER LINES AND PROBABILITIES -------------------
    # # FUNCTION - calculate_corners_lines_and_odds()

    # # Home lines/odds
    # h_main_line, h_minor_line, h_major_line, h_prob_under_main, h_prob_over_main, \
    # h_prob_under_min, h_prob_over_min, h_prob_under_maj, h_prob_over_maj = calculate_home_away_lines_and_odds(home_prediction)
    # # Away lines/odds
    # a_main_line, a_minor_line, a_major_line, a_prob_under_main, a_prob_over_main, \
    # a_prob_under_min, a_prob_over_min, a_prob_under_maj, a_prob_over_maj = calculate_home_away_lines_and_odds(away_prediction)


    # with cl1:
    # # Output the results
    #     st.write(f"Main Line: {h_main_line}")
    #     st.caption(f"Probability Under: {h_prob_under_main}")
    #     st.caption(f"Probability Over: {h_prob_over_main}")
    #     st.info(f'Home Odds - Under {h_main_line}: {round(1 / h_prob_under_main, 2)}')
    #     st.info(f'Home Odds - Over {h_main_line}: {round(1 / h_prob_over_main, 2)}')

    #     show_alternate_lines_h = st.checkbox('Show alternate home lines', label_visibility = 'visible')  # WIDGET
    #     if show_alternate_lines_h:
    #         st.write(f'Minor Line: [{h_minor_line}] - Under: {round(1 / h_prob_under_min, 2)}, Over: {round(1 / h_prob_over_min, 2)}')
    #         st.write(f'Major Line: [{h_major_line}] - Under: {round(1 / h_prob_under_maj, 2)}, Over: {round(1 / h_prob_over_maj, 2)}')
    #         st.write("---")

    # with cl3:
    # # Output the results
    #     st.write(f"Main Line: {a_main_line}")
    #     st.caption(f"Mixed Probability Under: {a_prob_under_main}")
    #     st.caption(f"Mixed Probability Over: {a_prob_over_main}")
    #     st.info(f'Away Odds - Under {a_main_line}: {round(1 / a_prob_under_main, 2)}')
    #     st.info(f'Away Odds - Over {a_main_line}: {round(1 / a_prob_over_main, 2)}')

    #     show_alternate_lines_a = st.checkbox('Show alternate away lines', label_visibility = 'visible')  # WIDGET
    #     if show_alternate_lines_a:
    #         st.write(f'Minor Line: [{a_minor_line}] - Under: {round(1 / a_prob_under_min, 2)}, Over: {round(1 / a_prob_over_min, 2)}')
    #         st.write(f'Major Line: [{a_major_line}] - Under: {round(1 / a_prob_under_maj, 2)}, Over: {round(1 / a_prob_over_maj, 2)}')
    #         st.write("---")

  
    # # --- Probability grid HC vs AC ---------------------------------------------------

    # Enter HC and AC exps' to return probability_grid, total_metrics_df (df of probability of each band), home_more_prob, equal_prob, away_more_prob (for matchups), total_metric_probabilities
    def calculate_probability_grid_hc_vs_ac(home_prediction, away_prediction):
        # Set the range for corners
        metric_range = np.arange(0, 30)

        # Initialize a DataFrame to store probabilities
        probability_grid = pd.DataFrame(index=metric_range, columns=metric_range)

        # HC adjustments
        home_mode = int(np.floor(home_prediction))  # Mode approximation by flooring the expected value
        bins_below_mode = home_mode
        reduction_per_bin = 0.05 / bins_below_mode if bins_below_mode > 0 else 0

        bins_above_mode = 7  # Bins from mode + 3 to mode + 9
        increase_factors = np.linspace(0.05, 0, bins_above_mode)  # Gradual scaling down of the 5% increase

        # AC adjustments
        away_mode = int(np.floor(away_prediction))  # Mode approximation for AC
        bins_below_away_mode = away_mode + 1  # From mode - 1 to 0
        increase_per_bin_away = 0.07 / bins_below_away_mode if bins_below_away_mode > 0 else 0

        bins_above_away_mode = 11  # From mode + 2 to mode + 12
        decrease_factors_away = np.linspace(0.07, 0, bins_above_away_mode + 1)  # Gradual scaling down of the 7% decrease

        # Calculate probabilities for each combination of home and away corners
        for home_metric in metric_range:
            for away_metric in metric_range:
                # Calculate Poisson probabilities
                poisson_home_prob = poisson.pmf(home_metric, home_prediction)
                poisson_away_prob = poisson.pmf(away_metric, away_prediction)

                # Calculate Negative Binomial probabilities
                nb_home_prob = nbinom.pmf(home_metric, home_prediction, home_prediction / (home_prediction + home_prediction))
                nb_away_prob = nbinom.pmf(away_metric, away_prediction, away_prediction / (away_prediction + away_prediction))

                # Calculate combined probability (50% from Poisson, 50% from Negative Binomial)
                combined_home_prob = 0.5 * poisson_home_prob + 0.5 * nb_home_prob
                combined_away_prob = 0.5 * poisson_away_prob + 0.5 * nb_away_prob

                # Adjust HC probabilities (Home corners)
                if home_metric < home_mode:
                    combined_home_prob *= (1 - reduction_per_bin)
                elif home_metric == home_mode:
                    pass  # Mode bin remains unchanged
                elif home_mode + 3 <= home_metric <= home_mode + 9:
                    offset = home_metric - (home_mode + 3)
                    combined_home_prob *= (1 + increase_factors[offset])


                # Adjust AC probabilities (Away corners)
                if 0 <= away_metric <= away_mode - 1:
                    combined_away_prob *= (1 + increase_per_bin_away)
                elif away_mode + 2 <= away_metric <= away_mode + 12:
                    offset = away_metric - (away_mode + 2)
                    combined_away_prob *= (1 - decrease_factors_away[offset])

                # Increase Away mode bin by 0.5%
                if away_metric == away_mode:
                    combined_away_prob *= 1.005  # Increase the away mode bin by 0.5%

                # Store the combined probabilities in the grid
                probability_grid.loc[home_metric, away_metric] = combined_home_prob * combined_away_prob

        # Normalize the probability grid to ensure it sums to 1
        probability_sum = probability_grid.values.sum()
        probability_grid /= probability_sum  # Normalization step

        # Calculate total probabilities after grid is normalized
        equal_prob = round(np.trace(probability_grid), 2)  # Probability where home = away
        home_more_mask = np.array(probability_grid.index)[:, None] > np.array(probability_grid.columns)
        away_more_mask = np.array(probability_grid.index)[:, None] < np.array(probability_grid.columns)

        home_more_prob = probability_grid.values[home_more_mask].sum()
        away_more_prob = probability_grid.values[away_more_mask].sum()

        # Calculate probabilities for total metrics (e.g., home_metric + away_metric)
        range_value = np.arange(0, 30)
        total_metric_probabilities = np.zeros(31)  # Array for outcomes 0-30
        for home_metric in range_value:
            for away_metric in range_value:
                total_metrics = home_metric + away_metric
                if total_metrics <= 30:
                    total_metric_probabilities[total_metrics] += probability_grid.loc[home_metric, away_metric]

        total_metric_probabilities /= total_metric_probabilities.sum()  # Ensure it sums to 1

        # Create a DataFrame to display the total metric probabilities
        total_metrics_df = pd.DataFrame({
            'Total Metrics': np.arange(len(total_metric_probabilities)),
            'Probability': total_metric_probabilities
        })

        return probability_grid, total_metrics_df, home_more_prob, equal_prob, away_more_prob, total_metric_probabilities
    # # ---------------------------------------------------------------

    # probability_grid, total_metrics_df, home_more_prob, equal_prob, away_more_prob, total_metric_probabilities = calculate_probability_grid_hc_vs_ac(home_prediction, away_prediction)

    # # st.write('probability_grid', probability_grid)    
    # # st.write('total_metrics_df', total_metrics_df)

    # # --- plot as histograms -------------------------------------------
    # # Extract marginal probabilities
    # home_probabilities = probability_grid.sum(axis=1)  # Sum rows for home corners
    # away_probabilities = probability_grid.sum(axis=0)  # Sum columns for away corners

    # with cl1:
    #     # WIDGET
    #     show_home_prob_dist = st.checkbox(f'Show expected probability distribution for Home {selected_metric}', label_visibility = 'visible')
    #     if show_home_prob_dist:
    #         # Histogram for Home Corners
    #         st.subheader(f"Expected Probability Distribution for Home {selected_metric}")
    #         st.bar_chart(home_probabilities)

    # with cl3:
    #     # WIDGET
    #     show_away_prob_dist = st.checkbox(f'Show expected probability distribution for Away {selected_metric}', label_visibility = 'visible')
    #     if show_away_prob_dist:
    #         # Histogram for Away Corners
    #         st.subheader(f"Expected Probability Distribution for Away {selected_metric}")
    #         st.bar_chart(away_probabilities)



    # # Display bands
    # with cl1:
    #     # WIDGET
    #     show_bands = st.checkbox('Show bands probability breakdown', label_visibility = 'visible')
    #     if show_bands:
    #         st.subheader(f"Probability Distribution for Total {selected_metric}")
    #         st.dataframe(total_metrics_df)



    # ------------- CALCULATE MAIN & ALT LINES & ODDS (TOTAL) --------------------------

    # takes home_prediction and away_prediction as args and returns lines (main, minor, major) and over/under %'s
    def calculate_totals_lines_and_odds(home_prediction, away_prediction, total_metrics_df):
        # Calculate main line & odds
        total_prediction = round(home_prediction + away_prediction, 2)
        tot_main_line = np.floor(total_prediction) + 0.5
        # Sum probabilities below and above this midpoint
        below_midpoint_p_main = total_metrics_df[total_metrics_df['Total Metrics'] <= np.floor(tot_main_line)]['Probability'].sum()
        above_midpoint_p_main = total_metrics_df[total_metrics_df['Total Metrics'] >= np.ceil(tot_main_line)]['Probability'].sum()

        # Calculate minor line & odds
        tot_minor_line = tot_main_line - 1
        # Sum probabilities below and above this midpoint
        below_midpoint_p_minor = total_metrics_df[total_metrics_df['Total Metrics'] <= np.floor(tot_minor_line)]['Probability'].sum()
        above_midpoint_p_minor = total_metrics_df[total_metrics_df['Total Metrics'] >= np.ceil(tot_minor_line)]['Probability'].sum()

        # Calculate major line & odds
        tot_major_line = tot_main_line + 1
        # Sum probabilities below and above this midpoint
        below_midpoint_p_major = total_metrics_df[total_metrics_df['Total Metrics'] <= np.floor(tot_major_line)]['Probability'].sum()
        above_midpoint_p_major = total_metrics_df[total_metrics_df['Total Metrics'] >= np.ceil(tot_major_line)]['Probability'].sum()

        # Calculate minor line 2 & odds
        tot_minor_line_2 = tot_main_line - 2
        # Sum probabilities below and above this midpoint
        below_midpoint_p_minor_2 = total_metrics_df[total_metrics_df['Total Metrics'] <= np.floor(tot_minor_line_2)]['Probability'].sum()
        above_midpoint_p_minor_2 = total_metrics_df[total_metrics_df['Total Metrics'] >= np.ceil(tot_minor_line_2)]['Probability'].sum()

        # Calculate major line & odds
        tot_major_line_2 = tot_main_line + 2
        # Sum probabilities below and above this midpoint
        below_midpoint_p_major_2 = total_metrics_df[total_metrics_df['Total Metrics'] <= np.floor(tot_major_line_2)]['Probability'].sum()
        above_midpoint_p_major_2 = total_metrics_df[total_metrics_df['Total Metrics'] >= np.ceil(tot_major_line_2)]['Probability'].sum()


        return (float(total_prediction), float(tot_main_line), float(tot_minor_line), float(tot_major_line), float(tot_minor_line_2), float(tot_major_line_2),\
               float(below_midpoint_p_main), float(above_midpoint_p_main), float(below_midpoint_p_minor), float(above_midpoint_p_minor), \
               float(below_midpoint_p_major), float(above_midpoint_p_major), \
               float(below_midpoint_p_minor_2), float(above_midpoint_p_minor_2), \
               float(below_midpoint_p_major_2), float(above_midpoint_p_major_2)
        )

    # # PASS THROUGH FUNCTION
    # total_prediction, tot_main_line, tot_minor_line, tot_major_line, tot_minor_line_2, tot_major_line_2, below_midpoint_p_main, \
    # above_midpoint_p_main, below_midpoint_p_minor, above_midpoint_p_minor, below_midpoint_p_major, \
    # above_midpoint_p_major, below_midpoint_p_minor_2, above_midpoint_p_minor_2, \
    # below_midpoint_p_major_2, above_midpoint_p_major_2 = calculate_totals_lines_and_odds(home_prediction, away_prediction, total_metrics_df)


    # with cl1:
    #     # Show odds of over/under mainline
    #     st.write("")
    #     st.subheader(f'Total {selected_metric}')
    #     st.success(f'Total prediction: **{total_prediction}**')
    #     st.write(f'Total Main Line: {tot_main_line}')
    #     st.info(f'Under {tot_main_line}: **{round(1 / below_midpoint_p_main, 2)}**')
    #     st.info(f'Over {tot_main_line}: **{round(1 / above_midpoint_p_main, 2)}**')

    #     show_alternate_lines_tot = st.checkbox('Show alternate toal lines')
    #     if show_alternate_lines_tot:
    #         st.write(f'Minor Line: [{tot_minor_line}] - Under: {round(1 / below_midpoint_p_minor, 2)}, Over: {round(1 / above_midpoint_p_minor, 2)}')
    #         st.write(f'Major Line: [{tot_major_line}] - Under: {round(1 / below_midpoint_p_major, 2)}, Over: {round(1 / above_midpoint_p_major, 2)}')
    #         st.write(f'Minor Line 2: [{tot_minor_line_2}] - Under: {round(1 / below_midpoint_p_minor_2, 2)}, Over: {round(1 / above_midpoint_p_minor_2, 2)}')
    #         st.write(f'Major Line 2: [{tot_major_line_2}] - Under: {round(1 / below_midpoint_p_major_2, 2)}, Over: {round(1 / above_midpoint_p_major_2, 2)}')
    #         st.write("---")


    #     show_tot_hist = st.checkbox(f'Show expected Total {selected_metric} distribution', label_visibility = 'visible')  # WIDGET
    #     if show_tot_hist:
    #         st.bar_chart(total_metric_probabilities)

    # show_prob_grid = st.checkbox(f'Show expected Total {selected_metric} probability grid', label_visibility = 'visible')  # WIDGET
    # if show_prob_grid:
    #     st.write(probability_grid)


    # with cl3:
    #     st.write("")
    #     st.write("")
    #     st.write("")
    #     st.write("")
    #     st.subheader(f'Team Most {selected_metric}')
    #     st.write(f'Home:', round(1 / home_more_prob, 2))
    #     st.write('Tie:', round(1 / equal_prob, 2))
    #     st.write('Away:', round(1 / away_more_prob, 2))



    # -------------------------------------------- CREATE ODDS FOR ALL UPCOMING FIXTURES --------------------------------------------------------------------

    st.write("")
    st.subheader(f'Generate odds for all upcoming {selected_league} matches (up to 7 days ahead)')

    column1,column2 = st.columns([1,2])

    with column1:
        # WIDGET
        margin_to_apply = st.number_input('Margin to apply:', step=0.01, value = 1.09, min_value=1.01, max_value=1.2, key='margin_to_apply', label_visibility = 'visible')
        bias_to_apply = st.number_input('Overs bias to apply (reduce overs & increase unders odds by a set %):', step=0.01, value = 1.06, min_value=0.95, max_value=1.1, key='bias_to_apply', label_visibility = 'visible')

    generate_odds_all_matches = st.button(f'Click to generate')

    if generate_odds_all_matches:
        with st.spinner("Odds being compiled..."):
            try:

                # GET FIXTURES WEEK AHEAD
                today = datetime.now()
                to_date = today + timedelta(days=7)
                from_date_str = today.strftime("%Y-%m-%d")
                to_date_str = to_date.strftime("%Y-%m-%d")
                MARKET_IDS = ['1', '5']             # WDW & Ov/Un
                BOOKMAKERS = ['8']                  # Pinnacle = 4, 365 = 8
                API_SEASON = CURRENT_SEASON[:4]


                df_fixtures = get_fixtures(league_id, from_date_str, to_date_str, API_SEASON)

                if df_fixtures.empty:
                    st.write("No data returned for the specified league and date range.")
                else:
                    df_fixts = df_fixtures[['Fixture ID', 'Date', 'Home Team', 'Away Team']]
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

                            # After using the temporary odds_df, delete it and call garbage collection
                            del odds_df
                            gc.collect()

                    # Display the collected odds
                    # st.write(all_odds_df)

                    # Use groupby and fillna to collapse rows and remove None values
                    df_collapsed = all_odds_df.groupby('Fixture ID').first().combine_first(
                        all_odds_df.groupby('Fixture ID').last()).reset_index()

                    # st.write(df_collapsed)


                    # Merge odds df_fixts with df_collapsed
                    df = df_fixts.merge(df_collapsed, on='Fixture ID')
                    df = df.dropna()

                    del df_collapsed
                    gc.collect()
                    # st.write(df)


                    #  ---------------  Create true wdw odds ---------------
                    # Convert columns to numeric (if they are strings or objects)
                    df['Home Win'] = pd.to_numeric(df['Home Win'], errors='coerce')
                    df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce')
                    df['Away Win'] = pd.to_numeric(df['Away Win'], errors='coerce')

                    df['margin'] = 1/df['Home Win'] + 1/df['Draw'] + 1/df['Away Win']

                    df['h_pc_true_raw'] = (1 / df['Home Win']) / df['margin']
                    df['d_pc_true_raw'] = (1 / df['Draw']) / df['margin'] 
                    df['a_pc_true_raw'] = (1 / df['Away Win']) / df['margin'] 

                    df[['h_pc_true', 'd_pc_true', 'a_pc_true']] = df.apply(
                        lambda row: calculate_true_from_true_raw(row['h_pc_true_raw'], row['d_pc_true_raw'], row['a_pc_true_raw'], row['margin']), 
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

                    # if any columns are None (ie havent played a home or away game yet
                    cols_to_check= ['H_h_for', 'H_h_ag', 'H_a_for', 'H_a_ag', 'A_h_for', 'A_h_ag', 'A_a_for', 'A_a_ag']
                    for col in cols_to_check:
                        if df[col].isnull().any():  # check if there are any missing values
                            df[col] = df[col].fillna(df[col].mean())
                            
                    # st.write(df)


                    # ------------------------ APPLY MODELS ---------------------------------------
                    df['h_lg_avg'] = h_lg_avg
                    df['a_lg_avg'] = a_lg_avg
                    # df['const'] = 1


                    # -----HOME -----------

                    # Creating a 2D array where each row is a sample, and each column is a feature
                    ml_inputs_array_h = np.array([
                        df['h_pc_true'], 
                        df['H_h_for'], 
                        df['H_a_for'], 
                        df['A_a_ag'],
                        df['A_h_ag'], 
                        df['h_lg_avg']
                    ]).T  # Transpose to make sure it's of shape (n_samples, n_features)

                    # Check for NaN values in ml_inputs_array_a
                    if np.any(np.isnan(ml_inputs_array_h)):
                        # If NaN values are found, handle them as per your requirement:
                        # Replace NaNs with a specific value (e.g., 0 or None), or drop rows with NaNs.
                        ml_inputs_array_h = np.nan_to_num(ml_inputs_array_h, nan=0)  # Replace NaNs with None

                    # Model Home
                    try:
                        poly_h = PolynomialFeatures(degree=2, include_bias=True)
                        X_poly_input_h = poly_h.fit_transform(ml_inputs_array_h)  # Transform the input features

                        # Predict using the model
                        corners_model_h_prediction = corners_model_h.predict(X_poly_input_h) * OVERS_BOOST


                        # Assign the predictions to the DataFrame - ** NAME THIS HEADER '_RAW' IF NEED TO ADD OVERS BIAS **
                        df['HC_Exp'] = np.round(corners_model_h_prediction, 2)

                        del ml_inputs_array_h
                        gc.collect()


                        # calculate_corners_lines_and_odds(prediction)
                        df[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = df.apply(
                            lambda row: calculate_home_away_lines_and_odds(row['HC_Exp'], selected_metric), 
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
                        df['HC_Exp'] = 0
                        df[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 
                            'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = 0


                    # ------ AWAY -----------

                    ml_inputs_array_a = np.array([
                        df['a_pc_true'], 
                        df['A_a_for'], 
                        df['A_h_for'], 
                        df['H_a_ag'],
                        df['H_h_ag'], 
                        df['a_lg_avg']
                    ]).T  # Transpose to make sure it's of shape (n_samples, n_features)

                    # Check for NaN values in ml_inputs_array_a
                    if np.any(np.isnan(ml_inputs_array_a)):
                        # If NaN values are found, handle them as per your requirement:
                        # Replace NaNs with a specific value (e.g., 0 or None), or drop rows with NaNs.
                        ml_inputs_array_a = np.nan_to_num(ml_inputs_array_a, nan=0)  # Replace NaNs with 0

                    # Model Away
                    try:
                        poly_a = PolynomialFeatures(degree=2, include_bias=True)
                        X_poly_input_a = poly_a.fit_transform(ml_inputs_array_a)  # Transform the input features


                        # Predict using the model
                        corners_model_a_prediction = corners_model_a.predict(X_poly_input_a) * OVERS_BOOST

                        del ml_inputs_array_a
                        gc.collect()

                        # Assign the predictions to the DataFrame - ** NAME THIS HEADER '_RAW' IF NEED TO ADD OVERS BIAS **
                        df['AC_Exp'] = np.round(corners_model_a_prediction, 2)

                        # calculate_corners_lines_and_odds(prediction)
                        df[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = df.apply(
                            lambda row: calculate_home_away_lines_and_odds(row['AC_Exp'], selected_metric), 
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
                        df['AC_Exp'] = 0
                        df[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 
                            'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = 0
                    

                    # --------  TOTAL ---------------

                    df[['TC_Exp', 'T_main_line', 'T_-1_line', 'T_+1_line', 'T_-2_line', 'T_+2_line','T_main_under_%', 
                        'T_main_over_%', 'T_-1_under_%', 'T_-1_over_%', 'T_+1_under_%', 
                        'T_+1_over_%', 'T_-2_under_%', 'T_-2_over_%', 'T_+2_under_%', 
                        'T_+2_over_%',]] = df.apply(
                        lambda row: calculate_totals_lines_and_odds(
                            row['HC_Exp'], 
                            row['AC_Exp'], 
                            total_metrics_df=calculate_probability_grid_hc_vs_ac(row['HC_Exp'], row['AC_Exp'])[1]
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

                    # st.write(df)
                    df[['H_most_%', 'Tie_%', 'A_most_%']] = df.apply(
                        lambda row: pd.Series(calculate_probability_grid_hc_vs_ac(row['HC_Exp'], row['AC_Exp'])[2:5]), 
                        axis=1, 
                        result_type='expand'
                    )


                    # -------------------------------------------------------

                    df['H_most'] = round(1 / df['H_most_%'], 2)
                    df['Tie'] = round(1 / df['Tie_%'], 2)
                    df['A_most'] = round(1 / df['A_most_%'], 2)

                    # Sub-select final columns
                    df_final = df[['Date', 'Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win',
                                'HC_Exp', 'h_main_line', 'h_main_un', 'h_main_ov', 
                                'h_-1_line', 'h_-1_un', 'h_-1_ov',
                                'h_+1_line', 'h_+1_un', 'h_+1_ov',
                                'AC_Exp', 'a_main_line', 'a_main_un', 'a_main_ov',
                                'a_-1_line', 'a_-1_un', 'a_-1_ov',
                                'a_+1_line', 'a_+1_un', 'a_+1_ov',
                                'TC_Exp', 'T_main_line', 'T_main_un', 'T_main_ov', 
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


                    del cols_to_add_margin
                    gc.collect()

                    # Create a copy of the DataFrame with the new columns added
                    df_final_wm = df_final.copy()

                    # Display the updated DataFrame
                    st.write(df_final_wm)

                    # warning if not all match  retrieved from API call matches the final df
                    if len(df) != len(fixt_id_list):
                        st.warning('Odds for 1 or more matches not currently available!')


                    # ---------  show simplified odds - just main lines  ---------
                    df_simple = df_final_wm[['Date', 'Home Team', 'Away Team', 'T_main_line', 'T_main_un_w.%', 'T_main_ov_w.%','h_main_line', 'h_main_un_w.%', 'h_main_ov_w.%', 'a_main_line', 'a_main_un_w.%', 'a_main_ov_w.%']]
            
                    #  ----- Calculate Daily Total CORNERS --------

                    # Convert to datetime
                    df_final_wm['Date'] = pd.to_datetime(df_final_wm['Date'], format="%d-%m-%y %H:%M", errors="coerce")

                    # Group by the day only (ignoring time)
                    df_final_wm['Day'] = df_final_wm['Date'].dt.date  # Extract just the date (day)
                    df_final_wm['Time'] = df_final_wm['Date'].dt.time  # Extract just the time (HH:MM:SS)

                    aggregated_corn = df_final_wm.groupby('Day').agg(
                        TC=('TC_Exp', 'sum'), 
                        Match_Count=('TC_Exp', 'size'),
                        Time=('Time', 'first')
                    ).reset_index()

                    df_result_corn = aggregated_corn[aggregated_corn['Match_Count'] >= 2]


                    # ------- Get increment prior to calling poisson functions for Daily Totals  --------------------------------

                    def calculate_increment(main_line):
                        """Determine increment based on main_line value."""
                        if main_line > 35:
                            return 3
                        elif main_line > 14:
                            return 2
                        return 1

                    # -------  Display Simple DF   --------------

                    st.write("---")

                    st.subheader('Main Lines')
                    st.write("")
                    st.write(df_simple)

                    st.subheader("", divider='blue')
                    st.subheader('Total Daily Corners')
                    st.write("")
                    if df_result_corn.shape[0] < 2:
                        st.caption('Less than two matches')

                    else:
                        st.write(df_result_corn)
                 

                        # Get poisson odds and lines for each day returned for Daily SOT
                        for _, row in df_result_corn.iterrows():
                            exp = row['TC'] * 1/OVERS_BOOST * TOTALS_BOOST
                            day = row['Day']
                            time = row['Time']
                            main_line = np.floor(exp) + 0.5

                            increment = calculate_increment(main_line)

                            line_minus_1 = main_line - increment
                            line_minus_2 = main_line - increment * 2
                            line_plus_1 = main_line + increment
                            line_plus_2 = main_line + increment * 2

                            probabilities = poisson_probabilities(exp, main_line, line_minus_1, line_plus_1, line_minus_2, line_plus_2)

                            st.write("")
                            st.write(f'##### Fixtures {day}')

                            margin_configured = ((margin_to_apply * 100 - 100) * 0.5) / 100   # margin to ADD to each side
                            # update probabiities dict by adding to each item
                            probabilities_marginated = {key: value + margin_configured for key, value in probabilities.items()}
                            #st.write(probabilities_marginated)

                            st.caption(f"{day} (with margin)")
                            st.write(f'(Line {line_plus_2}) - Over', round(1 / probabilities_marginated[f'over_plus_2 {line_plus_2}'], 2), f'Under', round(1 / probabilities_marginated[f'under_plus_2 {line_plus_2}'], 2))
                            st.write(f'(Line {line_plus_1}) - Over', round(1 / probabilities_marginated[f'over_plus_1 {line_plus_1}'], 2), f'Under', round(1 / probabilities_marginated[f'under_plus_1 {line_plus_1}'], 2))
                            st.write(f'**(Main Line {main_line}) - Over**', round(1 / probabilities_marginated[f'over_main {main_line}'], 2), f'**Under**', round(1 / probabilities_marginated[f'under_main {main_line}'], 2))
                            st.write(f'(Line {line_minus_1}) - Over', round(1 / probabilities_marginated[f'over_minus_1 {line_minus_1}'], 2), f'Under', round(1 / probabilities_marginated[f'under_minus_1 {line_minus_1}'], 2))
                            st.write(f'(Line {line_minus_2}) - Over', round(1 / probabilities_marginated[f'over_minus_2 {line_minus_2}'], 2), f'Under', round(1 / probabilities_marginated[f'under_minus_2 {line_minus_2}'], 2))
                            st.write("")
                        
                            columns = [
                                        'EVENT TYPE', 'SPORT', 'CATEGORY', 'COMPETITION', 'EVENT NAME', 
                                        'MARKET TYPE NAME', 'LINE', 'SELECTION NAME', 'PRICE', 'START DATE', 
                                        'START TIME', 'OFFER START TIME', 'OFFER END DATE', 'OFFER END TIME', 
                                        'PUBLISHED'
                                    ]

                            # Get today's date in YYYY-MM-DD format
                            today_date = datetime.today().strftime('%Y-%m-%d')

                            # Create an empty DataFrame with 6 rows and specified columns
                            df_csv = pd.DataFrame(index=range(6), columns=columns)

                            # Set the top 6 rows of specific columns
                            df_csv['EVENT TYPE'].iloc[:6] = 'Special'
                            df_csv['SPORT'].iloc[:6] = 'Football'
                            df_csv['CATEGORY'].iloc[:6] = 'Special Offer'
                            df_csv['COMPETITION'].iloc[:6] = 'Daily League Corners'
                            df_csv['MARKET TYPE NAME'].iloc[:6] = 'Daily Corners O/U {line}'


                            df_csv.loc[[0, 2, 4], 'SELECTION NAME'] = 'Over {line}'
                            df_csv.loc[[1, 3, 5], 'SELECTION NAME'] = 'Under {line}'

                            # Assign 'LINE' values
                            df_csv.loc[[0, 1], 'LINE'] = line_minus_1
                            df_csv.loc[[2, 3], 'LINE'] = main_line
                            df_csv.loc[[4, 5], 'LINE'] = line_plus_1

                            df_csv['OFFER START TIME'].iloc[:6] = '09:00:00'
                            df_csv['PUBLISHED'].iloc[:6] = 'NO'
                            df_csv['START DATE'] = today_date  # Today's date
                            df_csv['OFFER END DATE'] = day  
                            df_csv['START TIME'] = time
                            df_csv['OFFER END TIME'] = time

                            # Assign 'PRICE' values (rounded to 2 decimal places)
                            df_csv.loc[0, 'PRICE'] = round(1 / probabilities_marginated[f'over_minus_1 {line_minus_1}'], 2)
                            df_csv.loc[1, 'PRICE'] = round(1 / probabilities_marginated[f'under_minus_1 {line_minus_1}'], 2)
                            df_csv.loc[2, 'PRICE'] = round(1 / probabilities_marginated[f'over_main {main_line}'], 2)
                            df_csv.loc[3, 'PRICE'] = round(1 / probabilities_marginated[f'under_main {main_line}'], 2)
                            df_csv.loc[4, 'PRICE'] = round(1 / probabilities_marginated[f'over_plus_1 {line_plus_1}'], 2)
                            df_csv.loc[5, 'PRICE'] = round(1 / probabilities_marginated[f'under_plus_1 {line_plus_1}'], 2)

                            # Generate the event name using selected_league, match count, and date
                            selected_league_revised = dict_api_to_bk_league_names.get(selected_league, selected_league) # if api league name different from BK league name
                            event_name = f"{selected_league_revised} ({row['Match_Count']} matches {day})"
                            df_csv['EVENT NAME'].iloc[:6] = event_name

                            # Converting multiple columns to string format
                            columns_to_convert = ['START DATE', 'START TIME', 'OFFER END DATE', 'OFFER END TIME']
                            df_csv[columns_to_convert] = df_csv[columns_to_convert].astype(str)

                            df_csv = df_csv.reset_index(drop=True)

                            # Store the dataframe for this date
                            # df_csv_list.append(df_csv)

                            st.write("")
                            st.write('Downloadable FMH upload file')

                            df_csv.set_index('EVENT TYPE', inplace=True)
                            st.write(df_csv)

                            st.write("---")

            except Exception as e:
                st.write(f'An error has occurred whilst compiling: {e}')


if __name__ == "__main__":
    main()
