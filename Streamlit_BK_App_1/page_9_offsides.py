import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize_scalar
# from sklearn.preprocessing import PolynomialFeatures
# import joblib
from mymodule.functions import get_fixtures,  calculate_home_away_lines_and_odds, calculate_true_from_true_raw, calculate_expected_team_goals_from_1x2_refined, poisson_probabilities, team_names_t1x2_to_BK_dict
import requests
import os
from dotenv import load_dotenv
import gc



CURRENT_SEASON = '2025-26'
LAST_SEASON = '2024-25'
OVERS_BOOST = 1.01
TOTALS_BOOST = 1.01


# Functions to calc offsides (more mem efficient than storing joblib files)
# Taken from OFFSIDES_MODELLING_PROJECT in github repo

def predict_home_offsides(HG_Exp, ht_h_r_av_f, at_a_r_av_ag):
    h_pred_raw = np.exp(-0.2340
                + 0.0607 * HG_Exp
                + 0.1257 * ht_h_r_av_f
                + 0.2692 * at_a_r_av_ag) 
    correction_adj = -0.0654 * h_pred_raw + 0.1216
    h_pred = h_pred_raw + correction_adj
    return h_pred

def predict_away_offsides(AG_Exp, ht_h_r_av_ag, at_a_r_av_f):
    a_pred_raw = np.exp(-0.3112
                + 0.1024 * AG_Exp
                + 0.2877 * ht_h_r_av_ag
                + 0.1250 * at_a_r_av_f) 
    correction_adj = -0.0500 * a_pred_raw + 0.0868
    a_pred = a_pred_raw + correction_adj   
    return a_pred

# -----------------------------------------------------------
# The model does not react enough to teams that play an agressively high line. Use this dict to multiple opposition offsides prediction
# Guide team average aginst and multiple to apply: Use t1x2 team graphics (offsides) - Total Avg Match
# 5+: 1.3
# 4.5 - 5: 1.25
# 4 - 4.5: 1.2
# 3.5 - 4: 1.15
# 3 - 3.5: + 1.1

outlier_teams_dict = {
    'Barcelona': 1.25,         # Oct - avg offsides against = 4.87
    'Ath Bilbao': 1.2,          # Oct - 4.27
    'Valencia': 1.1,            # Oct - 3.5
    'Ein Frankfurt': 1.15,     # Oct - 3.84
    'Nantes': 1.1,              # Oct - avg offsides against = 3.4
    'Aston Villa': 1.1
}

# ------------------------------------------------------------

# key = current gw, value is perc of last season
game_week_decay_dict = {
    1: 1,
    2: 0.95,
    3: 0.87,
    4: 0.77,
    5: 0.67,
    6: 0.57,
    7: 0.47,
    8: 0.38,
    9: 0.31,
    10: 0.25,
    11: 0.20,
    12: 0.17,
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
      #  'Corners': ['HC', 'AC', 'TC'],
      #  'Fouls': ['HF', 'AF', 'TF'],
      #  'Shots on Target': ['HST', 'AST', 'TST'],
        'Offsides': ['H_Off', 'A_Off', 'T_Off'],
    }


    # Capture user selections  # WIDGET
    selected_league = st.sidebar.selectbox('Select League', options=list(league_options.values()), label_visibility = 'visible')
    selected_metric = 'Offsides'
 
    df = df[df['League'] == [key for key, value in league_options.items() if value == selected_league][0]]           


    this_df = df[(df['Season'] == CURRENT_SEASON)]  # remove all matches that are not current season
    last_df = df[(df['Season'] == LAST_SEASON)] 

    # delete df to free space
    del df
    gc.collect()

    # team_options = sorted(this_df['HomeTeam'].unique().tolist())

    # -----------------------------------------------------------------------

    st.header(f'{selected_metric} Model - {selected_league}', divider='blue')

    show_model_info = st.checkbox('Model Info')
    if show_model_info:
        st.caption('''
                 Evaluation metrics show non-robust model performance. Likely due to teams alternating tactics depending on oppsition.
                 Not recommended to publish standalone lines unless for marketing purposes or if offering an overs price only with extra margin added.
                 Fine to use as a reference to position ourselves within an established market or 'overs-only' offering with large margin
                 ''')

    # get fixtures
    league_id = leagues_dict.get(selected_league)

    # st.write(this_df)
    ssn_avg_this = round(this_df['T_Off'].mean(), 2)
    ssn_avg_last = round(last_df['T_Off'].mean(), 2)


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

    # this prevents leagues which have play-offs having those games allocated to last season's table
    last_options_df = last_options_df[last_options_df['MP'] > 10]

    # Display last season DataFrame
    show_last_ssn_stats = st.checkbox(f'Show last season {selected_metric} stats', label_visibility = 'visible')  # WIDGET
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


    h_lg_avg = round(this_df[metric_columns[0]].mean(), 2)  # HC or H_SOT
    a_lg_avg = round(this_df[metric_columns[1]].mean(), 2)     # AC or A_SOT


  
    # --- Probability grid HC vs AC ---------------------------------------------------

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

                # Calculate combined probability (70% from Poisson, 30% from Negative Binomial)
                combined_home_prob = 0.7 * poisson_home_prob + 0.3 * nb_home_prob
                combined_away_prob = 0.7 * poisson_away_prob + 0.3 * nb_away_prob

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
        range_value = np.arange(0, 15)
        total_metric_probabilities = np.zeros(16)  # Array for outcomes 0-30
        for home_metric in range_value:
            for away_metric in range_value:
                total_metrics = home_metric + away_metric
                if total_metrics <= 15:
                    total_metric_probabilities[total_metrics] += probability_grid.loc[home_metric, away_metric]

        total_metric_probabilities /= total_metric_probabilities.sum()  # Ensure it sums to 1

        # Create a DataFrame to display the total metric probabilities
        total_metrics_df = pd.DataFrame({
            'Total Metrics': np.arange(len(total_metric_probabilities)),
            'Probability': total_metric_probabilities
        })

        return probability_grid, total_metrics_df, home_more_prob, equal_prob, away_more_prob, total_metric_probabilities
    

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



    # -------------------------------------------- CREATE ODDS FOR ALL UPCOMING FIXTURES --------------------------------------------------------------------

    st.subheader(f'Generate odds for all upcoming {selected_league} matches (up to 7 days ahead)')

    column1, column2, _ = st.columns([1.5,1.5,1])

    with column1:
        # WIDGET
        margin_to_apply = st.number_input('Margin to apply:', step=0.01, value = 1.10, min_value=1.01, max_value=1.2, key='margin_to_apply', label_visibility = 'visible')
        bias_to_apply = st.number_input('Overs bias to apply (reduce overs & increase unders odds by a set %):', step=0.01, value = 1.15, min_value=0.95, max_value=1.30, key='bias_to_apply', label_visibility = 'visible')
        is_bst = st.toggle('Set time outputs if BST(-1hr). Unselected = UTC', value=False)

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

        ###########################  FUDGE CODE TO TEST HOW FAR TO PUSH OUT OVERS  - remove once tested ##########
        over_perc_mults = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]
        overs_multiple = st.selectbox('[TEST] Select multiple to increase overs expectation - applied to Home/Away exps',over_perc_mults)
        ###########################################################################################################

    generate_odds_all_matches = st.button(f'Click to generate')

    if generate_odds_all_matches:
        with st.spinner("Odds being compiled..."):
            try:
                from_date_str = today.strftime("%Y-%m-%d")
                to_date_str = up_to_date.strftime("%Y-%m-%d")
                MARKET_IDS = ['1', '5']             # WDW & Ov/Un
                BOOKMAKERS = ['4']                  # Pinnacle = 4, 365 = 8
                API_SEASON = CURRENT_SEASON[:4]

                df_fixtures = get_fixtures(league_id, from_date_str, to_date_str, API_SEASON)
                if df_fixtures.empty:
                    st.write("No data returned for the specified league and date range.")
                else:
                    df_fixts = df_fixtures[['Fixture ID', 'Date', 'Home Team', 'Away Team']]
                    # st.write('582',df_fixts)

                    fixt_id_list = list(df_fixts['Fixture ID'].unique())

                if not st.secrets:
                    load_dotenv()
                    API_KEY = os.getenv("API_KEY_FOOTBALL_API")

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
                                 'Under 3.5': None,                                             
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
                            if not odds_df.empty:
                                all_odds_df = pd.concat([all_odds_df, odds_df], ignore_index=True)

                            # After using the temporary odds_df, delete it and call garbage collection
                            del odds_df
                            gc.collect()

                    # Display the collected odds
                    # st.write('666', all_odds_df) 

                    # Use groupby and fillna to collapse rows and remove None values
                    df_collapsed = all_odds_df.groupby('Fixture ID').first().combine_first(
                        all_odds_df.groupby('Fixture ID').last()).reset_index()

                    # st.write('672', df_collapsed) 
                

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


                    # Merge odds df_fixts with df_collapsed
                    df = df_fixts.merge(df_collapsed, on='Fixture ID')

                    del df_collapsed
                    gc.collect()
                    
                    # st.write('681',df) 
                    if df.empty:
                        st.write('Odds currently unavailable from API') 


                    #  ---------------  Create true wdw odds ---------------
                    # Convert columns to numeric (if they are strings or objects)
                    df['Home Win'] = pd.to_numeric(df['Home Win'], errors='coerce')
                    df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce')
                    df['Away Win'] = pd.to_numeric(df['Away Win'], errors='coerce')

                    df['Over 2.5'] = pd.to_numeric(df['Over 2.5'], errors='coerce')
                    df['Under 2.5'] = pd.to_numeric(df['Under 2.5'], errors='coerce')

                    
                    df[['hg_ex', 'ag_ex']] = df.apply(
                        lambda row: calculate_expected_team_goals_from_1x2_refined(row['Home Win'], row['Draw'], row['Away Win'], row['Over 2.5'], row['Under 2.5']), 
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

                    # st.write('723:', df) 

                    # outlier_team_dict - fudge to account for aggressive high line teams
                    df['outlier_mult_h'] = df['Home Team'].map(outlier_teams_dict).fillna(1)
                    df['outlier_mult_a'] = df['Away Team'].map(outlier_teams_dict).fillna(1)

                    # st.write('1058', df) 

                    # ------------------------ APPLY MODELS ---------------------------------------


                    df['HO_Exp'] = round(predict_home_offsides(df['hg_ex'], df['H_h_for'], df['A_a_ag']) * df['outlier_mult_a'], 2)
                    df['AO_Exp'] = round(predict_away_offsides(df['ag_ex'], df['H_h_ag'], df['A_a_for']) * df['outlier_mult_h'], 2)
                    df['TO_Exp'] = df['HO_Exp'] + df['AO_Exp']
                    # st.write(df) 

                    ##############################  TEMP OVERS FUDGE ADJ - remove once tested ##################
                    df['HO_Exp'] = round(df['HO_Exp'] * overs_multiple,2)
                    df['AO_Exp'] = round(df['AO_Exp'] * overs_multiple,2)
                    #############################################################################################

                    # st.write('739', df[['HO_Exp', 'AO_Exp', 'TO_Exp']]) 

                    try:
                        # calculate_corners_lines_and_odds(prediction)
                        df[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = df.apply(
                            lambda row: calculate_home_away_lines_and_odds(row['HO_Exp'], selected_metric), 
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
                        df['HO_Exp'] = 0
                        df[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 
                            'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = 0


                    # ------ AWAY -----------

                    try:
                        # calculate_corners_lines_and_odds(prediction)
                        df[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = df.apply(
                            lambda row: calculate_home_away_lines_and_odds(row['AO_Exp'], selected_metric), 
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
                        df['AO_Exp'] = 0
                        df[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 
                            'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = 0
                    
                    # st.write('784',df) 
                    # --------  TOTAL ---------------

                    df[['TO_Exp', 'T_main_line', 'T_-1_line', 'T_+1_line', 'T_-2_line', 'T_+2_line','T_main_under_%', 
                        'T_main_over_%', 'T_-1_under_%', 'T_-1_over_%', 'T_+1_under_%', 
                        'T_+1_over_%', 'T_-2_under_%', 'T_-2_over_%', 'T_+2_under_%', 
                        'T_+2_over_%',]] = df.apply(
                        lambda row: calculate_totals_lines_and_odds(
                            row['HO_Exp'], 
                            row['AO_Exp'], 
                            total_metrics_df=calculate_probability_grid_hc_vs_ac(row['HO_Exp'], row['AO_Exp'])[1]
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

                    # st.write('811', df)
                    df[['H_most_%', 'Tie_%', 'A_most_%']] = df.apply(
                        lambda row: pd.Series(calculate_probability_grid_hc_vs_ac(row['HO_Exp'], row['AO_Exp'])[2:5]), 
                        axis=1, 
                        result_type='expand'
                    )


                    # -------------------------------------------------------

                    df['H_most'] = round(1 / df['H_most_%'], 2)
                    df['Tie'] = round(1 / df['Tie_%'], 2)
                    df['A_most'] = round(1 / df['A_most_%'], 2)

                    # Sub-select final columns
                    df_final = df[['Date', 'Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win',
                                'HO_Exp', 'h_main_line', 'h_main_un', 'h_main_ov', 
                                'h_-1_line', 'h_-1_un', 'h_-1_ov',
                                'h_+1_line', 'h_+1_un', 'h_+1_ov',
                                'AO_Exp', 'a_main_line', 'a_main_un', 'a_main_ov',
                                'a_-1_line', 'a_-1_un', 'a_-1_ov',
                                'a_+1_line', 'a_+1_un', 'a_+1_ov',
                                'TO_Exp', 'T_main_line', 'T_main_un', 'T_main_ov', 
                                'T_-1_line', 'T_-1_un', 'T_-1_ov',
                                'T_+1_line', 'T_+1_un', 'T_+1_ov',
                                'T_-2_line', 'T_-2_un', 'T_-2_ov',
                                'T_+2_line', 'T_+2_un', 'T_+2_ov',
                            #    '<9.5', '10-12', '13+',
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
                        st.warning('Odds for 1 or more matches not currently available')

                    # --------------------------------------------------------------------

                    # FORMAT EACH MATCH FOR FMH UPLOAD

                    # firstly allign streamlit team names with BK team names
                    df_final_wm['Home Team Alligned'] = df_final_wm['Home Team'].map(team_names_t1x2_to_BK_dict).fillna(df_final_wm['Home Team'])
                    df_final_wm['Away Team Alligned'] = df_final_wm['Away Team'].map(team_names_t1x2_to_BK_dict).fillna(df_final_wm['Away Team'])

                    # CREATE AN INDIVIDUAL DF FOR EACH MATCH

                    today_date = datetime.today().strftime('%Y-%m-%d')

                    columns = [
                            'EVENT TYPE', 'SPORT', 'CATEGORY', 'COMPETITION', 'EVENT NAME', 
                            'MARKET TYPE NAME', 'LINE', 'SELECTION NAME', 'PRICE', 'START DATE', 
                            'START TIME', 'OFFER START DATE', 'OFFER START TIME', 'OFFER END DATE', 'OFFER END TIME', 
                            'PUBLISHED'
                        ]
                    
                    fmh_comp_dict = {
                        'Spain La Liga': 'LaLiga',
                        'England Premier': 'Premier League',
                        'Italy Serie A': 'Serie A',
                        'Germany Bundesliga': 'Bundesliga',
                        'France Ligue 1': 'Ligue 1',
                        'South Africa Premier': 'Premier League',
                        'Scotland Premier': 'Premiership',
                        'Netherlands Eredivisie': 'Eredivisie',
                        'Belgium Jupiler': 'Pro League',
                        'England Championship': 'Championship',
                        'England League One': 'League One',
                        'England League Two': 'League Two'
                    }

                    # --- Preprocess Date column once ---
                    df_final_wm["Date"] = pd.to_datetime(df_final_wm["Date"], format="%d-%m-%y %H:%M")
                    df_final_wm["START DATE"] = df_final_wm["Date"].dt.strftime("%Y-%m-%d")

                    # Adjust START TIME depending on BST toggle (vectorized)
                    if is_bst:
                        df_final_wm["START TIME"] = (df_final_wm["Date"] - pd.Timedelta(hours=1)).dt.strftime("%H:%M:%S")
                    else:
                        df_final_wm["START TIME"] = df_final_wm["Date"].dt.strftime("%H:%M:%S")

                    rows_list = []  # store each row

                    for idx, row in df_final_wm.iterrows():
                        # Create an empty DataFrame with 9 rows and specified columns
                        df_row = pd.DataFrame(index=range(3), columns=columns)              # Change to 9 when individual teams added

                        # create category_lg variable
                        if selected_league.startswith("South Africa"):
                            category_lg = "South Africa"
                        else:
                            category_lg = selected_league.split(" ")[0]

                        # create competition variable
                        competition = fmh_comp_dict.get(selected_league)

                        # create event_name variable
                        # extract Home Team and Away Team, make BK team name compatible, make as a vs b and store
                        event_name = row['Home Team Alligned'] + " vs " + row["Away Team Alligned"]

                        # Set the specific columns
                        df_row['EVENT TYPE'].iloc[:3] = 'Match' # make [:9] when team offsides offered
                        df_row['SPORT'].iloc[:3] = 'Football'
                        df_row['CATEGORY'].iloc[:3] = category_lg
                        df_row['COMPETITION'].iloc[:3] = competition
                        df_row['EVENT NAME'].iloc[:3] = event_name

                        df_row['MARKET TYPE NAME'].iloc[:3] = 'Total offsides {line} Over'
                        # df_row['MARKET TYPE NAME'].iloc[3:6] = '{competitor1} total offsides {line} Over'
                        # df_row['MARKET TYPE NAME'].iloc[6:9] = '{competitor2} total offsides {line} Over'

                        df_row['LINE'].iloc[0] = row['T_-1_line']
                        df_row['LINE'].iloc[1] = row['T_main_line']
                        df_row['LINE'].iloc[2] = row['T_+1_line']
                        # df_row['LINE'].iloc[3] = row['h_-1_line']
                        # df_row['LINE'].iloc[4] = row['h_main_line']
                        # df_row['LINE'].iloc[5] = row['h_+1_line']
                        # df_row['LINE'].iloc[6] = row['a_-1_line']
                        # df_row['LINE'].iloc[7] = row['a_main_line']
                        # df_row['LINE'].iloc[8] = row['a_+1_line']

                        df_row['SELECTION NAME'].iloc[:3] = 'over {line}'  # make [:9] when team offsides offered

                        df_row['PRICE'].iloc[0] = row['T_-1_ov_w.%']
                        df_row['PRICE'].iloc[1] = row['T_main_ov_w.%']
                        df_row['PRICE'].iloc[2] = row['T_+1_ov_w.%']
                        # df_row['PRICE'].iloc[3] = row['h_-1_ov_w.%']
                        # df_row['PRICE'].iloc[4] = row['h_main_ov_w.%']
                        # df_row['PRICE'].iloc[5] = row['h_+1_ov_w.%']
                        # df_row['PRICE'].iloc[6] = row['a_-1_ov_w.%']
                        # df_row['PRICE'].iloc[7] = row['a_main_ov_w.%']
                        # df_row['PRICE'].iloc[8] = row['a_+1_ov_w.%']

                        # Dates & Times (already preprocessed)
                        start_date = row["START DATE"]
                        start_time = row["START TIME"]  # already adjusted for BST/UTC

                        df_row['START DATE'] = start_date
                        df_row['START TIME'] = start_time
                        df_row['OFFER START DATE'] = today_date #3
                        df_row['OFFER START TIME'] = '09:00:00' #4
                        df_row['OFFER END DATE'] = start_date
                        df_row['OFFER END TIME'] = start_time
                        df_row['PUBLISHED'] = 'YES' #7

                        # Finally, append to list
                        rows_list.append(df_row)

                    # Concatenate all blocks into one DataFrame
                    df_fmh_format = pd.concat(rows_list, ignore_index=True)
                    st.subheader('FMH Format')
                    st.write(df_fmh_format)

            
                    # #  ----- Calculate Daily Total OFFSIDES --------

                    # # Convert to datetime
                    # df_final_wm['Date'] = pd.to_datetime(df_final_wm['Date'], format="%d-%m-%y %H:%M", errors="coerce")

                    # # Group by the day only (ignoring time)
                    # df_final_wm['Day'] = df_final_wm['Date'].dt.date  # Extract just the date (day)

                    # aggregated_offs = df_final_wm.groupby('Day').agg(
                    #     TO=('TO_Exp', 'sum'), 
                    #     Match_Count=('TO_Exp', 'size')
                    # ).reset_index()

                    # df_result_offs = aggregated_offs[aggregated_offs['Match_Count'] >= 2]


                    # # ------- Get increment prior to calling poisson functions for Daily Totals  --------------------------------

                    # def calculate_increment(main_line):
                    #     """Determine increment based on main_line value."""
                    #     if main_line > 35:
                    #         return 3
                    #     elif main_line > 14:
                    #         return 2
                    #     return 1


                    # # ---------------------------------------------------

            #         st.subheader("", divider='blue')
            #         st.subheader('Total Daily Offsides')
            #         st.write("")
            #         # if df_result_offs.shape[0] < 2:
            #         #     st.caption('Less than two matches')

            #         st.write(df_result_offs)
            #         st.warning('Check correct number of fixtures have been logged for each day', icon="⚠️")

            #         # Get poisson odds and lines for each day returned for Daily SOT
            #         for _, row in df_result_offs.iterrows():
            #             exp = row['TO'] * 1/OVERS_BOOST * TOTALS_BOOST
            #             day = row['Day']
            #             main_line = np.floor(exp) + 0.5

            #             increment = calculate_increment(main_line)

            #             line_minus_1 = main_line - increment
            #             line_minus_2 = main_line - increment * 2
            #             line_plus_1 = main_line + increment
            #             line_plus_2 = main_line + increment * 2

            #             probabilities = poisson_probabilities(exp, main_line, line_minus_1, line_plus_1, line_minus_2, line_plus_2)

            #             st.caption(f"{day} (100% Prices)")
            #             st.write(f'(Line {line_plus_2}) - Over', round(1 / probabilities[f'over_plus_2 {line_plus_2}'], 2), f'Under', round(1 / probabilities[f'under_plus_2 {line_plus_2}'], 2))
            #             st.write(f'(Line {line_plus_1}) - Over', round(1 / probabilities[f'over_plus_1 {line_plus_1}'], 2), f'Under', round(1 / probabilities[f'under_plus_1 {line_plus_1}'], 2))
            #             st.write(f'**(Main Line {main_line}) - Over**', round(1 / probabilities[f'over_main {main_line}'], 2), f'**Under**', round(1 / probabilities[f'under_main {main_line}'], 2))
            #             st.write(f'(Line {line_minus_1}) - Over', round(1 / probabilities[f'over_minus_1 {line_minus_1}'], 2), f'Under', round(1 / probabilities[f'under_minus_1 {line_minus_1}'], 2))
            #             st.write(f'(Line {line_minus_2}) - Over', round(1 / probabilities[f'over_minus_2 {line_minus_2}'], 2), f'Under', round(1 / probabilities[f'under_minus_2 {line_minus_2}'], 2))
            #             st.write("")



            except Exception as e:
                st.write(f'An error occured: {e}')

    # -------------------------------------------------------------------------------

    # Compile any match / competition  

    st.write("---")
    with st.expander('Single match pricer (any competition)'):
        
        st.write("")
        st.caption('Fill in match odds and O/U. Use corner-stats.com to get Home, Away & Competition stats. If early season mix in with previous.')
        st.write('Enter Match Odds:')
        c1,c2,c3,c4, c5 = st.columns([1,1,1,1,5])
        with c1:
            try:
                h_odds = float(st.text_input('Home Odds', value = 2.10, label_visibility = 'visible'))  # WIDGET
            except ValueError:
                    st.error("Please enter a valid number.")
                    return
        with c2:
            try:
                d_odds = float(st.text_input('Draw Odds', value = 3.40, label_visibility = 'visible'))  # WIDGET
            except ValueError:
                    st.error("Please enter a valid number.")
                    return
        with c3:
            try:
                a_odds = float(st.text_input('Away Odds', value = 3.50, label_visibility = 'visible'))  # WIDGET
            except ValueError:
                    st.error("Please enter a valid number.")
                    return
        
        with c4:
            st.write("")
            st.write("")
            margin = round(1/h_odds + 1/d_odds + 1/a_odds, 2)
            st.write('Margin:', margin)


        h_pc_true_raw = 1/(h_odds * margin)
        d_pc_true_raw = 1/(d_odds * margin)
        a_pc_true_raw = 1/(a_odds * margin)


        # Error message if < 100 %
        if margin < 1:
            st.warning('Margin must be > 1.00 !')

        #  -----------   Ov/Und Odds and HG/AG Expectation calculation  -----------------
        col1,col2,col3, _ , _ = st.columns([1,1,1,1,5])
        with col1:
            try:
                ov_odds = float(st.text_input('Over 2.5 Odds', value = 1.9, label_visibility = 'visible')) # WIDGET
            except ValueError:
                    st.error("Please enter a valid number.")
        with col2:
            try:
                un_odds = float(st.text_input('Under 2.5 Odds', value = 1.9, label_visibility = 'visible')) # WIDGET
            except ValueError:
                    st.error("Please enter a valid number.")

        margin_ou = round(1/ov_odds + 1/un_odds, 2)

        ov_pc_true_raw = 1/(ov_odds * margin_ou)
        un_pc_true_raw = 1/(un_odds * margin_ou)

        with col3:
            st.write("")
            st.write("")
            st.write("")
            st.write('Margin:', margin_ou)


        # Error message if < 100 %
        if margin_ou < 1:
            st.warning('Odds must be > 1.00 !')

        # --------------------------------------------------------


        _ , d_pc_true, _ = calculate_true_from_true_raw(h_pc_true_raw, d_pc_true_raw, a_pc_true_raw, margin)

        # takes match and goals odds as args and returns hg and ag
        hg_ex, ag_ex = calculate_expected_team_goals_from_1x2_refined(h_pc_true_raw, d_pc_true_raw, a_pc_true_raw, ov_pc_true_raw , un_pc_true_raw)

        # st.write('h_pc_true, d_pc_true, a_pc_true', h_pc_true, d_pc_true, a_pc_true)

        cls1, _, cls3 = st.columns([7,1,7])
        with cls1:
            st.subheader('Home')
            ht_h_for = st.number_input('Avg home team home Offsides - for')
            at_a_ag = st.number_input('Avg away team away Offsides - against')
            st.write("---")

        with cls3:
            st.subheader('Away')
            at_a_for = st.number_input('Avg away team away Offsides - for')
            ht_h_ag = st.number_input('Avg home team home Offsides - against')
            st.write("---")

        h_off_exp = predict_home_offsides(hg_ex, ht_h_for, at_a_ag)
        a_off_exp = predict_away_offsides(ag_ex, ht_h_ag, at_a_for)


        # -------------  Additional factors  ---------
        with cls1:
            is_neutral = st.selectbox('Is this on a neutral pitch?', ['No', 'Yes'])
            is_extra_time = st.selectbox('Is extra-time possible?', ['No', 'Yes'])

        # home teams have 0.516 of total offsides, away team have 0.484
        # apply below factors if match is on a neutral (0.5 / 0.516) 
        is_neutral_factor_home = 0.97 if is_neutral=='Yes' else 1 
        is_neutral_factor_away = 1.03 if is_neutral=='Yes' else 1  

        # Is extra time possible 
        if is_extra_time == 'Yes':
            extra_time_factor = d_pc_true * 0.33 + 1     # (extra time is 31/93 (0.33) of total offsides)
        else:
            extra_time_factor = 1


        # ----------------------------

        home_prediction = round(h_off_exp * extra_time_factor * is_neutral_factor_home, 2)
        
        with cls1:
            st.success(f'Home Prediction: {home_prediction}')
            show_mult_home = st.checkbox('Show home multiples')
            if show_mult_home:
                st.write('ET mult', extra_time_factor, 'Neutral mult', is_neutral_factor_home)


        away_prediction = round(a_off_exp * extra_time_factor * is_neutral_factor_away, 2)

        with cls3:
            for i in range(11):
                st.write("")

            st.success(f'Away Prediction: {away_prediction}')
            show_mult_away = st.checkbox('Show away multiples')
            if show_mult_away:
                st.write('ET mult', extra_time_factor, 'Neutral mult', is_neutral_factor_away)



        df_single = pd.DataFrame([{
            'H_Off_Exp': home_prediction,
            'A_Off_Exp': away_prediction
        }])

        # st.write(df_single)

        # calculate_sot_lines_and_odds(prediction) - HOME
        df_single[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = df_single.apply(
            lambda row: calculate_home_away_lines_and_odds(row['H_Off_Exp'], selected_metric), 
            axis=1, result_type='expand')
        
        # calculate_corners_lines_and_odds(prediction) - AWAY
        df_single[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = df_single.apply(
            lambda row: calculate_home_away_lines_and_odds(row['A_Off_Exp'], selected_metric), 
            axis=1, result_type='expand')
        
        # st.write(df_single)

        df_single['h_main_un'] = round(1 / df_single['h_main_under_%'], 2)
        df_single['h_main_ov'] = round(1 / df_single['h_main_over_%'], 2)
        df_single['h_-1_un'] = round(1 / df_single['h_-1_under_%'], 2)
        df_single['h_-1_ov'] = round(1 / df_single['h_-1_over_%'], 2)
        df_single['h_+1_un'] = round(1 / df_single['h_+1_under_%'], 2)
        df_single['h_+1_ov'] = round(1 / df_single['h_+1_over_%'], 2)


        df_single['a_main_un'] = round(1 / df_single['a_main_under_%'], 2)
        df_single['a_main_ov'] = round(1 / df_single['a_main_over_%'], 2)
        df_single['a_-1_un'] = round(1 / df_single['a_-1_under_%'], 2)
        df_single['a_-1_ov'] = round(1 / df_single['a_-1_over_%'], 2)
        df_single['a_+1_un'] = round(1 / df_single['a_+1_under_%'], 2)
        df_single['a_+1_ov'] = round(1 / df_single['a_+1_over_%'], 2)
        
        # st.write(df_single)


        # --------  TOTAL ---------------

        df_single[['T_Off_Exp', 'T_main_line', 'T_-1_line', 'T_+1_line', 'T_-2_line', 'T_+2_line','T_main_under_%', 
            'T_main_over_%', 'T_-1_under_%', 'T_-1_over_%', 'T_+1_under_%', 
            'T_+1_over_%', 'T_-2_under_%', 'T_-2_over_%', 'T_+2_under_%', 
            'T_+2_over_%',]] = df_single.apply(
            lambda row: calculate_totals_lines_and_odds(
                row['H_Off_Exp'], 
                row['A_Off_Exp'], 
                total_metrics_df=calculate_probability_grid_hc_vs_ac(row['H_Off_Exp'], row['A_Off_Exp'])[1]
            ),
            axis=1, 
            result_type='expand'
        )

        df_single['T_main_un'] = round(1 / df_single['T_main_under_%'], 2)
        df_single['T_main_ov'] = round(1 / df_single['T_main_over_%'], 2)
        df_single['T_-1_un'] = round(1 / df_single['T_-1_under_%'], 2)
        df_single['T_-1_ov'] = round(1 / df_single['T_-1_over_%'], 2)
        df_single['T_+1_un'] = round(1 / df_single['T_+1_under_%'], 2)
        df_single['T_+1_ov'] = round(1 / df_single['T_+1_over_%'], 2)

        # Sub-select final columns
        df_single_final = df_single[[
                    'H_Off_Exp', 'h_main_line', 'h_main_un', 'h_main_ov', 
                    'h_-1_line', 'h_-1_un', 'h_-1_ov',
                    'h_+1_line', 'h_+1_un', 'h_+1_ov',
                    'A_Off_Exp', 'a_main_line', 'a_main_un', 'a_main_ov',
                    'a_-1_line', 'a_-1_un', 'a_-1_ov',
                    'a_+1_line', 'a_+1_un', 'a_+1_ov',
                    'T_Off_Exp', 'T_main_line', 'T_main_un', 'T_main_ov', 
                    'T_-1_line', 'T_-1_un', 'T_-1_ov',
                    'T_+1_line', 'T_+1_un', 'T_+1_ov',
                    ]].copy()

        
        st.subheader('Lines & Odds (100%)')
        st.write(df_single_final)


if __name__ == "__main__":
    main()