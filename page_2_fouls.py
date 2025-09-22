import streamlit as st
import pandas as pd
# import altair as alt
import numpy as np
from datetime import datetime, timedelta
import time
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom
from sklearn.preprocessing import PolynomialFeatures
import joblib
from mymodule.functions import get_fixtures,  calculate_home_away_lines_and_odds, poisson_probabilities, calculate_true_from_true_raw, team_names_t1x2_to_BK_dict
import requests
import os
from dotenv import load_dotenv


dict_api_to_bk_league_names = {
     'England Premier':'England Premier League',
     'Spain La Liga' : 'Spain LaLiga',
 }

CURRENT_SEASON = '2025-26'
LAST_SEASON = '2024-25'
OVERS_BOOST = 1.03 # increase all overs expectations by this amount as a foundation. 26.5 > 27.3. Odds change outputs also dafaulted on front-end.
TOTALS_BOOST = 1.02 # increase daily totals by this factor

fouls_model_h = joblib.load('models/fouls/fouls_home_neg_binom_4.pkl')
fouls_model_a = joblib.load('models/fouls/fouls_away_neg_binom_4.pkl')

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


# --------------- Probability grid HC vs AC -----------------

# Enter HC and AC exps' to return probability_grid, total_metrics_df (df of probability of each band), home_more_prob, equal_prob, away_more_prob (for matchups), total_metric_probabilities
def calculate_probability_grid_hc_vs_ac(home_prediction, away_prediction):
    # Set the range for corners
    metric_range = np.arange(0, 50)

    # Initialize a DataFrame to store probabilities
    probability_grid = pd.DataFrame(index=metric_range, columns=metric_range)

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
    range_value = np.arange(0, 50)
    total_metric_probabilities = np.zeros(51)  # Array for outcomes 0-30
    for home_metric in range_value:
        for away_metric in range_value:
            total_metrics = home_metric + away_metric
            if total_metrics <= 50:
                total_metric_probabilities[total_metrics] += probability_grid.loc[home_metric, away_metric]

    total_metric_probabilities /= total_metric_probabilities.sum()  # Ensure it sums to 1

    # Create a DataFrame to display the total metric probabilities
    total_metrics_df = pd.DataFrame({
        'Total Metrics': np.arange(len(total_metric_probabilities)),
        'Probability': total_metric_probabilities
    })

    return probability_grid, total_metrics_df, home_more_prob, equal_prob, away_more_prob, total_metric_probabilities


# ------------- CALCULATE MAIN & ALT LINES & ODDS (TOTAL) --------------------------

# takes home_prediction, away_prediction and total metrics df as args and returns lines (main, minor, major) and over/under %'s
# different function to that used for home and away
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

    # st.write(df)

    # Sidebar for user input
    st.sidebar.title('Select Data Filters')

    # Define selection options
    league_options = {
        # 'All_leagues': 'ALL',  # Uncomment for future development
        'Premier League': 'England Premier',
        'La Liga': 'Spain La Liga',
        'Bundesliga': 'Germany Bundesliga',
        'Ligue 1': 'France Ligue 1',
        'Serie A': 'Italy Serie A',
        'Premier Soccer League': 'South African Premier',
        'Premiership': 'Scotland Premier',
     #   'Eredivisie': 'Netherlands Eredivisie',
     #   'Jupiler Pro League': 'Belgium Jupiler',
     #   'Primeira Liga': 'Portugal Liga I',
        'Championship': 'England Championship',
     #   '2. Bundesliga': 'Germany 2 Bundesliga',
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
        "Scotland Premier": '179',
        'South African Premier': '288'
    }

    metric_options = {
        'Corners': ['HC', 'AC', 'TC'],
        'Fouls': ['HF', 'AF', 'TF'],
        'Shots on Target': ['HST', 'AST', 'TST'],
      #  'Shots': ['HS', 'AS', 'TS'],
    }


    # Capture user selections
    selected_league = st.sidebar.selectbox('Select League', options=list(league_options.values()), label_visibility = 'visible')
    selected_metric = 'Fouls'
 
    df = df[df['League'] == [key for key, value in league_options.items() if value == selected_league][0]]           

    this_df = df[(df['Season'] == CURRENT_SEASON)]  # remove all matches that are not current season
    last_df = df[(df['Season'] == LAST_SEASON)] 

    # -----------------------------------------------------------------------

    st.header(f'{selected_metric} Model', divider='blue')

    show_model_info = st.checkbox('Model Info')
    if show_model_info:
        st.caption('''
                 Good model evaluation metrics (Home - R2 0.25, MSE 12.34; Away - R2 0.23, MSE 13.12). Lines and prices all good to publish. 
                 Margin set to 10% given high variance of outputs
                 and likelihood of competitor line differences.
                 ''')

    # get fixtures
    league_id = leagues_dict.get(selected_league)

    # st.write(this_df)
    ssn_avg_this = round(this_df['TF'].mean(), 2)
    ssn_avg_last = round(last_df['TF'].mean(), 2)

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
    show_this_ssn_stats = st.checkbox(f'Show current season {selected_metric} stats')
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

    # this prevents leagues which have play-offs having those games allocated to last season's table
    last_options_df = last_options_df[last_options_df['MP'] > 10]

    # # Display last season DataFrame
    show_last_ssn_stats = st.checkbox(f'Show last season {selected_metric} stats')
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

    # st.write(teams_to_remove)

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


    show_df_mix = st.checkbox(f'Show team {selected_metric} stats (weighted current & previous season)')
    if show_df_mix:
        st.write(df_mix)
        st.caption('''
                 Current season and previous season statistics are merged based on a weighting of number of games through the current season.
                 Previous season data decays logarithmically from 100% at game 1 to 0 % by game 24. Teams new to a division are allocated
                 an initial defaulted previous season 1st or 3rd league quantile value (depending if promoted or relegated in), so predictions for those teams may be less reliable early season.
                 ''')


    # Get metric columns for the selected metric
    metric_columns = metric_options[selected_metric][:2]  # First two columns only, HC and AC for 'Corners'


    # --------------------- Get Ref Data -----------------
    # st.write(df)
    # Step 1: Clean the 'Referee' column
    df['Referee'] = df['Referee'].str.split(',').str[0]  # Keep only the part before the comma
    # Step 2: Group by Referee and calculate average TF
    df_avg_tf_per_ref = df.groupby('Referee')['TF'].mean().reset_index()
    df_avg_tf_per_ref.columns = ['Referee', 'Avg_Fouls']  # Rename columns for clarity
    # Step 3: Calculate the overall average TF
    overall_avg_tf = df['TF'].mean()
    # Step 4: Calculate the multiple
    df_avg_tf_per_ref['Multiple'] = round(df_avg_tf_per_ref['Avg_Fouls'] / overall_avg_tf, 2)
    # Calculate mean and standard deviation of the 'Multiple' column
    mean_multiple = df_avg_tf_per_ref['Multiple'].mean()
    std_multiple = df_avg_tf_per_ref['Multiple'].std()
    # Define the thresholds
    thresholds = {
        "0.5_above": mean_multiple + 0.5 * std_multiple,
        "1_above": mean_multiple + 1 * std_multiple,
        "1.5_above": mean_multiple + 1.5 * std_multiple,
        "2_above": mean_multiple + 2 * std_multiple,
        "0.5_below": mean_multiple - 0.5 * std_multiple,
        "1_below": mean_multiple - 1 * std_multiple,
        "1.5_below": mean_multiple - 1.5 * std_multiple,
        "2_below": mean_multiple - 2 * std_multiple,
    }
    # Define conditions and corresponding factor values
    conditions = [
        (df_avg_tf_per_ref['Multiple'] > thresholds["2_above"]),
        (df_avg_tf_per_ref['Multiple'] > thresholds["1.5_above"]),
        (df_avg_tf_per_ref['Multiple'] > thresholds["1_above"]),
        (df_avg_tf_per_ref['Multiple'] > thresholds["0.5_above"]),
        (df_avg_tf_per_ref['Multiple'] < thresholds["2_below"]),
        (df_avg_tf_per_ref['Multiple'] < thresholds["1.5_below"]),
        (df_avg_tf_per_ref['Multiple'] < thresholds["1_below"]),
        (df_avg_tf_per_ref['Multiple'] < thresholds["0.5_below"]),
    ]

    factors = [
        1.05,  # More than 2 SD above
        1.04,  # More than 1.5 SD above
        1.03,  # More than 1 SD above
        1.02,  # More than 0.5 SD above
        0.95,  # More than 2 SD below
        0.96,  # More than 1.5 SD below
        0.97,  # More than 1 SD below
        0.98,  # More than 0.5 SD below
    ]

    # Default factor is 1.00 for values within 0.5 SD of the mean
    default_factor = 1.00

    # Use np.select to apply the conditions
    df_avg_tf_per_ref['Factor'] = np.select(conditions, factors, default=default_factor)

    # Make ref names consistent initial.surname
    df_avg_tf_per_ref['Referee'] = df_avg_tf_per_ref['Referee'].apply(
    lambda name: name if (len(name.split()) == 2 and name[1] == '.' and name[0].isupper()) 
    else f"{name.split()[0][0].upper()}. {name.split()[1]}" if len(name.split()) == 2 
    else name
    )

    # Handle any duplicate ref names (if some matches were logged as full name and others as initial.surname)
    # Group by 'Referee' and calculate the mean for specified columns
    columns_to_average = ['Avg_Fouls', 'Multiple', 'Factor']
    averaged = df_avg_tf_per_ref.groupby('Referee', as_index=False)[columns_to_average].mean()
    df_avg_tf_per_ref = averaged

    # Display the result
    # st.write(df_avg_tf_per_ref)


    h_lg_avg = round(this_df[metric_columns[0]].mean(), 2)  # HC or H_SOT
    a_lg_avg = round(this_df[metric_columns[1]].mean(), 2)     # AC or A_SOT



    # -------------------------------------------- CREATE ODDS FOR ALL UPCOMING FIXTURES --------------------------------------------------------------------


    st.subheader(f'Generate Odds for all upcoming {selected_league} matches (up to 7 days ahead)')

    column1, _ = st.columns([1,2])

    with column1:
        margin_to_apply = st.number_input('Margin to apply:', step=0.01, value = 1.10, min_value=1.01, max_value=1.2, key='margin_to_apply', label_visibility = 'visible')
        # over bias set to 1.07 pre overs only being published
        bias_to_apply = st.number_input('Overs bias to apply (reduce overs & increase unders odds by a set %):', step=0.01, value = 1.15, min_value=1.00, max_value=1.30, key='bias_to_apply', label_visibility = 'visible')
        is_bst = st.toggle('Set time outputs if BST(-1hr). Unselected = UTC', value=True)

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
                    df_fixts = df_fixtures[['Fixture ID', 'Date', 'Referee', 'Home Team', 'Away Team']]
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
                                'Over 2.5': None,
                                'Under 2.5': None
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

                    # Display the collected odds
                    # st.write('all odds df 670', all_odds_df)

                    # Use groupby and fillna to collapse rows and remove None values
                    df_collapsed = all_odds_df.groupby('Fixture ID').first().combine_first(
                        all_odds_df.groupby('Fixture ID').last()).reset_index()

                    # st.write(df_collapsed)

                    # Merge odds df_fixts with df_collapsed
                    df = df_fixts.merge(df_collapsed, on='Fixture ID')
                    df = df.dropna(subset=['Home Win', 'Draw', 'Away Win', 'Over 2.5', 'Under 2.5'])
                    # st.write(df)
                    if df.empty:
                        st.write('Odds currently unavailable from API') 

                    # ---------------- Incorporate Distance/Derby factor --------

                    # 1. Now check for LOCAL derbies.

                    df_dist_grid = pd.read_csv(f'data/coordinates/dist_grid_{selected_league}.csv')
                    # st.write(df_dist_grid)
                    # Setting 'Unnamed:0' as the index of dist_grid for easier lookup
                    df_dist_grid.set_index('Unnamed: 0', inplace=True)

                    # Now, loop over each row in df to extract the distance from dist_grid
                    df['Dist'] = df.apply(lambda row: df_dist_grid.at[row['Home Team'], row['Away Team']], axis=1)

                    # Function to assign Derby_mult based on Dist value
                    def get_derby_mult(dist):
                        if dist < 8:
                            return 1.03
                        elif 8 <= dist < 15:
                            return 1.02
                        elif 15 <= dist < 24:
                            return 1.015
                        elif 24 <= dist < 33:                       
                            return 1.01
                        elif 33 <= dist < 40:
                            return 1.00
                        else:
                            return 1.00
                        
                    df['Derby_mult'] = df['Dist'].apply(get_derby_mult)


                    # 1. Check for NON local derbies first 1st, reference csv
                    df_non_local_derby = pd.read_csv('data/derbies_non_local.csv')
                    # st.write(df_non_local_derby)

                    # Merge df with df_non_local_derby on matching Home Team and Away Team
                    merged_df = df.merge(df_non_local_derby[['Home Team', 'Away Team', 'Mult']], 
                                        left_on=['Home Team', 'Away Team'], 
                                        right_on=['Home Team', 'Away Team'], 
                                        how='left')
                    
                    # st.write('merged_df 723', merged_df)

                    # Only update 'Derby_mult' if 'Mult' is not None (ie is a non local derby) 
                    df['Derby_mult'] = np.where(merged_df['Mult'].notna(), merged_df['Mult'], df['Derby_mult'])
                    # df['Derby_mult'] = df['Derby_mult'].fillna(1)

                    # st.write('df 729', df)

                    # ---------------- Incorporate Ref Factor --------------

                    df = df.merge(df_avg_tf_per_ref[['Referee', 'Factor']], on='Referee', how='left')
                    df.rename(columns={'Factor': 'Ref_mult'}, inplace=True)
                    df['Ref_mult'] = df['Ref_mult'].fillna(1)
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

                    # st.write('df 770', df)


                    # ------------------------ APPLY MODELS ---------------------------------------
                    df['h_lg_avg'] = h_lg_avg
                    df['a_lg_avg'] = a_lg_avg
                    # df['const'] = 1


                    # Get features mix (interaction terms of features used in modelling)
                    # df['H_mix'] = df['H_h_for'] * 0.3 + df['H_a_for'] * 0.15 + df['A_a_ag'] * 0.2 + df['A_h_ag'] * 0.15 + df['h_lg_avg'] * 0.2
                    # df['A_mix'] = df['A_a_for'] * 0.3 + df['A_h_for'] * 0.15 + df['H_a_ag'] * 0.2 + df['H_h_ag'] * 0.15 + df['a_lg_avg'] * 0.2

                    # ----- CREATE ARRAY TO PAASS INTO MODELS -----------

                    # Creating a 2D array where each row is a sample, and each column is a feature
                    ml_inputs_array = np.array([
                        df['h_pc_true'], 
                        df['a_pc_true'],
                        df['H_h_for'],
                        df['H_h_ag'],
                        df['H_a_for'],
                        df['H_a_ag'],
                        df['A_h_for'],
                        df['A_h_ag'],
                        df['A_a_for'],
                        df['A_a_ag'],
                    ]).T  # Transpose to make sure it's of shape (n_samples, n_features)


                    # Check for NaN values in ml_inputs_array_a
                    if np.any(np.isnan(ml_inputs_array)):
                        # If NaN values are found, handle them as per your requirement:
                        # Replace NaNs with a specific value (e.g., 0 or None), or drop rows with NaNs.
                        ml_inputs_array = np.nan_to_num(ml_inputs_array, nan=0)  # Replace NaNs with None

                    try:
                        # Model Home
                        poly_array = PolynomialFeatures(degree=2, include_bias=True)

                        # Check if there are NaN values in ml_inputs_array_h
                        if np.any(np.isnan(ml_inputs_array)):
                            raise ValueError("Input data contains NaN values, skipping prediction.")

                        # Transform the input features
                        X_poly_input = poly_array.fit_transform(ml_inputs_array)  # Transform the input features

                        # Predict using the HOME FOULS model
                        fouls_model_h_prediction = fouls_model_h.predict(X_poly_input) * OVERS_BOOST

                        # Assign the predictions to the DataFrame
                        df['HF_Exp_initial'] = np.round(fouls_model_h_prediction * df['Derby_mult'] * df['Ref_mult'], 2)

                        # ---   Fudge to handle big home fav/dog  -----------------

                        hf_sup_reduce_1 = 0.97  # < 1.15
                        hf_sup_reduce_2 = 0.98  # 1.15 ≤ x < 1.3
                        hf_sup_reduce_3 = 0.99  # < 1.55 but >= 1.3

                        hf_dog_boost_1 = 1.03 # > 14
                        hf_dog_boost_2 = 1.02 # 8 < x <= 14
                        hf_dog_boost_3 = 1.01 # 4.5 < x <= 8

                        conditions = [
                            df['Home Win'] < 1.15,                               # strong home favorite
                            (df['Home Win'] >= 1.15) & (df['Home Win'] < 1.3),   # moderately big home favorite
                            (df['Home Win'] >= 1.30) & (df['Home Win'] < 1.55),  # medium home favorite
                            df['Home Win'] > 14,                                 # big home underdog
                            (df['Home Win'] > 8) & (df['Home Win'] <= 14),       # moderately big home underdog
                            (df['Home Win'] > 4.5) & (df['Home Win'] <= 8)       # medium home underdog
                        ]

                        choices = [
                            hf_sup_reduce_1,
                            hf_sup_reduce_2,
                            hf_sup_reduce_3,
                            hf_dog_boost_1,
                            hf_dog_boost_2,
                            hf_dog_boost_3
                        ]

                        df['HF_Exp'] = round(df['HF_Exp_initial'] * np.select(conditions, choices, default=1.0), 2)
                        # --------------------------------------------------                        

                        # Calculate additional metrics using the prediction
                        df[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 
                            'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = df.apply(
                            lambda row: calculate_home_away_lines_and_odds(row['HF_Exp'], selected_metric), 
                            axis=1, result_type='expand'
                        )

                        df['h_main_un'] = round(1 / df['h_main_under_%'], 2)
                        df['h_main_ov'] = round(1 / df['h_main_over_%'], 2)
                        df['h_-1_un'] = round(1 / df['h_-1_under_%'], 2)
                        df['h_-1_ov'] = round(1 / df['h_-1_over_%'], 2)
                        df['h_+1_un'] = round(1 / df['h_+1_under_%'], 2)
                        df['h_+1_ov'] = round(1 / df['h_+1_over_%'], 2)

                    except Exception as e:
                        #st.write(f"An error occurred: {e}")
                        # If an error occurs, assign None to the predictions and related columns
                        df['HF_Exp'] = 0
                        df[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 
                            'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = 0
                    

                    # ------ AWAY -----------

                    try:
                        # Model AWAY
                        # Predict using the model
                        fouls_model_a_prediction = fouls_model_a.predict(X_poly_input) * OVERS_BOOST

                        # Assign the predictions to the DataFrame - ** NAME THIS HEADER '_RAW' IF NEED TO ADD OVERS BIAS **
                        df['AF_Exp_initial'] = np.round(fouls_model_a_prediction * df['Derby_mult'] * df['Ref_mult'], 2)

                        # -------- Fudge to handle big away fav/dog ------------

                        af_sup_reduce_1 = 0.97  # < 1.3
                        af_sup_reduce_2 = 0.98  # 1.30 ≤ x < 1.45
                        af_sup_reduce_3 = 0.99  # 1.45 ≤ x < 1.65

                        af_dog_boost_1 = 1.03 # > 18
                        af_dog_boost_2 = 1.02 # 11 < x <= 18
                        af_dog_boost_3 = 1.01 # 5.9 < x <= 11

                        conditions = [
                            df['Away Win'] < 1.30,                               # strong away favorite
                            (df['Away Win'] >= 1.30) & (df['Away Win'] < 1.45),  # moderately big away favorite
                            (df['Away Win'] >= 1.45) & (df['Away Win'] < 1.65),  # medium away favorite
                            df['Away Win'] > 18,                                 # big away underdog
                            (df['Away Win'] > 11) & (df['Away Win'] <= 18),       # moderately big away underdog
                            (df['Away Win'] > 5.9) & (df['Away Win'] <= 11)       # medium away underdog
                        ]

                        choices = [
                            af_sup_reduce_1,
                            af_sup_reduce_2,
                            af_sup_reduce_3,
                            af_dog_boost_1,
                            af_dog_boost_2,
                            af_dog_boost_3
                        ]

                        df['AF_Exp'] = round(df['AF_Exp_initial'] * np.select(conditions, choices, default=1.0), 2)
                        # -----------------------------------------------------

                        # calculate_corners_lines_and_odds(prediction)
                        df[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = df.apply(
                            lambda row: calculate_home_away_lines_and_odds(row['AF_Exp'], selected_metric), 
                            axis=1, result_type='expand')
                        
                        df['a_main_un'] = round(1 / df['a_main_under_%'], 2)
                        df['a_main_ov'] = round(1 / df['a_main_over_%'], 2)
                        df['a_-1_un'] = round(1 / df['a_-1_under_%'], 2)
                        df['a_-1_ov'] = round(1 / df['a_-1_over_%'], 2)
                        df['a_+1_un'] = round(1 / df['a_+1_under_%'], 2)
                        df['a_+1_ov'] = round(1 / df['a_+1_over_%'], 2)

                    except Exception as e:
                        # st.write(f"An error occurred: {e}")
                        # If an error occurs, assign None to the predictions and related columns
                        df['AF_Exp'] = 0
                        df[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 
                            'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = 0
                    

                    # --------  TOTAL ---------------

                    df[['TF_Exp', 'T_main_line', 'T_-1_line', 'T_+1_line', 'T_-2_line', 'T_+2_line','T_main_under_%', 
                        'T_main_over_%', 'T_-1_under_%', 'T_-1_over_%', 'T_+1_under_%', 
                        'T_+1_over_%', 'T_-2_under_%', 'T_-2_over_%', 'T_+2_under_%', 
                        'T_+2_over_%',]] = df.apply(
                        lambda row: calculate_totals_lines_and_odds(
                            row['HF_Exp'], 
                            row['AF_Exp'], 
                            total_metrics_df=calculate_probability_grid_hc_vs_ac(row['HF_Exp'], row['AF_Exp'])[1]
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
                        lambda row: pd.Series(calculate_probability_grid_hc_vs_ac(row['HF_Exp'], row['AF_Exp'])[2:5]), 
                        axis=1, 
                        result_type='expand'
                    )


                    df['H_most'] = round(1 / df['H_most_%'], 2)
                    df['Tie'] = round(1 / df['Tie_%'], 2)
                    df['A_most'] = round(1 / df['A_most_%'], 2)

                    # Sub-select final columns
                    df_final = df[['Date', 'Home Team', 'Away Team','Home Win', 'Draw', 'Away Win', 'Derby_mult', 'Ref_mult',
                                'HF_Exp', 'h_main_line', 'h_main_un', 'h_main_ov', 
                                'h_-1_line', 'h_-1_un', 'h_-1_ov',
                                'h_+1_line', 'h_+1_un', 'h_+1_ov',
                                'AF_Exp', 'a_main_line', 'a_main_un', 'a_main_ov',
                                'a_-1_line', 'a_-1_un', 'a_-1_ov',
                                'a_+1_line', 'a_+1_un', 'a_+1_ov',
                                'TF_Exp', 'T_main_line', 'T_main_un', 'T_main_ov', 
                                'T_-1_line', 'T_-1_un', 'T_-1_ov',
                                'T_+1_line', 'T_+1_un', 'T_+1_ov',
                                'T_-2_line', 'T_-2_un', 'T_-2_ov',
                                'T_+2_line', 'T_+2_un', 'T_+2_ov',
                                'H_most', 'Tie', 'A_most'
                                ]]

                    # st.write('True odds:', df_final)

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
                        #       'H_most', 'Tie', 'A_most'
                    ]


                    # Apply margin & bias for '_un' and '_ov' columns
                    for col in cols_to_add_margin:
                        if col.endswith('_ov'):  # For '_ov' columns, divide by margin_to_apply
                            df_final = df_final.assign(**{f'{col}_w.%': df_final[col].apply(lambda x: round(x / margin_to_apply / bias_to_apply, 2))})
                        elif col.endswith('_un'):  # For '_un' columns, multiply by bias_to_apply
                            df_final = df_final.assign(**{f'{col}_w.%': df_final[col].apply(lambda x: round(x / margin_to_apply * bias_to_apply, 2))})


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
                    st.write(df_final_wm)

                    # warning if not all match  retrieved from API call matches the final df
                    if len(df) != len(fixt_id_list):
                        st.warning('Odds for 1 or more matches not currently available!')


                    # -----------------------------------------------------------------------------------------------------------------
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
                        df_row = pd.DataFrame(index=range(9), columns=columns)

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
                        df_row['EVENT TYPE'].iloc[:9] = 'Match'
                        df_row['SPORT'].iloc[:9] = 'Football'
                        df_row['CATEGORY'].iloc[:9] = category_lg
                        df_row['COMPETITION'].iloc[:9] = competition
                        df_row['EVENT NAME'].iloc[:9] = event_name

                        df_row['MARKET TYPE NAME'].iloc[:3] = 'Total fouls {line} Over'
                        df_row['MARKET TYPE NAME'].iloc[3:6] = '{competitor1} total fouls {line} Over'
                        df_row['MARKET TYPE NAME'].iloc[6:9] = '{competitor2} total fouls {line} Over'

                        df_row['LINE'].iloc[0] = row['T_-2_line']
                        df_row['LINE'].iloc[1] = row['T_main_line']
                        df_row['LINE'].iloc[2] = row['T_+2_line']
                        df_row['LINE'].iloc[3] = row['h_-1_line']
                        df_row['LINE'].iloc[4] = row['h_main_line']
                        df_row['LINE'].iloc[5] = row['h_+1_line']
                        df_row['LINE'].iloc[6] = row['a_-1_line']
                        df_row['LINE'].iloc[7] = row['a_main_line']
                        df_row['LINE'].iloc[8] = row['a_+1_line']

                        df_row['SELECTION NAME'].iloc[:9] = 'over {line}'

                        df_row['PRICE'].iloc[0] = row['T_-2_ov_w.%']
                        df_row['PRICE'].iloc[1] = row['T_main_ov_w.%']
                        df_row['PRICE'].iloc[2] = row['T_+2_ov_w.%']
                        df_row['PRICE'].iloc[3] = row['h_-1_ov_w.%']
                        df_row['PRICE'].iloc[4] = row['h_main_ov_w.%']
                        df_row['PRICE'].iloc[5] = row['h_+1_ov_w.%']
                        df_row['PRICE'].iloc[6] = row['a_-1_ov_w.%']
                        df_row['PRICE'].iloc[7] = row['a_main_ov_w.%']
                        df_row['PRICE'].iloc[8] = row['a_+1_ov_w.%']

                        # Dates & Times (already preprocessed)
                        start_date = row["START DATE"]
                        start_time = row["START TIME"]

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

                    # ------------------------------------------------------------------------------------------------------------------
                    #  ----- Calculate Daily Total FOULS  --------

                    # Convert to datetime
                    df_final_wm['Date'] = pd.to_datetime(df_final_wm['Date'])

                    # Group by the day only (ignoring time)
                    df_final_wm['Day'] = df_final_wm['Date'].dt.date  # Extract just the date (day)
                    df_final_wm['Time'] = df_final_wm['Date'].dt.time  # Extract just the time (HH:MM:SS)

                    aggregated_fl = df_final_wm.groupby('Day').agg(
                        TF=('TF_Exp', 'sum'), 
                        Match_Count=('TF_Exp', 'size'),
                        Time=('Time', 'first')
                    ).reset_index()


                    df_result_fl = aggregated_fl[aggregated_fl['Match_Count'] >= 2]


                    # ------- Get increment prior to calling poisson functions for Daily Totals  --------------------------------

                    def calculate_increment(main_line):
                        """Determine increment based on main_line value."""
                        if main_line > 35:
                            return 3
                        elif main_line > 14:
                            return 2
                        return 1

                    # -------  Display Simple DFs and Daily Fouls --------------

                    # st.write("---")
                    # st.subheader("Lines to Publish")
                    # st.write("")
                    # st.write("##### Over & Under - Total")
                    # st.write(df_simple_o_and_u)

                    # st.write("")
                    # st.write("##### Over & Under - Home")
                    # st.write(df_simple_o_and_u_home)

                    # st.write("")
                    # st.write("##### Over & Under - Away")
                    # st.write(df_simple_o_and_u_away)

                    # st.write("")
                    # st.write("##### Overs only - Total")
                    # st.write(df_simple_only_overs)

                    # st.write("")
                    # st.write("##### Overs only - Home")
                    # st.write(df_simple_only_overs_home)

                    # st.write("")
                    # st.write("##### Overs only - Away")
                    # st.write(df_simple_only_overs_away)

                    st.subheader("", divider='blue')
                    st.subheader('Total Daily Fouls')
                    st.write("")

                    # if df_result_fl.shape[0] < 2:
                    #     st.caption('Less than two matches')

                    st.write(df_result_fl)
                    st.warning('Check correct number of fixtures have been logged for each day', icon="⚠️")
                    st.write("")

                    # Get poisson odds and lines for each day returned for Daly Goals
                    for _, row in df_result_fl.iterrows():
                        exp = row['TF'] * 1/OVERS_BOOST * TOTALS_BOOST  # remove individual match overs_boost factor, then mult by totals boost
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
                        #st.write(probabilities)

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

                        # st.write(df_result_fl)

                        # columns = [
                        #     'EVENT TYPE', 'SPORT', 'CATEGORY', 'COMPETITION', 'EVENT NAME', 
                        #     'MARKET TYPE NAME', 'LINE', 'SELECTION NAME', 'PRICE', 'START DATE', 
                        #     'START TIME', 'OFFER START DATE', 'OFFER START TIME', 'OFFER END DATE', 'OFFER END TIME', 
                        #     'PUBLISHED'
                        # ]

                        # # Get today's date in YYYY-MM-DD format
                        # today_date = datetime.today().strftime('%Y-%m-%d')

                        # Create an empty DataFrame with 6 rows and specified columns
                        df_csv = pd.DataFrame(index=range(6), columns=columns)

                        # Set the top 6 rows of specific columns
                        df_csv['EVENT TYPE'].iloc[:6] = 'Special'
                        df_csv['SPORT'].iloc[:6] = 'Football'
                        df_csv['CATEGORY'].iloc[:6] = 'Special Offer'
                        df_csv['COMPETITION'].iloc[:6] = 'Daily League Fouls'
                        df_csv['MARKET TYPE NAME'].iloc[:6] = 'Daily Fouls O/U {line}'
                        # df_csv['SELECTION NAME'].iloc[:6] = 'Daily Fouls {O/U} line'

                        df_csv.loc[[0, 2, 4], 'SELECTION NAME'] = 'Over {line}'
                        df_csv.loc[[1, 3, 5], 'SELECTION NAME'] = 'Under {line}'

                        # Assign 'LINE' values
                        df_csv.loc[[0, 1], 'LINE'] = line_minus_1
                        df_csv.loc[[2, 3], 'LINE'] = main_line
                        df_csv.loc[[4, 5], 'LINE'] = line_plus_1


                        df_csv['START DATE'] = day #1
                        df_csv['START TIME'] = time #2
                        df_csv['OFFER START DATE'] = today_date #3
                        df_csv['OFFER START TIME'].iloc[:6] = '09:00:00' #4
                        df_csv['OFFER END DATE'] = day  #5
                        df_csv['OFFER END TIME'] = time #6
                        df_csv['PUBLISHED'].iloc[:6] = 'NO' #7

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

                        st.write("")

            except Exception as e:
                st.write(f'An error has occurred whilst compiling: {e}')  

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

     

        h_pc_true, d_pc_true, a_pc_true = calculate_true_from_true_raw(h_pc_true_raw, d_pc_true_raw, a_pc_true_raw, margin)

        # st.write('h_pc_true, d_pc_true, a_pc_true', h_pc_true, d_pc_true, a_pc_true)

        cls1, _, cls3 = st.columns([7,1,7])
        with cls1:
            st.subheader('Home')
            ht_h_for = st.number_input('Avg home team home Fouls - for')
            ht_a_for = st.number_input('Avg home team away Fouls - for')
            ht_a_ag = st.number_input('Avg home team home Fouls - against')
            ht_h_ag = st.number_input('Avg home team away Fouls - against')

            st.write("---")

        with cls3:
            st.subheader('Away')
            at_a_for = st.number_input('Avg away team away Fouls - for')
            at_h_for = st.number_input('Avg away team home Fouls - for')
            at_a_ag = st.number_input('Avg away team away Fouls - against')
            at_h_ag = st.number_input('Avg away team home Fouls - against')

            st.write("---")

  

        # Creating a 2D array where each row is a sample, and each column is a feature
        ml_inputs_array_single = np.array([[
            h_pc_true, 
            a_pc_true,
            ht_h_for,
            ht_h_ag, 
            ht_a_for,
            ht_a_ag, 
            at_h_for, 
            at_h_ag, 
            at_a_for, 
            at_a_ag, 
        ]])

        # Model Home
        poly_array = PolynomialFeatures(degree=2, include_bias=True)

        # Check if there are NaN values in ml_inputs_array_h
        if np.any(np.isnan(ml_inputs_array_single)):
            raise ValueError("Input data contains NaN values, skipping prediction.")

        # Transform the input features
        X_poly_input = poly_array.fit_transform(ml_inputs_array_single)  # Transform the input features

        # Predict using the HOME FOULS model
        fouls_model_h_prediction = fouls_model_h.predict(X_poly_input) * OVERS_BOOST
        fouls_model_a_prediction = fouls_model_a.predict(X_poly_input) * OVERS_BOOST

        # st.write(fouls_model_h_prediction, fouls_model_a_prediction)


        # -------------  Additional factors  ---------
        with cls1:
            is_neutral = st.selectbox('Is this on a neutral pitch?', ['No', 'Yes'])
            is_extra_time = st.selectbox('Is extra-time possible?', ['No', 'Yes'])
            is_big_cup = st.selectbox('Is this a high stakes match?', ['No - Average', 'Yes - Above average', 'Yes - Very high', 'No - Below average'])
            is_derby = st.selectbox('Is this a derby?', ['No - Standard match', 'Yes - Above average', 'Yes - Very high', 'No - Sub-average (friendly)'])

        # home teams have 0.49 of total SOT, away team have 0.51
        # apply below factors if match is on a neutral 
        is_neutral_factor_home = 1.02 if is_neutral=='Yes' else 1 
        is_neutral_factor_away = 0.98 if is_neutral=='Yes' else 1  

        # Is extra time possible 
        if is_extra_time == 'Yes':
            extra_time_factor = d_pc_true * 0.34 * 1.05 + 1     # (extra time is 31/93 (0.33) of total SOT, increase by 1.05 for more space/tired legs)
        else:
            extra_time_factor = 1


        if is_big_cup == 'Yes - Very high':
            big_cup_factor = 1.03
        elif is_big_cup == 'Yes - Above average':
            big_cup_factor = 1.02
        elif is_big_cup == 'No - Average':
            big_cup_factor = 1
        else:
            big_cup_factor = 0.98   


        if is_derby == 'Yes - Very high':
            derby_factor = 1.04
        elif is_derby == 'Yes - Above average':
            derby_factor = 1.02
        elif is_derby == 'No - Standard match':
            derby_factor = 1
        else:
            derby_factor = 0.96  

        # ----------------------------

        # Add additional factors
        home_prediction = fouls_model_h_prediction * extra_time_factor * is_neutral_factor_home * big_cup_factor * derby_factor
        home_prediction = round(float(home_prediction), 2)
        
        with cls1:
            st.success(f'Home Prediction: {home_prediction}')
            show_mult_home = st.checkbox('Show home multiples')
            if show_mult_home:
                st.write('ET mult', extra_time_factor, 'Neutral mult', is_neutral_factor_home, 'High Stakes mult', big_cup_factor, 'Derby mult', derby_factor)


        # Add additional factors
        away_prediction = fouls_model_a_prediction * extra_time_factor * is_neutral_factor_away * big_cup_factor * derby_factor
        away_prediction = round(float(away_prediction), 2)
        total_prediction = home_prediction + away_prediction

        with cls3:
            for i in range(21):
                st.write("")

            st.success(f'Away Prediction: {away_prediction}')
            show_mult_away = st.checkbox('Show away multiples')
            if show_mult_away:
                st.write('ET mult', extra_time_factor, 'Neutral mult', is_neutral_factor_away, 'High Stakes mult', big_cup_factor, 'Derby mult', derby_factor)


        df_single = pd.DataFrame([{
            'HF_Exp': home_prediction,
            'AF_Exp': away_prediction,
            'TF_Exp': total_prediction,
        }])

        # st.write(df_single)

        # calculate_sot_lines_and_odds(prediction) - HOME
        df_single[['h_main_line', 'h_-1_line', 'h_+1_line', 'h_main_under_%', 'h_main_over_%', 'h_-1_under_%', 'h_-1_over_%', 'h_+1_under_%', 'h_+1_over_%']] = df_single.apply(
            lambda row: calculate_home_away_lines_and_odds(row['HF_Exp'], selected_metric), 
            axis=1, result_type='expand')
        
        # calculate_corners_lines_and_odds(prediction) - AWAY
        df_single[['a_main_line', 'a_-1_line', 'a_+1_line', 'a_main_under_%', 'a_main_over_%', 'a_-1_under_%', 'a_-1_over_%', 'a_+1_under_%', 'a_+1_over_%']] = df_single.apply(
            lambda row: calculate_home_away_lines_and_odds(row['AF_Exp'], selected_metric), 
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

        df_single[['TF_Exp', 'T_main_line', 'T_-1_line', 'T_+1_line', 'T_-2_line', 'T_+2_line','T_main_under_%', 
            'T_main_over_%', 'T_-1_under_%', 'T_-1_over_%', 'T_+1_under_%', 
            'T_+1_over_%', 'T_-2_under_%', 'T_-2_over_%', 'T_+2_under_%', 
            'T_+2_over_%',]] = df_single.apply(
            lambda row: calculate_totals_lines_and_odds(
                row['HF_Exp'], 
                row['AF_Exp'], 
                total_metrics_df=calculate_probability_grid_hc_vs_ac(row['HF_Exp'], row['AF_Exp'])[1]
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
                    'HF_Exp', 'h_main_line', 'h_main_un', 'h_main_ov', 
                    'h_-1_line', 'h_-1_un', 'h_-1_ov',
                    'h_+1_line', 'h_+1_un', 'h_+1_ov',
                    'AF_Exp', 'a_main_line', 'a_main_un', 'a_main_ov',
                    'a_-1_line', 'a_-1_un', 'a_-1_ov',
                    'a_+1_line', 'a_+1_un', 'a_+1_ov',
                    'TF_Exp', 'T_main_line', 'T_main_un', 'T_main_ov', 
                    'T_-1_line', 'T_-1_un', 'T_-1_ov',
                    'T_+1_line', 'T_+1_un', 'T_+1_ov',
                    ]].copy()

        
        st.subheader('Lines & Odds (100%)')
        st.write(df_single_final)     


                    
if __name__ == "__main__":
    main()