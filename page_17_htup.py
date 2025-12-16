
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from mymodule.functions import get_fixtures, calculate_true_from_true_raw, calculate_expected_team_goals_from_1x2_refined, team_names_t1x2_to_BK_dict, calc_prob_matrix


# https://dashboard.api-football.com/soccer/ids
# Dictionary to map league names to their IDs
leagues_dict = {
    "England Premier": '39', # eng premier
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
    "UEFA Champions League": "2",
    "UEFA Europa League": "3",
    "UEFA Conference League": '848'
}

reverse_dict = {v: k for k, v in leagues_dict.items()}

API_SEASON = '2025'
df_ou = pd.read_csv('data/over_under_exp_conversion.csv')

def main():

    st.header('HTUP Pricing', divider='blue')

    options = st.multiselect(
        "Select competitions to price",
        list(leagues_dict.keys())
    )

    # Extract league IDs for the selected leagues
    leagues_selected_ids = [leagues_dict[league] for league in options]


    st.subheader(f'Generate odds for all upcoming matches (up to 7 days ahead)')

    column1,column2, _ = st.columns([1.5,1.5,1])

    with column1:
        # margin_to_apply = st.number_input('Margin to apply:', step=0.01, value = 1.08, min_value=1.05, max_value=1.2, key='margin_to_apply')
        # over bias initially set to 1.07 pre over only being published
        is_bst = st.toggle('Set time outputs if BST(-1hr). Unselected = UTC', value=False)

        margin_selection = st.selectbox(
            'Select margin to apply to 1.5-2.00 price range:',
            options=['Standard Negative Margin (-4% EV)',
                      'Low Negative Margin (-2% EV)', 
                      'High Negative Margin (-6% EV)',
                        'True Margin (0% EV)',
                        'Small Positive Margin (+2% EV)' 
            ],
            index=0
        )

        st.write("")


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
                from_date_str = today.strftime("%Y-%m-%d")
                to_date_str = up_to_date.strftime("%Y-%m-%d")
                MARKET_IDS = ['1', '5']             # WDW & Ov/Un
                BOOKMAKERS = ['4']                  # Pinnacle = 4, 365 = 8, Uni = 16, BF = 3


                all_leagues_dfs = []

                for league_id in leagues_selected_ids:
                    df_fixtures = get_fixtures(league_id, from_date_str, to_date_str, API_SEASON)
                    # st.write('625', df_fixtures)
                    if df_fixtures.empty:
                        st.write("No data returned for the specified league and date range.")
                    else:
                        # Proceed with the next steps if data is available
                        df_fixts = df_fixtures[['Fixture ID', 'Date', 'Home Team', 'Away Team']]
                        fixt_id_list = list(df_fixts['Fixture ID'].unique())

                        if "rapidapi" in st.secrets and "API_KEY_FOOTBALL_API" in st.secrets["rapidapi"]:
                            API_KEY = st.secrets["rapidapi"]["API_KEY_FOOTBALL_API"]
                        else:
                            load_dotenv()
                            API_KEY = os.getenv("API_KEY_FOOTBALL_API")

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

                        # st.write('706', fixt_id_list)

                        # Iterate through each fixture ID and get odds
                        for fixture_id in fixt_id_list:
                            for market_id in MARKET_IDS:
                                odds_df = get_odds(fixture_id, market_id, BOOKMAKERS)
                                # st.write('712',odds_df)
                                if not odds_df.empty:
                                    all_odds_df = pd.concat([all_odds_df, odds_df], ignore_index=True)

                        # Display the collected odds
                        # st.write('717', all_odds_df)

                        # Use groupby and fillna to collapse rows and remove None values
                        df_collapsed = all_odds_df.groupby('Fixture ID').first().combine_first(
                            all_odds_df.groupby('Fixture ID').last()).reset_index()
                        
                        # st.write('723', df_collapsed)

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
                        # st.write('767', df_collapsed)

                        ###########################################################################################

                        # st.write('771', df_fixts)
                        # Merge odds df_fixts with df_collapsed
                        df = df_fixts.merge(df_collapsed, on='Fixture ID')
                        # st.write('775', df)

                        if df.empty:
                            st.write('Odds currently unavailable from API') 

                        # ------  Add league name column --------
                        comp_name = reverse_dict.get(league_id, "Unknown")  # id is the current league id in the loop
                        df['Competition'] = comp_name

                        # # -----  Split Date and Time column ---
                        # if df['Date'].dtype == 'object' and df['Date'].str.contains(' ').any():
                        #     df[['Date', 'Time']] = df['Date'].str.split(' ', expand=True)

                        # ------- Extract Country -------
                        df['Country'] = df['Competition'].str.split(' ', n=1).str[0]


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
                        
                        #    Apply the function row by row and expand the results into two columns
                        df[['hgx', 'agx']] = df.apply(
                            lambda row: calculate_expected_team_goals_from_1x2_refined(
                                1/row['h_pc_true'],
                                1/row['d_pc_true'],
                                1/row['a_pc_true'],
                                1/row['ov_pc_true'],
                                1/row['un_pc_true']
                            ),
                            axis=1,
                            result_type='expand'  # Ensures the tuple is split into two separate columns
                        )                       

                        df['Gl_Exp'] = df['hgx'] + df['agx']

                        df['0-0'] = poisson.pmf(0, df['hgx']) * poisson.pmf(0, df['agx']) * 1.1

                        df['h_nxt_gl'] = (df['hgx'] / df['Gl_Exp']) * (1 - df['0-0'])
                        df['a_nxt_gl'] = (df['agx'] / df['Gl_Exp']) * (1 - df['0-0'])


                        # Display the updated DataFrame
                        # st.write('304', df)

                        all_leagues_dfs.append(df)

                        # st.write('all_leagues_dfs', all_leagues_dfs)

                        # # warning if not all match  retrieved from API call matches the final df
                        # if len(df) != len(fixt_id_list):
                        #     st.warning('Odds for 1 or more matches not currently available!')

                # 👉 Concatenate all DataFrames after the loop
                if all_leagues_dfs:
                    combined_df = pd.concat(all_leagues_dfs, ignore_index=True)
                    st.write("### ✅ Combined DataFrame for all leagues:")
                    # st.dataframe(combined_df)
                else:
                    st.warning("No data collected from any league.")     

                # add next goal odds
                # combined_df['hgx'] = combined_df['Gl_Exp'] / 2
                # combined_df['h_next_gl'] = 

                # ------------ HTEP FUNCTIONS (top-level) ----------------

                def calculate_win_by_x_probs(supremacy, goals_exp, max_goals, draw_lambda, f_half_perc):
                    # build prob matrices
                    prob_matrix_ft, prob_matrix_1h, prob_matrix_2h, hg, ag = calc_prob_matrix(supremacy, goals_exp, max_goals, draw_lambda, f_half_perc)

                    home_win_prob = np.sum(np.tril(prob_matrix_ft, -1))  # Home win (lower triangle of the matrix excluding diagonal)
                    draw_prob = np.sum(np.diag(prob_matrix_ft))  # Draw (diagonal of the matrix)
                    away_win_prob = np.sum(np.triu(prob_matrix_ft, 1))  # Away win (upper triangle of the matrix excluding diagonal)

                    # st.write('home_win_prob', home_win_prob, 'draw_prob', draw_prob, 'away_win_prob', away_win_prob)
                    # Calculate HT/FT probs prep calculations

                    # Away wins 2nd half probability
                    away_win_2h = (prob_matrix_2h[0,1] + prob_matrix_2h[0,2] + prob_matrix_2h[0,3] + prob_matrix_2h[0,4] + prob_matrix_2h[0,5] + prob_matrix_2h[0,6] + prob_matrix_2h[0,7] + prob_matrix_2h[0,8] + prob_matrix_2h[0,9] + prob_matrix_2h[0,10] +
                                prob_matrix_2h[1,2] + prob_matrix_2h[1,3] + prob_matrix_2h[1,4] + prob_matrix_2h[1,5] + prob_matrix_2h[1,6] + prob_matrix_2h[1,7] + prob_matrix_2h[1,8] +
                                prob_matrix_2h[2,3] + prob_matrix_2h[2,4] + prob_matrix_2h[2,5] + prob_matrix_2h[2,6] + prob_matrix_2h[2,7] +
                                prob_matrix_2h[3,4] + prob_matrix_2h[3,5] + prob_matrix_2h[3,6] + prob_matrix_2h[3,7] +
                                prob_matrix_2h[4,5] + prob_matrix_2h[4,6] + prob_matrix_2h[4,7] +
                                prob_matrix_2h[5,6] + prob_matrix_2h[5,7])

                    # Draw 1h/2h probability
                    draw_1h = prob_matrix_1h[0,0] + prob_matrix_1h[1,1] + prob_matrix_1h[2,2] + prob_matrix_1h[3,3] + prob_matrix_1h[4,4] + prob_matrix_1h[5,5]
                    draw_2h = prob_matrix_2h[0,0] + prob_matrix_2h[1,1] + prob_matrix_2h[2,2] + prob_matrix_2h[3,3] + prob_matrix_2h[4,4] + prob_matrix_2h[5,5]

                    # Home 2h probability
                    home_win_2h = (prob_matrix_2h[1,0] + prob_matrix_2h[2,0] + prob_matrix_2h[3,0] + prob_matrix_2h[4,0] + prob_matrix_2h[5,0] + prob_matrix_2h[6,0] + prob_matrix_2h[7,0] + prob_matrix_2h[8,0] + prob_matrix_2h[9,0] + prob_matrix_2h[10,0] +
                                    prob_matrix_2h[2,1] + prob_matrix_2h[3,1] + prob_matrix_2h[4,1] + prob_matrix_2h[5,1] + prob_matrix_2h[6,1] + prob_matrix_2h[7,1] + prob_matrix_2h[8,1] + prob_matrix_2h[9,1] +
                                    prob_matrix_2h[3,2] + prob_matrix_2h[4,2] + prob_matrix_2h[5,2] + prob_matrix_2h[6,2] + prob_matrix_2h[7,2] + prob_matrix_2h[8,2] +
                                    prob_matrix_2h[4,3] + prob_matrix_2h[5,3] + prob_matrix_2h[6,3] + prob_matrix_2h[7,3] +
                                    prob_matrix_2h[5,4] + prob_matrix_2h[6,4] + prob_matrix_2h[7,4] +
                                    prob_matrix_2h[6,5] + prob_matrix_2h[7,5])

                    home_win_1h_by_1 = prob_matrix_1h[1,0] + prob_matrix_1h[2,1] + prob_matrix_1h[3,2] + prob_matrix_1h[4,3] + prob_matrix_1h[5,4] + prob_matrix_1h[6,5]
                    home_win_1h_by_2 = prob_matrix_1h[2,0] + prob_matrix_1h[3,1] + prob_matrix_1h[4,2] + prob_matrix_1h[5,3] + prob_matrix_1h[6,4] + prob_matrix_1h[7,5]
                    home_win_1h_by_3 = prob_matrix_1h[3,0] + prob_matrix_1h[4,1] + prob_matrix_1h[5,2] + prob_matrix_1h[6,3] + prob_matrix_1h[7,4] + prob_matrix_1h[8,5]
                    home_win_1h_by_4 = prob_matrix_1h[4,0] + prob_matrix_1h[5,1] + prob_matrix_1h[6,2] + prob_matrix_1h[7,3] + prob_matrix_1h[8,4] + prob_matrix_1h[9,5]

                    away_win_1h_by_1 = prob_matrix_1h[0,1] + prob_matrix_1h[1,2] + prob_matrix_1h[2,3] + prob_matrix_1h[3,4] + prob_matrix_1h[4,5] + prob_matrix_1h[5,6]
                    away_win_1h_by_2 = prob_matrix_1h[0,2] + prob_matrix_1h[1,3] + prob_matrix_1h[2,4] + prob_matrix_1h[3,5] + prob_matrix_1h[4,6] + prob_matrix_1h[5,7]
                    away_win_1h_by_3 = prob_matrix_1h[0,3] + prob_matrix_1h[1,4] + prob_matrix_1h[2,5] + prob_matrix_1h[3,6] + prob_matrix_1h[4,7] + prob_matrix_1h[5,8]
                    away_win_1h_by_4 = prob_matrix_1h[0,4] + prob_matrix_1h[1,5] + prob_matrix_1h[2,6] + prob_matrix_1h[3,7] + prob_matrix_1h[4,8] + prob_matrix_1h[5,9]

                    home_win_2h_by_1 = prob_matrix_2h[1,0] + prob_matrix_2h[2,1] + prob_matrix_2h[3,2] + prob_matrix_2h[4,3] + prob_matrix_2h[5,4] + prob_matrix_2h[6,5]
                    home_win_2h_by_2 = prob_matrix_2h[2,0] + prob_matrix_2h[3,1] + prob_matrix_2h[4,2] + prob_matrix_2h[5,3] + prob_matrix_2h[6,4] + prob_matrix_2h[7,5]
                    home_win_2h_by_3 = prob_matrix_2h[3,0] + prob_matrix_2h[4,1] + prob_matrix_2h[5,2] + prob_matrix_2h[6,3] + prob_matrix_2h[7,4] + prob_matrix_2h[8,5]
                    home_win_2h_by_4 = prob_matrix_2h[4,0] + prob_matrix_2h[5,1] + prob_matrix_2h[6,2] + prob_matrix_2h[7,3] + prob_matrix_2h[8,4] + prob_matrix_2h[9,5]

                    away_win_2h_by_1 = prob_matrix_2h[0,1] + prob_matrix_2h[1,2] + prob_matrix_2h[2,3] + prob_matrix_2h[3,4] + prob_matrix_2h[4,5] + prob_matrix_2h[5,6]
                    away_win_2h_by_2 = prob_matrix_2h[0,2] + prob_matrix_2h[1,3] + prob_matrix_2h[2,4] + prob_matrix_2h[3,5] + prob_matrix_2h[4,6] + prob_matrix_2h[5,7]
                    away_win_2h_by_3 = prob_matrix_2h[0,3] + prob_matrix_2h[1,4] + prob_matrix_2h[2,5] + prob_matrix_2h[3,6] + prob_matrix_2h[4,7] + prob_matrix_2h[5,8]
                    away_win_2h_by_4 = prob_matrix_2h[0,4] + prob_matrix_2h[1,5] + prob_matrix_2h[2,6] + prob_matrix_2h[3,7] + prob_matrix_2h[4,8] + prob_matrix_2h[5,9]

                    home_win_prob_fh = np.sum(np.tril(prob_matrix_1h, -1))  # Home win (lower triangle of the matrix excluding diagonal)
                    away_win_prob_fh = np.sum(np.triu(prob_matrix_1h, 1))  # Away win (upper triangle of the matrix excluding diagonal)
                    draw_prob_fh = draw_prob = np.sum(np.diag(prob_matrix_1h))  # Draw (diagonal of the matrix)

                    # st.write('home_win_prob_fh', home_win_prob_fh, 'draw_prob_fh', draw_prob_fh, 'away_win_prob_fh', away_win_prob_fh)

                    # HT/FT probabilities
                    HHp = 1 / (1 / home_win_prob_fh * (1 / (draw_2h + home_win_2h))) * 1.02
                    DHp = draw_prob_fh * home_win_2h
                    AHp = home_win_prob - HHp - DHp

                    HDp =  (1/(1/home_win_1h_by_1 * 1/away_win_2h_by_1) + 
                        1/(1/home_win_1h_by_2 * 1/away_win_2h_by_2) +
                        1/(1/home_win_1h_by_3 * 1/away_win_2h_by_3) +
                        1/(1/home_win_1h_by_4 * 1/away_win_2h_by_4)) * 1.10  

                    ADp =  (1/(1/away_win_1h_by_1 * 1/home_win_2h_by_1) + 
                        1/(1/away_win_1h_by_2 * 1/home_win_2h_by_2) +
                        1/(1/away_win_1h_by_3 * 1/home_win_2h_by_3) +
                        1/(1/away_win_1h_by_4 * 1/home_win_2h_by_4)) * 1.10  
                    DDp = draw_prob - HDp - ADp
                    
                    AAp = 1 / (1 / away_win_prob_fh * (1 / (draw_2h + away_win_2h))) * 1.03  
                    DAp = 1 / (1 / draw_prob_fh * (1 / away_win_2h))
                    HAp = away_win_prob - AAp - DAp

                    return DHp, AHp, HDp, ADp, DAp, HAp

                
                # crate supremacy column
                combined_df['Sup'] = combined_df['hgx'] - combined_df['agx']

                 
                combined_df[[
                    'DHp', 'AHp', 'HDp', 'ADp', 'DAp', 'HAp'
                    ]] = combined_df.apply(
                    lambda r: calculate_win_by_x_probs(r['Sup'], r['Gl_Exp'], max_goals=10, draw_lambda=0.08, f_half_perc=44),
                    axis=1, result_type='expand'
                )

                # st.write('412', combined_df)
                combined_df['htup_h_p'] = combined_df['h_pc_true_raw'] + combined_df['HDp'] + combined_df['HAp'] 
                combined_df['htup_a_p'] = combined_df['a_pc_true_raw'] + combined_df['ADp'] + combined_df['AHp']
                combined_df['htup_x_p'] = combined_df['d_pc_true_raw'] + combined_df['DHp'] + combined_df['DAp']

                combined_df['htup_h_odds_true'] = round(1 / combined_df['htup_h_p'], 2)
                combined_df['htup_a_odds_true'] = round(1 / combined_df['htup_a_p'], 2)
                combined_df['htup_x_odds_true'] = round(1 / combined_df['htup_x_p'], 2)

                # combined_df['htep_h_odds_marg_raw'] = round(combined_df['htep_h_odds_true'] / margin_to_apply, 2)
                # combined_df['htep_a_odds_marg_raw'] = round(combined_df['htep_a_odds_true'] / margin_to_apply, 2)
                # combined_df['htep_x_odds_marg_raw'] = round(combined_df['htep_x_odds_true'] / margin_to_apply, 2)

                # st.write('425', combined_df)


                # # ---   Conditions to handle margin distribution fav/dog after initial margin added  -----------------
                # # HOME

                # Multipliers
                if margin_selection == 'Standard Negative Margin (-4% EV)':
                    multipliers = [
                    1, 1, 1.04, 1.04, 0.96, 0.95, 0.90, 0.85, 0.80,
                    0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3  # add more if needed
                    ]
                elif margin_selection == 'Low Negative Margin (-2% EV)':
                    multipliers = [
                    1, 1, 1.02, 1.02, 0.96, 0.95, 0.90, 0.85, 0.80,
                    0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3  # add more if needed
                    ]
                elif margin_selection == 'High Negative Margin (-6% EV)':
                    multipliers = [
                        1, 1, 1.06, 1.06, 0.96, 0.95, 0.90, 0.85, 0.80,
                        0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3  # add more if needed
                    ]
                elif margin_selection == 'True Margin (0% EV)':
                    multipliers = [
                        1, 1, 1.00, 1.00, 0.96, 0.95, 0.90, 0.85, 0.80,
                        0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3  # add more if needed
                    ]
                elif margin_selection == 'Small Positive Margin (+2% EV)':
                    multipliers = [
                        1, 1, 0.98, 0.98, 0.96, 0.95, 0.90, 0.85, 0.80,
                        0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3  # add more if needed
                    ]

                # Define your ranges as tuples (lower_bound, upper_bound)
                # Note: np.select checks conditions in order, so ranges should not overlap
                ranges = [
                    (0, 1.35),      # 1   
                    (1.35, 1.51),   # 1
                    (1.51, 1.76),   # 1.02  - offering negative EV
                    (1.76, 2.01),   # 1.02  - offering negative EV
                    (2.01, 2.51),   # 0.96
                    (2.51, 3.51),   # 0.95
                    (3.51, 5.01),   # 0.9
                    (5.01, 7.01),   # 0.85 
                    (7.01, 8.51),   # 0.80 
                    (8.51, 10.01),  # 0.75
                    (10.01, 12.51),  # 0.70
                    (12.51, 15.01),  # 0.65
                    (15.01, 20.01),  # 0.60
                    (20.01, 30.01),  # 0.55  
                    (30.01, 50.01), # 0.50 
                    (50.01, 100.01), # 0.40
                    (101.01, 99999.01) # 0.30
                ]

                def get_multiplier(series):
                    conditions = [
                        (series >= low) & (series < high)
                        for (low, high) in ranges
                    ]
                    return np.select(conditions, multipliers, default=1.0)
                

                combined_df['htup_h_odds_marg'] = (
                    combined_df['htup_h_odds_true'] * get_multiplier(combined_df['htup_h_odds_true'])
                ).round(2)

                combined_df['htup_x_odds_marg'] = (
                    combined_df['htup_x_odds_true'] * get_multiplier(combined_df['htup_x_odds_true'])
                ).round(2)

                combined_df['htup_a_odds_marg'] = (
                    combined_df['htup_a_odds_true'] * get_multiplier(combined_df['htup_a_odds_true'])
                ).round(2)
     


                # -------- ensure final odds are never above true -----------
                combined_df['htup_h_odds_marg'] = combined_df['htup_h_odds_marg'].clip(upper=combined_df['Home Win']) 
                combined_df['htup_x_odds_marg'] = combined_df['htup_x_odds_marg'].clip(upper=combined_df['Draw']) 
                combined_df['htup_a_odds_marg'] = combined_df['htup_a_odds_marg'].clip(upper=combined_df['Away Win']) 

                st.write(combined_df)

                # # ----------------------  FMH Upload Format  ------------------------

                df1= combined_df.copy()

                                    # firstly allign streamlit team names with BK team names
                df1['Home Team Alligned'] = df1['Home Team'].map(team_names_t1x2_to_BK_dict).fillna(df1['Home Team'])
                df1['Away Team Alligned'] = df1['Away Team'].map(team_names_t1x2_to_BK_dict).fillna(df1['Away Team'])

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
                    'England League Two': 'League Two',
                    'UEFA Champions League': 'Champions League',
                    'UEFA Europa League': 'Europa League',
                    'UEFA Conference League': 'Conference League'
                }

                # --- Preprocess Date column once ---
                df1['Date'] = pd.to_datetime(df1['Date'], format='%d-%m-%y %H:%M', errors='coerce', dayfirst=True)
                df1["START DATE"] = df1["Date"].dt.strftime("%Y-%m-%d")
            
                # Adjust START TIME depending on BST toggle (vectorized)
                if is_bst:
                    df1["START TIME"] = (df1["Date"] - pd.Timedelta(hours=1)).dt.strftime("%H:%M:%S")
                else:
                    df1["START TIME"] = df1["Date"].dt.strftime("%H:%M:%S")

                rows_list = []  # store each row

                for idx, row in df1.iterrows():
                    # Create an empty DataFrame with 9 rows and specified columns
                    df_row = pd.DataFrame(index=range(3), columns=columns)

                    # Extract the competition name from the dataframe row
                    competition_name = row['Competition']

                    # create category_lg variable
                    if competition_name.startswith("South Africa"):
                        category_lg = "South Africa"
                    else:
                        category_lg = competition_name.split(" ")[0]

                    # create competition variable
                    competition_mapped = fmh_comp_dict.get(competition_name)

                    # create event_name variable
                    # extract Home Team and Away Team, make BK team name compatible, make as a vs b and store
                    event_name = row['Home Team Alligned'] + " vs " + row["Away Team Alligned"]

                    # Set the specific columns
                    df_row['EVENT TYPE'].iloc[:3] = 'Match'
                    df_row['SPORT'].iloc[:3] = 'Football'
                    df_row['CATEGORY'].iloc[:3] = category_lg
                    df_row['COMPETITION'].iloc[:3] = competition_mapped
                    df_row['EVENT NAME'].iloc[:3] = event_name

                    df_row['MARKET TYPE NAME'].iloc[:3] = 'Half Time Early Payout Both'
                #     # df_row['MARKET TYPE NAME'].iloc[3:6] = '{competitor1} total shots {line} Over'
                #     # df_row['MARKET TYPE NAME'].iloc[6:9] = '{competitor2} total shots {line} Over'

                    df_row['LINE'].iloc[:3] = 'N'
                #     # df_row['LINE'].iloc[1] = row['T_main_line']
                #     # df_row['LINE'].iloc[2] = row['T_+1_line']
                #     # df_row['LINE'].iloc[3] = row['h_-1_line']
                #     # df_row['LINE'].iloc[4] = row['h_main_line']
                #     # df_row['LINE'].iloc[5] = row['h_+1_line']
                #     # df_row['LINE'].iloc[6] = row['a_-1_line']
                #     # df_row['LINE'].iloc[7] = row['a_main_line']
                #     # df_row['LINE'].iloc[8] = row['a_+1_line']

                    df_row['SELECTION NAME'].iloc[0] = '{competitor1}'
                    df_row['SELECTION NAME'].iloc[1] = '{competitor2}'
                    df_row['SELECTION NAME'].iloc[2] = 'draw'

                    df_row['PRICE'].iloc[0] = row['htup_h_odds_marg']
                    df_row['PRICE'].iloc[1] = row['htup_a_odds_marg']
                    df_row['PRICE'].iloc[2] = row['htup_x_odds_marg'] 
                #     # df_row['PRICE'].iloc[3] = row['h_-1_ov_w.%']
                #     # df_row['PRICE'].iloc[4] = row['h_main_ov_w.%']
                #     # df_row['PRICE'].iloc[5] = row['h_+1_ov_w.%']
                #     # df_row['PRICE'].iloc[6] = row['a_-1_ov_w.%']
                #     # df_row['PRICE'].iloc[7] = row['a_main_ov_w.%']
                #     # df_row['PRICE'].iloc[8] = row['a_+1_ov_w.%']

                    # Dates & Times (already preprocessed)
                    start_date = row["START DATE"]
                    start_time = row["START TIME"] # already adjusted for BST/UTC

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
                df_fmh_format = df_fmh_format.set_index('EVENT TYPE')
                st.subheader('FMH Format')
                st.write(df_fmh_format)

                         
                        
            except Exception as e:
                st.write(f'An error has occurred whilst compiling: {e}')   



if __name__ == "__main__":
    main()