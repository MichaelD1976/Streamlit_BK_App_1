# import streamlit as st
# import pandas as pd
# # import altair as alt
# import numpy as np
# from datetime import datetime, timedelta
# import time
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from scipy.stats import poisson, nbinom
# from sklearn.preprocessing import PolynomialFeatures
# import joblib
# from mymodule.functions import get_fixtures,  calculate_home_away_lines_and_odds, poisson_probabilities, calc_prob_matrix
# import requests
# import os
# from dotenv import load_dotenv


# dict_api_to_bk_league_names = {
#      'England Premier':'England Premier League',
#      'Spain La Liga' : 'Spain LaLiga',
#  }

# CURRENT_SEASON = '2024'

# # ------------- Load the CSV file -----------------
# @st.cache_data
# def load_data():
#     time.sleep(1)
#     df_ou = pd.read_csv('data/over_under_exp_conversion.csv')
#     df_dnb = pd.read_csv('data/dnb_sup_conversion.csv')
#     # Convert 'Date' column to datetime format
#     return df_ou, df_dnb


# # --------------------  FIND TRUE 1X2 PRICES ------------------------------------- 
# # set marg_pc_move (amount to change the fav when odds-on eg 1.2 > 1.22 instead of 1.3)
# # complicated code. In testing it handles well transforming short price with margin > true price 

# def calculate_true_from_true_raw(h_pc_true_raw , d_pc_true_raw , a_pc_true_raw, margin):
#     marg_pc_remove = 0
#     if h_pc_true_raw > 0.90 or a_pc_true_raw > 0.90:
#         marg_pc_remove = 1    
#     elif h_pc_true_raw > 0.85 or a_pc_true_raw > 0.85:
#         marg_pc_remove = 0.85
#     elif h_pc_true_raw > 0.80 or a_pc_true_raw > 0.80:
#         marg_pc_remove = 0.7
#     elif h_pc_true_raw > 0.75 or a_pc_true_raw > 0.75:
#         marg_pc_remove = 0.6
#     elif h_pc_true_raw > 0.6 or a_pc_true_raw > 0.6:
#         marg_pc_remove = 0.5
#     elif h_pc_true_raw > 0.50 or a_pc_true_raw > 0.50:
#         marg_pc_remove = 0.4

#     if h_pc_true_raw >= a_pc_true_raw:
#         h_pc_true = h_pc_true_raw * (((marg_pc_remove * ((margin - 1) * 100)) / 100) + 1)
#         d_pc_true = d_pc_true_raw - ((h_pc_true - h_pc_true_raw) * 0.4) 
#         if h_pc_true + d_pc_true > 1:               # if greater than 100% (makes away price negative)
#             d_pc_true = 1 - h_pc_true - 0.0025
#             a_pc_true = 0.0025                              # make away price default 400
#         a_pc_true = 1 - h_pc_true - d_pc_true
#         if a_pc_true < 0:
#             a_pc_true = 0.0025
#     else:
#         a_pc_true = a_pc_true_raw * (((marg_pc_remove * ((margin - 1) * 100)) / 100) + 1)
#         d_pc_true= d_pc_true_raw - ((a_pc_true - a_pc_true_raw) * 0.4)
#         if a_pc_true + d_pc_true > 1:
#             d_pc_true = 1 - a_pc_true - 0.0025
#             h_pc_true = 0.0025
#         h_pc_true = 1 - a_pc_true - d_pc_true
#         if h_pc_true < 0:
#             h_pc_true = 0.0025

#     h_pc_true = round(h_pc_true, 2)
#     d_pc_true = round(d_pc_true, 2)
#     a_pc_true = round(a_pc_true, 2)

#     return (float(h_pc_true), float(d_pc_true), float(a_pc_true))

# def calc_sup_from_dnb():
#     pass
#     return()



# # --------------------------------------------------------------------

# def main():
#     with st.spinner('Loading Data...'):
#         df_ou, df_dnb = load_data()


#     # Sidebar for user input
#     st.sidebar.title('Select Data Filters')

#     # Define selection options
#     league_options = {
#         # 'All_leagues': 'ALL',  # Uncomment for future development
#         'Premier League': 'England Premier',
#         'La Liga': 'Spain La Liga',
#         'Bundesliga': 'Germany Bundesliga',
#         'Ligue 1': 'France Ligue 1',
#         'Serie A': 'Italy Serie A',
#         'Premier Soccer League': 'South Africa Premier',
#         'Premiership': 'Scotland Premier',
#         'Eredivisie': 'Netherlands Eredivisie',
#         'Jupiler Pro League': 'Belgium Jupiler',
#         'Primeira Liga': 'Portugal Liga I',
#         'Championship': 'England Championship',
#         '2. Bundesliga': 'Germany 2 Bundesliga',
#         'League One': 'England League One',
#         'League Two': 'England League Two',
#     }
    
#         # Dictionary to map league names to their IDs
#     leagues_dict = {
#         "England Premier": '39',
#         "Spain La Liga": '140',
#         "Germany Bundesliga": '78',
#         "Italy Serie A": '135',
#         "France Ligue 1": '61',
#         'England Championship': '40',
#         'England League One': '41',
#         'England League Two': '42',
#         "Germany 2 Bundesliga": '79',
#         "Netherlands Eredivisie": "88",
#         "Belgium Jupiler": "144",
#         "Portugal Liga I": '94',
#         "Scotland Premier": '179',
#         "South Africa Premier": "288",
#     }



#     # Capture user selections
#     selected_league = st.sidebar.selectbox('Select League', options=list(league_options.values()), label_visibility = 'visible')


#     # -----------------------------------------------------------------------

#     st.header('Chance Mix', divider='blue')


#     # get fixtures
#     league_id = leagues_dict.get(selected_league)


#     df_dnb.drop(['dnb price'], axis=1, inplace=True)
#     df_ou.drop(['Exp', 'Under', 'Over', 'Un2.5_%'], axis=1, inplace=True)
#     df_ou.drop_duplicates(subset='Ov2.5_%', inplace=True)
   

#     # -------------------------------------------- CREATE ODDS FOR ALL UPCOMING FIXTURES --------------------------------------------------------------------

#     # st.write("---")
#     st.subheader(f'Generate odds for all upcoming {selected_league} matches (up to 5 days ahead)')

#     column1,column2 = st.columns([1.5,2])

#     with column1:
#         margin_to_apply = st.number_input('Margin to apply - four winning outcomes so a 12% total market margin is 412%, displayed here as 4.12:', step=0.01, value = 4.12, min_value=4.02, max_value=4.2, key='margin_to_apply')
#         # bias_to_apply = st.number_input('Overs bias to apply (reduce overs & increase unders odds by a set %):', step=0.01, value = 1.07, min_value=1.00, max_value=1.12, key='bias_to_apply')


#     generate_odds_all_matches = st.button(f'Click to generate')

#     if generate_odds_all_matches:
#         with st.spinner("Odds being compiled..."):
#             try:
#                 # GET FIXTURES WEEK AHEAD
#                 today = datetime.now()
#                 to_date = today + timedelta(days=5)
#                 from_date_str = today.strftime("%Y-%m-%d")
#                 to_date_str = to_date.strftime("%Y-%m-%d")
#                 MARKET_IDS = ['1', '5']             # WDW & Ov/Un
#                 BOOKMAKERS = ['8']                  # Pinnacle = 4, 365 = 8
#                 API_SEASON = CURRENT_SEASON[:4]

                
#                 df_fixtures = get_fixtures(league_id, from_date_str, to_date_str, API_SEASON)

#                 if df_fixtures.empty:
#                     st.write("No data returned for the specified league and date range.")
#                 else:
#                     # Proceed with the next steps if data is available
#                     df_fixts = df_fixtures[['Fixture ID', 'Date', 'Home Team', 'Away Team']]
#                     fixt_id_list = list(df_fixts['Fixture ID'].unique())
                    
#                     if not st.secrets:
#                         load_dotenv()
#                         API_KEY = os.getenv("API_KEY_FOOTBALL_API")

#                     else:
#                         # Use Streamlit secrets in production
#                         API_KEY = st.secrets["rapidapi"]["API_KEY_FOOTBALL_API"]

#                     @st.cache_data
#                     def get_odds(fixture_id, market_id, bookmakers):
#                         url = "https://api-football-v1.p.rapidapi.com/v3/odds"
#                         headers = {
#                             "X-RapidAPI-Key": API_KEY,
#                             "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
#                         }
#                         querystring = {
#                             "fixture": fixture_id,
#                             "bet": market_id,
#                             "timezone": "Europe/London"
#                         }

#                         response = requests.get(url, headers=headers, params=querystring)
#                         data = response.json()

#                         if 'response' in data and data['response']:
#                             odds_dict = {
#                                 'Fixture ID': fixture_id,
#                                 'Home Win': None,
#                                 'Draw': None,
#                                 'Away Win': None,
#                                 'Over 2.5': None,
#                                 'Under 2.5': None
#                             }

#                             # Loop through bookmakers
#                             for bookmaker_data in data['response'][0].get('bookmakers', []):
#                                 if str(bookmaker_data['id']) in bookmakers:
#                                     # Loop through each market (bet) offered by the bookmaker
#                                     for bet_data in bookmaker_data['bets']:
#                                         if bet_data['id'] == int(market_id):  # Ensure it's the selected market
#                                             # Extract the outcomes (selections) and their corresponding odds
#                                             for value in bet_data['values']:
#                                                 selection = value['value']
#                                                 odd = value['odd']
                                                
#                                                 # Assign the odds based on the selection type
#                                                 if selection == 'Home':
#                                                     odds_dict['Home Win'] = odd
#                                                 elif selection == 'Draw':
#                                                     odds_dict['Draw'] = odd
#                                                 elif selection == 'Away':
#                                                     odds_dict['Away Win'] = odd
#                                                 elif selection == 'Over 2.5':
#                                                     odds_dict['Over 2.5'] = odd
#                                                 elif selection == 'Under 2.5':
#                                                     odds_dict['Under 2.5'] = odd

#                             # Create a DataFrame with a single row containing all the odds
#                             odds_df = pd.DataFrame([odds_dict])
#                             return odds_df

#                         # Return empty DataFrame if no data is found
#                         return pd.DataFrame()

#                     # Collect odds for all fixtures
#                     all_odds_df = pd.DataFrame()  # DataFrame to collect all odds

#                     # st.write(fixt_id_list)

#                     # Iterate through each fixture ID and get odds
#                     for fixture_id in fixt_id_list:
#                         for market_id in MARKET_IDS:
#                             odds_df = get_odds(fixture_id, market_id, BOOKMAKERS)
#                             # st.write(odds_df)
#                             if not odds_df.empty:
#                                 all_odds_df = pd.concat([all_odds_df, odds_df], ignore_index=True)

#                     # Display the collected odds
#                     # st.write(all_odds_df)

#                     # Use groupby and fillna to collapse rows and remove None values
#                     df_collapsed = all_odds_df.groupby('Fixture ID').first().combine_first(
#                         all_odds_df.groupby('Fixture ID').last()).reset_index()


#                     # Merge odds df_fixts with df_collapsed
#                     df = df_fixts.merge(df_collapsed, on='Fixture ID')
#                     df = df.dropna()
#                     # st.write(df)

#                     #  ---------------  Create true wdw odds ---------------

#                     # Convert columns to numeric (if they are strings or objects)
#                     df['Home Win'] = pd.to_numeric(df['Home Win'], errors='coerce')
#                     df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce')
#                     df['Away Win'] = pd.to_numeric(df['Away Win'], errors='coerce')

#                     df['O_2.5'] = pd.to_numeric(df['Over 2.5'], errors='coerce')
#                     df['U_2.5'] = pd.to_numeric(df['Under 2.5'], errors='coerce')


#                     df['margin_wdw'] = 1/df['Home Win'] + 1/df['Draw'] + 1/df['Away Win']
#                     df['margin_ou'] = 1/df['O_2.5'] + 1/df['U_2.5']


#                     df['h_pc_true_raw'] = (1 / df['Home Win']) / df['margin_wdw']
#                     df['d_pc_true_raw'] = (1 / df['Draw']) / df['margin_wdw'] 
#                     df['a_pc_true_raw'] = (1 / df['Away Win']) / df['margin_wdw'] 

#                     df['ov_pc_true'] = round((1 / df['O_2.5']) / df['margin_ou'], 2)
#                     df['un_pc_true'] = round((1 / df['U_2.5']) / df['margin_ou'], 2)

#                     df[['h_pc_true', 'd_pc_true', 'a_pc_true']] = df.apply(
#                         lambda row: calculate_true_from_true_raw(row['h_pc_true_raw'], row['d_pc_true_raw'], row['a_pc_true_raw'], row['margin_wdw']), 
#                         axis=1, result_type='expand')
                    


#                     # Function to add goal exp column to df
#                     def get_gl_exp_value(row, df_ou):
#                         # Extract the 'ov_pc_true' value from the current row in df
#                         ov_pc_true = row['ov_pc_true']
                        
#                         # Locate the row in df_ou where 'Ov2.5_%' matches the 'ov_pc_true' value
#                         gl_exp_value = df_ou.loc[df_ou['Ov2.5_%'] == ov_pc_true, 'Exp1']
                        
#                         # If there's no matching row, return NaN or a default value
#                         return gl_exp_value.values[0] if not gl_exp_value.empty else None
                    

#                     # Apply the function to each row in df to create a new column 'gl_exp'
#                     df['Gl_Exp'] = df.apply(lambda row: get_gl_exp_value(row, df_ou), axis=1)

#                     # ------------------

#                     df['dnb_%'] = round(df['h_pc_true'] / (df['h_pc_true'] + df['a_pc_true']), 2)


#                     # Function to sup from dnb
#                     def get_sup_fr_dnb(row, df_dnb):
#                         # Extract the dnb % from current row dnb
#                         dnb_pc = row['dnb_%']

#                         # Locate the row in df_dnb where 'dnb_%' matches the 'dnb_pc' value
#                         sup = df_dnb.loc[df_dnb['dnb %'] == dnb_pc, 'Sup']

#                         # If there's no matching row, return NaN or a default value
#                         return sup.iloc[0] if not sup.empty else None
                    
#                     # Apply the function to each row in df to create a new column 'gl_exp'
#                     df['Sup'] = df.apply(lambda row: get_sup_fr_dnb(row, df_dnb), axis=1)

#                     df['HG'] = round((df['Gl_Exp'] / 2) + (0.5 * df['Sup']), 2)
#                     df['AG'] = round((df['Gl_Exp'] / 2) - (0.5 * df['Sup']), 2)


#                     # -------------- Now generate scoreline probabilities and add them to df

#                     MAX_GOALS = 7
#                     DRAW_LAMBDA = 0.08
#                     F_HALF_PERC = 44

#                     # Helper function to extract the scoreline probabilities
#                     def extract_scoreline_probs(row):
#                         # Call the function and extract only the first return (FT matrix)
#                         prob_matrix_ft = calc_prob_matrix(
#                             row['Sup'], row['Gl_Exp'], MAX_GOALS, DRAW_LAMBDA, F_HALF_PERC
#                         )[0]
                        
#                         # Flatten matrix to a dict like {'0-0': 0.123, '0-1': 0.111, ...}
#                         return {
#                             f"{i}-{j}": prob_matrix_ft[i, j]
#                             for i in range(prob_matrix_ft.shape[0])
#                             for j in range(prob_matrix_ft.shape[1])
#                         }

#                     # Apply to each row and get list of dicts
#                     prob_dicts = df.apply(extract_scoreline_probs, axis=1)

#                     # Convert dicts to DataFrame
#                     scoreline_probs_df = pd.DataFrame(prob_dicts.tolist())

#                     # Merge into original df
#                     df = pd.concat([df, scoreline_probs_df], axis=1)


#                     # ---------- Now generate Chance Mix 1.5 markets with summed probabilities  -----------

#                     def is_home_win(h, a):
#                         return h > a

#                     def is_draw(h, a):
#                         return h == a

#                     def is_away_win(h, a):
#                         return h < a

#                     def total_goals(h, a):
#                         return h + a

#                     # Define a function to process each row (match)
#                     def compute_chance_mix_markets(row):
#                         markets = {
#                             '1 or Over 1.5': 0,
#                             'X or Over 1.5': 0,
#                             '2 or Over 1.5': 0,
#                             '1 or Under 1.5': 0,
#                             'X or Under 1.5': 0,
#                             '2 or Under 1.5': 0,
#                         }
                        
#                         for col in row.index:
#                             if '-' not in col:  # skip non-scoreline columns
#                                 continue
                            
#                             try:
#                                 h, a = map(int, col.split('-'))
#                                 prob = row[col]
#                             except:
#                                 continue
                            
#                             over_15 = total_goals(h, a) > 1.5
#                             under_15 = not over_15

#                             # Check and add to each market
#                             if is_home_win(h, a) or over_15:
#                                 markets['1 or Over 1.5'] += prob
#                             if is_draw(h, a) or over_15:
#                                 markets['X or Over 1.5'] += prob
#                             if is_away_win(h, a) or over_15:
#                                 markets['2 or Over 1.5'] += prob

#                             if is_home_win(h, a) or under_15:
#                                 markets['1 or Under 1.5'] += prob
#                             if is_draw(h, a) or under_15:
#                                 markets['X or Under 1.5'] += prob
#                             if is_away_win(h, a) or under_15:
#                                 markets['2 or Under 1.5'] += prob

#                         return pd.Series(markets)

#                     # Apply to scoreline probability DataFrame
#                     chance_mix_15_df = scoreline_probs_df.apply(compute_chance_mix_markets, axis=1)

#                     # transform to odds from %
#                     chance_mix_15_df = round(1 / chance_mix_15_df, 2)

#                     st.write(chance_mix_15_df)

#                     # Merge into original df
#                     df = pd.concat([df, chance_mix_15_df], axis=1)


#                     # ------------  Same for Chance Mix 2.5  --------------------------------

#                     def compute_chance_mix_markets_25(row):
#                         markets = {
#                             '1 or Over 2.5': 0,
#                             'X or Over 2.5': 0,
#                             '2 or Over 2.5': 0,
#                             '1 or Under 2.5': 0,
#                             'X or Under 2.5': 0,
#                             '2 or Under 2.5': 0,
#                         }
                        
#                         for col in row.index:
#                             if '-' not in col:
#                                 continue
#                             try:
#                                 h, a = map(int, col.split('-'))
#                                 prob = row[col]
#                             except:
#                                 continue
                            
#                             over_25 = (h + a) > 2.5
#                             under_25 = not over_25

#                             if h > a or over_25:
#                                 markets['1 or Over 2.5'] += prob
#                             if h == a or over_25:
#                                 markets['X or Over 2.5'] += prob
#                             if h < a or over_25:
#                                 markets['2 or Over 2.5'] += prob

#                             if h > a or under_25:
#                                 markets['1 or Under 2.5'] += prob
#                             if h == a or under_25:
#                                 markets['X or Under 2.5'] += prob
#                             if h < a or under_25:
#                                 markets['2 or Under 2.5'] += prob

#                         return pd.Series(markets)

#                     # Apply to scoreline_probs_df
#                     chance_mix_25_df = scoreline_probs_df.apply(compute_chance_mix_markets_25, axis=1)

#                     # transform to odds from %
#                     chance_mix_25_df = round(1 / chance_mix_25_df, 2)
                     
#                     # Merge with main DataFrame
#                     df = pd.concat([df, chance_mix_25_df], axis=1)


#                     # ---------- Apply initial margin to true chance mix odds ------------
#                     # assign an initial margin which reduces true odds by a staggered amount depending on how short the price is
#                     # for example 1.13 > 1.11, 3.5 > 3.3 etc

#                     initial_margin_dic = {
#                         (1.01, 1.20): 1.02,
#                         (1.21, 1.40): 1.03,
#                         (1.41, 1.60): 1.04,
#                         (1.61, 2.00): 1.05,
#                         (2.01, 3.00): 1.06,
#                         (3.01, 5.00): 1.08,
#                         (5.01, 10.00): 1.10,
#                     }

#                     # Function to get margin for a given value
#                     def get_initial_margin(val):
#                         for (low, high), margin in initial_margin_dic.items():
#                             if low <= val <= high:
#                                 return margin
#                         return 1.2  # default value
                    
#                     # Apply to DataFrame
#                     def apply_initial_margin(df):
#                         return df.applymap(lambda x: x / get_initial_margin(x) if pd.notnull(x) and get_initial_margin(x) else np.nan)
                    
#                     # ------------------------------------------------------------------

#                     # Now apply initial margin to both chance mix markets
#                     marg_15_df = apply_initial_margin(chance_mix_15_df)
#                     marg_25_df = apply_initial_margin(chance_mix_25_df)


#                     # ----  Re-scale odds to margin set by user (1.5)  -----------
#                     # Now the initial marginated odds need to be re-calibrated to the desired user overall margin eg 412%

#                     cols_to_adjust = [
#                         '1 or Over 1.5',
#                         'X or Over 1.5',
#                         '2 or Over 1.5',
#                         '1 or Under 1.5',
#                         'X or Under 1.5',
#                         '2 or Under 1.5'
#                     ]

#                     # Recalculate the odds so their inverse sums to `margin_to_apply`
#                     def recalibrate_odds(row):
#                         # Get inverse probabilities (i.e. implied %s)
#                         inv_probs = [1 / row[col] for col in cols_to_adjust]
                        
#                         # Current overround
#                         current_sum = sum(inv_probs)
                        
#                         # Scaling factor
#                         factor = margin_to_apply / current_sum if current_sum != 0 else 1
                        
#                         # Apply factor and return recalibrated odds (rounded)
#                         new_odds = [(1 / (inv * factor)) for inv in inv_probs]
#                         return pd.Series([round(o, 2) for o in new_odds], index=cols_to_adjust)

#                     # Apply to DataFrame
#                     marg_15_df[cols_to_adjust] = marg_15_df.apply(recalibrate_odds, axis=1)

#                     # Optionally, update the '%' column to reflect new implied sum
#                     marg_15_df['%'] = marg_15_df[cols_to_adjust].apply(lambda row: sum(1 / row[col] for col in cols_to_adjust), axis=1)



#                     # ----  Re-scale odds to margin set by user (2.5)  -----------
#                     # Now the initial marginated odds need to be re-calibrated to the desired user overall margin eg 412%

#                     # reset columns for 2.5 market
#                     cols_to_adjust = [
#                         '1 or Over 2.5',
#                         'X or Over 2.5',
#                         '2 or Over 2.5',
#                         '1 or Under 2.5',
#                         'X or Under 2.5',
#                         '2 or Under 2.5'
#                     ]

#                     # Apply to DataFrame
#                     marg_25_df[cols_to_adjust] = marg_25_df.apply(recalibrate_odds, axis=1)

#                     # Optionally, update the '%' column to reflect new implied sum
#                     marg_25_df['%'] = marg_25_df[cols_to_adjust].apply(lambda row: sum(1 / row[col] for col in cols_to_adjust), axis=1)


#                     # ------------------------------------------------------------


#                     st.write(marg_15_df)
#                     st.write(marg_25_df)

#                     df_matches = df[['Date', 
#                                        'Home Team', 
#                                        'Away Team']]
#                     st.write(df_matches)

#                     # Merge with main DataFrame
#                     final_df_15 = pd.concat([df_matches, marg_15_df], axis=1)
#                     final_df_25 = pd.concat([df_matches, marg_25_df], axis=1)

#                     st.write(final_df_15)
#                     st.write(final_df_25)

#                     # st.write(df)

#                     # df_chance_mix = df[['Date', 
#                     #                    'Home Team', 
#                     #                    'Away Team', 
#                     #                     '1 or Over 1.5',
#                     #                      'X or Over 1.5',
#                     #                        '2 or Over 1.5',
#                     #                          '1 or Under 1.5',
#                     #                            'X or Under 1.5',
#                     #                              '2 or Under 1.5',
#                     #                    '1 or Over 2.5',
#                     #                      'X or Over 2.5',
#                     #                        '2 or Over 2.5',
#                     #                          '1 or Under 2.5',
#                     #                            'X or Under 2.5',
#                     #                              '2 or Under 2.5']]
#                     # st.write(df_chance_mix)

#             except Exception as e:
#                 st.write(f'An error has occurred whilst compiling: {e}')   


                

# if __name__ == "__main__":
#     main()