
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from mymodule.functions import get_fixtures, calculate_true_from_true_raw



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
    # CL
    # Uefa
    # 
}

API_SEASON = '2025'
df_ou = pd.read_csv('data/over_under_exp_conversion.csv')
df_dnb = pd.read_csv('data/dnb_sup_conversion.csv')


def main():

    st.header('1 UP Multi-Competition Pricing', divider='blue')

    options = st.multiselect(
        "Select competitions to price",
        list(leagues_dict.keys())
    )

    st.write("You selected:", options)


    # Extract league IDs for the selected leagues
    leagues_selected_ids = [leagues_dict[league] for league in options]

    st.write("League IDs:", leagues_selected_ids)



    st.subheader(f'Generate odds for all upcoming matches (up to 7 days ahead)')

    column1,column2, _ = st.columns([1.5,1.5,1])

    with column1:
        margin_to_apply = st.number_input('Margin to apply:', step=0.01, value = 1.10, min_value=1.05, max_value=1.2, key='margin_to_apply')
        # over bias initially set to 1.07 pre over only being published
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

    generate_odds_all_matches = st.button(f'Click to generate')

    if generate_odds_all_matches:
        with st.spinner("Odds being compiled..."):
            try:
                from_date_str = today.strftime("%Y-%m-%d")
                to_date_str = up_to_date.strftime("%Y-%m-%d")
                MARKET_IDS = ['1', '5']             # WDW & Ov/Un
                BOOKMAKERS = ['4']                  # Pinnacle = 4, 365 = 8, Uni = 16, BF = 3


                all_leagues_dfs = []

                for id in leagues_selected_ids:
                    df_fixtures = get_fixtures(id, from_date_str, to_date_str, API_SEASON)
                    # st.write('625', df_fixtures)
                    if df_fixtures.empty:
                        st.write("No data returned for the specified league and date range.")
                    else:
                        # Proceed with the next steps if data is available
                        df_fixts = df_fixtures[['Fixture ID', 'Date', 'Home Team', 'Away Team']]
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

                        df['dnb_perc_h'] = round(df['h_pc_true_raw'] / (df['h_pc_true_raw'] + df['a_pc_true_raw']), 2)

                        df[['h_pc_true', 'd_pc_true', 'a_pc_true']] = df.apply(
                            lambda row: calculate_true_from_true_raw(row['h_pc_true_raw'], row['d_pc_true_raw'], row['a_pc_true_raw'], row['margin_wdw']), 
                            axis=1, result_type='expand')
                        

                        # Function to add goal exp column to df
                        def get_gl_exp_value(row, df_ou):
                            # Extract the 'ov_pc_true' value from the current row in df
                            ov_pc_true = row['ov_pc_true']
  
                            # Locate the row in df_ou where 'Ov2.5_%' matches the 'ov_pc_true' value
                            gl_exp_value = df_ou.loc[df_ou['Ov2.5_%'] == ov_pc_true, 'Exp1']
                            
                            # If there's no matching row, return NaN or a default value
                            return gl_exp_value.values[0] if not gl_exp_value.empty else None
                        
                        st.write(df_dnb)
                        #Function to add supremacy to df
                        def get_match_sup(row, df_dnb):
                            if pd.isna(row.get('dnb_perc_h')):
                                return None
                            dnb_perc_h = row['dnb_perc_h']
                            if 'dnb %' not in df_dnb.columns or df_dnb.empty:
                                return None
                            idx = (df_dnb['dnb %'] - dnb_perc_h).abs().idxmin()
                            return df_dnb.at[idx, 'Sup']


                        # Apply the function to each row in df to create a new column 'gl_exp'
                        df['Gl_Exp'] = df.apply(lambda row: get_gl_exp_value(row, df_ou), axis=1)
                        df['Sup'] = df.apply(lambda row: get_match_sup(row, df_dnb), axis=1)

                        df['hgx'] = round(df['Gl_Exp'] / 2 + 0.5 * df['Sup'], 2)
                        df['agx'] = round(df['Gl_Exp'] / 2 - 0.5 * df['Sup'], 2)

                        df['0-0'] = poisson.pmf(0, df['hgx']) * poisson.pmf(0, df['agx']) * 1.1

                        df['h_nxt_gl'] = (df['hgx'] / df['Gl_Exp']) * (1 - df['0-0'])
                        df['a_nxt_gl'] = (df['agx'] / df['Gl_Exp']) * (1 - df['0-0'])


                        # Display the updated DataFrame
                        st.write(df)

                        all_leagues_dfs.append(df)

                        # # warning if not all match  retrieved from API call matches the final df
                        # if len(df) != len(fixt_id_list):
                        #     st.warning('Odds for 1 or more matches not currently available!')

                # 👉 Concatenate all DataFrames after the loop
                if all_leagues_dfs:
                    combined_df = pd.concat(all_leagues_dfs, ignore_index=True)
                    st.write("### ✅ Combined DataFrame for all leagues:")
                    st.dataframe(combined_df)
                else:
                    st.warning("No data collected from any league.")     


                # add next goal odds
                # combined_df['hgx'] = combined_df['Gl_Exp'] / 2
                # combined_df['h_next_gl'] = 


            # ------------ 1-UP FUNCTIONS (top-level) ----------------

                def calculate_win_given_one_nil(hg, ag, minute_of_goal=29):
                    """
                    Returns (P(Home wins | 1-0 at minute_of_goal), P(Away wins | 0-1 at minute_of_goal))
                    Uses Poisson remainder with small adjustments for leading/trailing team.
                    """
                    minutes_remaining = 93 - minute_of_goal
                    if minutes_remaining <= 0:
                        return 1.0, 1.0  # after match end — defensive fallback

                    # rates per 93 minutes
                    home_rate = hg / 93.0
                    away_rate = ag / 93.0

                    rem_home_xg = home_rate * minutes_remaining
                    rem_away_xg = away_rate * minutes_remaining

                    # adjustments for lead/trail
                    adj_home_xg_lead = rem_home_xg * 0.95
                    adj_away_xg_trail = rem_away_xg * 1.05

                    adj_home_xg_trail = rem_home_xg * 1.05
                    adj_away_xg_lead = rem_away_xg * 0.95

                    max_goals = 7
                    win_prob_home = 0.0
                    win_prob_away = 0.0

                    for i in range(max_goals + 1):          # additional goals by home after lead
                        for j in range(max_goals + 1):      # additional goals by away after lead
                            # Home leads 1-0 -> final_home = 1+i, final_away = j
                            prob_home_lead = poisson.pmf(i, adj_home_xg_lead) * poisson.pmf(j, adj_away_xg_trail)
                            if (1 + i) > j:
                                win_prob_home += prob_home_lead

                            # Away leads 0-1 -> final_home = i, final_away = 1+j
                            prob_away_lead = poisson.pmf(i, adj_home_xg_trail) * poisson.pmf(j, adj_away_xg_lead)
                            if (1 + j) > i:
                                win_prob_away += prob_away_lead

                    return win_prob_home, win_prob_away
                
                
                combined_df[['h_win_given_1_0_up', 'a_win_given_0_1_up']] = combined_df.apply(
                    lambda r: calculate_win_given_one_nil(r['hgx'], r['agx'], minute_of_goal=29),
                    axis=1, result_type='expand'
                )

                combined_df['h_1_Up_p'] = combined_df['h_nxt_gl'] + combined_df['h_pc_true'] - (combined_df['h_nxt_gl'] * combined_df['h_win_given_1_0_up'])
                combined_df['a_1_Up_p'] = combined_df['a_nxt_gl'] + combined_df['a_pc_true'] - (combined_df['a_nxt_gl'] * combined_df['a_win_given_0_1_up'])

                combined_df['h_1_Up_odds_true'] = round(1 / combined_df['h_1_Up_p'], 2)
                combined_df['a_1_Up_odds_true'] = round(1 / combined_df['a_1_Up_p'], 2)

                combined_df['h_1_Up_odds_marg_raw'] = round(combined_df['h_1_Up_odds_true'] / margin_to_apply, 2)
                combined_df['a_1_Up_odds_marg_raw'] = round(combined_df['a_1_Up_odds_true'] / margin_to_apply, 2)


                # ---   Conditions to handle margin distribution fav/dog after initial margin added  -----------------
                # HOME

                mult_1 = 1.08  # < 1.15
                mult_2 = 1.06  # 1.15 ≤ x < 1.3
                mult_3 = 1.04  # < 1.30 but >= 1.55
                mult_4 = 1.02 # < 1.55 but >= 1.70
                mult_5 = 0.75 # > 10
                mult_6 = 0.82 # 6 < x <= 10
                mult_7 = 0.87 # 3.75 < x <= 6
                mult_8 = 0.92 # 2.7 < x <= 3.75

                conditions = [
                    combined_df['h_1_Up_odds_marg_raw'] < 1.15,             # strong home favorite
                    (combined_df['h_1_Up_odds_marg_raw'] >= 1.15) & (combined_df['h_1_Up_odds_marg_raw'] < 1.3),  # moderately big home favorite
                    (combined_df['h_1_Up_odds_marg_raw'] >= 1.30) & (combined_df['h_1_Up_odds_marg_raw'] < 1.55),  # medium home favorite
                    (combined_df['h_1_Up_odds_marg_raw'] >= 1.55) & (combined_df['h_1_Up_odds_marg_raw'] < 1.70),
                    combined_df['h_1_Up_odds_marg_raw'] > 10,               # big home underdog
                    (combined_df['h_1_Up_odds_marg_raw'] > 6) & (combined_df['h_1_Up_odds_marg_raw'] <= 10),       # moderately big home underdog
                    (combined_df['h_1_Up_odds_marg_raw'] > 3.75) & (combined_df['h_1_Up_odds_marg_raw'] <= 6),       # medium home underdog
                    (combined_df['h_1_Up_odds_marg_raw'] > 2.7) & (combined_df['h_1_Up_odds_marg_raw'] <= 3.75)       
                ]

                choices = [
                    mult_1,
                    mult_2,
                    mult_3,
                    mult_4,
                    mult_5,
                    mult_6,
                    mult_7,
                    mult_8,
                ]

                combined_df['h_1_Up_marg_odds_final'] = round(combined_df['h_1_Up_odds_marg_raw'] * np.select(conditions, choices, default=1.0), 2)


                # ----  AWAY  -----
                conditions = [
                    combined_df['a_1_Up_odds_marg_raw'] < 1.15,             # strong home favorite
                    (combined_df['a_1_Up_odds_marg_raw'] >= 1.15) & (combined_df['a_1_Up_odds_marg_raw'] < 1.3),  # moderately big home favorite
                    (combined_df['a_1_Up_odds_marg_raw'] >= 1.30) & (combined_df['a_1_Up_odds_marg_raw'] < 1.55),  # medium home favorite
                    (combined_df['a_1_Up_odds_marg_raw'] >= 1.55) & (combined_df['a_1_Up_odds_marg_raw'] < 1.70),
                    combined_df['a_1_Up_odds_marg_raw'] > 10,               # big home underdog
                    (combined_df['a_1_Up_odds_marg_raw'] > 6) & (combined_df['a_1_Up_odds_marg_raw'] <= 10),       # moderately big home underdog
                    (combined_df['a_1_Up_odds_marg_raw'] > 3.75) & (combined_df['a_1_Up_odds_marg_raw'] <= 6),       # medium home underdog
                    (combined_df['a_1_Up_odds_marg_raw'] > 2.7) & (combined_df['a_1_Up_odds_marg_raw'] <= 3.75)       
                ]

                choices = [
                    mult_1,
                    mult_2,
                    mult_3,
                    mult_4,
                    mult_5,
                    mult_6,
                    mult_7,
                    mult_8,
                ]

                combined_df['a_1_Up_marg_odds_final'] = round(combined_df['a_1_Up_odds_marg_raw'] * np.select(conditions, choices, default=1.0), 2)

                st.write(combined_df)

                # TODO
                '''
                add league column
                separate out time as own column
                harmonize team names
                remove superfluous columns
                make bst/utc adjustable

                format fmh
                '''




                         
                        
            except Exception as e:
                st.write(f'An error has occurred whilst compiling: {e}')   



if __name__ == "__main__":
    main()