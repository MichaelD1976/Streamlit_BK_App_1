import streamlit as st
import pandas as pd
import altair as alt
import requests
import time
from mymodule.functions import get_fixtures
from datetime import datetime, timedelta
import gc
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from dotenv import load_dotenv
import os


CURRENT_SEASON = '2025-26'
# year_options = ['2024-25', '2023-24']

# Define league options
league_options = {
    'E0': 'England Premier',
    # 'D1': 'Germany Bundesliga',
    # 'SP1': 'Spain La Liga',
    # 'I1': 'Italy Serie A',
    # 'F1': 'France Ligue 1',
    # 'N1': 'Netherlands Eredivisie',
    # 'B1': 'Belgium Jupiler',
    # 'P1': 'Portugal Liga I',
    'E1': 'England Championship',
    'E2': 'England League One',
    'E3': 'England League Two',
#    'SC0': 'Scotland Premier',
#     'D2': 'Germany 2 Bundesliga',
#     'SP2': 'Spain La Liga 2',
#     'I2': 'Italy Serie B',
# #    'SP2': 'France Ligue 2',
#     'All Leagues': '* All Leagues *',  # Add 'All Leagues' option
}

team_naming_dict = {
    "Nott'm Forest": "Nottm Forest",
    "Peterboro": "Peterborough",
    'Sheffield United': 'Sheffield Utd',
    'Dundee United': 'Dundee Utd',
    'Man United': 'Man Utd',
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
    "Scotland Championship": '180',
    "Scotland League One": '181',
}




# Load the CSV file
@st.cache_data
def load_data():
    time.sleep(2)
    df = pd.read_csv('data/outputs_processed/teams/raw_football_data_current_week.csv')
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df



def main():
    with st.spinner('Loading Data...'):
        df = load_data()

    # Sidebar for user input
    st.sidebar.title('Select Data Filters')

    # Define selection options
    league_options = {
        'E0': 'England Premier',
        # 'D1': 'Germany Bundesliga',
        # 'SP1': 'Spain La Liga',
        # 'I1': 'Italy Serie A',
        # 'F1': 'France Ligue 1',
        # 'N1': 'Netherlands Eredivisie',
        # 'B1': 'Belgium Jupiler',
        # 'P1': 'Portugal Liga I',
        'E1': 'England Championship',
        'E2': 'England League One',
        'E3': 'England League Two',
        'SC0': 'Scotland Premier',
    #    'SC1': 'Scotland Championship',
    #    'SC2': 'Scotland League One',
    #    'SC3': 'Scotland League Two',
        # 'D2': 'Germany 2 Bundesliga',
        # 'I2': 'Italy Serie B',
        # 'SP2': 'Spain La Liga 2',
        # 'F2': 'France Ligue 2',
        # 'T1': 'Turkey Super Lig',
        # 'G1': 'Greece Super League'
    }

    # year_options = [
    #                 '2025-26',
    #                 '2024-25',
    # ]

    # Capture user selections from sidebar
    selected_league = st.sidebar.selectbox('Select League', options=list(league_options.values()), label_visibility = 'visible') # WIDGET
    # selected_year = st.sidebar.selectbox('Select Year', options=year_options, label_visibility = 'visible') # WIDGET

        # Function to apply filters
    def apply_filters(df, league):
        df['Date'] = pd.to_datetime(df['Date'])
        # Filter by league
        df = df[df['Div'] == [key for key, value in league_options.items() if value == league][0]]
        # Filter by year

        # if year == '2025-26':
        #     start_date, end_date = datetime(2025, 8, 1), datetime(2026, 7, 1)
        # elif year == '2024-25':
        #     start_date, end_date = datetime(2024, 8, 1), datetime(2025, 7, 1)

        # df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        return df

    # Apply filters
    filtered_df = apply_filters(df, selected_league)

    st.header(f'High Draw Matches - {selected_league}', divider='blue')

    # st.write(filtered_df)
        
    del df
    gc.collect()

    filtered_df_2 = filtered_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H','B365D', 'B365A']]
    filtered_df_2['TG'] = filtered_df_2['FTHG'] + filtered_df_2['FTAG']
    filtered_df_2.reset_index(drop=True, inplace=True)

    # st.write(filtered_df_2)


    # -----  Create a dataframe showing the number of matches played and mean total goals (TG) for each team in the selected league and year ----- #

    # Create one dataframe for home teams
    home_df = filtered_df_2[['HomeTeam', 'TG']].copy()
    home_df = home_df.rename(columns={'HomeTeam': 'Team'})

    # Create one dataframe for away teams
    away_df = filtered_df_2[['AwayTeam', 'TG']].copy()
    away_df = away_df.rename(columns={'AwayTeam': 'Team'})

    # Stack them together
    teams_df = pd.concat([home_df, away_df], ignore_index=True)

    # Aggregate
    tg_df = (
        teams_df
        .groupby('Team')
        .agg(
            Matches_Played=('TG', 'count'),
            Mean_TG=('TG', 'mean')
        )
        .reset_index()
    )

    # st.write(tg_df)

    # ----- Create a dataframe showing the number of matches played and mean B365 odds for home wins and away wins for each team  ----- #

    # --- Home stats ---
    home_stats = (
        filtered_df_2.groupby('HomeTeam')
        .agg(
            Home_Matches=('HomeTeam', 'count'),
            Avg_B365H=('B365H', 'mean')
        )
        .reset_index()
        .rename(columns={'HomeTeam': 'Team'})
    )

    # --- Away stats ---
    away_stats = (
        filtered_df_2.groupby('AwayTeam')
        .agg(
            Away_Matches=('AwayTeam', 'count'),
            Avg_B365A=('B365A', 'mean')
        )
        .reset_index()
        .rename(columns={'AwayTeam': 'Team'})
    )

    # --- Merge together ---
    av_odds_df = pd.merge(home_stats, away_stats, on='Team', how='outer')
    av_odds_df['Av_odds'] = av_odds_df[['Avg_B365H', 'Avg_B365A']].mean(axis=1)

    # st.write('196', av_odds_df)

    # Merge Mean_TG (and Matches_Played if you want) into av_odds_df
    av_odds_df = av_odds_df.merge(
        tg_df[['Team', 'Mean_TG', 'Matches_Played']],
        on='Team',
        how='left'   # use left so you keep all teams from av_odds_df
    )

    # st.write('217', av_odds_df)

    # # ---   Conditions to handle different average odds and the average expected goals based on those odds  -----------------

  
    goal_exp = [
        2.9, 2.85, 2.8, 2.75, 2.7, 2.65, 2.6, 2.6, 2.55, 2.5, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.95, 3.1
        ]

    # Define your ranges as tuples (lower_bound, upper_bound)
    # Note: np.select checks conditions in order, so ranges should not overlap
    ranges = [
        (1.3, 1.5),   # 2.9
        (1.5, 1.6),      # 2.85 
        (1.6, 1.7),   # 2.8
        (1.7, 1.8),   # 2.75
        (1.8, 1.9),   # 2.7
        (1.9, 2),   # 2.65
        (2, 2.1),   # 2.6
        (2.1, 2.2),   # 2.6
        (2.2, 2.4),   # 2.55
        (2.4, 2.6),   # 2.5
        (2.6, 3.0),   # 2.45
        (3.0, 3.4),   # 2.5
        (3.4, 3.8),   # 2.55
        (3.8, 4.4),   # 2.6
        (4.4, 5),   # 2.65  
        (5, 5.5),  # 2.7
        (5.5, 6),  # 2.75
        (6, 7),  # 2.8
        (7,9),  # 2.85
        (9, 11),  # 2.95
        (11, 15),  # 3.1
    ]

    def get_multiplier(series):
        conditions = [
            (series >= low) & (series < high)
            for (low, high) in ranges
        ]
        return np.select(conditions, goal_exp, default=3.2)
    

    av_odds_df['g_exp'] = get_multiplier(av_odds_df['Av_odds'])
    av_odds_df['g_exp'] = av_odds_df['g_exp'].round(2)

    av_odds_df['Mean_TG/g_exp'] = av_odds_df['Mean_TG'] / av_odds_df['g_exp']
    av_odds_df['Mean_TG/g_exp_scl'] = av_odds_df['Mean_TG/g_exp'] / av_odds_df['Mean_TG/g_exp'].mean()

    av_odds_df['Team'] = av_odds_df['Team'].map(team_naming_dict).fillna(av_odds_df['Team'])

    
    # st.write('253', av_odds_df)


    # -------------------------------------------- CREATE ODDS FOR ALL UPCOMING FIXTURES --------------------------------------------------------------------

    # get fixtures
    league_id = leagues_dict.get(selected_league)


    # st.write("---")
    st.subheader(f'Generate fixtures for upcoming {selected_league} matches (up to 7 days ahead)')

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
        with st.spinner("Fetching fixtures..."):
            try:
                from_date_str = today.strftime("%Y-%m-%d")
                to_date_str = up_to_date.strftime("%Y-%m-%d")
                MARKET_IDS = ['1', '5']             # WDW & Ov/Un
                BOOKMAKERS = ['4']                  # Pinnacle = 4, 365 = 8, Uni = 16, BF = 3
                API_SEASON = CURRENT_SEASON[:4]

                
                df_fixtures = get_fixtures(league_id, from_date_str, to_date_str, API_SEASON)
                # st.write('304', df_fixtures)
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
                    
                    # st.write('271', df_collapsed)

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
                    # st.write('442', df)


                    # Create a mapping from team → scaled ratio
                    team_ratio_map = av_odds_df.set_index('Team')['Mean_TG/g_exp_scl']

                    # Map for HomeTeam
                    df['Home_mean_tg/g_exp'] = df['Home Team'].map(team_ratio_map)

                    # Map for AwayTeam
                    df['Away_mean_tg/g_exp'] = df['Away Team'].map(team_ratio_map)

                    df['Goal_PR'] = (df['Home_mean_tg/g_exp'] * df['Away_mean_tg/g_exp']) 

                    # st.write('454', df)

                    # Define the bins and labels
                    bins = [-np.inf, 0.8, 0.9, 1.1, 1.2, np.inf]
                    labels = ['V LOW GOALS', 'Low', 'ok', 'High', 'V HIGH GOALS']

                    # Create the Outcome column
                    df['Outcome'] = pd.cut(df['Goal_PR'], bins=bins, labels=labels, right=True)

                    # If you want a string dtype rather than categorical
                    df['Outcome'] = df['Outcome'].astype(str)
                    # st.write(df)
                    df = df[['Date', 'Home Team', 'Away Team', 'Outcome', 'Goal_PR', 'Home_mean_tg/g_exp', 'Away_mean_tg/g_exp']]


                    st.write(df)

                    st.caption('''
                            The Goal_PR column is calculated as the product of the scaled Mean_TG/g_exp values for the home and away teams. 
                            The Outcome column categorises matches into different goal expectation levels based on the Goal_PR value. 
                            Mean_TG/g_exp values are derived from the average total goals (Mean_TG) divided by the expected goals (g_exp) based on average odds and corresponding expected goals, 
                            so supremacy of teams is factored in.
                            '''
                            )



                    # # Filter only caution matches
                    # caution_df = df[df['Outcome'].isin(['V LOW GOALS', 'Low'])]

                    # st.markdown("### ⚠️ Matches flagged as low goals (high draw)")

                    # # Loop through the rows
                    # for idx, row in caution_df.iterrows():
                    #     if row['Outcome'] == 'V LOW GOALS':
                    #         st.error(f"{row['Home Team']} vs {row['Away Team']} — {row['Outcome']}")
                    #     elif row['Outcome'] == 'Low':
                    #         st.warning(f"{row['Home Team']} vs {row['Away Team']} — {row['Outcome']}")


 

                    # if df.empty:
                    #     st.write('Odds currently unavailable from API') 



            except Exception as e:
                st.error(f"An error occurred while fetching odds: {e}")


if __name__ == "__main__":
    main()



# TODO graphic to show team 'Goal_PR' ratings
# TODO add lower scottish leagues
# TODO add low goal 'Matches flagged'
# TODO add high goal 'Matches flagged'
# TODO add derbies