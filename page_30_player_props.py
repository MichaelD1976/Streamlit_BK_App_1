import streamlit as st
import pandas as pd
import altair as alt
import time
import gc
from mymodule.functions import get_fixtures, calculate_expected_team_goals_from_1x2_refined
import requests
import joblib
from dotenv import load_dotenv
import os
from scipy.stats import poisson


API_SEASON = '2026'

# Dictionary to map league names to their IDs
leagues_dict = {
    "England Premier": '39',
    "Spain La Liga": '140',
    "Germany Bundesliga": '78',
    "Italy Serie A": '135',
    "France Ligue 1": '61',
    "England Championship": '40',
    "England League One": '41', 
    "England League Two": '42'  
}

sot_model_h = joblib.load('models/sot/sot_home_poisson.pkl')
sot_model_a = joblib.load('models/sot/sot_away_poisson.pkl')


# Load the CSV file
@st.cache_data
def load_data(per_90=True):
    time.sleep(0.5)  # Simulate a delay for loading
    df = pd.read_csv('data/outputs_processed/players/processed_f_api_combined_all_seasons_standard.csv')
    return df

def main():

    with st.spinner('Loading Data...'):

        df = load_data()


    # =========================================================
    # INITIALISE SESSION STATE
    # =========================================================

    if 'selected_league' not in st.session_state:
        st.session_state.selected_league = 'England Premier'

    if 'selected_season_range' not in st.session_state:
        st.session_state.selected_season_range = 'All seasons'

    if 'selected_team' not in st.session_state:
        st.session_state.selected_team = None

    if 'selected_player' not in st.session_state:
        st.session_state.selected_player = None

    if 'selected_metric' not in st.session_state:
        st.session_state.selected_metric = 'Goals'


    # =========================================================
    # SIDEBAR
    # =========================================================

    st.sidebar.title('Select Data Filters')


    # =========================================================
    # FILTER BY LEAGUE
    # =========================================================

    league_options = [
        'England Premier',
        'Germany Bundesliga',
        'Spain La Liga',
        'Italy Serie A',
        'France Ligue 1',
        'England League One',
        'England League Two',
    ]

    selected_league = st.sidebar.selectbox(
        'Select League',
        options=league_options,
        key='selected_league'
    )


    # =========================================================
    # FILTER BY SEASON RANGE
    # =========================================================

    season_range_options = [
        'Current season',
        'Last 2 seasons',
        'Last 3 seasons',
        'All seasons'
    ]

    selected_season_range = st.sidebar.selectbox(
        'Select Seasons',
        options=season_range_options,
        key='selected_season_range'
    )


    # Determine seasons to use
    if selected_season_range == 'Current season':
        selected_years = ['2026-27']

    elif selected_season_range == 'Last 2 seasons':
        selected_years = ['2026-27', '2025-26']

    elif selected_season_range == 'Last 3 seasons':
        selected_years = ['2026-27', '2025-26', '2024-25']

    else:
        # All available seasons
        selected_years = df['Season'].dropna().unique().tolist()


    # =========================================================
    # FILTER DATA BY LEAGUE
    # =========================================================

    filtered_df = df[
        df['League'] == selected_league
    ]


    # =========================================================
    # FILTER DATA BY SEASONS
    # =========================================================

    filtered_df = filtered_df[
        filtered_df['Season'].isin(selected_years)
]

    # =========================================================
    # FILTER BY TEAM
    # =========================================================

    team_options = sorted(
        filtered_df['Team'].dropna().unique().tolist()
    )

    if team_options:

        # Preserve existing team if it is still available
        if st.session_state.selected_team not in team_options:
            st.session_state.selected_team = team_options[0]

        team_index = team_options.index(
            st.session_state.selected_team
        )

        selected_team = st.sidebar.selectbox(
            'Select Team',
            options=team_options,
            index=team_index,
            key='selected_team'
        )

        filtered_df_squad = filtered_df[
            filtered_df['Team'] == selected_team
        ]

    else:

        st.session_state.selected_team = None
        st.session_state.selected_player = None

        selected_team = None
        filtered_df_squad = filtered_df.iloc[0:0]


    # =========================================================
    # FILTER BY PLAYER
    # =========================================================

    player_options = sorted(
        filtered_df_squad['Player'].dropna().unique().tolist()
    )

    if player_options:

        # Preserve existing player if it is still available
        if st.session_state.selected_player not in player_options:
            st.session_state.selected_player = player_options[0]

        player_index = player_options.index(
            st.session_state.selected_player
        )

        selected_player = st.sidebar.selectbox(
            'Select Player',
            options=player_options,
            index=player_index,
            key='selected_player'
        )

        filtered_df_player = filtered_df_squad[
            filtered_df_squad['Player'] == selected_player
        ]

    else:

        st.session_state.selected_player = None

        selected_player = None
        filtered_df_player = filtered_df_squad.iloc[0:0]


# =========================================================
# FILTER FOR METRIC
# =========================================================

    metrics = [
        'Goals', 'Assists', 'Shots On', 'Shots Total', 'Shots on/gl',
        'Fouls Drawn', 'Fouls Committed', 'Tackles Total', 'Blocks', 'Passes Total',
        'Passes Key', 'Dribbles Attempted', 'Dribbles Success',
        'Interceptions', 'Yellow Cards', 'Red Cards', 'Duels Total',
        'Duels Won',
    ]


    # Ensure selected metric remains valid
    if st.session_state.selected_metric not in metrics:
        st.session_state.selected_metric = 'Goals'


    selected_metric = st.sidebar.selectbox(
        'Select Metric for Analysis',
        options=metrics,
        key='selected_metric'
    )


    # =========================================================
    # CLEANUP
    # =========================================================

    del df
    gc.collect()


    # =========================================================
    # DISPLAY
    # =========================================================

    st.header('Player Prop Pricing', divider='red')

    st.subheader(
        f'{selected_player} - Player {selected_metric}: '
        f'{selected_season_range}'
    )


    # =========================================================
    # SHOW RAW DATA
    # =========================================================

    if st.checkbox(
        'Show raw filtered data',
        key='show_raw_data',
        label_visibility='visible'
    ):

        st.write(
            f'Squad Stats for {selected_team}: '
            f'{selected_season_range}'
        )

        st.dataframe(
            filtered_df_squad,
            use_container_width=True
        )

        st.write(
            f'Stats for {selected_player}: '
            f'{selected_season_range}'
        )

        st.dataframe(
            filtered_df_player,
            use_container_width=True
        )

        st.write("")

# -------------------------------------------------------------------------------------------------
    '''
        The below code calculates the % share of the selected metric for the selected player compared to the average of the team and the average of the 
        player's position. It also creates a hypothetical starting XI based on positional averages and calculates the expected share of the selected player 
        in that XI.

        All squad players have their per 90 calculated - total metric / total minutes (Avg per 90)
        Of all the defenders, midfielders, and attackers, the average per 90 is calculated for each position (positional benchmark)
        We then construct 2 starting XIs (4-4-2 and 4-3-3) using the positional averages, and replace one of the players in the XI (in the selected players position) 
        with the selected player to see how much of the total expected production they would contribute.

        The final expected share is the average of the two formations.

    '''

    # =========================================================
    # PLAYER VS TEAM VS POSITION — PER 90
    # =========================================================

    if selected_player is not None and not filtered_df_player.empty:

        # Get selected player's position
        selected_position = filtered_df_player['Position'].iloc[0]

        # -----------------------------------------------------
        # FUNCTION TO CALCULATE PER 90
        # -----------------------------------------------------

        def per_90(data, metric):

            total_stat = data[metric].sum()
            total_minutes = data['Minutes'].sum()

            if total_minutes == 0:
                return 0

            return (total_stat / total_minutes) * 90


        # -----------------------------------------------------
        # PLAYER PER 90
        # -----------------------------------------------------

        player_average = per_90(
            filtered_df_player,
            selected_metric
        )


        # -----------------------------------------------------
        # TEAM PER 90
        # -----------------------------------------------------

        team_average = per_90(
            filtered_df_squad,
            selected_metric
        )


        # -----------------------------------------------------
        # POSITION PER 90
        # -----------------------------------------------------

        position_df = filtered_df_squad[
            filtered_df_squad['Position'] == selected_position
        ]

        position_average = per_90(
            position_df,
            selected_metric
        )


        # -----------------------------------------------------
        # CREATE DATAFRAME FOR PLOT
        # -----------------------------------------------------

        comparison_df = pd.DataFrame({
            'Category': [
                selected_player,
                f'{selected_team} Average',
                f'{selected_position} Average'
            ],
            'Average': [
                player_average,
                team_average,
                position_average
            ]
        })


        # -----------------------------------------------------
        # HORIZONTAL BAR CHART
        # -----------------------------------------------------

        comparison_df['Type'] = [
            'Player',
            'Team',
            'Position'
        ]

        bars = (
            alt.Chart(comparison_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    'Average:Q',
                    title=f'{selected_metric} per 90'
                ),
                y=alt.Y(
                    'Category:N',
                    title='',
                    sort=None
                ),
                color=alt.Color(
                    'Type:N',
                    scale=alt.Scale(
                        domain=[
                            'Player',
                            'Team',
                            'Position'
                        ],
                        range=[
                            '#1f77b4',
                            '#ff7f0e',
                            '#2ca02c'
                        ]
                    ),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip(
                        'Category:N',
                        title=''
                    ),
                    alt.Tooltip(
                        'Average:Q',
                        title=f'{selected_metric} per 90',
                        format='.2f'
                    )
                ]
            )
        )

        labels = (
            alt.Chart(comparison_df)
            .mark_text(
                align='left',
                baseline='middle',
                dx=5
            )
            .encode(
                x='Average:Q',
                y=alt.Y(
                    'Category:N',
                    sort=None
                ),
                text=alt.Text(
                    'Average:Q',
                    format='.2f'
                )
            )
        )

        chart = (
            (bars + labels)
            .properties(
                title=f'{selected_metric} per 90: '
                      f'Player vs Team vs Position',
                height=200
            )
            .configure_view(
                strokeWidth=0
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True
        )



    # ====== CREATE TABLE OF PLAYERS WITH MOST MINS FOR SELECTED SEASON RANGE AND METRIC ======

    # =========================================================
    # PLAYER SUMMARY
    # =========================================================

    outfield_squad = filtered_df_squad[
        filtered_df_squad['Position'].str.lower() != 'goalkeeper'
    ].copy()

    outfield_squad['Position'] = (
        outfield_squad['Position']
        .str.lower()
    )


    # ---------------------------------------------------------
    # Total minutes and metric for every player
    # ---------------------------------------------------------

    player_summary = (
        outfield_squad
        .groupby('Player')
        .agg(
            Minutes=('Minutes', 'sum'),
            Metric=(selected_metric, 'sum')
        )
        .reset_index()
    )


    # ---------------------------------------------------------
    # Determine primary position
    # ---------------------------------------------------------

    position_minutes = (
        outfield_squad
        .groupby(['Player', 'Position'])['Minutes']
        .sum()
        .reset_index()
    )

    primary_positions = (
        position_minutes
        .sort_values(
            ['Player', 'Minutes'],
            ascending=[True, False]
        )
        .drop_duplicates('Player')
        [['Player', 'Position']]
    )


    player_summary = player_summary.merge(
        primary_positions,
        on='Player',
        how='left'
    )


    # ---------------------------------------------------------
    # Per 90
    # ---------------------------------------------------------

    player_summary['Per 90'] = (
        player_summary['Metric']
        / player_summary['Minutes']
        * 90
    )


    # =========================================================
    # TOP 15 TABLE
    # =========================================================

    top_15_players = (
        player_summary
        .sort_values('Minutes', ascending=False)
        .head(15)
    )

    st.subheader('Top 15 Outfield Players by Minutes')

    st.dataframe(
        top_15_players[
            ['Player', 'Position', 'Minutes', 'Metric', 'Per 90']
        ],
        use_container_width=True,
        hide_index=True
    )


    # =========================================================
    # POSITIONAL BENCHMARK
    # =========================================================

    minimum_minutes = 900

    eligible_players = player_summary[
        player_summary['Minutes'] >= minimum_minutes
    ].copy()


    position_averages = (
        eligible_players
        .groupby('Position')
        .agg(
            Players=('Player', 'count'),
            Average_Per_90=('Per 90', 'mean')
        )
        .reset_index()
    )

    position_averages['Average_Per_90'] = (
        position_averages['Average_Per_90'].round(2)
    )


    if st.checkbox('Show Positional Benchmarks', key='show_positional_benchmarks'):

        st.write('Positional Benchmarks')

        st.dataframe(
            position_averages,
            use_container_width=True,
            hide_index=True
        )

    # =========================================================
    # HYPOTHETICAL STARTING XI
    # =========================================================

    # Convert positional averages into a simple lookup
    position_rates = (
        position_averages
        .set_index('Position')['Average_Per_90']
        .to_dict()
    )


    # ---------------------------------------------------------
    # Selected player's rate
    # ---------------------------------------------------------

    selected_player_row = player_summary[
        player_summary['Player'] == selected_player
    ]

    if not selected_player_row.empty:

        selected_player_rate = (
            selected_player_row.iloc[0]['Per 90']
        )

        selected_player_position = (
            selected_player_row.iloc[0]['Position']
        )

    else:

        selected_player_rate = None
        selected_player_position = None


    # ---------------------------------------------------------
    # Calculate hypothetical XI rate
    # ---------------------------------------------------------

    def calculate_xi_rate(
        formation,
        selected_position,
        selected_rate,
        position_rates
    ):

        if formation == '4-4-2':
            defenders = 4
            midfielders = 4
            attackers = 2

        elif formation == '4-3-3':
            defenders = 4
            midfielders = 3
            attackers = 3

        else:
            return None


        defender_rate = position_rates.get(
            'defender',
            0
        )

        midfielder_rate = position_rates.get(
            'midfielder',
            0
        )

        attacker_rate = position_rates.get(
            'attacker',
            0
        )


        # Start with the normal positional XI
        total_rate = (
            defenders * defender_rate
            + midfielders * midfielder_rate
            + attackers * attacker_rate
        )


        # Replace one positional-average player
        # with the selected player
        if selected_position == 'defender':

            total_rate -= defender_rate
            total_rate += selected_rate

        elif selected_position == 'midfielder':

            total_rate -= midfielder_rate
            total_rate += selected_rate

        elif selected_position == 'attacker':

            total_rate -= attacker_rate
            total_rate += selected_rate


        return total_rate


    # =========================================================
    # PLAYER EXPECTED SHARE
    # =========================================================

    average_share = None # initialise

    if selected_player_rate is not None:

        rate_442 = calculate_xi_rate(
            '4-4-2',
            selected_player_position,
            selected_player_rate,
            position_rates
        )

        rate_433 = calculate_xi_rate(
            '4-3-3',
            selected_player_position,
            selected_player_rate,
            position_rates
        )


        share_442 = (
            selected_player_rate / rate_442
            if rate_442 > 0
            else 0
        )

        share_433 = (
            selected_player_rate / rate_433
            if rate_433 > 0
            else 0
        )


        average_share = (
            share_442 + share_433
        ) / 2


        st.subheader(
            f'{selected_player} Expected Team Share'
        )


        col1, col2, col3 = st.columns(3)


        with col1:
            st.metric(
                '4-4-2 Share',
                f'{share_442:.1%}'
            )

        with col2:
            st.metric(
                '4-3-3 Share',
                f'{share_433:.1%}'
            )

        with col3:
            st.metric(
                'Average Expected Share',
                f'{average_share:.1%}'
            )

    # =========================================================
    if st.checkbox('Show calculation logic'):

        st.info(
            """The expected team share estimates what percentage of a team's
            expected production for the selected metric could be attributed to
            the selected player if they were included in the starting XI. The
            calculation uses the player's total output and minutes over the
            selected period to calculate their individual per-90 rate. Their
            primary position is determined by where they played the most minutes.
            Positional benchmarks are calculated from eligible outfield players
            with at least 900 minutes, excluding goalkeepers. Two hypothetical
            starting XIs are then modelled — a 4-4-2 and a 4-3-3 — using the
            average per-90 rate for each position, with the selected player
            replacing the average player in their position. The player's rate is
            divided by the total expected production of each XI to calculate
            their expected share, and the final percentage is the average of the
            two formations."""
        )

    st.write('---')
    # =========================================================

    st.subheader('Player Expectation for Next Fixture')


    # Player wasn't found in player_summary
    if average_share is None:

        st.warning(
            f"Unable to calculate an expected share for {selected_player}. "
        )

        st.stop()


    # Player was found, but calculated share is zero/negative
    if average_share <= 0:

        st.warning(
            'Average Share must be greater than 0.'
        )

        st.stop()

    elif average_share >0.5:
        st.warning(
            'Average share is disproportionately high. Check data and/or manually adjust the lower Expected Team Share value'
        )


    average_share = st.number_input(
        'Average Expected Player Share %:',
        value=average_share,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )

    expected_minutes = st.number_input(
        'Expected Minutes to Play (defaulted to 87):',
        value=87,
        min_value=50,
        max_value=90,
        step=1,
    )

    player_form_boost = st.number_input(
        f"Is {selected_player} playing better/worse than his 'average' data might suggest? If so apply a % adjustment:",
        value=1.0,
        min_value=0.9,
        max_value=1.1,
        step=0.01,
    )

    # except:
    #     st.error("Selected player must have an expected share of > 0")
    #     st.stop

    if st.button("Run Next Match Calculation"):

        league_id = leagues_dict.get(selected_league)
        from_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        to_date = (pd.Timestamp.now() + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        df_fixtures = get_fixtures(league_id, from_date, to_date, API_SEASON)

        # if df_fixtures empty then tell user and stop code
        if df_fixtures.empty:
            st.write('No upcoming fixtures available (code line:815)') 
            st.stop()

        # st.write('818', df_fixtures)

        try:
            fixture_id = int(df_fixtures[(df_fixtures['Home Team'] == selected_team) | (df_fixtures['Away Team'] == selected_team)]['Fixture ID'].values[0])
        except:
            st.write('Next fixture unavailable (code line:823)')
            st.stop()

        # st.write(fixture_id)



        # -------------- get fixture odds'


        def get_odds(fixture_id, market_ids, bookmakers):
            load_dotenv()

            API_KEY = os.getenv('API_KEY_FOOTBALL_API')

            url = "https://api-football-v1.p.rapidapi.com/v3/odds"

            headers = {
                "X-RapidAPI-Key": API_KEY,
                "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
            }

            result = {
                "Fixture_ID": fixture_id,
                "Home": None,
                "Draw": None,
                "Away": None
            }

            for market_id in market_ids:

                querystring = {
                    "fixture": fixture_id,
                    "bet": str(market_id),
                    "timezone": "Europe/London"
                }

                response = requests.get(
                    url,
                    headers=headers,
                    params=querystring
                )

                data = response.json()

                # st.write(f"Market {market_id} status:", response.status_code)
                # st.write(f"Market {market_id} response:", data)

                if 'response' not in data or not data['response']:
                    continue

                fixture_data = data['response'][0]

                for bookmaker_data in fixture_data.get('bookmakers', []):

                    if str(bookmaker_data['id']) not in bookmakers:
                        continue

                    for bet_data in bookmaker_data.get('bets', []):

                        if bet_data['id'] != int(market_id):
                            continue

                        # Market 1 - Match Winner
                        if int(market_id) == 1:

                            for value in bet_data.get('values', []):

                                selection = value['value']
                                odd = value['odd']

                                if selection == 'Home':
                                    result['Home'] = odd

                                elif selection == 'Draw':
                                    result['Draw'] = odd

                                elif selection == 'Away':
                                    result['Away'] = odd

                        # Market 5 - Over/Under
                        elif int(market_id) == 5:

                            wanted_selections = {
                                "Over 2.5",
                                "Under 2.5",
                                "Over 3.5",
                                "Under 3.5"
                            }

                            for value in bet_data.get('values', []):

                                selection = value['value']
                                odd = value['odd']

                                if selection == "Over 2.5":
                                    result["Over 2.5"] = odd

                                elif selection == "Under 2.5":
                                    result["Under 2.5"] = odd

                                elif selection == "Over 3.5":
                                    result["Over 3.5"] = odd

                                elif selection == "Under 3.5":
                                    result["Under 3.5"] = odd

            return pd.DataFrame([result])


        market_ids = ['1', '5']
        bookmaker_ids = ['4']

        # Find the selected fixture
        fixture_row = df_fixtures[
            df_fixtures['Fixture ID'] == fixture_id
        ].iloc[0]

        odds_df = get_odds(
            fixture_id,
            market_ids,
            bookmaker_ids
        )

        # st.write('947', odds_df)
        if odds_df['Home'].values == None:
            st.write(f'Odds for next match unavailable (code line:949)')
            st.stop()

        # Add team names from df_fixtures
        odds_df.insert(
            1,
            "Home Team",
            fixture_row["Home Team"]
        )

        odds_df.insert(
            2,
            "Away Team",
            fixture_row["Away Team"]
        )

        st.caption('Next fixture:')

        if odds_df.empty:
            st.write('Next fixture odds unavailable (code line: 970)')
            st.stop()
        else:
            st.write(odds_df)


        # filter only metrics which can be used for prop pricing
        applicable_metric_list = ['Goals', 
                                  'Assists', 
                                  'Shots On', 
                                  'Shots Total', 
                                  'Fouls Committed']
        
        if selected_metric not in applicable_metric_list:
            st.warning(f'{selected_metric} currently unavailable for player prop pricing')
            st.stop()

        # ====== CALCULATE TEAM XG ===========#

        home_odds = float(odds_df['Home'].values[0])
        draw_odds = float(odds_df['Draw'].values[0])
        away_odds = float(odds_df['Away'].values[0])
        over_2_5_odds = float(odds_df['Over 2.5'].values[0])
        under_2_5_odds = float(odds_df['Under 2.5'].values[0])

        # ===== Calculate GOALS metric ======#
        hxg, axg = calculate_expected_team_goals_from_1x2_refined(home_odds, draw_odds, away_odds, over_2_5_odds, under_2_5_odds)

        if odds_df['Home Team'].values[0] == selected_team:
            team_xg = hxg
        else:
            team_xg = axg

        # ===== Calculate Assists metric ======# 
        # no code needed - output works for assists without any changes needed

     # ===== Calculate SOT metric ======#   
        if selected_metric == 'Shots On': 
            # using formula from 'Relationships Analysis' -- ALL LEAGUES USED
            # y_h = 0.177x2 + 0.905x + 4.162
            # y_a = 0.237x2 -1.029x + 4.00

            sup = hxg - axg
            sot_exp_h = round((0.177 * sup * sup) + (0.905 * sup) + 4.162, 2)
            sot_exp_a = round((0.237 * sup * sup) - (1.029 * sup) + 4.00, 2)

            if odds_df['Home Team'].values[0] == selected_team:
                team_xg = sot_exp_h
            else:
                team_xg = sot_exp_a


    # ===== Calculate Shots Total metric ======#   
        if selected_metric == 'Shots Total': 
            # using formula from 'Relationships Analysis' -- ALL LEAGUES USED

            sup = hxg - axg
            shots_tot_h = round((0.156 * sup * sup) + (2.368 * sup) + 12.650, 2)
            shots_tot_a = round((0.347 * sup * sup) - (2.314 * sup) + 11.677, 2)

            if odds_df['Home Team'].values[0] == selected_team:
                team_xg = shots_tot_h
            else:
                team_xg = shots_tot_a


    # ===== Calculate Fouls metric ======# 
        if selected_metric == 'Fouls Committed': 
            # using formula from 'Relationships Analysis'
            # y_h = -0.281x2 - 0.111x + 12.325
            # y_a = -0.372x2 + 0.292x + 12.719

            sup = hxg - axg

            if selected_league == 'England Premier':
                fouls_exp_h = round((-0.444 * sup * sup) + (0.245 * sup) + 11.212, 2)
                fouls_exp_a = round((-0.386 * sup * sup) + (0.780 * sup) + 11.492, 2)

            elif selected_league == 'Spain':
                fouls_exp_h = round((-0.406 * sup * sup) - (0.923 * sup) + 13.207, 2)
                fouls_exp_a = round((-0.534 * sup * sup) + (0.616 * sup) + 12.794, 2)

            elif selected_league == 'Italy':
                fouls_exp_h = round((-0.364 * sup * sup) - (0.005 * sup) + 12.824, 2)
                fouls_exp_a = round((-0.596 * sup * sup) + (0.300 * sup) + 13.522, 2)

            elif selected_league == 'Germany':
                fouls_exp_h = round((-0.065 * sup * sup) - (0.458 * sup) + 11.041, 2)
                fouls_exp_a = round((-0.276 * sup * sup) + (0.631 * sup) + 11.129, 2) 

            elif selected_league == 'France':
                fouls_exp_h = round((-0.065 * sup * sup) - (0.458 * sup) + 11.041, 2)
                fouls_exp_a = round((-0.276 * sup * sup) + (0.631 * sup) + 11.129, 2)    

            else: # use 'All Leagues' formula
                fouls_exp_h = round((-0.281 * sup * sup) - (0.111 * sup) + 12.325, 2)
                fouls_exp_a = round((-0.372 * sup * sup) + (0.292 * sup) + 12.719, 2)


            if odds_df['Home Team'].values[0] == selected_team:
                team_xg = fouls_exp_h
            else:
                team_xg = fouls_exp_a


    # ========  Calculate Player Expected Share =============
        player_expected_share = team_xg * player_form_boost * (average_share /90 * expected_minutes)

        st.write(f"Expected {selected_metric} for {selected_team}: {team_xg:.2f}")
        st.write(f"Expected Share for {selected_player}: {player_expected_share:.2f}")

        # Write poisson formulas given player_expected_share lambda and over 0.5, over 1.5 and over 2.5 probabilities
        fudge_boost_05 = 1.01  # add an insurance % to modelled output
        fudge_boost_15 = 1.02
        fudge_boost_25 = 1.03

        prob_over_05 =poisson.sf(0, player_expected_share * fudge_boost_05 )
        prob_over_15 = poisson.sf(1, player_expected_share * fudge_boost_15 )
        prob_over_25 = poisson.sf(2, player_expected_share * fudge_boost_25 )

        true_odds_05 = round(1/prob_over_05, 2)
        true_odds_15 = round(1/prob_over_15, 2)
        true_odds_25 = round(1/prob_over_25, 2)

        st.write(f'{selected_player} true odds over 0.5 {selected_metric}:', true_odds_05)
        st.write(f'{selected_player} true odds over 1.5 {selected_metric}:', true_odds_15)
        st.write(f'{selected_player} true odds over 2.5 {selected_metric}:', true_odds_25)

   



   
    # check player injury status and if he played last match
    # check if player played last match






if __name__ == '__main__':
    main()