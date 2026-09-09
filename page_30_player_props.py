import streamlit as st
import pandas as pd
import altair as alt
# import plotly.express as px
import time
import gc
import plotly.graph_objects as go


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
        'France Ligue 1'
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

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=comparison_df['Average'],
                y=comparison_df['Category'],
                orientation='h',
                text=comparison_df['Average'].round(2),
                textposition='auto',
                marker_color=[
                    '#1f77b4',   # Player
                    '#ff7f0e',   # Team
                    '#2ca02c'    # Position
                ],
                showlegend=False
            )
        )

        fig.update_layout(
            title=f'{selected_metric} per 90: Player vs Team vs Position',
            xaxis_title=f'{selected_metric} per 90',
            yaxis_title='',
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig,
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
        width='stretch',
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
            width='stretch',
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
    # =========================================================