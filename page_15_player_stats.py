# https://docs.streamlit.io/
# https://docs.streamlit.io/develop/api-reference
# https://www.youtube.com/watch?v=D0D4Pa22iG0&t=41s

import streamlit as st
import pandas as pd
import altair as alt
# import plotly.express as px
import time
import gc
# import plotly.graph_objects as go


# Load the CSV file
@st.cache_data
def load_data(per_90=True):
    time.sleep(0.5)  # Simulate a delay for loading
    if per_90:
        df = pd.read_csv('data/outputs_processed/players/processed_f_api_combined_all_seasons_per90.csv')
    else:
        df = pd.read_csv('data/outputs_processed/players/processed_f_api_combined_all_seasons_standard.csv')
    return df

def main():
    # Toggle for showing data per 90 minutes
    per_90_toggle = st.toggle('Show data per 90 minutes', value=True)  # Default is True

    with st.spinner('Loading Data...'):
        # Load the data
        df = load_data(per_90=per_90_toggle)

    # Sidebar for user input
    st.sidebar.title('Select Data Filters')

    # Define selection options
    league_options = [
        'England Premier', 'Germany Bundesliga', 'Spain La Liga', 'Italy Serie A',
        'France Ligue 1', 'Netherlands Eredivisie', 'Belgium Jupiler', 'Portugal Liga I',
        'Scotland Premier', 'England Championship', 'England League One', 'England League Two'
    ]
    # To add 'ALL LEAGUES functionality
    league_options_with_all = league_options # + ['**All Leagues**']


    # Default league selection
    default_league = 'England Premier'  # Default to England Premier
    selected_league = st.sidebar.selectbox('Select League', options=league_options_with_all, index=league_options_with_all.index(default_league))

    # Year options
    year_options = ['2024-25', '2023-24']
    default_year = '2024-25'  # Default to the latest season
    selected_year = st.sidebar.selectbox('Select Year', options=year_options, index=year_options.index(default_year))

    # Filter the DataFrame by season
    df = df[df['Season'] == selected_year]

    # Handle multi-league selection
    if selected_league == '**All Leagues**':
        selected_leagues = st.sidebar.multiselect('Filter Leagues to Include', league_options, placeholder="Choose leagues", default=league_options[:4])  # Default to first 3 leagues
        num_players = st.sidebar.number_input('Select number of players to filter', min_value=5, max_value=50, value=20)  # Default is 10 players
        if not selected_leagues:
            st.warning("Please select at least one league.")
            return
        filtered_df = df[df['League'].isin(selected_leagues)]
        # Set default team value when selecting multiple leagues
        selected_team = 'Selected Leagues'
    else:
        # Filter for single-league selection
        filtered_df = df[(df['League'] == selected_league) & (df['Season'] == selected_year)]
        # Team selection for single-league filtering
        team_options = sorted(filtered_df['Team'].unique().tolist())
        default_team = team_options[0] if team_options else None  # Default to the first team
        selected_team = st.sidebar.selectbox('Select Team', options=team_options, index=0 if team_options else None)
        # Filter by team for single-league
        filtered_df = filtered_df[filtered_df['Team'] == selected_team]


    # Drop players with fewer than 180 minutes played
    filtered_df = filtered_df[filtered_df['Minutes'] > 180]
    # st.write(filtered_df)

    # Define metrics for heatmap
    advanced_stat_leagues = ['England Premier', 'Germany Bundesliga', 'Spain La Liga', 'Italy Serie A', '**All Leagues**']
    advanced_stat_seasons = ['2024-25']

    if selected_league in advanced_stat_leagues and selected_year in advanced_stat_seasons:
        heatmap_metrics = [
            'Goals', 'XG', 'Assists', 'Shots On', 'Shots Total', 
            'Fouls Drawn', 'Fouls Committed', 'Tackles Total', 'Blocks', 'Passes Total',
            'Passes Key', 'Dribbles Attempted', 'Dribbles Success', 'Minutes', 'Age',
            'Interceptions', 'Attacking Carries', 'Attacking Passes', 'Attacking Receives',
            'Non_Pen_XG', 'X_Assists', 'Yellow Cards', 'Red Cards', 'Duels Total',
            'Duels Won', 'Height', 'Weekly Wages $'
        ]
    else:
        heatmap_metrics = [
            'Goals', 'Assists', 'Shots On', 'Shots Total',
            'Fouls Drawn', 'Fouls Committed', 'Tackles Total', 'Blocks', 'Passes Total',
            'Passes Key', 'Dribbles Attempted', 'Dribbles Success', 'Minutes', 'Age',
            'Interceptions', 'Yellow Cards', 'Red Cards', 'Duels Total',
            'Duels Won', 'Height'
        ]

    # Default metric
    default_metric = 'Goals'  # Default to 'Goals'
    selected_heatmap_metric = st.sidebar.selectbox('Select Metric for Analysis', options=heatmap_metrics, index=heatmap_metrics.index(default_metric))

    # If multi-leagues chosen, filter the top X players and alter 'team name'
    if selected_league == '**All Leagues**':
        filtered_df = filtered_df.sort_values(by=selected_heatmap_metric, ascending=False).head(num_players)
        filtered_df['Player'] = filtered_df['Player'] + " (" + filtered_df['Team'] + ")"
        # Update the "Team" column to indicate multiple leagues
        filtered_df['Team'] = 'Selected Leagues'
    else:
        # Sort for single-league filtering
        filtered_df = filtered_df.sort_values(by=selected_heatmap_metric, ascending=False)

    # Display results
    if filtered_df.empty:
        st.write("No data currently available for the selected league and season.")
        st.stop()


    # # Display filtered data and charts
    # st.header(f'{selected_team} - Player {selected_heatmap_metric} {selected_year}')
    # if per_90_toggle:
    #     st.write('Player data based on per 90 mins. Minimum of 180 minutes played for inclusion.')
    # else:
    #     st.write('All player data. Minimum of 180 minutes played for inclusion.')

    # Cleanup
    del df
    gc.collect()



    # Display filtered data and charts
    st.header(f'{selected_team} - Player {selected_heatmap_metric} {selected_year}', divider='blue')
    if per_90_toggle:
        st.write('Player data based on per 90 mins. Minimum of 180 minutes played for inclusion')
    else:
        st.write('All player data. Minimum of 180 minutes played for inclusion')


    # Checkbox to show or hide the raw data # WIDGET
    if st.checkbox('Show raw player data', value=True, label_visibility = 'visible'):
        if per_90_toggle:
            st.write(f"Player Stats (per 90) for {selected_team} {selected_year}")
            st.dataframe(filtered_df)
            if selected_league not in advanced_stat_leagues:
                st.caption(f'Metrics currently unavailable for {selected_league} : XG, Non_Pen_XG, X_Assists, Attacking Carries, Attacking Passes, Attacking Receives')
                st.write("")
        else:
            st.write(f"Player Stats for {selected_team} {selected_year}")
            st.dataframe(filtered_df)
            if selected_league not in advanced_stat_leagues:
                st.caption(f'Metrics currently unavailable for {selected_league} : XG, Non_Pen_XG, X_Assists, Attacking Carries, Attacking Passes, Attacking Receives')
                st.write("")

    # handle cases when 'All leagues' selected and filtered chosen leagues do not possess selected_metric
    if filtered_df[selected_heatmap_metric].sum() == 0:
        st.write('Filtered leagues do not currently contain selected metric data')
        st.stop()
    # st.write(filtered_df[selected_heatmap_metric])




    # # ----------------------------------------------------------------------------------------
    # Function to plot the heatmap (chart 1)
    def plot_heatmap(df, metric):
        heatmap_chart = alt.Chart(df).mark_rect().encode(
            x=alt.X('Player:N', title='Player', sort=None),
            y=alt.Y(f'{metric}:Q', title=metric),
            color=alt.Color(f'{metric}:Q', scale=alt.Scale(scheme='redyellowgreen', domain=[df[metric].min(), df[metric].max()]), title='Value'),
            tooltip=['Player:N', f'{metric}:Q']
        ).properties(
            width=700,
            height=500,
            title=f'Player {metric}'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14,
            labelPadding=10,
            titlePadding=15,
        )
        return heatmap_chart
    

    # # Function to plot the bar chart (Chart 2)
    # def plot_bar_chart(df, metric):
    #     sorted_df = df.sort_values(by=metric, ascending=False)
    #     bar_color = '#1f77b4'  # Change as needed
    #     bar_chart = alt.Chart(sorted_df).mark_bar(color=bar_color).encode(
    #         x=alt.X(f'{metric}:Q', title=metric),
    #         y=alt.Y('Player:N', sort='-x', title='Player'),
    #         tooltip=['Player', metric]
    #     ).properties(
    #         width=800,
    #         height=600,
    #         title=""
    #     ).configure_axis(
    #         labelFontSize=12,
    #         titleFontSize=14,
    #         labelPadding=10,
    #         titlePadding=15,
    #     )
    #     return bar_chart
    
    
    # Display heatmap and bar chart
    #st.subheader(f"Bar Charts showing {selected_team} {selected_heatmap_metric}")
    if selected_heatmap_metric in heatmap_metrics:
        st.altair_chart(plot_heatmap(filtered_df, selected_heatmap_metric), use_container_width=False)
        st.write("----")
        # st.altair_chart(plot_bar_chart(filtered_df, selected_heatmap_metric))
    else:
        st.write("Select a valid metric for analysis.")

    # # -----------------------------------
    # st.write("---")

    # # Radar Chart for Player Comparison (Chart 3)
    # st.subheader('Radar Chart for Player Comparison')
    # radar_metrics = heatmap_metrics[:18]
    # # WIDGET
    # lc, rc = st.columns([4,5])
    # with lc:
    #     selected_radar_players = st.multiselect('Select single or multiple players for stats comparison', options=filtered_df['Player'].unique(), placeholder='Select player/s', label_visibility = 'visible')


    # def plot_radar_chart(df, players, metrics):
    #     # Compute min-max scaling and retain original values
    #     scaled_data = []
    #     for player in players:
    #         row = df[df['Player'] == player].iloc[0]
    #         scaled = []
    #         original = []
    #         for metric in metrics:
    #             val = row[metric]
    #             original.append(val)

    #             team_min = df[metric].min()
    #             team_max = df[metric].max()
    #             if team_max - team_min == 0:
    #                 scaled_val = 0.5
    #             else:
    #                 scaled_val = (val - team_min) / (team_max - team_min)
    #             scaled.append(scaled_val)

    #         scaled_data.append({
    #             'player': player,
    #             'scaled': scaled,
    #             'original': original
    #         })

    #     fig = go.Figure()

    #     for i, entry in enumerate(scaled_data):
    #         fig.add_trace(go.Scatterpolar(
    #             r=entry['scaled'],
    #             theta=metrics,
    #             fill='toself',
    #             name=entry['player'],
    #             customdata=[[o] for o in entry['original']],
    #             hovertemplate='<b>%{text}</b><br>%{theta}<br>' +
    #                         'Scaled: %{r:.2f}<br>Original: %{customdata[0]:.2f}<extra></extra>',
    #             text=[entry['player']] * len(metrics),
    #             line=dict(color=['#4682B4', '#CD5C5C', '#32CD32'][i % 3])
    #         ))

    #     fig.update_layout(
    #         title='Radar Chart Comparison (Min-Max Scaled)',
    #         polar=dict(
    #             radialaxis=dict(visible=True, range=[0, 1]),
    #         ),
    #         annotations=[
    #             dict(
    #                 text=(
    #                     "Each metric is scaled: 0 = min, 1 = max across selected players.<br>"
    #                     "Hover to see original metric values."
    #                 ),
    #                 showarrow=False,
    #                 xref="paper",
    #                 yref="paper",
    #                 x=0.5,
    #                 y=1.1,
    #                 xanchor="center",
    #                 yanchor="bottom",
    #                 font=dict(size=12, color="gray"),
    #             )
    #         ],
    #         width=650,
    #         height=650
    #     )

    #     return fig

    # if selected_radar_players:
    #     st.plotly_chart(plot_radar_chart(filtered_df, selected_radar_players, radar_metrics), use_container_width=False)
    # else:
    #     st.write("")

    # st.write("---")
    # # ---------------------------------------------------------------------------

    # # Comparison scatter plot (Chart 4)
    # st.subheader('Compare Two Metrics')
    # metrics = heatmap_metrics

    # # Default to the first and second metrics # WIDGET
    # # default_metric_1 = metrics[0]
    # default_metric_1 = selected_heatmap_metric
    # default_metric_2 = metrics[1]
    # lc2, rc2 = st.columns([4,5])
    # with lc2:
    #     comparison_metric_1 = st.selectbox('Select the first metric for comparison (x-axis)', options=metrics, index=metrics.index(default_metric_1), label_visibility = 'visible')
    #     comparison_metric_2 = st.selectbox('Select the second metric for comparison (y-axis)', options=metrics, index=metrics.index(default_metric_2), label_visibility = 'visible')

    # # -------------------Control scatter background theme ---------------------

    # # Dropdown or toggle for manual theme selection (Optional)
    # # WIDGET
    # theme = st.radio("Select background theme", options=['Dark', 'Light'], index=0, label_visibility = 'visible')

    # # Define color themes
    # themes = {
    #     "Dark": {
    #         "background_color": '#0E1117',
    #         "text_color": 'white',
    #         "point_color": '#AAAAAA',
    #         "gridline_color": '#444444',
    #         "axis_color": 'white'
    #     },
    #     "Light": {
    #         "background_color": 'white',
    #         "text_color": 'black',
    #         "point_color": '#333333',
    #         "gridline_color": '#CCCCCC',
    #         "axis_color": 'black'
    #     }
    # }

    # # Select theme colors
    # theme_colors = themes[theme]

    #     # ----------------------------------------------------------------------------

    # chart_title = (f'{comparison_metric_1} vs {comparison_metric_2} - {selected_team} {selected_year}')

    # if comparison_metric_1 and comparison_metric_2:
    #     if comparison_metric_1 == comparison_metric_2:
    #         st.warning("Please select different metrics for comparison.")
    #     else:
    #         comparison_df = filtered_df[[comparison_metric_1, comparison_metric_2, 'Player']].dropna()
    #         comparison_df = comparison_df[(comparison_df[comparison_metric_1] >= 0) & (comparison_df[comparison_metric_2] >= 0)]

    #         # Calculate min and max for each metric
    #         x_min, x_max = comparison_df[comparison_metric_1].min(), comparison_df[comparison_metric_1].max()
    #         y_min, y_max = comparison_df[comparison_metric_2].min(), comparison_df[comparison_metric_2].max()

    #         chart = alt.Chart(comparison_df).mark_point(filled=True).encode(
    #         x=alt.X(
    #             f'{comparison_metric_1}:Q',
    #             title=comparison_metric_1,
    #             scale=alt.Scale(domain=[x_min, x_max])  # Set custom domain for x-axis
    #         ),
    #         y=alt.Y(
    #             f'{comparison_metric_2}:Q',
    #             title=comparison_metric_2,
    #             scale=alt.Scale(domain=[y_min, y_max])  # Set custom domain for y-axis
    #         ),
    #             tooltip=[f'{comparison_metric_1}:Q', f'{comparison_metric_2}:Q', 'Player:N'],
    #             color=alt.value(theme_colors['point_color']) 
    #         ).properties(
    #             width=600,
    #             height=500,
    #             title=chart_title
    #         )

    #         text = alt.Chart(comparison_df).mark_text(align='left', dx=5, dy=-5, fontSize=10, color=theme_colors['text_color']).encode(
    #             x=alt.X(f'{comparison_metric_1}:Q'),
    #             y=alt.Y(f'{comparison_metric_2}:Q'),
    #             text=alt.Text('Player:N', format=''),
    #         )
    #                 # Combine the chart and text
    #         combined_chart = chart + text

    #         # Set background color for the combined chart
    #         combined_chart = combined_chart.properties(
    #         background = theme_colors['background_color']
    #         ).configure_axis(
    #             labelColor= theme_colors['axis_color'],
    #             titleColor= theme_colors['axis_color']
    #         )

    #         st.altair_chart(combined_chart)
    # else:
    #     st.write("Select two different metrics to compare.")

    # st.write("---")

    # show_definitions = st.checkbox('Metric definitions')
    # if show_definitions:
    #     st.caption('''
    # - XG: Expected Goals scored (including penalty kicks).
    # - Shots On: Shots on Target.
    # - Fouls Drawn: Fouls committed by an opposition player.
    # - Passes Key: Passes directly resulting in a shot.
    # - Blocks: Blocking the ball by standing in its path.
    # - Attacking Carries: Travelling with the ball into the attacking penalty area or ten yards beyond the furthest point in the previous six passes.
    # - Attacking Passes: Completed passes into the attacking penalty area or ten yards beyond the furthest point in the previous six passes.
    # - Attacking Receives: Attacking Passes received.
    # - Non_Pen_XG: Non-penalty Expected Goals scored.
    # - X_Assists: Expected assisted goals.
    # - Duels: Ground or aerial 1v1 player challenges.                         
    #                ''')
    # # --------------------------------------------------------


if __name__ == '__main__':
    main()