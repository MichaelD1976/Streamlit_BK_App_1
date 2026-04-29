
import streamlit as st
import pandas as pd
import time
import altair as alt


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
        '**All Leagues**', 'England Premier', 'Germany Bundesliga', 'Spain La Liga', 'Italy Serie A',
        'France Ligue 1', 'Netherlands Eredivisie', 'Belgium Jupiler', 'Portugal Liga I',
        'Scotland Premier', 'England Championship'
    ]



    # Default league selection
    default_league = '**All Leagues**'  # Default to England Premier
    selected_league = st.sidebar.selectbox('Select League', options=league_options, index=league_options.index(default_league))

    # Year options
    year_options = ['2025-26', '2024-25']
    default_year = '2025-26'  # Default to the latest season
    selected_year = st.sidebar.selectbox('Select Year', options=year_options, index=year_options.index(default_year))

    # Filter the DataFrame by season
    df = df[df['Season'] == selected_year]

    # Handle multi-league selection
    if selected_league == '**All Leagues**':
        selected_leagues = st.sidebar.multiselect('Filter Leagues to Include', league_options, placeholder="Choose leagues", default=league_options[1:])  # Default to first 3 leagues
        # num_players = st.sidebar.number_input('Select number of players to filter', min_value=5, max_value=50, value=20)  # Default is 10 players
        if not selected_leagues:
            st.warning("Please select at least one league.")
            return
        filtered_df = df[df['League'].isin(selected_leagues)]
        # Set default team value when selecting multiple leagues
        selected_team = 'Selected Leagues'
    else:
        # Filter for single-league selection
        filtered_df = df[(df['League'] == selected_league) & (df['Season'] == selected_year)]



    # Drop players with fewer than 180 minutes played
    filtered_df = filtered_df[filtered_df['Minutes'] > 180]
    # st.write(filtered_df)


    df_nig = filtered_df[filtered_df['Nation'] == 'Nigeria'].copy()

    st.subheader("Nigerian Player Stats", divider='blue')

    st.write(df_nig)

    # ------------------  Written Summary  -------------------
    st.subheader("Summary of Nigerian Players")
    if df_nig.empty:
        st.write("No Nigerian players found in the selected league and season.")
    else:
        highest_rating = df_nig['Rating'].max().round(2)
        highest_rating_player = df_nig[df_nig['Rating'] == highest_rating]['Player'].values[0]
        highest_goals = df_nig['Goals'].max()
        highest_goals_player = df_nig[df_nig['Goals'] == highest_goals]['Player'].values[0]
        highest_assists = df_nig['Assists'].max()
        highest_assists_player = df_nig[df_nig['Assists'] == highest_assists]['Player'].values[0]  
        highest_key_passes = df_nig['Passes Key'].max()
        highest_key_passes_player = df_nig[df_nig['Passes Key'] == highest_key_passes]['Player'].values[0]
        highest_dribbles_success = df_nig['Dribbles Success'].max()
        highest_dribbles_success_player = df_nig[df_nig['Dribbles Success'] == highest_dribbles_success]['Player'].values[0]

        st.write(f"""
                 
                 In the selected leagues for {selected_year} season, **{highest_goals_player}** has the highest goals with {highest_goals}, 
                 while **{highest_assists_player}** has the most assists with {highest_assists}.
                    **{highest_key_passes_player}** has the most key passes with {highest_key_passes}, 
                    and **{highest_dribbles_success_player}** has the highest dribble success rate with {highest_dribbles_success}.
                    **{highest_rating_player}** has the highest team rating of {highest_rating}.

                 """            
                 )



    # -------------------  Bar Chart  -------------------
    st.write("")
    heatmap_metrics = [
            'Goals', 'Assists', 'Shots On', 'Shots Total',
            'Fouls Drawn', 'Fouls Committed', 'Tackles Total', 'Blocks', 'Passes Total',
            'Passes Key', 'Dribbles Attempted', 'Dribbles Success', 'Minutes', 'Age',
            'Interceptions', 'Yellow Cards', 'Red Cards', 'Duels Total', 'Rating',
            'Duels Won', 'Height'
        ]

    # Default metric
    default_metric = 'Goals'  # Default to 'Goals'
    selected_heatmap_metric = st.selectbox('Select Bar Chart Metric', options=heatmap_metrics, index=heatmap_metrics.index(default_metric))

    filtered_df_2 = df_nig.sort_values(by=selected_heatmap_metric, ascending=False)

        # Display results
    if filtered_df_2.empty:
        st.write("No data currently available for the selected league and season.")
        st.stop()


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
    

    # Display heatmap and bar chart
    #st.subheader(f"Bar Charts showing {selected_team} {selected_heatmap_metric}")
    if selected_heatmap_metric in heatmap_metrics:
        st.altair_chart(plot_heatmap(filtered_df_2, selected_heatmap_metric), use_container_width=False)
        st.write("----")
        # st.altair_chart(plot_bar_chart(filtered_df, selected_heatmap_metric))
    else:
        st.write("Select a valid metric for analysis.")


if __name__ == "__main__":
    main()