import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import time
from mymodule.functions import get_table # get_topscorer
import gc


# ------------- Load the CSV file -----------------
@st.cache_data
def load_data():
    time.sleep(0.5)
    df = pd.read_csv('data/outputs_processed/teams/api-football_master_teams.csv')
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    return df

# -------------------------------------------

def main():
    with st.spinner('Loading Data...'):
        df = load_data()

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
        'Serie A': 'Italy Serie A',
        'Ligue 1': 'France Ligue 1',
        'Eredivisie': 'Netherlands Eredivisie',
        'Jupiler Pro League': 'Belgium Jupiler',
        'Primeira Liga': 'Portugal Liga I',
        'Premiership': 'Scotland Premier',
        'Championship': 'England Championship',
        'League One': 'England League One',
        'League Two': 'England League Two',
    #    '2. Bundesliga': 'Germany 2 Bundesliga',
    #    'Serie B': 'Italy Serie B',
    #    'Segunda Division': 'Spain La Liga 2',
    #    'Ligue 2': 'France Ligue 2',
    #    'Super Lig': 'Turkey Super Lig',
    #    'Super League 1': 'Greece Super League'
    #    'Premier Soccer League': 'South Africa Premier'
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
    #    "Italy Serie B": '136',
    #    "Spain La Liga 2": '141',
    #    "France Ligue 2": '62',
    #    'Turkey Super Lig': '203',
    #    "Greece Super League": '197',
       # "South Africa Premier": "288" # available in data to offer. Commented out due to low level of league, would look odd
    }

    year_options = [
                    '2025-26',
                    '2024-25',
                    '2023-24',
    ]

    metric_options = {
        'Goals': ['HG', 'AG', 'TG'],
        'Corners': ['HC', 'AC', 'TC'],
        'Fouls': ['HF', 'AF', 'TF'],
        'Shots on Target': ['HST', 'AST', 'TST'],
        'Shots': ['HS', 'AS', 'TS'],
        'Shots In Box': ['HS_in_Box', 'AS_in_Box', 'TS_in_Box'],
        'Shots Out Box': ['HS_out_Box', 'AS_out_Box', 'TS_out_Box'],
        'Offsides': ['H_Off', 'A_Off', 'T_Off'],
        'Possession': ['H_Pos', 'A_Pos', 'T_Pos'],
        # 'XG': ['HxG', 'AxG', 'TxG'],
        'Passing': ['H_Pass', 'A_Pass', 'T_Pass'],
        'Yellow Cards': ['HY', 'AY', 'TY'],
        'Red Cards': ['HR', 'AR', 'TR']
    }

    # WIDGET
    # Capture user selections
    selected_league = st.sidebar.selectbox('Select League', options=list(league_options.values()), label_visibility = 'visible')
    selected_year = st.sidebar.selectbox('Select Year', options=year_options, label_visibility = 'visible')
    selected_metric = st.sidebar.selectbox('Select Metric', options=list(metric_options.keys()), label_visibility = 'visible')

    # Function to apply filters
    def apply_filters(df, league, year):
        # Filter by league
        if league != 'ALL':
            df = df[df['League'] == [key for key, value in league_options.items() if value == league][0]]
        
        # Filter by year
        if year != 'ALL':
            if year == '2025-26':
                start_date, end_date = datetime(2025, 8, 1), datetime(2026, 7, 1)
            elif year == '2024-25':
                start_date, end_date = datetime(2024, 8, 1), datetime(2025, 7, 1)
            elif year == '2023-24':
                start_date, end_date = datetime(2023, 8, 1), datetime(2024, 7, 1)
            # elif year == '2022-23':
            #     start_date, end_date = datetime(2022, 8, 1), datetime(2023, 7, 1)
            # elif year == '2021-22':
            #     start_date, end_date = datetime(2021, 8, 1), datetime(2022, 7, 1)
            # elif year == '2020-21':
            #     start_date, end_date = datetime(2020, 8, 1), datetime(2021, 7, 1)
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        return df

    # Apply filters
    filtered_df = apply_filters(df, selected_league, selected_year)
    # st.write(filtered_df)
    del df
    gc.collect()

    if filtered_df.empty:
        st.subheader(f"Data for {selected_league} {selected_year} currently unavailable.")
        return
    
    elif filtered_df[metric_options[selected_metric][2]].sum() == 0:
            st.subheader(f"Data for {selected_league} {selected_metric} - {selected_year} currently unavailable.")
            return

    st.header(f'{selected_league} {selected_year} {selected_metric}', divider='red')

    show_data = st.checkbox('Show filtered data', label_visibility = 'visible')
    if show_data:
        st.write(filtered_df)


# -------------------------------------------------------------------------------
    # Prepare data for the first chart
    home_metric = metric_options[selected_metric][0]
    away_metric = metric_options[selected_metric][1]
    total_metric = metric_options[selected_metric][2]

    home_f_df = filtered_df.groupby('HomeTeam')[home_metric].mean().reset_index()
    away_f_df = filtered_df.groupby('AwayTeam')[away_metric].mean().reset_index()

    home_ag_df = filtered_df.groupby('HomeTeam')[away_metric].mean().reset_index()
    away_ag_df = filtered_df.groupby('AwayTeam')[home_metric].mean().reset_index()

    # Round the values to 2 decimal places
    home_f_df[home_metric] = home_f_df[home_metric].round(2)
    away_f_df[away_metric] = away_f_df[away_metric].round(2)

    home_ag_df[away_metric] = home_ag_df[away_metric].round(2)
    away_ag_df[home_metric] = away_ag_df[home_metric].round(2)

    home_f_df.columns = ['Team', 'Home_Mean']
    away_f_df.columns = ['Team', 'Away_Mean']
    home_ag_df.columns = ['Team', 'Away_Mean']
    away_ag_df.columns = ['Team', 'Home_Mean']

    combined_f_df = pd.merge(home_f_df, away_f_df, on='Team', how='outer').fillna(0)
    combined_ag_df = pd.merge(home_ag_df, away_ag_df, on='Team', how='outer').fillna(0)  
    combined_f_df['Total_Mean'] = combined_f_df['Home_Mean'] + combined_f_df['Away_Mean']
    combined_ag_df['Total_Mean'] = combined_ag_df['Home_Mean'] + combined_ag_df['Away_Mean']

    # Calculate averages for display
    home_avg_for = round(home_f_df['Home_Mean'].mean(), 2)
    away_avg_for = round(away_f_df['Away_Mean'].mean(), 2)
    total_avg = home_avg_for + away_avg_for

    # Plotting the first chart using a melted chart with y-axis removed -- FOR
    melted_combined_f_df = pd.melt(combined_f_df, id_vars=['Team'], value_vars=['Home_Mean', 'Away_Mean'],
                                var_name='Type', value_name='Value')

    # After melting the dataframe
    melted_combined_f_df = pd.melt(combined_f_df, id_vars=['Team'], value_vars=['Home_Mean', 'Away_Mean'],
                                var_name='Type', value_name='Value')

    # Create the team order list sorted by total descending
    team_order_for = combined_f_df.sort_values('Total_Mean', ascending=False)['Team'].tolist()

    # Use this order in your Altair chart
    bars_for = alt.Chart(melted_combined_f_df).mark_bar().encode(
        x=alt.X('Team:N', sort=team_order_for, title=''),
        y=alt.Y('Value:Q', title=f'Average {selected_metric}', axis=None),
        color=alt.Color('Type:N', title='Type', scale=alt.Scale(domain=['Home_Mean', 'Away_Mean'], range=['#d62728', '#ff9896']))
    ).properties(
        width=alt.Step(40),
        height=500
    )


    # Melt the 'against' DataFrame
    melted_combined_ag_df = pd.melt(combined_ag_df, id_vars=['Team'], value_vars=['Home_Mean', 'Away_Mean'],
                                    var_name='Type', value_name='Value')

    # Sort teams by total descending for 'against'
    team_order_against = combined_ag_df.sort_values('Total_Mean', ascending=False)['Team'].tolist()

    # Define bars_against with sorted x-axis
    bars_against = alt.Chart(melted_combined_ag_df).mark_bar().encode(
        x=alt.X('Team:N', sort=team_order_against, title=''),
        y=alt.Y('Value:Q', title=f'Average {selected_metric}', axis=None),
        color=alt.Color('Type:N', title='Type', scale=alt.Scale(domain=['Home_Mean', 'Away_Mean'], range=['#d62728', '#ff9896']))
    ).properties(
        width=alt.Step(40),
        height=500
    )

    # ------ MAIN DISPLAY -------
    # ------ STANDINGS -----------

    # WIDGET
    show_table = st.checkbox('Show League Standings', label_visibility = 'visible')
    if show_table:
        league_table_df = get_table(leagues_dict[selected_league], selected_year[:4])  
        st.dataframe(league_table_df, height=750)

    # show_scorer_standings = st.checkbox('Show Topsorer Standings')
    # if show_scorer_standings:
    #     league_table_df = get_topscorer(leagues_dict[selected_league], selected_year[:4])  
    #     st.dataframe(league_table_df, height=750)


    # -------- METRICS CHARTs ------------------

    st.markdown(f'''
                <div style="font-size:18px; font-weight:bold;">
                    Average Total {selected_metric}: <span style="color:red;">{total_avg:.2f}</span> 
                    (Home: {home_avg_for:.2f} / Away: {away_avg_for:.2f})
                </div>
                ''', unsafe_allow_html=True)
    st.write('---')
    
    # if st.checkbox(f'Show {selected_metric} team data'):
    #     st.write(combined_f_df)
    #     st.write(filtered_df)
    #st.write("----")


    # ------ DISPLAY CHART 1 --------

    # capture which chart to show, chosen metric FOR or AGAINST   # WIDGET
    left_column, _ = st.columns([3, 6])
    with left_column:
        selection_for_or_against = st.selectbox(f'Select Home & Away {selected_metric} Chart - For / Against', options=['For', 'Against'], label_visibility = 'hidden' )

    if selection_for_or_against == 'For':
        chosen_chart = bars_for
    else:
        chosen_chart = bars_against

    # Display filtered data and charts
    st.subheader(f'Average Home and Away {selected_metric} - {selection_for_or_against}')
    st.altair_chart(chosen_chart.configure_view(clip=False), use_container_width=False)
    st.write("----")

    # -------------------------

    # Prepare data for the second chart

    home_for_df = filtered_df.groupby('HomeTeam')[home_metric].mean().reset_index()
    away_for_df = filtered_df.groupby('AwayTeam')[away_metric].mean().reset_index()
    home_against_df = filtered_df.groupby('HomeTeam')[away_metric].mean().reset_index()
    away_against_df = filtered_df.groupby('AwayTeam')[home_metric].mean().reset_index()

    home_for_df.columns = ['Team', 'Home_For']
    away_for_df.columns = ['Team', 'Away_For']
    home_against_df.columns = ['Team', 'Home_Against']
    away_against_df.columns = ['Team', 'Away_Against']

    home_for_avg = pd.merge(home_for_df, away_for_df, on='Team', how='outer').fillna(0)
    against_avg = pd.merge(home_against_df, away_against_df, on='Team', how='outer').fillna(0)

    home_for_avg['For'] = home_for_avg['Home_For'] + home_for_avg['Away_For']
    against_avg['Against'] = against_avg['Home_Against'] + against_avg['Away_Against']

    combined_for_against_avg = pd.merge(home_for_avg[['Team', 'For']], against_avg[['Team', 'Against']], on='Team', how='outer').fillna(0)

    # Round the For and Against values to 2 decimal places
    combined_for_against_avg['For'] = combined_for_against_avg['For'].round(2)
    combined_for_against_avg['Against'] = combined_for_against_avg['Against'].round(2)

    # Calculate the correct For and Against averages
    combined_for_against_avg['For'] = combined_for_against_avg['For'] / 2
    combined_for_against_avg['Against'] = combined_for_against_avg['Against'] / 2

    # Calculate total (For + Against) for ordering
    combined_for_against_avg['Total'] = combined_for_against_avg['For'] + combined_for_against_avg['Against']


    st.write("")
    st.subheader(f'Total Average Match {selected_metric}')

    # Dropdown for ordering
    c1, _ = st.columns([3,6])
    with c1:
        order_by = st.selectbox("Order plot by:", options=['Total', 'For', 'Against'], index=0)


    # Create order list based on dropdown
    team_order_breakdown = combined_for_against_avg.sort_values(order_by, ascending=False)['Team'].tolist()

    # Melt the dataframe
    melted_breakdown_df = pd.melt(combined_for_against_avg, id_vars=['Team'], value_vars=['For', 'Against'],
                                var_name='Type', value_name='Value')

    # Create chart with dynamic sorting
    breakdown_bars = alt.Chart(melted_breakdown_df).mark_bar().encode(
        x=alt.X('Team:N', sort=team_order_breakdown, title=''),
        y=alt.Y('Value:Q', title=f'Average {selected_metric}'),
        color=alt.Color('Type:N', title='Type', scale=alt.Scale(domain=['For', 'Against'], range=['#5e2a8a', '#cba0d1']))
    ).properties(
        width=alt.Step(40),
        height=500
    )

    # Display chart
    st.write("")
    st.altair_chart(breakdown_bars.configure_view(clip=False), use_container_width=False)
    st.write("----")


    # ----- PREPARE CHART 3 ----------------------
     # WIDGET
    if total_metric == 'T_Pos':
        with left_column:
            poss_choice = st.selectbox('Select Home or Away Possession', options=['Home', 'Away'], label_visibility = 'visible')
            if poss_choice == 'Home':
                total_metric = home_metric
            else:
                total_metric = away_metric
        bin_step = 0.02

    elif total_metric == 'T_Pass':
        bin_step = 25

    elif total_metric == 'TxG':
        bin_step = 0.5

    else:
        bin_step = 1
            
    histogram = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X(total_metric, 
                bin=alt.Bin(step=bin_step, extent=[filtered_df[total_metric].min(), filtered_df[total_metric].max()]),
                title=f'Total {selected_metric}'),
        y=alt.Y('count()', title='Frequency'),
        tooltip=[total_metric, 'count()']
    ).properties(
        width=500,
        height=500
    )

    line = alt.Chart(filtered_df).transform_density(
        total_metric,
        as_=[total_metric, 'density']
    ).mark_line().encode(
        x=alt.X(total_metric, title=f'Total {selected_metric}'),
        y='density:Q'
    )

    # -------------- DISPLAY CHART 3 -----------------  

    # handle possession selected metric
    if total_metric == 'H_Pos':
        st.subheader(f'Frequency of Home {selected_metric}, All Matches')
    elif total_metric == 'A_Pos':
        st.subheader(f'Frequency of Away {selected_metric}, All Matches')
    elif total_metric != 'TR':
        st.subheader(f'Frequency of {selected_metric}, All Matches')

    if total_metric != 'TR':
        st.altair_chart(histogram + line, use_container_width=False)
        st.write("----")

# # -------------------------------------------------------------

   
#     # ---- Prepare data for the chart 4: Running average of total team goals for each team **** PART1 *****

#     lc, _ = st.columns([4,5])


#     with lc:                                                  
#         st.subheader(f'Moving Average {selected_metric}')    
#         avg_type = st.selectbox(
#             'Select Moving Average Type',
#             options=['Season-to-date', '5-game rolling', '10-game rolling']
#         )


#     # Create DataFrames for home and away goals
#     home_df = filtered_df[['Date', 'HomeTeam', home_metric]].rename(columns={'HomeTeam': 'Team', home_metric: 'Total'})
#     away_df = filtered_df[['Date', 'AwayTeam', away_metric]].rename(columns={'AwayTeam': 'Team', away_metric: 'Total'})

#     # Concatenate the DataFrames
#     combined_goals_df = pd.concat([home_df, away_df])

#     # Sort by team and date
#     combined_goals_df = combined_goals_df.sort_values(by=['Team', 'Date'])

#     # Calculate running average
#     if avg_type == 'Season-to-date':
#         combined_goals_df['running_avg'] = (
#             combined_goals_df.groupby('Team')['Total']
#             .expanding()
#             .mean()
#             .reset_index(level=0, drop=True)
#         )
#     elif avg_type == '5-game rolling':
#         combined_goals_df['running_avg'] = (
#             combined_goals_df.groupby('Team')['Total']
#             .rolling(window=5, min_periods=1)
#             .mean()
#             .reset_index(level=0, drop=True)
#         )
#     elif avg_type == '10-game rolling':
#         combined_goals_df['running_avg'] = (
#             combined_goals_df.groupby('Team')['Total']
#             .rolling(window=10, min_periods=1)
#             .mean()
#             .reset_index(level=0, drop=True)
#         )

#     # Calculate the match number for each team
#     combined_goals_df['match_num'] = combined_goals_df.groupby('Team').cumcount() + 1


#     # --------- DISPLAY SELECTION BOX CHART 4 ---------------

#     # Dropdown to select team for running average


#     with lc:
#         selected_team = st.selectbox('Select Team', options=combined_goals_df['Team'].unique())



#     # Filter data for the selected team
#     team_running_avg_df = combined_goals_df[combined_goals_df['Team'] == selected_team]

#     if selected_metric in ['Corners', 'Shots on Target', 'Shots In Box', 'Shots Out Box']:
#         yaxlow, yaxhigh = 2, 12

#     elif selected_metric in ['Fouls', 'Shots']:
#         yaxlow, yaxhigh = 6, 20

#     elif selected_metric in ['Red Cards']:
#         yaxlow, yaxhigh = 0, 0.8

#     elif selected_metric in ['Possession']:
#         yaxlow, yaxhigh = 0.2, 0.8

#     elif selected_metric in ['Passing']:
#          yaxlow, yaxhigh = 200, 700  

#     else:
#         yaxlow, yaxhigh = 0, 5.0

#     # Altair chart configuration for the running average
#     line_chart = alt.Chart(team_running_avg_df).mark_line().encode(
#         x=alt.X('match_num:Q', title='Match Number', axis=alt.Axis(format='d')),
#         y=alt.Y('running_avg:Q', title='Running Avg For ', scale=alt.Scale(domain=[yaxlow, yaxhigh])),
#         color=alt.ColorValue("#4ACE3E"),
#         tooltip=['match_num', 'running_avg']
#     ).properties(
#         width=600,
#         height=500
#     )
# # ***************************

    # # Create a second DataFrame for 'against' goals running average  *** PART 2 ***
    # home_against_df = filtered_df[['Date', 'HomeTeam', away_metric]].rename(columns={'HomeTeam': 'Team', away_metric: 'Total'})
    # away_against_df = filtered_df[['Date', 'AwayTeam', home_metric]].rename(columns={'AwayTeam': 'Team', home_metric: 'Total'})

    # # Concatenate the DataFrames for 'against' goals
    # combined_against_df = pd.concat([home_against_df, away_against_df])

    # # Sort by team and date
    # combined_against_df = combined_against_df.sort_values(by=['Team', 'Date'])

    # # Calculate running average for 'against' goals
    # if avg_type == 'Season-to-date':
    #     combined_against_df['running_avg_a'] = (
    #         combined_against_df.groupby('Team')['Total']
    #         .expanding()
    #         .mean()
    #         .reset_index(level=0, drop=True)
    #     )
    # elif avg_type == '5-game rolling':
    #     combined_against_df['running_avg_a'] = (
    #         combined_against_df.groupby('Team')['Total']
    #         .rolling(window=5, min_periods=1)
    #         .mean()
    #         .reset_index(level=0, drop=True)
    #     )
    # elif avg_type == '10-game rolling':
    #     combined_against_df['running_avg_a'] = (
    #         combined_against_df.groupby('Team')['Total']
    #         .rolling(window=10, min_periods=1)
    #         .mean()
    #         .reset_index(level=0, drop=True)
    #     )

    # # Calculate the match number for each team
    # combined_against_df['match_num'] = combined_against_df.groupby('Team').cumcount() + 1

    # # Merge the two DataFrames on 'Team' and 'match_num'
    # merged_df = pd.merge(combined_goals_df, combined_against_df, on=['Team', 'match_num', 'Date'], suffixes=('', '_a'))

    # # Filter data for the selected team
    # team_running_avg_df = merged_df[merged_df['Team'] == selected_team]


    # # Altair chart configuration for the 'against' running average

    # df_melted = team_running_avg_df.melt(
    #     id_vars=['match_num', 'Date'],
    #     value_vars=['running_avg', 'running_avg_a'],
    #     var_name='Type',
    #     value_name='Value'
    # )

    # # Dynamically label the lines based on selected_metric
    # df_melted['Type'] = df_melted['Type'].map({
    #     'running_avg': f'{selected_metric} For',
    #     'running_avg_a': f'{selected_metric} Against'
    # })



    # # Build the chart
    # combined_chart = alt.Chart(df_melted).mark_line().encode(
    #     x=alt.X('match_num:Q', title='Match Number'),
    #     y=alt.Y('Value:Q',
    #             title=f'Rolling Average {selected_metric}',
    #             scale=alt.Scale(domain=[yaxlow, yaxhigh])),
    #     color=alt.Color('Type:N',
    #                     title='',
    #                     scale=alt.Scale(
    #                         domain=[f'{selected_metric} For', f'{selected_metric} Against'],
    #                         range=['green', 'red'])),
    #     tooltip=['Date', 'Type', 'Value']
    # ).properties(
    #     width=600,
    #     height=500,
    #     title=f'{selected_team}: {avg_type} average - {selected_metric}'
    #)

    # # ----------- DISPLAY CHART 4 & RAW DATA OPTION CHECKBOX ---------------
    # st.write("")

  
    # # Display the combined chart
    # st.altair_chart(combined_chart, use_container_width=False)   

    # if st.checkbox('Show Running Average Raw Data'):
    #     st.write(team_running_avg_df)

        
    # st.write("----")


    # # --------- Comaprison Scatter ---------------------------------
    # # st.write(combined_for_against_avg)


    # st.subheader(f'Comparison of Average {selected_metric} For and Against')

    # # Assuming comparison_df has columns 'For', 'Against', and 'Team'
    # comparison_df = combined_for_against_avg[['For', 'Against', 'Team']].dropna()
    # comparison_df.rename(columns={'For': 'Avg For', 'Against': 'Avg Against'}, inplace=True)

    # # Get the minimum and maximum values for 'For' and 'Against' to adjust the axis range dynamically
    # for_min = comparison_df['Avg For'].min()
    # for_max = comparison_df['Avg For'].max()
    # against_min = comparison_df['Avg Against'].min()
    # against_max = comparison_df['Avg Against'].max()



    #     # -------------------Control scatter background theme ---------------------

    # # WIDGET
    # # Dropdown or toggle for manual theme selection (Optional)
    # theme = st.radio("Select Background Theme", options=['Dark', 'Light'], index=0, label_visibility = 'visible')

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


    #     # -------------------------------------------------

        
    #     # # Colors for dark mode
    #     # background_color = '#0E1117'  # Streamlit dark theme background color
    #     # text_color = 'white'  # Light gray for text on dark background
    #     # point_color = '#AAAAAA'  # Slightly lighter gray for points on dark background
    #     # gridline_color = '#444444'  # Subtle gridline color for dark background
    #     # axis_color = 'white'  # Axis label color for dark background


    # # Create scatter plot with dynamically adjusted axis ranges
    # chart = alt.Chart(comparison_df).mark_point(filled=True).encode(
    #     x=alt.X('Avg For:Q', title='Avg For', scale=alt.Scale(domain=[for_min, for_max])),  # Dynamic x-axis range
    #     y=alt.Y('Avg Against:Q', title='Avg Against', scale=alt.Scale(domain=[against_min, against_max])),  # Dynamic y-axis range
    #     tooltip=['Avg For:Q', 'Avg Against:Q', 'Team:N'],
    #     color=alt.value(theme_colors['point_color'])  # Use point color that blends well with the background
    # ).properties(
    #     width=600,
    #     height=500,
    # )

    # # Adding labels (Team names) with softer text color for blending
    # text = alt.Chart(comparison_df).mark_text(
    #     align='left', 
    #     dx=5, dy=-5,  # Adjust positioning
    #     fontSize=10, 
    #     color=theme_colors['text_color']
    # ).encode(
    #     x=alt.X('Avg For:Q'),
    #     y=alt.Y('Avg Against:Q'),
    #     text=alt.Text('Team:N')
    # )

    # # Combine the chart and text (labels)
    # combined_chart = chart + text

    # # Set background color and axis label style that blend with the dark background
    # combined_chart = combined_chart.properties(
    #     background=theme_colors['background_color']  # Seamless blending background
    # ).configure_axis(
    #     labelColor=theme_colors['axis_color'],
    #     titleColor=theme_colors['axis_color'],
    #     gridColor=theme_colors['gridline_color']
    # )

    # # Render the chart in Streamlit
    # st.altair_chart(combined_chart, use_container_width=False)   


    # # with right_column:
    # display_comparison_df = comparison_df[['Team', 'Avg For', 'Avg Against']]
    # display_comparison_df = round(display_comparison_df[['Team','Avg For','Avg Against']], 2)
    # display_comparison_df['Diff'] = round(display_comparison_df['Avg For'] - display_comparison_df['Avg Against'], 2)
    # display_comparison_df = display_comparison_df.sort_values(['Diff'], ascending = False).reset_index(drop=True)
    # display_comparison_df.index = display_comparison_df.index +1

    # st.write('---')
    # st.subheader(f'{selected_metric} Table')
    # st.dataframe(display_comparison_df, height=737)



    # --------------------  Show comparison all league for selected metric  -------------
    # st.write("---")
    # st.subheader(f"Comparison of {selected_metric} across all available leagues for {selected_year}")

    # show_all_leagues = st.checkbox(f'Show average {selected_metric} - all leagues')
    # if show_all_leagues:
    #     # filter data
    #     df_2 = pd.read_csv('data/outputs_processed/teams/api-football_master_teams.csv')

    #     # remove South Africa rows
    #     df_2 = df_2[df_2['League'] != 'Premier Soccer League']
        
    #     # Ensure 'Date' column is in datetime format
    #     df_2['Date'] = pd.to_datetime(df_2['Date'], errors='coerce')  # Convert to datetime, invalid values become NaT
    #     # st.write(df_2)
    #     # Function to apply filters
    #     def apply_year_filter(df, year):
    #             # Filter by year           
    #             if year == '2025-26':
    #                 df = df[df['Season'] == '2025-26']
    #             if year == '2024-25':
    #                 df = df[df['Season'] == '2024-25']
    #             elif year == '2023-24':
    #                 df = df[df['Season'] == '2023-24']
            
    #             return df

    #     # Apply year filter
    #     filtered_df_2 = apply_year_filter(df_2, selected_year)
    #     # st.write(filtered_df_2)

    #     # modify back possession
    #     if total_metric == 'H_Pos':
    #         total_metric = 'T_Pos'

    #     # Create a new DataFrame grouped by 'League' and calculate the mean of the selected metric
    #     league_avg_df = filtered_df_2.groupby('League', as_index=False)[total_metric].mean()

    #     # Now, sort by the newly renamed column, ensuring it exists
    #     league_avg_df = league_avg_df.sort_values(by=league_avg_df.columns[1], ascending=False)

    #     # Convert league names based on the league_options dictionary
    #     league_avg_df['League'] = league_avg_df['League'].map(league_options)

    #     # Reset the index of the DataFrame
    #     league_avg_df = league_avg_df.reset_index(drop=True)
    #     mean_metric = round(league_avg_df[league_avg_df.columns[1]].mean(), 2)

    #     # Display the new DataFrame
    #     st.write("")
    #     st.dataframe(league_avg_df)
    #     st.write(f'Average {selected_metric}:', mean_metric)
    #     st.write("")

    #     # Create an Altair bar chart with the dynamic selected metric
    #     chart = (
    #         alt.Chart(league_avg_df)
    #         .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
    #         .encode(
    #             y=alt.Y('League:N', sort='-x', title='League'),  # Switch x and y
    #             x=alt.X(f'{total_metric}:Q', title=f'{total_metric}'),
    #             color=alt.Color(f'{total_metric}:Q', scale=alt.Scale(scheme='blues')),
    #             tooltip=['League', total_metric]
    #         )
    #         .properties(width=700, height=alt.Step(30))  # Adjust width and step size
    #     )


    #     # Display the Altair chart in Streamlit
    #     st.altair_chart(chart, use_container_width=False)


    
# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()