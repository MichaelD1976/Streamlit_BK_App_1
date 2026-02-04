import streamlit as st
import pandas as pd
import time
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
import altair as alt
from mymodule.functions import generate_team_fst, get_injuries_by_team, get_transfers_by_team

CURRENT_SEASON = '2025' # for api injuries call
CURRENT_SEASON_FST_FORMAT = '2025-26' # for generate_fst downgrades. If not current season, use spring adjustments

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
    # st.toggle returns True if toggled on, False otherwise
    per_90_toggle = st.toggle('Show data per 90 minutes', value=True)  # Default is True
    
    with st.spinner('Loading Data...'):
        # Call the load_data function with the toggle value
        df = load_data(per_90=per_90_toggle)

    # Sidebar for user input
    st.sidebar.title('Select Data Filters')

    year_options = ['2025-26', '2024-25', '2023-24']

    # Define selection options
    league_options = ['England Premier',
                      'Germany Bundesliga',
                    'Spain La Liga',
                    'Italy Serie A',
                    'France Ligue 1',
                    'Netherlands Eredivisie',
                    'Belgium Jupiler',
                    'Portugal Liga I',
                    'Scotland Premier',
                    'England Championship',
                    'England League One',
                    'England League Two'
                    ]
    
    # WIDGET
    selected_league = st.sidebar.selectbox('Select League', options=league_options, label_visibility = 'visible')

    # Filter the DataFrame based on the selected league
    df_filtered_by_league = df[df['League'] == selected_league]

    # WIDGET
    selected_year = st.sidebar.selectbox('Select Year', options=year_options, label_visibility = 'visible')

     # Filter the DataFrame based on the selected league and season
    df_filtered_by_league_and_year = df_filtered_by_league[df_filtered_by_league['Season'] == selected_year]

    team_options = sorted(df_filtered_by_league_and_year['Team'].unique().tolist())
    # st.write(df_filtered_by_league_and_year)

    # WIDGET
    selected_team = st.sidebar.selectbox('Select Team', options=team_options, label_visibility = 'visible')


    # Function to apply filters
    def apply_filters(df, league, year, team):
        if league:
            df = df[df['League'] == league]
        if year:
            df = df[df['Season'] == year]
        if team:
            df = df[df['Team'] == team]
        return df

    # Apply filters
    filtered_df = apply_filters(df, selected_league, selected_year, selected_team).copy()
    # Convert 'Birth_date' column to the desired string format
    filtered_df['Birth_date'] = pd.to_datetime(filtered_df['Birth_date'], errors='coerce')

    # Handle NaT values (invalid dates) by replacing them with an empty string or a default value
    filtered_df['Birth_date'] = filtered_df['Birth_date'].dt.strftime('%Y-%m-%d').fillna('')

    if filtered_df.empty:
        st.write(f"Data for {selected_league} {selected_year} currently unavailable.")
        return

    # Drop duplicate player names if any
    filtered_df = filtered_df[~filtered_df.duplicated(subset=['Player'], keep='first')]
    filtered_df.drop(['Season','Team','League'], axis=1, inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.index = filtered_df.index + 1
    filtered_df['Rating'] = round(filtered_df['Rating'], 1)

    # Remove columns for non advanced league eg xg,attacking carries etc
    advanced_stat_leagues = ['England Premier', 'Germany Bundesliga', 'Spain La Liga', 'Italy Serie A']

    if selected_league not in advanced_stat_leagues:
        filtered_df.drop([ 'npxG+xAG', 'XG', 'Non_Pen_XG', 'X_Assists', 'Attacking Carries', 'Attacking Receives', 'Attacking Passes'], axis=1, inplace=True)
        #st.caption(f'Metrics currently unavailable for {selected_league} : XG, Non_Pen_XG, X_Assists, Attacking Carries, Attacking Passes, Attacking Receives')
    # 'npxG+AG', 
    st.write("---")


    # ------------DF CONFIG ------------------------------

    st.header(f'{selected_team} Squad {selected_year}', divider='red')
    st.write('Players with > 90 total minutes shown')
    st.write('')

    # make a copy to remove from displayed squad table
    filtered_df_squad = filtered_df.drop(['Player_id'], axis = 1).copy()

    # Configure Ag-Grid options
    gb = GridOptionsBuilder.from_dataframe(filtered_df_squad)

    # Freeze the 'Player' column
    gb.configure_column('Player', pinned='left')

    # Light green background color for the entire DataFrame
    light_green = '#e6ffe6'  # Light green color

    # Cell style for all cells
    cell_style = {
        'backgroundColor': light_green,  # Light green color for all cells
        'color': 'black'                  # Default text color
    }

    # Dynamically set column widths based on header text lengths
    for column in filtered_df_squad.columns:
        # Calculate width based on header text length
        header_length = len(column) * 10  # Adjust multiplier for better fitting
        gb.configure_column(column, width=max(100, header_length))  # Minimum width of 100

    # Apply the same cell style to all columns
    gb.configure_default_column(cellStyle=cell_style)
    gb.configure_column('Player', cellStyle={'color': 'black', 'backgroundColor': '#FFB6C1'})

    # Build grid options
    grid_options = gb.build()

    # Display the DataFrame with styles applied in Streamlit
    filtered_df_squad = filtered_df_squad.replace([np.inf, -np.inf], 0)
    filtered_df_squad = filtered_df_squad.fillna("")
    AgGrid(filtered_df_squad, gridOptions=grid_options, enable_enterprise_modules=False, height=600)
    st.write("---")

    # ---------- Generate fst via function ------------------------------ 

    # st.write(filtered_df)
    # function returns fst & squad player ratings. Not currently utilising squad ratings so returned as '_'
    df_r, _ = generate_team_fst(filtered_df, selected_year)


    # # -------- ADD SCALED SHOTS AS ATTACK RATING ------------

    # Step 1: Find the min and max values in the 'Shots Total' column
    if not per_90_toggle:
        min_shots = (filtered_df['Shots Total'] / filtered_df['Minutes']).min()
        max_shots = (filtered_df['Shots Total'] / filtered_df['Minutes']).max()

        min_goals = (filtered_df['Goals'] / filtered_df['Minutes']).min()
        max_goals = (filtered_df['Goals'] / filtered_df['Minutes']).max()

        min_passes_key = (filtered_df['Passes Key'] / filtered_df['Minutes']).min()
        max_passes_key = (filtered_df['Passes Key'] / filtered_df['Minutes']).max()

        min_dribbles_suc = (filtered_df['Dribbles Success'] / filtered_df['Minutes']).min()
        max_dribbles_suc = (filtered_df['Dribbles Success'] / filtered_df['Minutes']).max()

    else:
        min_shots = filtered_df['Shots Total'].min()
        max_shots = filtered_df['Shots Total'].max()

        min_goals = filtered_df['Goals'].min()
        max_goals = filtered_df['Goals'].max()

        min_passes_key = filtered_df['Passes Key'].min()
        max_passes_key = filtered_df['Passes Key'].max()

        min_dribbles_suc = filtered_df['Dribbles Success'].min()
        max_dribbles_suc = filtered_df['Dribbles Success'].max()

    if not per_90_toggle:
        # Step 2: Apply Min-Max Scaling
        filtered_df['Shots_Rating'] = (((filtered_df['Shots Total'] / filtered_df['Minutes']) - min_shots) / (max_shots - min_shots)) * 10
        filtered_df['Goals_Rating'] = (((filtered_df['Goals'] / filtered_df['Minutes']) - min_goals) / (max_goals - min_goals)) * 10
        filtered_df['Passes_Key_Rating'] = (((filtered_df['Passes Key'] / filtered_df['Minutes'])- min_passes_key) / (max_passes_key - min_passes_key)) * 10
        filtered_df['Dribbles_Success_Rating'] = (((filtered_df['Dribbles Success'] / filtered_df['Minutes']) - min_dribbles_suc) / (max_dribbles_suc - min_dribbles_suc)) * 10

    else:
        # Step 2: Apply Min-Max Scaling
        filtered_df['Shots_Rating'] = ((filtered_df['Shots Total']  - min_shots) / (max_shots - min_shots)) * 10
        filtered_df['Goals_Rating'] = ((filtered_df['Goals']  - min_goals) / (max_goals - min_goals)) * 10
        filtered_df['Passes_Key_Rating'] = ((filtered_df['Passes Key'] - min_passes_key) / (max_passes_key - min_passes_key)) * 10
        filtered_df['Dribbles_Success_Rating'] = ((filtered_df['Dribbles Success'] - min_dribbles_suc) / (max_dribbles_suc - min_dribbles_suc)) * 10

    filtered_df['Attack_Rating_1'] = (filtered_df['Shots_Rating'] * 0.7) + (filtered_df['Goals_Rating'] * 0.1) + (filtered_df['Passes_Key_Rating'] * 0.1) +  (filtered_df['Dribbles_Success_Rating'] * 0.1)


    # Now AR based on position

    # Define conditions based on 'Position'
    conditions = [
        filtered_df['Position'] == 'Goalkeeper',
        filtered_df['Position'] == 'Defender',
        filtered_df['Position'] == 'Midfielder',
        filtered_df['Position'] == 'Attacker'
    ]

    # Define corresponding values for each condition
    values = [0, 2.5, 6.5, 9]

    # Apply conditions using np.select
    filtered_df['Attack_Rating_2'] = np.select(conditions, values, default=5)  # default=5 for other positions

    # Merge Attack_Rating_1 and Attack_Rating_2 (Mix 30:70)

    filtered_df['Attack Rating'] = round(filtered_df['Attack_Rating_1'] * 0.5 + filtered_df['Attack_Rating_2'] * 0.5, 1)



    # Step 4: Merge the scaled values into df_r
    df_r = df_r.merge(filtered_df[['Player', 'Attack Rating']], on='Player', how='left')

 
    # # Display the DataFrame with styles applied in Streamlit
    col1, _,col3, _ = st.columns([10,1,6,1])

    with col3:
        st.write("")
        st.write("")
        # WIDGET
        show_d_grade_info = st.checkbox('Downgrade/Classification Info', label_visibility = 'visible')
        if show_d_grade_info:
            st.caption(
                """
                    Player Rating reflects the significance of each player's contribution to the team's performance (during the selected season only) and this rating translates to an automated classification of that player's likely relative importance within the full squad.  
                    A player's downgrade indicates the estimated reduction in team supremacy, measured in hundredths of a goal. For instance, a downgrade of -9 means the team’s supremacy shifts from 0.00 to -0.09 if the player does not start.
                    Downgrades are calculated assuming a match with zero supremacy and average goals, where the replacement player is rated as "Regular." If the substitute is rated lower, the downgrade may increase accordingly.
                    These values scale proportionally for matches with higher supremacy or total goals.
                    Players who have been sold or are out due to long-term injuries may still appear in Best XI. 
                    While their ratings gradually decay over time, their classification remains relevant due to their prior impact on matches, influence on overall team ratings and the potential effect of their absence on team performance.
                    All ratings are data-derived.
                """
            )
            st.write('Player classification and downgrade values around midseason:')
            st.caption('"Key +" : -12')
            st.caption('"Key" : -9')
            st.caption('"Important +" : -6')
            st.caption('"Important" : -4')
            st.caption('"Regular +" : -2')
            st.caption('"Regular" : -1')
            st.caption("Downgrade values increase as the season progresses and player classifications become less volatile")


    # ------- Build FST df -------------

    # get the top 1 GK
    df_goalkeepers = df_r[df_r['Position'] == 'Goalkeeper'].nlargest(1, 'Rating')

    # Get the top 4 Defenders
    df_defenders = df_r[df_r['Position'] == 'Defender'].nlargest(4, 'Rating')

    # Get the top 5 Midfielders
    df_midfielders = df_r[df_r['Position'] == 'Midfielder'].nlargest(5, 'Rating')

    # Get the top 3 Attackers
    df_attackers = df_r[df_r['Position'] == 'Attacker'].nlargest(3, 'Rating')

    # 13 players selected - now concat and filter best 11 - this is so teams with fewer attackers still return 11 players, 10 outfield
    # Concatenate the three results into a single dataframe
    df_fst = pd.concat([df_goalkeepers, df_defenders, df_midfielders, df_attackers])
    df_fst = df_fst.nlargest(11, 'Rating')
    df_fst = df_fst.sort_values(by='Attack Rating', ascending=True)

    df_fst = df_fst.drop(['Player_id'], axis = 1)

    # Reset index for cleanliness (optional)
    df_fst.reset_index(drop=True, inplace=True)

    # --- Create coloured DF for df_fst -------------------

    # Configure Ag-Grid options
    gb3 = GridOptionsBuilder.from_dataframe(df_fst)

    # Freeze the 'Player' column
    gb3.configure_column('Player', pinned='left')

    # Light green background color for the entire DataFrame
    light_purple = '#ffccff'  # Light green color

    # Cell style for all cells
    cell_style = {
        'backgroundColor': light_purple,  # Light green color for all cells
        'color': 'black'                  # Default text color
    }

    # # Dynamically set column widths based on header text lengths
    # for column in df_r.columns:
    #     # Calculate width based on header text length
    #     header_length = len(column) * 10  # Adjust multiplier for better fitting
    #     gb.configure_column(column, width=max(100, header_length))  # Minimum width of 100

    # Apply the same cell style to all columns
    gb3.configure_default_column(cellStyle=cell_style)
    gb3.configure_column('Player', cellStyle={'color': 'black', 'backgroundColor': '#FFB6C1'})
    gb3.configure_column('Downgrade', maxWidth=120)
    gb3.configure_column('Attack Rating', maxWidth=120)
    gb3.configure_column('Rating', maxWidth=95)
    gb3.configure_column('Classification', maxWidth=120)
    gb3.configure_column('Position', maxWidth=130)

    # # Build grid options
    grid_options = gb3.build()

    with col1:
        st.write("")
        st.write(f'**{selected_team} Best XI - {selected_year}**')

        for col in df_fst.select_dtypes(['category']):
            df_fst[col] = df_fst[col].astype(str)  # convert to object (string)

        # st.write(df_fst)
        df_fst = df_fst.replace([np.inf, -np.inf], 0)
        df_fst = df_fst.fillna("")
        AgGrid(df_fst, gridOptions=grid_options, enable_enterprise_modules=False, height=375)

    # # ------------------ Wage Bill -------------------------------------

    # st.write("---")
    # wages_bill_leagues = ['England Premier', 'Spain La Liga', 'France Ligue 1', 'Italy Serie A', 'Germany Bundesliga']
    # if selected_league in wages_bill_leagues:
    #     show_wages_bill = st.checkbox('Show wage bill')
    #     if show_wages_bill:
    #         df_wages_raw = pd.read_csv(f'data/outputs_raw/players/wages_{selected_year}.csv')
    #         df_wages_filtered = df_wages_raw[df_wages_raw['Team'] == selected_team]
    #         df_wages_filtered = df_wages_filtered[['Player', 'Weekly Wages $']]
    #         # st.write(df_wages_filtered)


    #         def plot_wage_bar_chart(df, top_n=20):
    #             # Clean and prepare data
    #             df = df.copy()
    #             df = df[['Player', 'Weekly Wages $']].dropna()
    #             df['Weekly Wages $'] = pd.to_numeric(df['Weekly Wages $'], errors='coerce')
    #             df = df.dropna(subset=['Weekly Wages $'])
    #             df['Player'] = df['Player'].astype(str)

    #             # Sort and limit to top N
    #             df = df.sort_values('Weekly Wages $', ascending=False).head(top_n)

    #             # Create bar chart
    #             chart = alt.Chart(df).mark_bar().encode(
    #                 x=alt.X(
    #                     'Player:N',
    #                     sort=df['Player'].tolist(),
    #                     title='',
    #                     axis=alt.Axis(
    #                         labelAngle=-45,
    #                         labelFontSize=12,
    #                         labelOverlap=False,
    #                         tickMinStep=1,         # Key setting to show all labels
    #                         ticks=True,
    #                         labelFlush=False
    #                     )
    #                 ),
    #                 y=alt.Y('Weekly Wages $:Q', title='Weekly Wages ($)'),
    #                 color=alt.Color('Weekly Wages $:Q', scale=alt.Scale(scheme='tealblues'), legend=None),
    #                 tooltip=[
    #                     alt.Tooltip('Player:N', title='Player'),
    #                     alt.Tooltip('Weekly Wages $:Q', title='Wage ($)', format=',')
    #                 ]
    #             ).properties(
    #                 width=max(800, 30 * len(df)),  # Adjust width based on number of players
    #                 height=500,
    #                 title=alt.TitleParams(
    #                     text='Top Paid Players – Weekly Wages',
    #                     fontSize=20,
    #                     anchor='start'
    #                 )
    #             ).configure_view(
    #                 stroke=None
    #             )

    #             return chart
            
            
    #         df_wages_filtered['Weekly Wages $'] = (
    #             df_wages_filtered['Weekly Wages $']
    #             .replace('[\$,]', '', regex=True)   # remove $ and commas
    #             .astype(float)                      # convert to float
    #         )
    #         st.altair_chart(plot_wage_bar_chart(df_wages_filtered), use_container_width=False)

    # else:
    #     pass

    # # ------------------  Show current injuries  ----------------------

    # # Get team id
    # df2 = pd.read_csv('data/coordinates/all_leagues.csv', encoding='ISO-8859-1')

    # # Extract API_ID where Team matches selected_team
    # api_id_values = int(df2.loc[df2['Team'] == selected_team, 'API_ID'].values)

    # api_id = str(int(api_id_values))

    # # Ensure API_ID is valid before calling the API
    # if api_id:

    #     # call function to return df of team injuries last 2 weeks, passing args team api_id and current season
    #     try:
    #         df_injuries = get_injuries_by_team(api_id, CURRENT_SEASON)

    #         # Check if DataFrame has exactly 3 columns before renaming
    #         if df_injuries.shape[1] == 3:
    #             df_injuries.columns = ["Player Name", "Reason", "Date"]
    #         else:
    #             st.write(f"Unexpected column count: {df_injuries.shape[1]}. Data might be incomplete.")

    #         show_recent_absences = st.checkbox('Show absent players (last two weeks)')
    #         if show_recent_absences:
    #             if df_injuries.empty:
    #                 st.write("No data available")  # Display message when empty
    #             else:
    #                 st.write(df_injuries)

    #     except Exception as e:
    #         pass # Display nothing

    # else:
    #     st.write(f"No API_ID found for team: {selected_team}")


    # # ------------------   Show recent transfers  ----------------------------------

    # # Ensure API_ID is valid before calling the API
    # if api_id:

    #     # call function to return df of team injuries last 2 weeks, passing args team api_id and current season
    #     df_transfers = get_transfers_by_team(api_id, CURRENT_SEASON)

    #     show_recent_transfers = st.checkbox('Show transfers (last twelve months)')
    #     if show_recent_transfers:
    #         if df_transfers.empty:
    #             st.write("No transfers")  # Display message when empty
    #         else:
    #             st.write(df_transfers)

    # else:
    #     st.write(f"No API_ID found for team: {selected_team}")

    # # ------------------  Wiki link -------------------------------------------------

    # wiki_link = df2.loc[df2['Team'] == selected_team, 'Wiki'].values[0]
    # tmkt_link = df2.loc[df2['Team'] == selected_team, 'Tmkt'].values[0]

    # if pd.notna(wiki_link) and wiki_link:  # Ensure it's not NaN or empty
    #     st.subheader('Web Links')
    #     with st.expander("Wikipedia"): # can add fbref also
    #         st.markdown(f'<iframe src="{wiki_link}" width="100%" height="600"></iframe>', unsafe_allow_html=True)
    #     st.write("")
    #     st.write('**Transfermkt** - ', tmkt_link)


# Ensure the main function is called when the script is run directly
if __name__ == "__main__":
    main()