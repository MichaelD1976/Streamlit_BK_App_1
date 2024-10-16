import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from scipy.stats import poisson
import math
import numpy as np


st.header("Daily Markets", divider='rainbow')

league_options = {
                'England Premier': 'eng1',
                'Germany Bundesliga': 'ger1',
                'Spain La Liga': 'spain1',
                'France Ligue 1': 'fra1',
                'Italy Serie A': 'italy1',
                'Sweden Allsvenskan': 'swe1',
                'Norway Eliteserien': 'nor1'
                }

market_options = [
                'Daily Goals',
                'Daily Corners',
                'Daily Shots OT'
                ]

OVER_BIAS = 1.02

def main():
    @st.cache_data
    def load_data():
        time.sleep(1)
        df = pd.read_csv(r'C:/Users/MikeD/Streamlit_BK_app/data/processed/next_fixts.csv')
        # Convert the 'timestamp' column to datetime
        # df['date'] = pd.to_datetime(df['date'], format='%Y:%m:%d %H:%M:%S')

        # # Extract just the date part (YYYY-MM-DD)
        df['date'] = pd.to_datetime(df['date']).dt.date  

        return df
    
    df = load_data()



    #st.sidebar.selectbox('Select League', league_options)
    selected_league = st.sidebar.radio('**Select League**', league_options.keys())
    st.sidebar.write("---")
    st.sidebar.write("")
    selected_metric = st.sidebar.radio('**Select Market**', market_options)

    # st.write(df)
    
    # Sidebar for 'from_date' input
    # Calculate max 'to_date' based on 'from_date'
    today = datetime.now().date()
    date_chosen = st.sidebar.date_input('Select a date', today)
 

    # apply filters
    def apply_filters(df, selected_league, date):
        league_code = league_options[selected_league]
        df1 = df[df['league_name'] == league_code]
        df2 = df1[df1['date'] == date]
        return df2
    
    filtered_df = apply_filters(df, selected_league, date_chosen)

    if filtered_df.empty:
        st.write(f"Data for {selected_league} currently unavailable.")   
    else:
        st.header(f'{selected_league} - {selected_metric}')
        st.subheader(f'Fixtures : {date_chosen}')
        st.write(filtered_df)
        #st.subheader(f'{selected_metric} :')
        if selected_metric == 'Daily Goals':
            total_day_goals = round(filtered_df['HG_Exp'].sum() + filtered_df['HG_Exp'].sum(), 2)
            if total_day_goals == 0:
                st.write('No lines available')
            else:
                st.write('Total Goals Exp:', total_day_goals)
                st.write('(with overs bias):', round(total_day_goals * OVER_BIAS, 2))

                # Calculate lines
                main_line = np.floor(total_day_goals) + 0.5
                if main_line > 20:
                    main_line_plus_1 = np.floor(total_day_goals) + 2.5
                    main_line_minus_1 = np.floor(total_day_goals) - 1.5

                elif main_line > 50:
                    main_line_plus_1 = np.floor(total_day_goals) + 3.5
                    main_line_minus_1 = np.floor(total_day_goals) - 2.5

                else:
                    main_line_plus_1 = np.floor(total_day_goals) + 1.5
                    main_line_minus_1 = np.floor(total_day_goals) - 0.5

                # Calculate Poisson probabilities
                poisson_lambda = total_day_goals

                # Poisson calcs
                under_prob_main =  poisson.cdf(main_line, poisson_lambda)
                under_prob_plus_1 = poisson.cdf(main_line_plus_1, poisson_lambda)
                under_prob_minus_1 = poisson.cdf(main_line_minus_1, poisson_lambda)

                over_prob_main = 1 - under_prob_main
                over_prob_plus_1 = 1 - under_prob_plus_1
                over_prob_minus_1 = 1 - under_prob_minus_1

                col1, col2, col3, col4 = st.columns([1,1,1,2])
                with col1:
                    st.subheader(f'Line {main_line_minus_1}')
                    st.write(f'Over: {round(1/over_prob_minus_1, 2)}')
                    st.write(f'Under: {round(1/under_prob_minus_1, 2)}')
                with col2:
                    st.subheader(f'Line {main_line}')
                    st.write(f'Over: {round(1/over_prob_main, 2)}')
                    st.write(f'Under: {round(1/under_prob_main, 2)}')
                with col3:
                    st.subheader(f'Line {main_line_plus_1}')
                    st.write(f'Over: {round(1/over_prob_plus_1, 2)}')
                    st.write(f'Under: {round(1/under_prob_plus_1, 2)}')


    #     LG_EX_H_SOT_P_G = 3.25
    #     LG_EX_A_SOT_P_G = 3.35
    #     df = filtered_df.copy()
    #     df['HG_fr_Av'] = ((-0.0122 * (df['HG_Exp'] ** 3)) + (0.118 * (df['HG_Exp'] ** 2 )) - ((0.459 * df['HG_Exp'])) + 1.445)
    #     df['H_SOT/G']  = df['HG_fr_Av'] * LG_EX_H_SOT_P_G
    #     df['AG_fr_Av'] = ((-0.084 * (df['AG_Exp'] ** 3)) + (0.507 * (df['AG_Exp'] ** 2 )) - ((1.145 * df['AG_Exp'])) + 1.76)
    #     df['A_SOT/G']  = df['AG_fr_Av'] * LG_EX_A_SOT_P_G       
    #     df['H_Sot'] = round(df['HG_Exp'] * df['H_SOT/G'], 2)
    #     df['A_Sot'] = round(df['AG_Exp'] * df['A_SOT/G'], 2)
    #     df['T_Sot'] = df['H_Sot'] + df['A_Sot']
    #     df = df[['home_name', 'away_name', 'H_Sot', 'A_Sot', 'T_Sot']]
    #     show_sot_df = st.checkbox(f'Show {selected_metric} expectations')
    #     if show_sot_df:
    #         st.write(df)

    # # elif selected_metric == 'Daily Goals':


    # else:
    #     st.write(f'{selected_metric} market not available yet')


if __name__ == '__main__':
    main()