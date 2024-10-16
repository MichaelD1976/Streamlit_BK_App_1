import streamlit as st
import pandas as pd
import time


st.set_page_config(layout="wide", initial_sidebar_state="expanded")


league_options = {
                'England Premier': 'eng1',
                'Germany Bundesliga': 'ger1',
                'Spain La Liga': 'spain1',
                'France Ligue 1': 'fra1',
                'Italy Serie A': 'italy1',
                'Sweden Allsvenskan': 'swe1',
                'Norway Eliteserien': 'nor1'
                }

page_options = [
    'Home', 
        'Daily Markets',
        'Stat Markets',
            'Goalscorer Matchbets',
            'League Matchbets',
                'Odds Calculators',
                'Team News',
                    ]


def main():

    st.sidebar.title('Select Page')
    selected_page = st.sidebar.selectbox('', page_options)
    st.sidebar.write("---")

    if selected_page == 'Home':

        st.header("Kingmakers Trading App", divider='blue')

        st.write("")
        st.write("")

        st.subheader('Market:')
        st.write('Select required market from dropdown menu')



    elif selected_page == 'Daily Markets':
        import page_2_daily_markets
        page_2_daily_markets.main()

    elif selected_page == 'Stat Markets':
        import page_3_stat_markets
        page_3_stat_markets.main()

    elif selected_page == 'Goalscorer Matchbets':
        import page_4_goalscorer_matchbets
        page_4_goalscorer_matchbets.main()

    elif selected_page == 'League Matchbets':
        import page_5_league_matchbets
        page_5_league_matchbets.main()    

    elif selected_page == 'Odds Calculators':
        import page_6_odds_calcs
        page_6_odds_calcs.main()

    elif selected_page == 'Team News':
        import page_7_team_news
        page_7_team_news.main()


if __name__ == '__main__':
    main()