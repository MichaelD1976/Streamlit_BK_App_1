import streamlit as st
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import logging

# Set Streamlit logger level to ERROR to suppress warnings
logging.getLogger('streamlit').setLevel(logging.ERROR)


st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# CSS to hide the Streamlit footer
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


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
                'Shots on Target',
                'Fouls',
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

        col1,col2,col3 = st.columns([1,8,1])
        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.image("images/KM_Logo-bgr.png", width=700)
        c1,c2,c3 = st.columns([1,11,1])
        with c2:
            # st.write("")
            st.write("")
            st.write("")
            # st.markdown("<h1 style='font-size: 80px;'>KingMakers Trading App</h1>", unsafe_allow_html=True)
            st.write("")
            st.write("")


        st.header("", divider='blue')

        st.write("")
        st.write("")

        # st.write('Select required market from dropdown menu')




    elif selected_page == 'Shots on Target':
        import page_1_sot
        page_1_sot.main()

    elif selected_page == 'Fouls':
        import page_2_fouls
        page_2_fouls.main()

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