import streamlit as st
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import logging

# Set Streamlit logger level to ERROR to suppress warnings
logging.getLogger('streamlit').setLevel(logging.ERROR)


st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# hide github link on home page
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# league_options = {
#                 'England Premier': 'eng1',
#                 'Germany Bundesliga': 'ger1',
#                 'Spain La Liga': 'spain1',
#                 'France Ligue 1': 'fra1',
#                 'Italy Serie A': 'italy1',
#                 'Sweden Allsvenskan': 'swe1',
#                 'Norway Eliteserien': 'nor1'
#                 }

page_options = [
                 'Home', 
                 'Models',
            #     'Shots on Target',
            #     'Fouls',
             #    'Corners',
            #     'Throw Ins',
             #    'Offsides',
            #    'Daily Totals',
            #    'Goalscorer Matchbets',
            #    'League Matchbets',
                 'Odds Calculators',
                 'Team News',
                ]


def main():

    st.sidebar.title('Select Page')
    selected_page = st.sidebar.selectbox('Choose a page', page_options, label_visibility = 'hidden')
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
            st.write("")
            st.write("""
                     **Updates**:
                     - Offsides Model added - okay to use as a reference to position ourselves or for specials creation
                     - Corners added - okay to use as a reference to position ourselves or for specials creation
                     - Shots on Target - successfully tested and good to use 
                     - Fouls model outputs still in testing - keep using auto pricing emails
                     """)
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


    # elif selected_page == 'Shots on Target':
    #     import page_1_sot
    #     page_1_sot.main()

    # elif selected_page == 'Fouls':
    #     import page_2_fouls
    #     page_2_fouls.main()

    # elif selected_page == 'Corners':
    #     import page_3_corners
    #     page_3_corners.main()

    elif selected_page == 'Models':
        import page_10_models_choose
        page_10_models_choose.main()

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

    # elif selected_page == 'Offsides':
    #     import page_9_offsides
    #     page_9_offsides.main()



if __name__ == '__main__':
    main()