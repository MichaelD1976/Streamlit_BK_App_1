import streamlit as st
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
                 'Derivatives',
            #     'Specials Pricing',
                 '1Up Pricing',
                 'HTUP Pricing',
                  'Player Stats',
            #     'Shots on Target',
            #     'Fouls',
             #    'Corners',
            #     'Throw Ins',
             #    'Offsides',
            #    'Daily Totals',
            #    'Goalscorer Matchbets',
            #    'League Matchbets',
                 'Team News',
            #     'Popular Matches'
            #     'Chance Mix Calc'
                 'Outright Sim',
                 'Odds Calculators',
                ]


def main():

    st.sidebar.title('Select Page')
    selected_page = st.sidebar.selectbox('Choose a page', page_options, label_visibility = 'hidden')
    st.sidebar.write("---")

    if selected_page == 'Home':

        _ ,col2, _ = st.columns([1,8,1])
        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.image("images/KM_Logo-bgr.png", width=700)
            st.write("Please start using this internal link for App access: https://tradingmodel-test.int.kingmakers.tech")
            st.write("")
            with st.expander('Updates Log'): 
                st.write(""" 
                     - 16/12/25 HTUP pricing page added    
                     - 17/11/25 1Up multi-competition pricing added
                     - 05/11/25 Models - Shots/Fouls head to head fmh format added
                     - 28/10/25 Models - BST toggle default to off. API cache_resource amendment
                     - 13/10/25 Models - new offsides model deployed (improvement in handling outlier teams)
                     - 22/09/25 Models - current and previous stat averages added for selected league
                     - 15/09/25 Player Stats 25-26 available
                     - 08/09/25 Models - BST/UTC toggle added. Offsides FMH format ready
                     - 27/08/25 Models - sot and fouls outputs amended for FMH upload format
                     - 20/08/25 New season - api params updated & new season outputs ready
                     - 21/07/25 Models - single match pricing function added (SOT/Fouls/Offsides)
                     - 09/07/25 Derivatives - HT Early Payout added
                     - 02/07/25 Player Stats - option to filter by 'All Leagues' added
                     - 18/06/25 Player Stats page added                  
                     - 11/06/25 Derivatives - 1 Up & 2 Up Markets added
                     - 02/06/25 Derivatives - facility to generate team exp goals from market odds added 
                     - 07/05/25 Derivatives - draw & 1H split parameters added
                     - 28/04/25 Derivatives page added
                     - 24/03/25 Outright simulator added
                     - 26/02/25 SA Data added
                     """)
        c1,c2,c3 = st.columns([1,11,1])
        with c2:
            # st.write("")
            st.write("")
            st.write("")
            # st.markdown("<h1 style='font-size: 80px;'>KingMakers Trading! App</h1>", unsafe_allow_html=True)
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

    elif selected_page == 'Outright Sim':
        import page_11_simulator
        page_11_simulator.main() 

    elif selected_page == 'Popular Matches':
        import page_12_popular_matches
        page_12_popular_matches.main() 

    elif selected_page == 'Derivatives':
        import page_13_derivatives
        page_13_derivatives.main() 

    elif selected_page == 'Chance Mix Calc':
        import page_14_chance_mix
        page_14_chance_mix.main() 
    
    elif selected_page == 'Player Stats':
        import page_15_player_stats
        page_15_player_stats.main() 

    elif selected_page == '1Up Pricing':
        import page_16_1up
        page_16_1up.main() 
    
    elif selected_page == 'HTUP Pricing':
        import page_17_htup
        page_17_htup.main() 

    # elif selected_page == 'Offsides':
    #     import page_9_offsides
    #     page_9_offsides.main()



if __name__ == '__main__':
    main()