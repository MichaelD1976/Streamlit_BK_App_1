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
                 'Pricing Tools',
            #     'Derivatives',
            #     'Specials Pricing',
                #  '1Up Pricing',
                #  'HTUP Pricing',
                  'Player Stats',
                  'Squad Data',
            #      'High Draw Matches',
            #     'Shots on Target',
            #     'Fouls',
             #    'Corners',
            #     'Throw Ins',
             #    'Offsides',
            #    'Daily Totals',
            #    'Goalscorer Matchbets',
            #    'League Matchbets',
                 'Team News',
                 'Fixtures',
            #     'Popular Matches'
            #     'Chance Mix Calc'
                 'Outright Sim',
            #     'Odds Calculators',
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
                     - 06/05/26 Nigerian Players summary page added within player stats page
                     - 29/04/26 Fixtures page - selection dropdowns, api links, team badges and fst's added
                     - 22/04/26 Fixtures page added
                     - 15/04/26 Derivatives - DC 1 Up market added 
                     - 26/03/26 High Draw Matches - plot and derby matches added
                     - 23/03/26 Home page/sidebar - selection dropdown and page restructuring. Models - GMT > BST default change.
                     - 16/03/26 High Draw Matches - Scottish Champ and League One data added
                     - 04/03/26 High Draw Matches - low goal individual matches flagged
                     - 26/02/26 Odds Calculators - Early Goals O/U payout tool added - defined goals per line
                     - 25/02/26 Odds Calculators - Early Goals O/U Payout tool added - any early goal
                     - 18/02/26 High draw matches page added
                     - 04/02/26 Squad Data - team best XI's added
                     - 03/02/26 Squad Data page added
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



    elif selected_page == 'Models':
        import page_10_models_choose
        page_10_models_choose.main()

    elif selected_page == 'Pricing Tools':
        import page_20_pricing_tools_choose
        page_20_pricing_tools_choose.main()

    elif selected_page == 'Team News':
        import page_7_team_news
        page_7_team_news.main()

    elif selected_page == 'Outright Sim':
        import page_11_simulator
        page_11_simulator.main() 
    

    # # remove this and swap with below for Nig players page
    # elif selected_page == 'Player Stats':
    #     import page_15_player_stats
    #     page_15_player_stats.main() 

    elif selected_page == 'Player Stats':
        import page_22_player_stats_choose
        page_22_player_stats_choose.main() 

    elif selected_page == 'Squad Data':
        import page_18_squad_data
        page_18_squad_data.main() 

    elif selected_page == 'Fixtures':
        import page_21_fixtures
        page_21_fixtures.main() 





if __name__ == '__main__':
    main()