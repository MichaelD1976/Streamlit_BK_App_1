import streamlit as st

tool_options = [
    'Derivatives',
    'Odds Calculators',
    'High Draw Matches', 
    '1UP',  
    'HTUP'
]

def main():

    selected_page = st.sidebar.selectbox("Select Pricing Tool", options=tool_options, index=0)
    st.sidebar.write("---")

    if selected_page == 'Derivatives':
        import page_13_derivatives
        page_13_derivatives.main()    

    elif selected_page == 'Odds Calculators':
        import page_6_odds_calcs
        page_6_odds_calcs.main()

    elif selected_page == 'High Draw Matches':
        import page_19_draw_matches
        page_19_draw_matches.main()

    elif selected_page == '1UP':
        import page_16_1up
        page_16_1up.main()

    elif selected_page == 'HTUP':
        import page_17_htup
        page_17_htup.main()


if __name__ == "__main__":
    main()