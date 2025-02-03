import streamlit as st

model_options = [
    'Shots on Target',
    'Fouls',
    'Offsides', 
    'Corners',  
]

def main():

    selected_page = st.sidebar.selectbox("Select a Model", options=model_options, index=0)

    if selected_page == 'Corners':
        import page_3_corners
        page_3_corners.main()    

    elif selected_page == 'Shots on Target':
        import page_1_sot
        page_1_sot.main()

    # elif selected_page == 'Shots Total':
    #     import page21_model_shots
    #     page21_model_shots.main()

    elif selected_page == 'Fouls':
        import page_2_fouls
        page_2_fouls.main()

    # elif selected_page == 'Yellow Cards':
    #     import page20_model_yellows
    #     page20_model_yellows.main()

    elif selected_page == 'Offsides':
        import page_9_offsides
        page_9_offsides.main()


if __name__ == "__main__":
    main()