import streamlit as st

player_stats_options = [
    'Leagues',
    'Nigerian Players',
]

def main():

    selected_page = st.sidebar.selectbox("Select Page Option", options=player_stats_options, index=0)
    st.sidebar.write("---")

    if selected_page == 'Leagues':
        import page_15_player_stats
        page_15_player_stats.main()    

    elif selected_page == 'Nigerian Players':
        import page_23_nigerian_players
        page_23_nigerian_players.main()



if __name__ == "__main__":
    main()