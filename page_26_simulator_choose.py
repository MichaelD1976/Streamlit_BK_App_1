import streamlit as st

sim_options = [
    'General Purpose',
 #   'League Specific',
    'Group Stage',
]

def main():

    selected_page = st.sidebar.selectbox("Select Simulator", options=sim_options, index=0)

    if selected_page == 'General Purpose':
        import page_11_simulator
        page_11_simulator.main()    

    # elif selected_page == 'League Specific':
    #     import page_16_simulator
    #     page_16_simulator.main()

    elif selected_page == 'Group Stage':
        import page_25_simulator_group
        page_25_simulator_group.main()


if __name__ == "__main__":
    main()