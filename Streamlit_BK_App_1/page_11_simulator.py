import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, rankdata


def main():
    st.header("Outright Winner Simulator", divider='blue')
    st.caption('''
                General purpose simulator for any market where selections have an estimated final outcome expectation.
               For example, Top Scorer - input selection player names and estimate of expected number of goals; League Outright - input teams and final points estimate. 
                ''')

    # Initialize session state
    if "num_selections" not in st.session_state:
        st.session_state.num_selections = 10
    if "selection_names" not in st.session_state:
        st.session_state.selection_names = [f"Outcome {i+1}" for i in range(st.session_state.num_selections)]
    if "expected_goals" not in st.session_state:
        st.session_state.expected_goals = [10.0] * st.session_state.num_selections
    if "num_simulations" not in st.session_state:
        st.session_state.num_simulations = 10000

    # User Input: Number of Selections
    c1,c2 = st.columns([1,3])
    with c1:
        st.session_state.num_selections = st.number_input("Number of Selections", min_value=2, max_value=50, value=st.session_state.num_selections, step=1)

    with c1:
        show_expectation_help = st.checkbox('Help to calculate expectation')
        if show_expectation_help:
            current_total = st.number_input('Current total', min_value=0, value=0, step=1)
            matches_remaining = st.number_input('Matches remaining', min_value=0, value=0, step=1)
            per_match_exp = st.number_input('Per match expectation', min_value=0.0, value=0.0, step=0.1)
            final_exp = round(current_total + matches_remaining * per_match_exp, 2)
            st.write('Expected number:', final_exp)

    # Adjust lists if num_selections changes
    if len(st.session_state.selection_names) < st.session_state.num_selections:
        st.session_state.selection_names += [f"Team {i+1}" for i in range(len(st.session_state.selection_names), st.session_state.num_selections)]
        st.session_state.expected_goals += [10.0] * (st.session_state.num_selections - len(st.session_state.expected_goals))
    elif len(st.session_state.selection_names) > st.session_state.num_selections:
        st.session_state.selection_names = st.session_state.selection_names[:st.session_state.num_selections]
        st.session_state.expected_goals = st.session_state.expected_goals[:st.session_state.num_selections]

    # Input Fields for Names & Expected Goals
    st.subheader("Enter Selection Names & Expected Goals/Points")
    col1, col2 = st.columns([3,2])

    for i in range(st.session_state.num_selections):
        with col1:
            st.session_state.selection_names[i] = st.text_input(f"Selection {i+1} Name", value=st.session_state.selection_names[i], key=f"name_{i}")
        with col2:
            st.session_state.expected_goals[i] = st.number_input(f"Expection for {st.session_state.selection_names[i]}", 
                                                                 min_value=0.0, value=st.session_state.expected_goals[i], step=0.1, key=f"goals_{i}")

    # Number of Simulations
    st.write("---")
    column1, column2 = st.columns([2,3])
    with column1:
        st.session_state.num_simulations = st.number_input("Number of Simulations", min_value=5000, max_value=1000000, value=st.session_state.num_simulations, step=1000)

    # Run Simulation Button
    if st.button("Run Simulation"):
        expected_goals = np.array(st.session_state.expected_goals)

        # Initialize progress bar
        progress_bar = st.progress(0)
        update_interval = max(st.session_state.num_simulations // 100, 1)

        # Simulate Poisson-distributed goals for each selection
        simulated_results = np.zeros((st.session_state.num_selections, st.session_state.num_simulations))

        for i in range(st.session_state.num_simulations):
            simulated_results[:, i] = poisson.rvs(mu=expected_goals)
            if i % update_interval == 0 or i == st.session_state.num_simulations - 1:
                progress_bar.progress((i + 1) / st.session_state.num_simulations)

        # Rank teams per simulation using unbiased ranking
        rankings = np.zeros_like(simulated_results, dtype=int)

        for i in range(st.session_state.num_simulations):
            rankings[:, i] = rankdata(-simulated_results[:, i], method='average') - 1  # 'average' method to handle ties fairly

        # Compute cumulative rank probabilities (Top X)
        top_x_probs = np.zeros((st.session_state.num_selections, st.session_state.num_selections))

        for team_idx in range(st.session_state.num_selections):
            for top_x in range(st.session_state.num_selections):
                top_x_probs[team_idx, top_x] = np.mean(rankings[team_idx, :] <= top_x)

        # Convert probabilities to odds
        top_x_odds = np.where(top_x_probs > 0, 100 / (top_x_probs * 100), np.nan)

        # Create DataFrames
        rank_labels = [f"Top {i+1}" for i in range(st.session_state.num_selections)]
        prob_df = pd.DataFrame(top_x_probs, index=st.session_state.selection_names, columns=rank_labels)
        odds_df = pd.DataFrame(top_x_odds, index=st.session_state.selection_names, columns=rank_labels)

        # Display the Results
        st.subheader("Probability of Finishing Position")
        st.dataframe(prob_df.style.format("{:.2%}"))

        st.subheader("Odds of Finishing Position")
        st.dataframe(odds_df.style.format("{:.2f}"))

if __name__ == "__main__":
    main()