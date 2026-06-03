import streamlit as st
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
from mymodule.functions import calc_prob_matrix
import random



def main():

    # ---------------------------------------------------
    # USER INPUTS
    # ---------------------------------------------------
    st.title("Group Stage Simulator")

    n_teams = st.number_input("Number of teams", min_value=2, max_value=24, value=4)

    teams = []
    ratings = []

    st.subheader("Teams & Ratings")
    st.caption("Enter team names and their corresponding ratings - a 10 point difference in rating equates to 0.1 supremacy difference. Include HAdv in the rating if present")

    for i in range(n_teams):
        col1, col2 = st.columns(2)

        with col1:
            team = st.text_input(f"Team {i+1}", key=f"team_{i}")
        with col2:
            rating = st.number_input(f"Rating {i+1}", key=f"rating_{i}", value=100)

        teams.append(team)
        ratings.append(rating)

    show_matches = st.checkbox("Show Match Odds Table")


    # ---------------------------------------------------
    # MATCH TABLE (ODDS + EXPECTATIONS)
    # ---------------------------------------------------
    def generate_match_table(teams, ratings):

        ratings_dict = dict(zip(teams, ratings))
        fixtures = list(itertools.combinations(teams, 2))

        draw_lambda = 0.08
        f_half_perc = 44
        max_goals = 9

        rows = []

        for a, b in fixtures:

            ra = ratings_dict[a]
            rb = ratings_dict[b]

            # Supremacy
            sup = (ra - rb) * 0.01

            # Goal Expectation
            GEx = 0.164 * (sup ** 2) - 0.019 * sup + 2.53

            # xG split
            home_xg = GEx / 2 + 0.5 * sup
            away_xg = GEx / 2 - 0.5 * sup

            # Probability matrix from external engine
            prob_matrix, _, _, _, _ = calc_prob_matrix(
                sup, GEx, max_goals, draw_lambda, f_half_perc
            )

            home_p = np.sum(np.tril(prob_matrix, -1))
            draw_p = np.sum(np.diag(prob_matrix))
            away_p = np.sum(np.triu(prob_matrix, 1))

            rows.append({
                "Team A": a,
                "Team B": b,
                "Supremacy": round(sup, 4),
                "GEx": round(GEx, 3),
                "Home_xG": round(home_xg, 2),
                "Away_xG": round(away_xg, 2),

                "Home Win Prob": round(home_p, 3),
                "Draw Prob": round(draw_p, 3),
                "Away Win Prob": round(away_p, 3),

                "Home Odds": round(1 / home_p, 2) if home_p > 0 else None,
                "Draw Odds": round(1 / draw_p, 2) if draw_p > 0 else None,
                "Away Odds": round(1 / away_p, 2) if away_p > 0 else None,
            })

        return pd.DataFrame(rows)
    
    # ---------------------------------------------------
    # MATCH ODDS TABLE
    # ---------------------------------------------------
    if show_matches:
        st.subheader("Match Odds & Expected Goals")

        match_df = generate_match_table(teams, ratings)
        st.dataframe(match_df.style.format(precision=2))

        # -----------------------------------------
        # TEAM EXPECTED GOALS (FOR / AGAINST)
        # -----------------------------------------

        team_xg = {t: {"xG_for": 0, "xG_against": 0} for t in teams}

        for _, row in match_df.iterrows():

            home = row["Team A"]
            away = row["Team B"]

            team_xg[home]["xG_for"] += row["Home_xG"]
            team_xg[home]["xG_against"] += row["Away_xG"]

            team_xg[away]["xG_for"] += row["Away_xG"]
            team_xg[away]["xG_against"] += row["Home_xG"]

        team_xg_df = pd.DataFrame([
            {
                "Team": t,
                "Expected Goals For": round(v["xG_for"], 2),
                "Expected Goals Against": round(v["xG_against"], 2),
            }
            for t, v in team_xg.items()
        ])

        st.subheader("Team Expected Goals Summary")
        st.dataframe(team_xg_df.style.format(precision=2))

    # ---------------------------------------------------
    # SIMULATION PARAMETERS
    # ---------------------------------------------------
    st.write("")
    st.write("---")
    n_sims = st.number_input("Simulations", min_value=1000, max_value=50000, value=10000, step=5000)

    run = st.button("Run Simulation")

    ratings_dict = dict(zip(teams, ratings))


    # ---------------------------------------------------
    # SAMPLING FUNCTIONS
    # ---------------------------------------------------
    def sample_match_outcome(home_p, draw_p, away_p):

        r = np.random.random()

        if r < home_p:
            return 1, 0
        elif r < home_p + draw_p:
            return 0, 0
        else:
            return 0, 1


    def sample_score(home_xg, away_xg):
        hg = np.random.poisson(max(home_xg, 0.1))
        ag = np.random.poisson(max(away_xg, 0.1))
        return hg, ag


    # ---------------------------------------------------
    # SINGLE SIMULATION
    # ---------------------------------------------------
    def run_single_sim(teams, ratings_dict):

        fixtures = list(itertools.combinations(teams, 2))

        table = {
            t: {"pts": 0, "gf": 0, "ga": 0}
            for t in teams
        }

        for a, b in fixtures:

            ra = ratings_dict[a]
            rb = ratings_dict[b]

            sup = (ra - rb) * 0.01
            GEx = 0.164 * (sup ** 2) - 0.019 * sup + 2.53

            home_xg = GEx / 2 + 0.5 * sup
            away_xg = GEx / 2 - 0.5 * sup

            prob_matrix, _, _, _, _ = calc_prob_matrix(
                sup, GEx, 9, 0.08, 44
            )

            home_p = np.sum(np.tril(prob_matrix, -1))
            draw_p = np.sum(np.diag(prob_matrix))
            away_p = np.sum(np.triu(prob_matrix, 1))

            # score simulation (for GD realism)
            hg, ag = sample_score(home_xg, away_xg)

            table[a]["gf"] += hg
            table[a]["ga"] += ag
            table[b]["gf"] += ag
            table[b]["ga"] += hg

            # outcome simulation (probability-driven)
            outcome = sample_match_outcome(home_p, draw_p, away_p)

            if outcome == (1, 0):
                table[a]["pts"] += 3
            elif outcome == (0, 1):
                table[b]["pts"] += 3
            else:
                table[a]["pts"] += 1
                table[b]["pts"] += 1

        for t in teams:
            table[t]["gd"] = table[t]["gf"] - table[t]["ga"]

        ranking = sorted(
            table.items(),
            key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]),
            reverse=True
        )

        return [t for t, _ in ranking]


    # ---------------------------------------------------
    # MONTE CARLO ENGINE
    # ---------------------------------------------------
    if run:

        position_counts = {
            t: defaultdict(int) for t in teams
        }

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(int(n_sims)):

            ranking = run_single_sim(teams, ratings_dict)

            for pos, team in enumerate(ranking):
                position_counts[team][pos + 1] += 1

            if i % max(1, n_sims // 100) == 0:
                progress_bar.progress((i + 1) / n_sims)
                status_text.text(f"Simulations: {i+1}/{n_sims}")

        progress_bar.progress(1.0)
        status_text.text("Simulation complete ✅")

        st.session_state["position_counts"] = position_counts
        st.session_state["n_sims"] = n_sims

        # ---------------------------------------------------
        # FINAL PROBABILITY TABLE
        # ---------------------------------------------------
        results = []

        for t in teams:

            row = {"Team": t}

            # position probabilities
            pos_probs = {}

            for pos in range(1, n_teams + 1):
                p = position_counts[t][pos] / n_sims
                pos_probs[pos] = p
                row[f"P({pos})"] = p

            # -----------------------------
            # DERIVED MARKETS
            # -----------------------------

            top2 = pos_probs.get(1, 0) + pos_probs.get(2, 0)
            top3 = pos_probs.get(1, 0) + pos_probs.get(2, 0) + pos_probs.get(3, 0)

            row["Top 2"] = top2
            row["Top 3"] = top3

            # -----------------------------
            # ODDS (all markets)
            # -----------------------------

            for pos in range(1, n_teams + 1):
                p = pos_probs[pos]
                row[f"Odds P({pos})"] = (1 / p) if p > 0 else None

            row["Odds Top 2"] = (1 / top2) if top2 > 0 else None
            row["Odds Top 3"] = (1 / top3) if top3 > 0 else None

            results.append(row)

            df = pd.DataFrame(results)

        st.subheader("Final Position Probabilities + Markets")

        numeric_cols = [
            c for c in df.columns
            if c != "Team"
        ]

        format_dict = {col: "{:.2f}" for col in numeric_cols}

        st.dataframe(df.style.format(format_dict))

    # --------------------
    # st.write("")
    # st.write("---")
    # st.subheader("Position Odds Explorer")

    # if "position_counts" not in st.session_state:
    #     st.info("Run the simulation first")
    #     return  # or st.stop()

    # position_counts = st.session_state["position_counts"]
    # n_sims = st.session_state["n_sims"]


    # team_selected = st.selectbox("Select team", teams)

    # position_selected = st.number_input(
    #     "Select position",
    #     min_value=1,
    #     max_value=n_teams,
    #     value=2
    # )

    # if team_selected:

    #     pos_probs = {
    #         pos: position_counts[team_selected][pos] / n_sims
    #         for pos in range(1, n_teams + 1)
    #     }

    #     # -----------------------------
    #     # ABOVE / BELOW SPLIT
    #     # -----------------------------

    #     p_above = sum(
    #         pos_probs[p]
    #         for p in range(position_selected, n_teams + 1)
    #     )

    #     p_below = sum(
    #         pos_probs[p]
    #         for p in range(1, position_selected)
    #     )

    #     # odds
    #     odds_above = 1 / p_above if p_above > 0 else None
    #     odds_below = 1 / p_below if p_below > 0 else None

    #     # -----------------------------
    #     # DISPLAY
    #     # -----------------------------

    #     st.markdown(f"### {team_selected} — Position Analysis")

    #     st.write(f"**P({position_selected} or better / worse)**")

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.metric(
    #             label=f"Finish {position_selected} or WORSE (positions {position_selected}-{n_teams})",
    #             value=f"{p_above:.2%}",
    #             delta=f"Odds: {odds_above:.2f}" if odds_above else "N/A"
    #         )

    #     with col2:
    #         st.metric(
    #             label=f"Finish ABOVE {position_selected} (positions 1-{position_selected-1})",
    #             value=f"{p_below:.2%}",
    #             delta=f"Odds: {odds_below:.2f}" if odds_below else "N/A"
    #         )


if __name__ == "__main__":
    main()