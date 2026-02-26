import streamlit as st
import numpy as np
import math
# from scipy.stats import poisson
# from scipy.optimize import minimize_scalar
from mymodule.functions import poisson_expectation, poisson_probabilities


def main():

    st.header('Miscellaneous Calculators', divider='blue')


    col1,col2,col3, col4 = st.columns([2.5,0.5,2.5,0.5])
    with col1:
        st.write("")
        st.write('Input Line and Over/Under Odds to generate Expectation')
        container_1 = st.container(border=True)
        with container_1:
            st.subheader('Line to Expectation Calculator')
            line = st.number_input('Input Line (half ball)', value=2.5, step=1.0, label_visibility='visible')
            if line % 1 != 0.5:
                st.error('Input a half ball line')
            else:
                pass

            odds1 = st.number_input('Input Over Odds', value = 1.85, label_visibility='visible')
            odds2 = st.number_input('Input Under Odds', value = 1.85, label_visibility='visible')

            if odds1 == 0 or odds2 == 0:
                # Handle the case where odds2 is zero
                st.write("Error: odds cannot be zero.")
                st.stop()


            else:
                perc_ov = 1 / odds1
                perc_un = 1 / odds2

            margin = round(perc_ov + perc_un, 2)
            true_ov = round(perc_ov / margin, 2)
            true_un = round(perc_un / margin, 2)
            margin_as_per = round(margin * 100, 2)
            st.write('Margin:', margin_as_per ) 

            expected_value = poisson_expectation(line, true_un)
            st.write("")
            st.success(f"**Expectation: {expected_value:.2f}**")

        st.write("")
        st.write("")

    # ---------------------------------------------------------------
    # Calculate alternative Lines given expectation

    with col3:
        st.write("")
        st.write('Input Expectation to generate lines and odds')
        container_2 = st.container(border=True)
        with container_2:
            st.subheader('Expectation to Line Calculator')
            st.write("")
            exp = st.number_input('Input Expectation', step=0.05, value=expected_value, label_visibility='visible')

            main_line = np.floor(exp) + 0.5

            if main_line > 35:
                increment = 3
            elif main_line > 14:
                increment = 2
            else:
                increment = 1

            line_minus_1 = main_line - increment
            line_minus_2 = main_line - increment * 2
            line_plus_1 = main_line + increment
            line_plus_2 = main_line + increment * 2

            #st.write('main line:', main_line, 'minus_line:', line_minus_1, 'plus_line:', line_plus_1)
            #st.write('minus_line_2:', line_minus_2, 'plus_line_2:', line_plus_2)

            probabilities = poisson_probabilities(exp, main_line, line_minus_1, line_plus_1, line_minus_2, line_plus_2)
            #st.write(probabilities)
            # access the key/values in probabilities dict
            st.write("")
            st.write("")

            st.caption("100% Prices")
            st.write(f'(Line {line_plus_2}) - Over', round(1 / probabilities[f'over_plus_2 {line_plus_2}'], 2), f'Under', round(1 / probabilities[f'under_plus_2 {line_plus_2}'], 2))

            st.write(f'(Line {line_plus_1}) - Over', round(1 / probabilities[f'over_plus_1 {line_plus_1}'], 2), f'Under', round(1 / probabilities[f'under_plus_1 {line_plus_1}'], 2))

            st.write(f'**(Main Line {main_line}) - Over**', round(1 / probabilities[f'over_main {main_line}'], 2), f'**Under**', round(1 / probabilities[f'under_main {main_line}'], 2))

            st.write(f'(Line {line_minus_1}) - Over', round(1 / probabilities[f'over_minus_1 {line_minus_1}'], 2), f'Under', round(1 / probabilities[f'under_minus_1 {line_minus_1}'], 2))

            st.write(f'(Line {line_minus_2}) - Over', round(1 / probabilities[f'over_minus_2 {line_minus_2}'], 2), f'Under', round(1 / probabilities[f'under_minus_2 {line_minus_2}'], 2))


    with col1:

        # Eary Goals Market Calculator - Calculate the true odds of over/under market with any single early goal before selected minute 

        # Cumulative goal distribution probabilities based on historical data
        GOAL_CUMULATIVE = {
            0: 0,
            5: 0.04,
            10: 0.08,
            15: 0.13,
            20: 0.18,
            25: 0.24,
            30: 0.30,
            35: 0.37,
            40: 0.43,
            45: 0.46,
            50: 0.54,
            55: 0.60,
            60: 0.67,
            65: 0.74,
            70: 0.80
        }


        def poisson_prob(k, lam):
            return (lam**k * math.exp(-lam)) / math.factorial(k)


        def poisson_under(max_goals, lam):
            return sum(poisson_prob(k, lam) for k in range(max_goals+1))


        def price_early_goals(lam, minute_cutoff, line):

            goals_needed = int(line + 0.5)

            # Over/Under probabilities
            p_under = poisson_under(goals_needed-1, lam)
            p_over = 1 - p_under

            # Early goal probability
            goal_share = GOAL_CUMULATIVE[minute_cutoff]

            p_early = 1 - math.exp(-lam * goal_share)

            # Offer probability
            p_offer = p_over + p_under * p_early

            odds = 1 / p_offer

            return p_offer, odds
        

        # Interface
        container_3 = st.container(border=True)
        with container_3:
            st.subheader("Over/Under with Early Single Goal")
            st.caption("Calculate the true odds of over/under market with an early goal pay-out (at least one goal)")
            st.write("")

            lam = st.number_input(
                "Expected Match Goals",
                value=2.7,
                step=0.05
            )


            line = st.selectbox(
                "Over/Under Line",
                [1.5,2.5,3.5]
            )

            minute_cutoff = st.selectbox(
                "Early Payout Minute",
                [0,5,10,15,20,25,30, 35]
            )


            prob, odds = price_early_goals(
                lam,
                minute_cutoff,
                line
            )

            # st.write(f"Probability Over {line}:", round(prob,2))
            # st.write(f"True Odds - Over {line} or Goal < {minute_cutoff} mins:", round(odds,2))
            st.caption(f"Odds of over {line} OR any goal before {minute_cutoff} mins:")
            st.success(f"**True Odds: {odds:.2f}**")


# -------------------------------------------------------------------------------------------------
    # Early Payout Pricer for Over/Under Markets where set number of goals needed for selected line eg 2.5 line, 2 goals score before selected min, 1.5 line 1 goal etc
    with col3:

        GOAL_CUMULATIVE = {
            0: 0.00,
            5: 0.04,
            10: 0.08,
            15: 0.13,
            20: 0.18,
            25: 0.24,
            30: 0.30,
            35: 0.37,
            40: 0.43,
            45: 0.46,
            50: 0.54,
            55: 0.60,
            60: 0.67,
            65: 0.74,
            70: 0.80
        }

        # -------------------------------------------------
        # POISSON PROBABILITY MASS FUNCTION
        # -------------------------------------------------
        # Returns:
        #   P(X = k) for a Poisson distribution with mean λ
        #
        # Formula:
        #   (λ^k * e^-λ) / k!
        # -------------------------------------------------

        def poisson(k, lam):
            return (lam ** k * math.exp(-lam)) / math.factorial(k)


        # -------------------------------------------------
        # POISSON CUMULATIVE DISTRIBUTION FUNCTION
        # -------------------------------------------------
        # Returns:
        #   P(X ≤ max_k)
        #
        # Used to compute:
        #   P(FT ≥ N+1) = 1 - P(FT ≤ N)
        # -------------------------------------------------

        def cumulative_poisson(max_k, lam):
            return sum(poisson(k, lam) for k in range(max_k + 1))


        def price_early_payout(lam, minute_cutoff, line):

            """
            Prices a market where the bet wins if:
            1) FT total ≥ line + 1 (over N)
            OR
            2) At least N goals are scored before the cutoff minute
            """
            # N = floor of line, e.g., 1.5 → 1, 2.5 → 2
            N = math.floor(line)

            # Split expected goals into early and late based on historical time share
            share = GOAL_CUMULATIVE[minute_cutoff]
            lam_early = lam * share
            lam_late = lam * (1 - share)

            p_over = 1 - cumulative_poisson(N, lam)

            p_under = cumulative_poisson(N, lam)  # P(FT ≤ N)
            p_early = 1 - cumulative_poisson(N-1, lam_early)  # P(Early ≥ N)
            p_offer = p_over + p_under * p_early

            odds = 1 / p_offer

            return p_offer, odds


        # -------------------------------------------------
        # STREAMLIT USER INTERFACE
        # -------------------------------------------------
        st.write("")
        st.write("")
        container_4 = st.container(border=True)
        with container_4:
            st.subheader("Over/Under with Early Defined Goals")
            st.caption("Calculate the true odds of over/under market with an early goal pay-out based on defined goals for selected line - 1.5 line 1 goal, 2.5 line 2 goals etc")

            # Total expected goals (λ)
            lam = st.number_input(
                "Expected Match Goals",
                min_value=0.5,
                max_value=6.0,
                value=2.7,
                step=0.1
            )


            # Select betting line
            line = st.selectbox(
                "Over/Under Line",
                [1.5, 2.5, 3.5, 4.5]
            )


            # Select early payout minute
            minute_cutoff = st.selectbox(
                "Early Payout Minute",
                list(GOAL_CUMULATIVE.keys())
            )


            # Calculate probability and fair odds
            prob, odds = price_early_payout(
                lam,
                minute_cutoff,
                line
            )


            # Display results
            st.caption(f"Odds of over {line} OR at least {math.floor(line)} goals before {minute_cutoff} mins:")
            # st.write("Win Probability:", round(prob, 4))
            # st.success("True Odds:", round(odds, 2))
            st.success(f"**True Odds: {odds:.2f}**")



if __name__ == '__main__':
    main()