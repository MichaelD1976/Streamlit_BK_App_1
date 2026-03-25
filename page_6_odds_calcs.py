import streamlit as st
import numpy as np
import math
# from scipy.stats import poisson
# from scipy.optimize import minimize_scalar
from mymodule.functions import poisson_expectation, poisson_probabilities


def main():

    st.header('Miscellaneous Calculators', divider='blue')


    col1, _,col3, _ = st.columns([2.5,0.5,2.5,0.5])
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

        def poisson(k, lam):
            return (lam ** k * math.exp(-lam)) / math.factorial(k)

        def cumulative_poisson(max_k, lam):
            return sum(poisson(k, lam) for k in range(max_k + 1))

        def price_early_payout(lam, minute_cutoff, line):

            N = math.floor(line)

            share = GOAL_CUMULATIVE[minute_cutoff]
            lam_early = lam * share
            lam_late = lam * (1 - share)

            # Standard over probability
            p_over = 1 - cumulative_poisson(N, lam)

            # Early-only path: exactly N early AND zero late
            p_early_exact = poisson(N, lam_early)
            p_late_zero = poisson(0, lam_late)

            p_offer = p_over + p_early_exact * p_late_zero

            odds = 1 / p_offer

            return p_offer, odds
            
        st.write("")
        st.write("")

        container_5 = st.container(border=True)
        with container_5:

            st.subheader("Over/Under with Early Defined Goals")
            st.caption("Calculate the true odds of over/under market with an early goal pay-out based on defined goals for selected line - 1.5 line 1 goal, 2.5 line 2 goals etc")

            lam = st.number_input(
                "Expected Match Goals",
                min_value=0.5,
                max_value=6.0,
                value=2.7,
                step=0.05,key=1,
            )

            line = st.selectbox(
                "Over/Under Line",
                [1.5, 2.5, 3.5], key=2
            )

            minute_cutoff = st.selectbox(
                "Early Payout Minute",
                # list(GOAL_CUMULATIVE.keys()), key=3      # Use if we want all 5 minute goal band options
                [20, 45, 65], key=3  # Use if we want to limit goal band options to those we are using - easier for testing
            )

            prob, odds = price_early_payout(lam, minute_cutoff, line)

            st.caption(
                f"Odds of over {line} OR exactly {math.floor(line)} goals before {minute_cutoff} mins and no more goals"
            )

            overs_fudge = 0.99
            st.success(f"True Odds: {odds * overs_fudge:.2f}")


    

if __name__ == '__main__':
    main()