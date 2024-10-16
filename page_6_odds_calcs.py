import streamlit as st
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar


def poisson_expectation(line, unders_prob):
    # Poisson is a discrete distribution, so we deal with integers below the line
    # We need to find lambda such that P(X <= floor(line)) is approximately unders_prob
    
    # Define the function we want to minimize: difference between CDF and unders_prob
    def objective(lmbda):
        return abs(poisson.cdf(int(line), lmbda) - unders_prob)
    
    # Use a scalar minimizer to find the lambda that minimizes the objective function
    result = minimize_scalar(objective, bounds=(0, 100), method='bounded')

    # Return the lambda (expected value) that best fits the given probabilities
    return result.x


def poisson_probabilities(expectation, main_line, line_minus_1, line_plus_1, line_minus_2, line_plus_2):
    # Calculate the cumulative probabilities (CDF) for under scenarios
    under_main = round(poisson.cdf(main_line, expectation), 2)
    under_minus_1 = round(poisson.cdf(line_minus_1, expectation), 2)
    under_plus_1 = round(poisson.cdf(line_plus_1, expectation), 2)
    under_minus_2 = round(poisson.cdf(line_minus_2, expectation), 2)
    under_plus_2 = round(poisson.cdf(line_plus_2, expectation), 2)
    
    # Over probabilities are 1 - CDF, or using the survival function
    over_main = round(poisson.sf(main_line, expectation), 2)  # sf(x, lambda) = 1 - cdf(x, lambda)
    over_minus_1 = round(poisson.sf(line_minus_1, expectation), 2)
    over_plus_1 = round(poisson.sf(line_plus_1, expectation), 2)
    over_minus_2 = round(poisson.sf(line_minus_2, expectation), 2)
    over_plus_2 = round(poisson.sf(line_plus_2, expectation), 2)

    # Return the probabilities as a dictionary for clarity
    return {
        f'under_main {main_line}': under_main,
        f'over_main {main_line}': over_main,
        f'under_minus_1 {line_minus_1}': under_minus_1,
        f'over_minus_1 {line_minus_1}': over_minus_1,
        f'under_plus_1 {line_plus_1}': under_plus_1,
        f'over_plus_1 {line_plus_1}': over_plus_1,
        f'under_minus_2 {line_minus_2}': under_minus_2,
        f'over_minus_2 {line_minus_2}': over_minus_2,
        f'under_plus_2 {line_plus_2}': under_plus_2,
        f'over_plus_2 {line_plus_2}': over_plus_2
    }




def main():

    st.header('Miscellaneous Calculators', divider='blue')


    col1,col2,col3, col4 = st.columns([2,0.5,2,1])
    with col1:
        st.write("")
        st.write('Input Line and Over/Under Odds to generate Expectation')
        container_1 = st.container(border=True)
        with container_1:
            st.subheader('Line to Expectation Calculator')
            line = st.number_input('Input Line (half ball)', value=2.5, step=1.0)
            if line % 1 != 0.5:
                st.error('Input a half ball line')
            else:
                pass

            odds1 = st.number_input('Input Over Odds', value = 1.85)
            odds2 = st.number_input('Input Under Odds', value = 1.85)

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
            exp = st.number_input('Input Expectation', step=0.05, value=expected_value)

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

if __name__ == '__main__':
    main()