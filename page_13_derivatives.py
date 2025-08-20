
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from mymodule.functions import calc_prob_matrix, calculate_expected_team_goals_from_1x2


def main():

    # ---------------  INITIAL MAIN PAGE HEADER LAYOUT  ------------------------------
    st.header(f'Derivative Odds Calculator', divider='blue')
    st.write(f'Select Match Supremacy & Goals Expectation to calculate pre-match market odds / probabilities')

    st.caption('''
               Derivative odds are supremacy/totals outputs based on bivariate poisson distributions (not past data). 
               Individual leagues and countries fit to varying bivariate lambda values for draw calculation. For simplicity, the lambda parameter is set here to 
               a constant 0.08 (with further adjustments for 1-1 scorelines) allowing for better league generalisability - increase for leagues with higher draw tendencies and vice versa.
               First Half Goal % is defaulted to multi-league average of 44%.
                ''')

    match_suprem, total_match_gls = 0.0, 2.5 

    # Call function to calculate values
    with st.expander('Supremacy and Goals Expectation from 1X2 & O/U Market Prices'):
        match_suprem, total_match_gls = calculate_expected_team_goals_from_1x2()

    # ---------------------- Sidebar --------------------

    # Supremacy slider in the sidebar # WIDGET
    supremacy = st.sidebar.slider('**Select Supremacy**', min_value=-2.6, max_value=3.1, value=match_suprem, step=0.02, label_visibility = 'visible')

    # Goals Exp slider in the sidebar # WIDGET
    goals_exp = st.sidebar.slider('**Select Goals Exp**', min_value=1.6, max_value=4.4, value=total_match_gls, step=0.02, label_visibility = 'visible')

    # 1h perc slider in the sidebar # WIDGET
    f_half_perc = st.sidebar.slider('**First Half Goal %**', min_value=42, max_value=48, value=44, step=1, label_visibility = 'visible')

    # draw lambda in the sidebar # WIDGET
    draw_lambda = st.sidebar.slider('**Draw Parameter**', min_value=0.00, max_value=0.26, value=0.08, step=0.02, label_visibility = 'visible') 

    s_half_perc = 100 - f_half_perc
    # Calculate 1st Half and 2nd Half Goals Exp
    f_half_g = round(goals_exp / 100 * f_half_perc, 2)
    s_half_g = round(goals_exp / 100 * s_half_perc, 2)

    max_goals = 10

    # call function from functions.py to return match prob matrices and hg/ag exp's
    prob_matrix_ft, prob_matrix_1h, prob_matrix_2h, hg, ag = calc_prob_matrix(supremacy, goals_exp, max_goals, draw_lambda, f_half_perc)


    # Display 1st Half and 2nd Half Goals Exp in a box
    st.sidebar.markdown("------")
    st.sidebar.markdown("### Generated Expectations")
    st.sidebar.markdown(
        f"""
        <div style="padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
            <strong>Supremacy:</strong> {supremacy}<br>
            <strong>Goals Exp:</strong> {goals_exp}<br><br>
            <strong>Home Goals Exp:</strong> {hg}<br>
            <strong>Away Goals Exp:</strong> {ag}<br>
            <strong>1st Half Goals Exp:</strong> {f_half_g}<br>
            <strong>2nd Half Goals Exp:</strong> {s_half_g}
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("------")

    # ----------------------------- MAIN SCREEN ------------------------ 

    # ------------------------------- MARKETS ----------------------

    # Calculate Win-Draw-Win market
    home_win_prob = np.sum(np.tril(prob_matrix_ft, -1))  # Home win (lower triangle of the matrix excluding diagonal)
    draw_prob = np.sum(np.diag(prob_matrix_ft))  # Draw (diagonal of the matrix)
    away_win_prob = np.sum(np.triu(prob_matrix_ft, 1))  # Away win (upper triangle of the matrix excluding diagonal)

    # Calculate Over/Under 

    # Calculate Over/Under 0.5 goals
    over_0_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 0])
    under_0_5 = 1 - over_0_5   
    
    # Calculate Over/Under 1.5 goals
    over_1_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 1])
    under_1_5 = 1 - over_1_5

    # Calculate Over/Under 2.5 goals
    over_2_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 2])
    under_2_5 = 1 - over_2_5

    # Calculate Over/Under 3.5 goals
    over_3_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 3])
    under_3_5 = 1 - over_3_5

    # Calculate Over/Under 4.5 goals
    over_4_5 = np.sum(prob_matrix_ft[(np.add.outer(range(max_goals+1), range(max_goals+1))) > 4])
    under_4_5 = 1 - over_4_5    

    # dictionary to reference values in selection and return probabilities
    lines_to_select_ov_un = {"2.5": [over_2_5, under_2_5],
                             "1.5": [over_1_5, under_1_5],
                             "3.5": [over_3_5, under_3_5],
                             "4.5": [over_4_5,under_4_5],
                             "0.5": [over_0_5,under_0_5],
    }

    # Both Teams to Score (Yes/No)
    btts_yes = np.sum(prob_matrix_ft[1:, 1:])
    btts_no = 1 - btts_yes

    # Double Chance market
    home_or_draw = home_win_prob + draw_prob
    away_or_draw = away_win_prob + draw_prob
    home_or_away = home_win_prob + away_win_prob

    # DNB
    dnb_home = home_win_prob / (home_win_prob + away_win_prob)
    dnb_away = 1 - dnb_home

    # Calculate Home Win to Nil and Away Win to Nil
    home_win_to_nil = sum(prob_matrix_ft[i, 0] for i in range(1, max_goals+1))
    away_win_to_nil = sum(prob_matrix_ft[0, j] for j in range(1, max_goals+1))

    # Calculate Win-Draw-Win market - 1H
    home_win_prob_fh = np.sum(np.tril(prob_matrix_1h, -1))  # Home win (lower triangle of the matrix excluding diagonal)
    draw_prob_fh = np.sum(np.diag(prob_matrix_1h))  # Draw (diagonal of the matrix)
    away_win_prob_fh = np.sum(np.triu(prob_matrix_1h, 1))  # Away win (upper triangle of the matrix excluding diagonal)

    # Handicap -1
    # Home Win by 2 or More Goals
    home_win_by_2_or_more = np.sum(prob_matrix_ft[i, j] for i in range(2, max_goals+1) for j in range(i-1))
    # Tie (Home Wins by Exactly 1 Goal)
    tie_home_win_by_1 = np.sum(prob_matrix_ft[i, j] for i in range(1, max_goals+1) for j in range(i) if i-j == 1)
    # Away Win or Draw
    away_win_or_draw = draw_prob + away_win_prob 

    # Handicap +1
    # Home Win or Draw (if home wins or if match is a draw, considering the handicap)
    home_win_or_draw = home_win_prob + draw_prob  # Sum of match draw + home win percentage
    # Tie (Away Wins by Exactly 1 Goal, which nullifies the -1 handicap)
    tie_away_win_by_1 = np.sum(prob_matrix_ft[i, j] for j in range(1, max_goals+1) for i in range(j) if j-i == 1)
    # Away Win by 2 or More Goals (considering the handicap of -1)
    away_win_by_2_or_more = np.sum(prob_matrix_ft[i, j] for j in range(2, max_goals+1) for i in range(j-1))


    # Calculate HT-FT market

    # Away wins 2nd half probability
    away_win_2h = (prob_matrix_2h[0,1] + prob_matrix_2h[0,2] + prob_matrix_2h[0,3] + prob_matrix_2h[0,4] + prob_matrix_2h[0,5] + prob_matrix_2h[0,6] + prob_matrix_2h[0,7] + prob_matrix_2h[0,8] + prob_matrix_2h[0,9] + prob_matrix_2h[0,10] +
                prob_matrix_2h[1,2] + prob_matrix_2h[1,3] + prob_matrix_2h[1,4] + prob_matrix_2h[1,5] + prob_matrix_2h[1,6] + prob_matrix_2h[1,7] + prob_matrix_2h[1,8] +
                prob_matrix_2h[2,3] + prob_matrix_2h[2,4] + prob_matrix_2h[2,5] + prob_matrix_2h[2,6] + prob_matrix_2h[2,7] +
                prob_matrix_2h[3,4] + prob_matrix_2h[3,5] + prob_matrix_2h[3,6] + prob_matrix_2h[3,7] +
                prob_matrix_2h[4,5] + prob_matrix_2h[4,6] + prob_matrix_2h[4,7] +
                prob_matrix_2h[5,6] + prob_matrix_2h[5,7])

    # Draw 1h/2h probability
    draw_1h = prob_matrix_1h[0,0] + prob_matrix_1h[1,1] + prob_matrix_1h[2,2] + prob_matrix_1h[3,3] + prob_matrix_1h[4,4] + prob_matrix_1h[5,5]
    draw_2h = prob_matrix_2h[0,0] + prob_matrix_2h[1,1] + prob_matrix_2h[2,2] + prob_matrix_2h[3,3] + prob_matrix_2h[4,4] + prob_matrix_2h[5,5]

    # Home 2h probability
    home_win_2h = (prob_matrix_2h[1,0] + prob_matrix_2h[2,0] + prob_matrix_2h[3,0] + prob_matrix_2h[4,0] + prob_matrix_2h[5,0] + prob_matrix_2h[6,0] + prob_matrix_2h[7,0] + prob_matrix_2h[8,0] + prob_matrix_2h[9,0] + prob_matrix_2h[10,0] +
                    prob_matrix_2h[2,1] + prob_matrix_2h[3,1] + prob_matrix_2h[4,1] + prob_matrix_2h[5,1] + prob_matrix_2h[6,1] + prob_matrix_2h[7,1] + prob_matrix_2h[8,1] + prob_matrix_2h[9,1] +
                    prob_matrix_2h[3,2] + prob_matrix_2h[4,2] + prob_matrix_2h[5,2] + prob_matrix_2h[6,2] + prob_matrix_2h[7,2] + prob_matrix_2h[8,2] +
                    prob_matrix_2h[4,3] + prob_matrix_2h[5,3] + prob_matrix_2h[6,3] + prob_matrix_2h[7,3] +
                    prob_matrix_2h[5,4] + prob_matrix_2h[6,4] + prob_matrix_2h[7,4] +
                    prob_matrix_2h[6,5] + prob_matrix_2h[7,5])

    # Create grouped 'win by' probabilities to aid HT-FT calculations

    home_win_1h_by_1 = prob_matrix_1h[1,0] + prob_matrix_1h[2,1] + prob_matrix_1h[3,2] + prob_matrix_1h[4,3] + prob_matrix_1h[5,4] + prob_matrix_1h[6,5]
    home_win_1h_by_2 = prob_matrix_1h[2,0] + prob_matrix_1h[3,1] + prob_matrix_1h[4,2] + prob_matrix_1h[5,3] + prob_matrix_1h[6,4] + prob_matrix_1h[7,5]
    home_win_1h_by_3 = prob_matrix_1h[3,0] + prob_matrix_1h[4,1] + prob_matrix_1h[5,2] + prob_matrix_1h[6,3] + prob_matrix_1h[7,4] + prob_matrix_1h[8,5]
    home_win_1h_by_4 = prob_matrix_1h[4,0] + prob_matrix_1h[5,1] + prob_matrix_1h[6,2] + prob_matrix_1h[7,3] + prob_matrix_1h[8,4] + prob_matrix_1h[9,5]

    away_win_1h_by_1 = prob_matrix_1h[0,1] + prob_matrix_1h[1,2] + prob_matrix_1h[2,3] + prob_matrix_1h[3,4] + prob_matrix_1h[4,5] + prob_matrix_1h[5,6]
    away_win_1h_by_2 = prob_matrix_1h[0,2] + prob_matrix_1h[1,3] + prob_matrix_1h[2,4] + prob_matrix_1h[3,5] + prob_matrix_1h[4,6] + prob_matrix_1h[5,7]
    away_win_1h_by_3 = prob_matrix_1h[0,3] + prob_matrix_1h[1,4] + prob_matrix_1h[2,5] + prob_matrix_1h[3,6] + prob_matrix_1h[4,7] + prob_matrix_1h[5,8]
    away_win_1h_by_4 = prob_matrix_1h[0,4] + prob_matrix_1h[1,5] + prob_matrix_1h[2,6] + prob_matrix_1h[3,7] + prob_matrix_1h[4,8] + prob_matrix_1h[5,9]

    home_win_2h_by_1 = prob_matrix_2h[1,0] + prob_matrix_2h[2,1] + prob_matrix_2h[3,2] + prob_matrix_2h[4,3] + prob_matrix_2h[5,4] + prob_matrix_2h[6,5]
    home_win_2h_by_2 = prob_matrix_2h[2,0] + prob_matrix_2h[3,1] + prob_matrix_2h[4,2] + prob_matrix_2h[5,3] + prob_matrix_2h[6,4] + prob_matrix_2h[7,5]
    home_win_2h_by_3 = prob_matrix_2h[3,0] + prob_matrix_2h[4,1] + prob_matrix_2h[5,2] + prob_matrix_2h[6,3] + prob_matrix_2h[7,4] + prob_matrix_2h[8,5]
    home_win_2h_by_4 = prob_matrix_2h[4,0] + prob_matrix_2h[5,1] + prob_matrix_2h[6,2] + prob_matrix_2h[7,3] + prob_matrix_2h[8,4] + prob_matrix_2h[9,5]

    away_win_2h_by_1 = prob_matrix_2h[0,1] + prob_matrix_2h[1,2] + prob_matrix_2h[2,3] + prob_matrix_2h[3,4] + prob_matrix_2h[4,5] + prob_matrix_2h[5,6]
    away_win_2h_by_2 = prob_matrix_2h[0,2] + prob_matrix_2h[1,3] + prob_matrix_2h[2,4] + prob_matrix_2h[3,5] + prob_matrix_2h[4,6] + prob_matrix_2h[5,7]
    away_win_2h_by_3 = prob_matrix_2h[0,3] + prob_matrix_2h[1,4] + prob_matrix_2h[2,5] + prob_matrix_2h[3,6] + prob_matrix_2h[4,7] + prob_matrix_2h[5,8]
    away_win_2h_by_4 = prob_matrix_2h[0,4] + prob_matrix_2h[1,5] + prob_matrix_2h[2,6] + prob_matrix_2h[3,7] + prob_matrix_2h[4,8] + prob_matrix_2h[5,9]


    # Probabilities for each HT-FT outcome

    HHp = 1 / (1 / home_win_prob_fh * (1 / (draw_2h + home_win_2h))) * 1.02
    DHp = draw_prob_fh * home_win_2h
    AHp = home_win_prob - HHp - DHp

    HDp =  (1/(1/home_win_1h_by_1 * 1/away_win_2h_by_1) + 
           1/(1/home_win_1h_by_2 * 1/away_win_2h_by_2) +
           1/(1/home_win_1h_by_3 * 1/away_win_2h_by_3) +
           1/(1/home_win_1h_by_4 * 1/away_win_2h_by_4)) * 1.10

    ADp =  (1/(1/away_win_1h_by_1 * 1/home_win_2h_by_1) + 
           1/(1/away_win_1h_by_2 * 1/home_win_2h_by_2) +
           1/(1/away_win_1h_by_3 * 1/home_win_2h_by_3) +
           1/(1/away_win_1h_by_4 * 1/home_win_2h_by_4)) * 1.10
    DDp = draw_prob - HDp - ADp
    
    AAp = 1 / (1 / away_win_prob_fh * (1 / (draw_2h + away_win_2h))) * 1.03
    DAp = 1 / (1 / draw_prob_fh * (1 / away_win_2h))
    HAp = away_win_prob - AAp - DAp
  
    # Calculate Clean sheet

    # Calculate Clean Sheet Probabilities
    home_clean_sheet_prob = np.sum(prob_matrix_ft[0, :])
    away_clean_sheet_prob = np.sum(prob_matrix_ft[:, 0])


    # Calculate Next Goal
    home_next_goal = hg/goals_exp * (1-prob_matrix_ft[0,0])
    away_next_goal = ag/goals_exp * (1-prob_matrix_ft[0,0])
    no_next_goal = prob_matrix_ft[0,0]

    # Calculate Win Either Half
    home_either_half = ((home_win_prob_fh * away_win_2h) + (home_win_prob_fh * draw_2h) + (home_win_prob_fh * home_win_2h) + 
                        (draw_prob_fh * home_win_2h) + (away_win_prob_fh * home_win_2h))
    away_either_half = ((away_win_prob_fh * home_win_2h) + (away_win_prob_fh * draw_2h) + (away_win_prob_fh * away_win_2h) + 
                        (draw_prob_fh * away_win_2h) + (home_win_prob_fh * away_win_2h))

    # Calculate Win Both Halves
    home_both_halves = home_win_prob_fh * home_win_2h           
    away_both_halves = away_win_prob_fh * away_win_2h


    # Calculate Asian Lines
    h_p_0_25 = 1 / ((1 - draw_prob / 2) / (home_win_prob + draw_prob / 2))
    a_m_0_25 = 1 / ((1 - draw_prob / 2) / away_win_prob)
    a_p_0_25 = 1 / ((1 - draw_prob / 2) / (away_win_prob + draw_prob / 2))
    h_m_0_25 = 1 / ((1 - draw_prob / 2) / home_win_prob)

    h_p_0_5 = home_or_draw
    a_m_0_5 = away_win_prob
    a_p_0_5 = away_or_draw
    h_m_0_5 = home_win_prob

    awb1 = prob_matrix_ft[0,1] + prob_matrix_ft[1,2] + prob_matrix_ft[2,3] + prob_matrix_ft[3,4] + prob_matrix_ft[4,5] + prob_matrix_ft[5,6]
    hwb1 = prob_matrix_ft[1,0] + prob_matrix_ft[2,1] + prob_matrix_ft[3,2] + prob_matrix_ft[4,3] + prob_matrix_ft[5,4] + prob_matrix_ft[6,5]

    h_p_0_75 = 1 / ((1 - awb1 / 2) / (1 - away_win_prob))
    a_m_0_75 = 1 / ((1 - awb1 / 2) / (away_win_prob - (awb1 / 2)))
    a_p_0_75 = 1 / ((1 - hwb1 / 2) / (1 - home_win_prob))
    h_m_0_75 = 1 / ((1 - hwb1 / 2) / (home_win_prob - (hwb1 / 2)))

    h_p_1_0 = 1 / ((1 - awb1) / (1 - away_win_prob))
    a_m_1_0 = 1 / ((1 - awb1) / (away_win_prob - awb1))
    a_p_1_0 = 1 / ((1 - hwb1) / (1 - home_win_prob))
    h_m_1_0 = 1 / ((1 - hwb1) / (home_win_prob - hwb1))

    h_p_1_25 = 1 / ((1 - (awb1 / 2)) / (1 - away_win_prob + (awb1 / 2)))
    a_m_1_25 = 1 / ((1 - (awb1 / 2)) / (away_win_prob - awb1))
    a_p_1_25 = 1 / ((1 - (hwb1 / 2)) / (1 - home_win_prob + (hwb1 / 2)))
    h_m_1_25 = 1 / ((1 - (hwb1 / 2)) / (home_win_prob - hwb1))

    h_p_1_5 = 1 / (1 / (1 - away_win_prob + awb1))
    a_m_1_5 = 1 / (1 / (away_win_prob - awb1))
    a_p_1_5 = 1 / (1/ (1 - home_win_prob + hwb1))
    h_m_1_5 = 1 / (1 / (home_win_prob - hwb1))

    awb2 = prob_matrix_ft[0,2] + prob_matrix_ft[1,3] + prob_matrix_ft[2,4] + prob_matrix_ft[3,5] + prob_matrix_ft[4,6] + prob_matrix_ft[5,7]
    hwb2 = prob_matrix_ft[2,0] + prob_matrix_ft[3,1] + prob_matrix_ft[4,2] + prob_matrix_ft[5,3] + prob_matrix_ft[6,4] + prob_matrix_ft[7,5]

    h_p_1_75 = 1 / ((1 - awb2 / 2) / (1 - away_win_prob + awb1))
    a_m_1_75 = 1 / ((1 - awb2 / 2) / (away_win_prob - awb1 - (awb2 / 2)))
    a_p_1_75 = 1 / ((1 - hwb2 / 2) / (1 - home_win_prob + hwb1))
    h_m_1_75 = 1 / ((1 - hwb2 / 2) / (home_win_prob - hwb1 - (hwb2 / 2)))

    h_p_2_0 = 1 / ((1 - awb2) / (1 - away_win_prob + awb1))
    a_m_2_0 = 1 / ((1 - awb2) / (away_win_prob - awb1 - awb2))
    a_p_2_0 = 1 / ((1 - hwb2) / (1 - home_win_prob + hwb1))
    h_m_2_0 = 1 / ((1 - hwb2) / (home_win_prob - hwb1 - hwb2))

    h_p_2_25 = 1 / ((1 - awb2 / 2) / (1 - away_win_prob + awb1 + (awb2 / 2)))
    a_m_2_25 = 1 / ((1 - awb2 / 2) / (away_win_prob - awb1 - awb2))
    a_p_2_25 = 1 / ((1 - hwb2 / 2) / (1 - home_win_prob + hwb1 + (hwb2 / 2)))
    h_m_2_25 = 1 / ((1 - hwb2 / 2) / (home_win_prob - hwb1 - hwb2))

    h_p_2_5 = 1 / (1 / (1 - away_win_prob + awb1 + awb2))
    a_m_2_5 = 1 / (1 / (away_win_prob - awb1 - awb2))
    a_p_2_5 = 1 / (1 / (1 - home_win_prob + hwb1 + hwb2))
    h_m_2_5 = 1 / (1 / (home_win_prob - hwb1 - hwb2))

    """
    Continue adding hacaps
    """

    # dictionary to reference values in selection and return probabilities
    lines_to_select = {"0": [dnb_home,dnb_away], 
                        "+0.25": [h_p_0_25, a_m_0_25], 
                        "-0.25": [h_m_0_25, a_p_0_25],
                        "+0.5": [h_p_0_5, a_m_0_5],
                        "-0.5": [h_m_0_5, a_p_0_5],
                        "+0.75": [h_p_0_75, a_m_0_75],
                        "-0.75": [h_m_0_75, a_p_0_75],
                        "+1.0": [h_p_1_0, a_m_1_0],
                        "-1.0": [h_m_1_0, a_p_1_0],
                        "+1.25": [h_p_1_25, a_m_1_25],
                        "-1.25": [h_m_1_25, a_p_1_25],
                        "+1.50": [h_p_1_5, a_m_1_5],
                        "-1.50": [h_m_1_5, a_p_1_5],
                        "+1.75": [h_p_1_75, a_m_1_75],
                        "-1.75": [h_m_1_75, a_p_1_75],
                        "+2.00": [h_p_2_0, a_m_2_0],
                        "-2.00": [h_m_2_0, a_p_2_0],
                        "+2.25": [h_p_2_25, a_m_2_25],
                        "-2.25": [h_m_2_25, a_p_2_25],
                        "+2.50": [h_p_2_5, a_m_2_5],
                        "-2.50": [h_m_2_5, a_p_2_5],
    }

    # Home No Bet
    home_no_bet_draw = draw_prob / (draw_prob + away_win_prob)          
    home_no_bet_away = away_win_prob / (draw_prob + away_win_prob)

    # Away No Bet
    away_no_bet_draw = draw_prob / (draw_prob + home_win_prob)          
    away_no_bet_home = home_win_prob / (draw_prob + home_win_prob)

    # Half Most Goals
    def calculate_highest_scoring_half(prob_matrix_1h, prob_matrix_2h):
        prob_first_half_higher = 0
        prob_second_half_higher = 0
        prob_draw = 0

        # Iterate over all possible goal combinations for both halves
        for i in range(len(prob_matrix_1h)):
            for j in range(len(prob_matrix_1h[i])):
                for k in range(len(prob_matrix_2h)):
                    for l in range(len(prob_matrix_2h[k])):
                        prob_1h = prob_matrix_1h[i][j]
                        prob_2h = prob_matrix_2h[k][l]
                        total_goals_1h = i + j
                        total_goals_2h = k + l

                        if total_goals_1h > total_goals_2h:
                            prob_first_half_higher += prob_1h * prob_2h
                        elif total_goals_1h < total_goals_2h:
                            prob_second_half_higher += prob_1h * prob_2h
                        else:
                            prob_draw += prob_1h * prob_2h

        return prob_first_half_higher, prob_second_half_higher, prob_draw
    
    # Odd/Even
    def calculate_even_and_odd_probabilities(prob_matrix_ft):
        even_probability = 0
        for i in range(len(prob_matrix_ft)):
            for j in range(len(prob_matrix_ft[i])):
                if (i + j) % 2 == 0:  # Check if the sum of goals is even
                    even_probability += prob_matrix_ft[i][j]
        odd_probability = 1 - even_probability
        return even_probability, odd_probability
    
    
    # Home Score Both Halves
    def calculate_home_to_score_both_halves(prob_matrix_1h, prob_matrix_2h):
        # For the first half, sum probabilities where the home team scores 1 or more goals (i ≥ 1),
        # while the away team's score (j) can be any value (from 0 to maximum).
        prob_home_first_half = sum(prob_matrix_1h[i][j]
                                for i in range(1, len(prob_matrix_1h))   # i = 1, 2, ..., max home goals
                                for j in range(len(prob_matrix_1h[i])))    # j = 0, 1, 2, ... (all away outcomes)

        # Similarly, for the second half:
        prob_home_second_half = sum(prob_matrix_2h[i][j]
                                    for i in range(1, len(prob_matrix_2h))
                                    for j in range(len(prob_matrix_2h[i])))
        
        prob_home_score_both_halves = prob_home_first_half * prob_home_second_half
        # Multiply the summed probabilities to get the overall probability
        return prob_home_score_both_halves
    
    # Away Score Both Halves
    def calculate_away_to_score_both_halves(prob_matrix_1h, prob_matrix_2h):
        # For the first half, sum probabilities where the away team scores 1 or more goals (j ≥ 1),
        # while the home team's score (i) can be any value (from 0 to maximum).
        prob_away_first_half = sum(prob_matrix_1h[i][j]
                                for i in range(len(prob_matrix_1h))   # i = 0, 1, ..., max home goals
                                for j in range(1, len(prob_matrix_1h[i])))    # j = 1, 2, ... (away team scores at least 1)

        # Similarly, for the second half:
        prob_away_second_half = sum(prob_matrix_2h[i][j]
                                    for i in range(len(prob_matrix_2h))
                                    for j in range(1, len(prob_matrix_2h[i])))
        
        prob_away_score_both_halves = prob_away_first_half * prob_away_second_half
        # Multiply the summed probabilities to get the overall probability
        return prob_away_score_both_halves
    

    # -----------  1 Up Functions ------------------------------

    def calculate_win_given_one_nil(hg, ag, minute_of_goal=29):
        """
        Estimate P(Win | 1-0 lead at a given minute) for both home and away teams.

        Parameters:
        - hg: Expected goals for the home team (full match)
        - ag: Expected goals for the away team (full match)
        - minute_of_goal: The minute at which the 1-0 lead is taken (default: 29)

        Returns:
        - (P(Home wins | 1-0), P(Away wins | 0-1))
        """

        minutes_remaining = 93 - minute_of_goal

        home_rate = hg / 93
        away_rate = ag / 93

        rem_home_xg = home_rate * minutes_remaining
        rem_away_xg = away_rate * minutes_remaining

        # Adjust for 1-0 scenario
        adj_home_xg_lead = rem_home_xg * 0.95
        adj_away_xg_trail = rem_away_xg * 1.05

        # Adjust for 0-1 scenario
        adj_home_xg_trail = rem_home_xg * 1.05
        adj_away_xg_lead = rem_away_xg * 0.95

        max_goals = 7

        win_prob_home = 0.0
        win_prob_away = 0.0

        for i in range(max_goals + 1):  # goals after lead
            for j in range(max_goals + 1):

                # Home leads 1-0
                final_home_1_0 = 1 + i
                final_away_1_0 = j
                prob_home_lead = poisson.pmf(i, adj_home_xg_lead) * poisson.pmf(j, adj_away_xg_trail)
                if final_home_1_0 > final_away_1_0:
                    win_prob_home += prob_home_lead

                # Away leads 0-1
                final_home_0_1 = i
                final_away_0_1 = 1 + j
                prob_away_lead = poisson.pmf(i, adj_home_xg_trail) * poisson.pmf(j, adj_away_xg_lead)
                if final_away_0_1 > final_home_0_1:
                    win_prob_away += prob_away_lead

        return win_prob_home, win_prob_away
    

    w_pb_given_1_up_h, w_pb_given_1_up_a = calculate_win_given_one_nil(hg, ag, minute_of_goal=29)
   

    # 1 Up - Main Calculation
    # Prob (bet wins) = P(1-0 at anytime) + [ Prob(Home win - P(1-0 and HW) ]
    def calculate_one_up(home_next_goal, home_win_prob, w_pb_given_1_up_h, away_next_goal, away_win_prob, w_pb_given_1_up_a):
        home_1_up = home_next_goal + home_win_prob - (home_next_goal * w_pb_given_1_up_h)
        away_1_up = away_next_goal + away_win_prob - (away_next_goal * w_pb_given_1_up_a)

        return home_1_up, away_1_up
    
    
    # ------------------  2 UP FUNCTIONS  --------------------
    
    def calculate_win_given_two_nil(hg, ag, minute_of_second_goal=42):
        """
        Estimate P(Win | 2-0 lead at a given minute) for both home and away teams
        using adjusted Poisson models.

        Parameters:
        - hg: Expected goals for the home team (full match)
        - ag: Expected goals for the away team (full match)
        - minute_of_second_goal: Minute at which team takes a 2-0 lead (default: 42)

        Returns:
        - Tuple: (home_win_given_2_0, away_win_given_0_2)
        """
        minutes_remaining = 93 - minute_of_second_goal

        home_rate = hg / 93
        away_rate = ag / 93

        rem_home_xg = home_rate * minutes_remaining
        rem_away_xg = away_rate * minutes_remaining

        # Game state adjustments: more defensive when 2-0 up
        adj_home_xg = rem_home_xg * 0.90
        adj_away_xg = rem_away_xg * 1.10

        adj_away_lead_home_xg = rem_home_xg * 1.10
        adj_away_lead_away_xg = rem_away_xg * 0.90

        max_goals = 7
        home_win_prob_given_2_0 = 0.0
        away_win_prob_given_0_2 = 0.0

        # Home team leading 2–0
        for i in range(max_goals + 1):  # home goals after 2-0
            for j in range(max_goals + 1):  # away goals
                prob = poisson.pmf(i, adj_home_xg) * poisson.pmf(j, adj_away_xg)
                final_home = 2 + i
                final_away = j
                if final_home > final_away:
                    home_win_prob_given_2_0 += prob

        # Away team leading 0–2
        for i in range(max_goals + 1):  # home goals after 0-2
            for j in range(max_goals + 1):  # away goals
                prob = poisson.pmf(i, adj_away_lead_home_xg) * poisson.pmf(j, adj_away_lead_away_xg)
                final_home = i
                final_away = 2 + j
                if final_away > final_home:
                    away_win_prob_given_0_2 += prob

        return home_win_prob_given_2_0, away_win_prob_given_0_2
    

    w_pb_given_2_up_h, w_pb_given_2_up_a = calculate_win_given_two_nil(hg, ag, minute_of_second_goal=42)


    # 2 Up - Main Calculation
    # Prob (bet wins) = P(2-0 at anytime) + [ Prob(Home win - P(2-0 and HW) ]
    def calculate_two_up(home_next_goal, home_win_prob, w_pb_given_2_up_h, away_next_goal, away_win_prob, w_pb_given_2_up_a):
        # calc initial going 2-0 up
        home_2_0_initial = home_next_goal * home_next_goal * 0.95
        away_2_0_initial = away_next_goal * away_next_goal * 0.95

        home_2_up = home_2_0_initial + home_win_prob - (home_2_0_initial * w_pb_given_2_up_h)
        away_2_up = away_2_0_initial + away_win_prob - (away_2_0_initial * w_pb_given_2_up_a)

        return home_2_up, away_2_up

    
    # HTEP (Half time early payout)
    htep_h = home_win_prob + HDp + HAp
    htep_a = away_win_prob + ADp + AHp
    htep_x = draw_prob + DHp + DAp


    # ------------------------ Layout ---------------------------------

    # --------------------------- CSS format of market selections odds/percentage format ----------
    def display_odds_with_percentage(title, odds, percentage):
        """
        Displays a line with formatted odds and percentage.

        Args:
        - title: The title or label for the line (e.g., "Home Win").
        - odds: The decimal odds value (1/prob)
        - percentage: The probability percentage value.
        """
        st.markdown(
            f"**{title}:** {round(odds, 2)} "
            f"<span style='font-size: smaller; font-style: italic;'>({percentage:.1%})</span>",
            unsafe_allow_html=True
        )
    # ------------------------------------------------------------------------------------------------

    # Create three columns
    left_column, middle_column, right_column = st.columns([2, 2, 2])

    # set color for market name
    market_name_color = '#4682B4' # 0d3b66

    with left_column:
        # Win-Draw-Win market
        st.markdown(f"<h4 style='color:{market_name_color};'>1X2</h4>", unsafe_allow_html=True)
        display_odds_with_percentage("Home Win", 1/home_win_prob, home_win_prob)
        display_odds_with_percentage("Draw", 1/draw_prob, draw_prob)
        display_odds_with_percentage("Away Win", 1/away_win_prob, away_win_prob)

        # Over/Under
    
        st.markdown(f"<h4 style='color:{market_name_color};'>Over/Under</h4>", unsafe_allow_html=True)
        selected_line_ov_un = st.selectbox('Select Over/Under Line', options=lines_to_select_ov_un.keys())
        selected_values_ov_un = lines_to_select_ov_un[selected_line_ov_un]
        # st.write(f"Odds for Line '{selected_line_ov_un}': (Home) {round(1/selected_values_ov_un[0],3)}, (Away) {round(1/selected_values_ov_un[1],3)}")
        display_odds_with_percentage(f"Over {selected_line_ov_un} Goals", 1/selected_values_ov_un[0], selected_values_ov_un[0])
        display_odds_with_percentage(f"Under {selected_line_ov_un} Goals", 1/selected_values_ov_un[1], selected_values_ov_un[1])

        # Asian Lines       
        st.markdown(f"<h4 style='color:{market_name_color};'>Asian Match Lines</h4>", unsafe_allow_html=True)
        selected_line = st.selectbox('Select Line (references home team)', options=lines_to_select.keys())
        selected_values = lines_to_select[selected_line]
        st.write(f"**Odds for Line {selected_line}**")
        display_odds_with_percentage(f"Home", 1/selected_values[0], selected_values[0])
        display_odds_with_percentage(f"Away", 1/selected_values[1], selected_values[1])

        # Both Teams to Score
        st.markdown(f"<h4 style='color:{market_name_color};'>Both Teams To Score</h4>", unsafe_allow_html=True)
        display_odds_with_percentage("Yes", 1/btts_yes, btts_yes)
        display_odds_with_percentage("No", 1/btts_no, btts_no)

        # Double Chance
        st.markdown(f"<h4 style='color:{market_name_color};'>Double Chance</h4>", unsafe_allow_html=True)
        display_odds_with_percentage("Home or Draw", 1/home_or_draw, home_or_draw)
        display_odds_with_percentage("Away or Draw", 1/away_or_draw, away_or_draw)
        display_odds_with_percentage("Home or Away", 1/home_or_away, home_or_away)

        # HTEP (Half Time Early Payout)
        st.markdown(f"<h4 style='color:{market_name_color};'>HT Early Payout</h4>", unsafe_allow_html=True)
        display_odds_with_percentage("Home HTEP", 1/htep_h, htep_h)
        display_odds_with_percentage("Draw HTEP", 1/htep_x, htep_x)
        display_odds_with_percentage("Away HTEP", 1/htep_a, htep_a)

    with middle_column:
        st.markdown(f"<h4 style='color:{market_name_color};'>Draw No Bet</h4>", unsafe_allow_html=True)
        display_odds_with_percentage("Home", 1/dnb_home, dnb_home)
        display_odds_with_percentage("Away", 1/dnb_away, dnb_away)

        st.markdown(f"<h4 style='color:{market_name_color};'>Win To Nil</h4>", unsafe_allow_html=True)
        display_odds_with_percentage("Home Win to Nil", 1/home_win_to_nil, home_win_to_nil)     
        display_odds_with_percentage("Away Win to Nil", 1/away_win_to_nil, away_win_to_nil) 

        st.markdown(f"<h4 style='color:{market_name_color};'>HT - 1X2</h4>", unsafe_allow_html=True)
        display_odds_with_percentage("Home Win HT", 1/home_win_prob_fh, home_win_prob_fh)
        display_odds_with_percentage("Draw HT", 1/draw_prob_fh, draw_prob_fh)
        display_odds_with_percentage("Away Win", 1/away_win_prob_fh, away_win_prob_fh)

        # Display the Handicap markets
        st.markdown(f"<h4 style='color:{market_name_color};'>Goal Handicap</h4>", unsafe_allow_html=True)
        goal_handicap_list = ['Home -1 Goal', 'Home +1 Goal']
        selected_line_hcap = st.selectbox('Select Line (references home team)', goal_handicap_list)
        if selected_line_hcap == 'Home -1 Goal':
            display_odds_with_percentage("Home -1", 1/home_win_by_2_or_more, home_win_by_2_or_more)
            display_odds_with_percentage("Hcap Tie", 1/tie_home_win_by_1, tie_home_win_by_1)
            display_odds_with_percentage("Away +1", 1/away_win_or_draw, away_win_or_draw)

        elif selected_line_hcap == 'Home +1 Goal':
            display_odds_with_percentage("Home +1", 1/home_win_or_draw, home_win_or_draw)
            display_odds_with_percentage("Hcap Tie", 1/tie_away_win_by_1, tie_away_win_by_1)
            display_odds_with_percentage("Away -1", 1/away_win_by_2_or_more, away_win_by_2_or_more)  

        # Next Goal
        st.markdown(f"<h4 style='color:{market_name_color};'>Next Goal</h4>", unsafe_allow_html=True)
        display_odds_with_percentage("Home Next Goal", 1/home_next_goal, home_next_goal)
        display_odds_with_percentage("Away Next Goal", 1/away_next_goal, away_next_goal) 
        display_odds_with_percentage("No Next Goal", 1/no_next_goal, no_next_goal)  

        # 1 Up
        st.markdown(f"<h4 style='color:{market_name_color};'>1 Up Early Payout</h4>", unsafe_allow_html=True)
        st.caption("Keep draw price as 1x2 market. Reduce outsider more heavily from 100% (eg 9.0 > 7.5; 4.2 > 3.5; 2.8 > 2.4)")
        home_1_up, away_1_up = calculate_one_up(home_next_goal, home_win_prob, w_pb_given_1_up_h, away_next_goal, away_win_prob, w_pb_given_1_up_a)
        # st.write(home_next_goal, home_win_prob, w_pb_given_1_up_h, away_next_goal, away_win_prob, w_pb_given_1_up_a)
        display_odds_with_percentage("Home 1 Up", 1/home_1_up, home_1_up)
        display_odds_with_percentage("Away 1 Up", 1/away_1_up, away_1_up)


        calculate_two_up(home_next_goal, home_win_prob, w_pb_given_2_up_h, away_next_goal, away_win_prob, w_pb_given_2_up_a)
        # 2 Up
        st.markdown(f"<h4 style='color:{market_name_color};'>2 Up Early Payout</h4>", unsafe_allow_html=True)
        st.caption("Keep draw price as 1x2 market. Reduce outsider more heavily from 100% (eg 9.0 > 7.5; 4.2 > 3.5; 2.8 > 2.4)")
        home_2_up, away_2_up = calculate_two_up(home_next_goal, home_win_prob, w_pb_given_2_up_h, away_next_goal, away_win_prob, w_pb_given_2_up_a)
        # st.write(home_next_goal, home_win_prob, w_pb_given_1_up_h, away_next_goal, away_win_prob, w_pb_given_1_up_a)
        display_odds_with_percentage("Home 2 Up", 1/home_2_up, home_2_up)
        display_odds_with_percentage("Away 2 Up", 1/away_2_up, away_2_up)

    with right_column:
        # Correct Score Grid

        # Display the grid as a DataFrame with a smaller size
        st.markdown(f"<h4 style='color:{market_name_color};'>Correct Score Odds</h4>", unsafe_allow_html=True)

        # Create a DataFrame from the probability matrix
        grid_df = pd.DataFrame(prob_matrix_ft, index=range(max_goals + 1), columns=range(max_goals + 1))

        # Create an amended version for display:
        # Step 1: Slice the DataFrame to a 6x6 grid to limit scorelines shown
        display_df = grid_df.iloc[:6, :6].copy()

        # Step 2: Convert the values to decimal odds (1/probability)
        # To handle potential division by zero, use np.inf for zero probabilities
        display_df = display_df.map(lambda x: 1/x if x > 0 else np.inf)

        # Add axis labels above the DataFrame to show which is home/away
        st.markdown("""
        <div style='text-align: left; font-size: small; color: grey;'>
            Home Goals (Rows) vs. Away Goals (Columns)
        </div>
        """, unsafe_allow_html=True)

        # Display the amended grid as a DataFrame in Streamlit with decimal odds
        st.dataframe(display_df.style.format("{:.2f}"), width=390, height=250)


        # HT-FT market
        st.markdown(f"<h4 style='color:{market_name_color};'>HT - FT</h4>", unsafe_allow_html=True)
        display_odds_with_percentage("Home/Home", 1/HHp, HHp)
        display_odds_with_percentage("Draw/Home", 1/DHp, DHp)
        display_odds_with_percentage("Away/Home", 1/AHp, AHp)
        display_odds_with_percentage("Home/Draw", 1/HDp, HDp)
        display_odds_with_percentage("Draw/Draw", 1/DDp, DDp)
        display_odds_with_percentage("Away/Draw", 1/ADp, ADp)
        display_odds_with_percentage("Home/Away", 1/HAp, HAp)
        display_odds_with_percentage("Draw/Away", 1/DAp, DAp)
        display_odds_with_percentage("Away/Away", 1/AAp, AAp)

        # OTHER MARKETS
        markets_list = ['Half Most Goals',
                         'Home No Bet',
                           'Away No Bet',
                             'Clean Sheet',
                             'Win Either Half',
                             'Win Both Halves',
                             'Score Both Halves',
                             'Odd Even FT'
                             ]   

        st.write("")
        st.markdown(f"<h4 style='color:{market_name_color};'>Extra Markets</h4>", unsafe_allow_html=True)
        selected_market = st.selectbox('Select Market', options=markets_list, placeholder="Choose a market", key='other_markets')
        if selected_market == 'Half Most Goals':
            prob_first_half_higher, prob_second_half_higher, prob_draw = calculate_highest_scoring_half(prob_matrix_1h, prob_matrix_2h)
            display_odds_with_percentage(f"First Half", 1/prob_first_half_higher, prob_first_half_higher)
            display_odds_with_percentage(f"Second Half", 1/prob_second_half_higher, prob_second_half_higher)
            display_odds_with_percentage(f"Tie", 1/prob_draw, prob_draw)

        elif selected_market == 'Clean Sheet':
            display_odds_with_percentage("Home Clean Sheet", 1/home_clean_sheet_prob, home_clean_sheet_prob)
            display_odds_with_percentage("Away Clean Sheet", 1/away_clean_sheet_prob, away_clean_sheet_prob)

        elif selected_market == 'Home No Bet':
            display_odds_with_percentage(f"Draw", 1/home_no_bet_draw, home_no_bet_draw)
            display_odds_with_percentage(f"Away", 1/home_no_bet_away, home_no_bet_away)

        elif selected_market == 'Away No Bet':
            display_odds_with_percentage(f"Home", 1/away_no_bet_home, away_no_bet_home)
            display_odds_with_percentage(f"Draw", 1/away_no_bet_draw, away_no_bet_draw)

        elif selected_market == 'Win Either Half':
            display_odds_with_percentage("Home Win Either Half", 1/home_either_half, home_either_half)
            display_odds_with_percentage("Away Win Either Half", 1/away_either_half, away_either_half)

        elif selected_market == 'Win Both Halves':
            display_odds_with_percentage("Home Win Both Halves", 1/home_both_halves, home_both_halves)
            display_odds_with_percentage("Away Win Both Halves", 1/away_both_halves, away_both_halves)

        elif selected_market == 'Odd Even FT':
            even_probability, odd_probability = calculate_even_and_odd_probabilities(prob_matrix_ft)
            display_odds_with_percentage(f"Even", 1/even_probability, even_probability)
            display_odds_with_percentage(f"Odd", 1/odd_probability, odd_probability)

        elif selected_market == 'Score Both Halves':
            prob_home_score_both_halves = calculate_home_to_score_both_halves(prob_matrix_1h, prob_matrix_2h)
            prob_away_score_both_halves = calculate_away_to_score_both_halves(prob_matrix_1h, prob_matrix_2h)
            display_odds_with_percentage(f"Home Score Both Halves", 1/prob_home_score_both_halves, prob_home_score_both_halves)
            display_odds_with_percentage(f"Away Score Both Halves", 1/prob_away_score_both_halves, prob_away_score_both_halves)




# ------------------------------------------
if __name__ == "__main__":
    main()
