import pandas as pd
import streamlit as st
# import altair as alt
from datetime import datetime
import numpy as np
from scipy.stats import poisson, nbinom
import requests
from unidecode import unidecode
from dotenv import load_dotenv
import os
from scipy.optimize import minimize_scalar

# dic to standardize api-football team names to t1x2 team names
team_names_api_to_t1x2_dict = {
   'Manchester City': 'Man City',
   'Manchester United': 'Man Utd',
   'Nottingham Forest': "Nottm Forest",
   'Hull City': 'Hull',
   'Oxford United': 'Oxford',
   'Sheffield Wednesday': 'Sheffield Weds',
   'Stoke City': 'Stoke',
   'Mansfield Town': 'Mansfield',
   'Bristol Rovers': 'Bristol Rvs',
   'Burton Albion': 'Burton',
   'Exeter City': 'Exeter',
   'Stockport County': 'Stockport',
   'Cambridge United': 'Cambridge',
   'Harrogate Town': 'Harrogate',
   'Salford City': 'Salford',
   'Swindon Town': 'Swindon',
    'Accrington ST': 'Accrington',
    'AS Roma': 'Roma',
    'Borussia Monchengladbach': "M'gladbach",
    'VfL Bochum': 'Bochum',
    'Borussia Dortmund': 'Dortmund',
    'Eintracht Frankfurt': 'Ein Frankfurt',
    '1899 Hoffenheim': 'Hoffenheim',
    'SC Freiburg': 'Freiburg',
    'VfB Stuttgart': 'Stuttgart',
    'FC Augsburg': 'Augsburg',
    'VfL Wolfsburg': 'Wolfsburg',
    'Bayern Munchen': 'Bayern Munich',
    'FSV Mainz 05': 'Mainz',
    'FC St. Pauli': 'St Pauli',
    '1. FC Heidenheim': 'Heidenheim',
    'Bayer Leverkusen': 'Leverkusen',   
    'Rayo Vallecano': 'Vallecano',
    'Real Sociedad': 'Sociedad',
    'Athletic Club': 'Ath Bilbao',
    'Atletico Madrid': 'Atl Madrid',
    'LE Havre': 'Le Havre',
    'Paris Saint Germain': 'Paris SG',
    'Stade Brestois 29': 'Brest',
    'Saint Etienne': 'St Etienne',   
    'Heart Of Midlothian': 'Hearts',
    'ST Mirren': 'St Mirren',
    'ST Johnstone': 'St Johnstone',
    'PEC Zwolle': 'Zwolle',
    'GO Ahead Eagles': 'Go Ahead Eagles',
    'NEC Nijmegen': 'Nijmegen',
    'Almere City FC': 'Almere City',
    'Fortuna Sittard': 'For Sittard',
    'Beerschot Wilrijk': 'Beerschot',
    'KV Mechelen': 'Mechelen',
    'Club Brugge KV': 'Club Brugge',
    'KVC Westerlo': 'Westerlo',
    'Union St. Gilloise': 'St Gilloise',
    'St. Truiden': 'St Truiden',
    'FC Porto': 'Porto',
    'GIL Vicente': 'Gil Vicente',
    'SC Braga': 'Braga',
    'Sporting CP': 'Sporting',
    'Hamburger SV': 'Hamburg',
    'FC Schalke 04': 'Schalke',
    'Fortuna Dusseldorf': 'Dusseldorf',
    '1. FC Kaiserslautern': 'Kaiserslautern',
    'Hannover 96': 'Hannover',
    'SV Elversberg': 'Elversberg',
    'SpVgg Greuther Furth': 'Greuther Furth',
    'SC Paderborn 07': 'Paderborn',
    '1. FC Nurnberg': 'Nurnberg',
    'Eintracht Braunschweig': 'Braunschweig',
    '1. FC Magdeburg': 'Magdeburg',
    'Karlsruher SC': 'Karlsruher',
    'VfL Osnabruck': 'Osnabruck',
    'Hertha BSC': 'Hertha',
    '1.FC Koln': 'FC Koln',
    'SSV Ulm 1846': 'Ulm',
    'SV Darmstadt 98': 'Darmstadt'
}

# ---------------------------------------------------


# Function to fetch data from the API for fixtures
@st.cache_data
def get_fixtures(league_id, from_date, to_date, API_SEASON):

    if not st.secrets:
        load_dotenv()
        API_KEY = os.getenv("API_KEY_FOOTBALL_API")

    else:
        # Use Streamlit secrets in production
        API_KEY = st.secrets["rapidapi"]["API_KEY_FOOTBALL_API"]


    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    
    querystring = {
        "league": league_id,
        "season": API_SEASON,
        "from": from_date,
        "to": to_date,
        "timezone": "Europe/London"
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    fixture_list = []
    if 'response' in data and data['response']:
        for fixture_data in data['response']:
            fixture = fixture_data.get('fixture', {})
            league = fixture_data.get('league', {})
            teams = fixture_data.get('teams', {})
            goals = fixture_data.get('goals', {})
            score = fixture_data.get('score', {})
            
            fixture_date_str = fixture.get('date', 'N/A')
            if fixture_date_str != 'N/A':
                fixture_date_obj = datetime.fromisoformat(fixture_date_str[:-6])
                formatted_date = fixture_date_obj.strftime('%d-%m-%y %H:%M')
            else:
                formatted_date = 'N/A'
                
            fixture_details = {
                'Fixture ID': fixture.get('id', 'N/A'),
                'Date': formatted_date,
                'Venue': fixture.get('venue', {}).get('name', 'N/A'),
                'City': fixture.get('venue', {}).get('city', 'N/A'),
                'Status': fixture.get('status', {}).get('long', 'N/A'),
                "Referee": fixture.get("referee", 'N/A'),
                'Home Team': teams.get('home', {}).get('name', 'N/A'),
                'Away Team': teams.get('away', {}).get('name', 'N/A'),
                'Home Team Logo': teams.get('home', {}).get('logo', 'N/A'),
                'Away Team Logo': teams.get('away', {}).get('logo', 'N/A'),
                'League Name': league.get('name', 'N/A'),
                'League Round': league.get('round', 'N/A'),
                'Goals Home': goals.get('home', 'N/A'),
                'Goals Away': goals.get('away', 'N/A'),
                'Halftime Score Home': score.get('halftime', {}).get('home', 'N/A'),
                'Halftime Score Away': score.get('halftime', {}).get('away', 'N/A'),
                'Fulltime Score Home': score.get('fulltime', {}).get('home', 'N/A'),
                'Fulltime Score Away': score.get('fulltime', {}).get('away', 'N/A')
            }
            
            fixture_list.append(fixture_details)

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(fixture_list) if fixture_list else pd.DataFrame()
 

    # Now apply team name team name conversion 
    if not df.empty:
        # Use the map function to standardize the team names using the dictionary
        df['Home Team'] = df['Home Team'].apply(unidecode)
        df['Away Team'] = df['Away Team'].apply(unidecode)
        df['Home Team'] = df['Home Team'].map(team_names_api_to_t1x2_dict).fillna(df['Home Team'])
        df['Away Team'] = df['Away Team'].map(team_names_api_to_t1x2_dict).fillna(df['Away Team'])

    return df


# ------------------------------------------------------------------------------

# calculates lines (main, minor, major) and odds based on single args (home or away prediction/expectation) [poss add arg 'n' (number from main line - higher exps use 2)]
# due to neg binom k factor (dispersion) to calculate 'Total' exp, add together probabilities generated (dont pass through summed exp's to this funct)
# Both corners and sot are distributed in an almost 50/50 split between neg binom and poisson. Also used for fouls.
# 
def calculate_home_away_lines_and_odds(prediction):

    if prediction == 0:
        # Handle the edge case where prediction is zero
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)   

    hc_k = prediction # the dispersion factor (see modelling)
    main_line = np.floor(prediction) + 0.5  # Main line rounded to nearest 0.5
    minor_line = main_line - 1
    major_line = main_line + 1
    # minor_line_2 = main_line - 2
    # major_line_2 = main_line + 2

        # Check for division by zero (p_success = prediction / (prediction + hc_k))
    try:
        p_success = prediction / (prediction + hc_k)
    except ZeroDivisionError:
        # If ZeroDivisionError occurs, return NaN values for all probabilities
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    
    # Probability of success for Negative Binomial
    p_success = prediction / (prediction + hc_k)

    # MAIN LINE 
    # Calculate the probability of being under the main line using Negative Binomial
    probability_under_nbinom_main = nbinom.cdf(np.floor(main_line), hc_k, p_success)
    probability_over_nbinom_main = 1 - probability_under_nbinom_main  # Total probability is 1
    # Calculate the probability of being under the main line using Poisson
    probability_under_poisson_main = poisson.cdf(np.floor(main_line), prediction)
    probability_over_poisson_main = 1 - probability_under_poisson_main  # Total probability is 1
    # Calculate the mixed probabilities (50/50)
    prob_under_main = round(0.5 * probability_under_nbinom_main + 0.5 * probability_under_poisson_main, 2)
    prob_over_main = round(0.5 * probability_over_nbinom_main + 0.5 * probability_over_poisson_main, 2)

    # MINOR LINE
    # Calculate the probability of being under the minor line using Negative Binomial
    probability_under_nbinom_min = nbinom.cdf(np.floor(minor_line), hc_k, p_success)
    probability_over_nbinom_min = 1 - probability_under_nbinom_min  # Total probability is 1
    # Calculate the probability of being under the minor line using Poisson
    probability_under_poisson_min = poisson.cdf(np.floor(minor_line), prediction)
    probability_over_poisson_min = 1 - probability_under_poisson_min  # Total probability is 1
    # Calculate the mixed probabilities (50/50)
    prob_under_min = round(0.5 * probability_under_nbinom_min + 0.5 * probability_under_poisson_min, 2)
    prob_over_min = round(0.5 * probability_over_nbinom_min + 0.5 * probability_over_poisson_min, 2)

    # # MINOR LINE 2
    # # Calculate the probability of being under the minor line 2 using Negative Binomial
    # probability_under_nbinom_min_2 = nbinom.cdf(np.floor(minor_line_2), hc_k, p_success)
    # probability_over_nbinom_min_2 = 1 - probability_under_nbinom_min_2  # Total probability is 1
    # # Calculate the probability of being under the minor line using Poisson
    # probability_under_poisson_min_2 = poisson.cdf(np.floor(minor_line_2), prediction)
    # probability_over_poisson_min_2 = 1 - probability_under_poisson_min_2  # Total probability is 1
    # # Calculate the mixed probabilities (50/50)
    # prob_under_min_2 = round(0.5 * probability_under_nbinom_min_2 + 0.5 * probability_under_poisson_min_2, 2)
    # prob_over_min_2 = round(0.5 * probability_over_nbinom_min_2 + 0.5 * probability_over_poisson_min_2, 2)

    # MAJOR LINE
    # Calculate the probability of being under the major line using Negative Binomial
    probability_under_nbinom_maj = nbinom.cdf(np.floor(major_line), hc_k, p_success)
    probability_over_nbinom_maj = 1 - probability_under_nbinom_maj  # Total probability is 1
    # Calculate the probability of being under the major line using Poisson
    probability_under_poisson_maj = poisson.cdf(np.floor(major_line), prediction)
    probability_over_poisson_maj = 1 - probability_under_poisson_maj  # Total probability is 1
    # Calculate the mixed probabilities (50/50)
    prob_under_maj = round(0.5 * probability_under_nbinom_maj + 0.5 * probability_under_poisson_maj, 2)
    prob_over_maj = round(0.5 * probability_over_nbinom_maj + 0.5 * probability_over_poisson_maj, 2)

    # # MAJOR LINE 2
    # # Calculate the probability of being under the major line 2 using Negative Binomial
    # probability_under_nbinom_maj_2 = nbinom.cdf(np.floor(major_line_2), hc_k, p_success)
    # probability_over_nbinom_maj_2 = 1 - probability_under_nbinom_maj_2  # Total probability is 1
    # # Calculate the probability of being under the major line using Poisson
    # probability_under_poisson_maj_2 = poisson.cdf(np.floor(major_line_2), prediction)
    # probability_over_poisson_maj_2 = 1 - probability_under_poisson_maj_2  # Total probability is 1
    # Calculate the mixed probabilities (50/50)
    # prob_under_maj_2 = round(0.5 * probability_under_nbinom_maj_2 + 0.5 * probability_under_poisson_maj_2, 2)
    # prob_over_maj_2 = round(0.5 * probability_over_nbinom_maj_2 + 0.5 * probability_over_poisson_maj_2, 2)


    return (float(main_line), float(minor_line), float(major_line), # float(minor_line_2), float(major_line_2), 
            float(prob_under_main), float(prob_over_main), 
            float(prob_under_min), float(prob_over_min), 
            # float(prob_under_min_2), float(prob_over_min_2),
            float(prob_under_maj), float(prob_over_maj))
            # float(prob_under_maj_2), float(prob_over_maj_2))


# ----------------------------------------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------------------------------------------------

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