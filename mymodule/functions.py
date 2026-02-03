import pandas as pd
import streamlit as st
# import altair as alt
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize
import requests
from unidecode import unidecode
from dotenv import load_dotenv
import os
import math
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
    '1. FC Koln': 'FC Koln',
    'SSV Ulm 1846': 'Ulm',
    'SV Darmstadt 98': 'Darmstadt'
}

team_names_t1x2_to_BK_dict = {
    'Chelsea' : 'Chelsea',
    'Fulham' : 'Fulham',
    'Man Utd' : 'Man Utd',
    'Burnley' : 'Burnley',
    'Tottenham': 'Tottenham',
    'Bournemouth': 'Bournemouth',
    'Sunderland': 'Sunderland',
    'Brentford': 'Brentford',
    'Leeds': 'Leeds',
    'Newcastle': 'Newcastle',
    'Brighton': 'Brighton',
    'Man City': 'Man City',
    'Nottm Forest': 'Nottingham',
    'West Ham': 'West Ham',
    'Liverpool': 'Liverpool',
    'Arsenal': 'Arsenal',
    'Everton': 'Everton',
    'Aston Villa': 'Aston Villa', 

    'QPR': 'Queens Park Rangers',
    'Sheffield Weds': 'Sheffield Wed',
    'Oxford': 'Oxford Utd',
    'Charlton': 'Charlton Athletic',
    'Bristol Rvs': 'Bristol Rovers',
    'Crawley Town': 'Crawley',
    'Fleetwood Town': 'Fleetwood',
    'Newport County': 'Newport',
    'Milton Keynes Dons': 'MK Dons',


    'Girona': 'Girona',
    'Villarreal': 'Villarreal',
    'Mallorca': 'Mallorca',
    'Valencia': 'Valencia',
    'Alaves': 'Alaves',
    'Celta Vigo': 'Celta Vigo',
    'Ath Bilbao': 'Athletic Bilbao',
    'Espanyol': 'Espanyol',
    'Elche': 'Elche',
    'Real Madrid': 'Real Madrid', # no change
    'Real Betis': 'Betis',
    'Atl Madrid': 'Atletico Madrid',
    'Levante': 'Levante',
    'Osasuna': 'Osasuna',
    'Sociedad': 'R. Sociedad',
    'Oviedo': 'R. Oviedo',
    'Sevilla': 'Sevilla',
    'Vallecano': 'Rayo Vallecano',
    'Barcelona': 'Barcelona',
    'Getafe': 'Getafe',
    
    'Rennes': 'Rennes',
    'Lens': 'Lens',
    'Monaco': 'Monaco',
    'Nice': 'Nice',
    'Brest': 'Brest',
    'Angers': 'Angers',
    'Auxerre': 'Auxerre',
    'Metz': 'Metz',
    'Nantes': 'Nantes',
    'Paris SG': 'PSG',
    'Marseille': 'Marseille',
    'Lyon': 'Lyon',
    'Lorient': 'Lorient',
    'Strasbourg': 'Strasbourg',
    'Toulouse': 'Toulouse',
    'Le Havre': 'Le Havre',
    'Lille': 'Lille',
    'Paris FC': 'Paris FC', # no change

    'Augsburg': 'Augsburg',
    'Bayern Munich': 'Bayern Munich', # same
    'Dortmund': 'Borussia Dortmund',
    'Ein Frankfurt': 'Eintracht Frankfurt',
    'Freiburg': 'Freiburg',
    'Heidenheim': 'Heidenheim',
    'Hoffenheim': 'Hoffenheim',
    'Leverkusen': 'Bayer Leverkusen',
    "M'gladbach": 'B. Monchengladbach',
    'Mainz': 'Mainz',
    'RB Leipzig': 'RB Leipzig', #same
    'St Pauli': 'St. Pauli',
    'Stuttgart': 'Stuttgart',
    'Union Berlin': 'Union Berlin', # same
    'Werder Bremen': 'Werder Bremen', # same
    'Wolsburg': 'Wolfsburg',
    'Hamburg': 'Hamburg',
    'FC Koln': 'Cologne',

    'AC Milan': 'Milan',
    'Cagliari': 'Cagliari',
    'Como': 'Como',
    'Genoa': 'Genoa',
    'Lazio': 'Lazio',
    'Lecce': 'Lecce',
    'Napoli': 'Napoli',
    'Roma': 'Roma',
    'Torino': 'Torino',
    'Verona': 'Hellas Verona',
    'Cremonese': 'Cremonese',
    'Sassuolo': 'Sassuolo',
    'Pisa': 'Pisa',

    'Kaizer Chiefs' : 'Kaizer Chiefs',
    'Richards Bay' : 'Richards Bay FC',
    'Chippa United' : 'Chippa United FC',
    'TS Galaxy' : 'TS Galaxy FC',
    'Orbit College' : 'Orbit College FC',
    'Sekhukhune United' : 'Sekhukhune United',
    'Durban City' : 'Durban City',
    'Golden Arrows' : 'Lamontville Golden Arrows',
    'Amazulu' : 'Amazulu FC',
    'Marumo Gallants' : 'Marumo Gallants FC',
    'Orlando Pirates' : 'Orlando Pirates FC',
    'Stellenbosch' : 'Stellenbosch FC',
    'Magesi' : 'Magesi FC',
    'Polokwane City' : 'Polokwane City',

        # CL/EL Teams
    'Olympiakos Piraeus': 'Olympiacos',
    'Sporting CP': 'Sporting',
    'Slavia Praha': 'Slavia Prague',
    'St Gilloise': 'Union St.Gilloise',
    'FC Copenhagen': 'Copenhagen',
    'Bodo/Glimt': 'Bodoe/Glimt',
    'PSV Eindhoven': 'PSV',

    'Ferencvaros': 'Ferencvarosi',
    'Dinamo Zagreb': 'GNK Dinamo Zagreb',
    'FC Midtjylland': 'Midtjylland',
    'FK Crvena Zvezda': 'Crvena Zvezda',
    'BSC Young Boys': 'Young Boys',
    'FC Basel 1893': 'Basel',
    'Red Bull Salzburg': 'Salzburg',
    'GO Ahead Eagles': 'GA Eagles',
    'Plzen': 'Viktoria Plzen',
    'Brann': 'SK Brann',

    # Turkey
    'Rizespor': 'Caykur Rizespor',
    'Fatih Karagumruk': 'Karagumruk',
    'Gaziantepspor': 'Gaziantep',

    # Greece
    'Kifisia': 'AE Kifisia FC',
    'Larisa': 'AE Larissa FC',
    'Asteras Atromitos': 'Atromitos Athinon',
    'Olympiakos': 'Olympiacos',
    'Panetolikos': 'Panetolikos Agrinio',
    'Levadiakos': 'APO Levadiakos FC',
    'Volos NFC': 'Volos NPS',
    'OFI': 'OFI Crete',
    'Aris': 'Aris Thessaloniki',
    'AEK': 'AEK Athens',
    'Panserraikos': 'Panserraikos FC',

    # Saudi
    'Al Khaleej Saihat': 'Al Khaleej Saihat FC',
    'Damac': 'Damac FC',
    'Al Taawon': 'Al Taawon FC',
    'Al Shabab': 'Al Shabab FC (KSA)',
    'Al-Ittihad FC': 'Al Ittihad Jeddah',
    'Al Riyadh': 'AL Riyadh',
    'Al-Fayha': 'AL Fayha FC',
    
}


CURRENT_SEASON_FST_FORMAT = '2025-26' # for generate_fst downgrades. If not current season, use spring adjustments


# ---------------------------------------------------


# Function to fetch data from the API for fixtures
@st.cache_resource
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
def calculate_home_away_lines_and_odds(prediction, model):

    if model == 'Offsides':
        poisson_weight = 0.7
        nb_weight = 0.3
    else:
        poisson_weight = 0.5
        nb_weight = 0.5

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


# --------------------  FIND TRUE 1X2 PRICES ------------------------------------- 
# set marg_pc_move (amount to change the fav when odds-on eg 1.2 > 1.22 instead of 1.3)
# complicated code. In testing it handles well transforming short price with margin > true price 

def calculate_true_from_true_raw(h_pc_true_raw , d_pc_true_raw , a_pc_true_raw, margin):
    marg_pc_remove = 0
    if h_pc_true_raw > 0.90 or a_pc_true_raw > 0.90:
        marg_pc_remove = 1    
    elif h_pc_true_raw > 0.85 or a_pc_true_raw > 0.85:
        marg_pc_remove = 0.85
    elif h_pc_true_raw > 0.80 or a_pc_true_raw > 0.80:
        marg_pc_remove = 0.7
    elif h_pc_true_raw > 0.75 or a_pc_true_raw > 0.75:
        marg_pc_remove = 0.6
    elif h_pc_true_raw > 0.6 or a_pc_true_raw > 0.6:
        marg_pc_remove = 0.5
    elif h_pc_true_raw > 0.50 or a_pc_true_raw > 0.50:
        marg_pc_remove = 0.4

    if h_pc_true_raw >= a_pc_true_raw:
        h_pc_true = h_pc_true_raw * (((marg_pc_remove * ((margin - 1) * 100)) / 100) + 1)
        d_pc_true = d_pc_true_raw - ((h_pc_true - h_pc_true_raw) * 0.4) 
        if h_pc_true + d_pc_true > 1:               # if greater than 100% (makes away price negative)
            d_pc_true = 1 - h_pc_true - 0.0025
            a_pc_true = 0.0025                              # make away price default 400
        a_pc_true = 1 - h_pc_true - d_pc_true
        if a_pc_true <= 0:
            a_pc_true = 0.0025
    else:
        a_pc_true = a_pc_true_raw * (((marg_pc_remove * ((margin - 1) * 100)) / 100) + 1)
        d_pc_true= d_pc_true_raw - ((a_pc_true - a_pc_true_raw) * 0.4)
        if a_pc_true + d_pc_true > 1:
            d_pc_true = 1 - a_pc_true - 0.0025
            h_pc_true = 0.0025
        h_pc_true = 1 - a_pc_true - d_pc_true
        if h_pc_true <= 0:
            h_pc_true = 0.0025

    h_pc_true = round(h_pc_true, 2)
    d_pc_true = round(d_pc_true, 2)
    a_pc_true = round(a_pc_true, 2)

    return (float(h_pc_true), float(d_pc_true), float(a_pc_true))

# ---------------------------------------------------------------------

@st.cache_resource
def get_odds(fixture_id, market_id, bookmakers):

    if not st.secrets:
        load_dotenv()
        API_KEY = os.getenv("API_KEY_FOOTBALL_API")

    else:
        # Use Streamlit secrets in production
        API_KEY = st.secrets["rapidapi"]["API_KEY_FOOTBALL_API"]


    url = "https://api-football-v1.p.rapidapi.com/v3/odds"
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    querystring = {
        "fixture": fixture_id,
        "bet": market_id,
        "timezone": "Europe/London"
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    if 'response' in data and data['response']:
        odds_dict = {
            'Fixture ID': fixture_id,
            'Home Win': None,
            'Draw': None,
            'Away Win': None,
            # 'Over 2.5': None,                 # goals not needed in ml model for corners. v.big odds on might not have 2.5
            # 'Under 2.5': None,                # line which will cause the whole match to be dropped in later script
        }

        # Loop through bookmakers
        for bookmaker_data in data['response'][0].get('bookmakers', []):
            if str(bookmaker_data['id']) in bookmakers:
                # Loop through each market (bet) offered by the bookmaker
                for bet_data in bookmaker_data['bets']:
                    if bet_data['id'] == int(market_id):  # Ensure it's the selected market
                        # Extract the outcomes (selections) and their corresponding odds
                        for value in bet_data['values']:
                            selection = value['value']
                            odd = value['odd']
                            
                            # Assign the odds based on the selection type
                            if selection == 'Home':
                                odds_dict['Home Win'] = odd
                            elif selection == 'Draw':
                                odds_dict['Draw'] = odd
                            elif selection == 'Away':
                                odds_dict['Away Win'] = odd
                            # elif selection == 'Over 2.5':     
                            #     odds_dict['Over 2.5'] = odd     
                            # elif selection == 'Under 2.5':
                            #     odds_dict['Under 2.5'] = odd


        # Create a DataFrame with a single row containing all the odds
        odds_df = pd.DataFrame([odds_dict])
        return odds_df
        
    # Return empty DataFrame if no data is found
    return pd.DataFrame()

# ---------------------------------------------------

# reconfi to pass lam0 and 1H goal split as args after user input
def calc_prob_matrix(supremacy, goals_exp, max_goals, draw_lambda, f_half_perc): # lam0 adjusts the draw likelihood for higher draw leagues

    # Calculate Home and Away Goals Expected Full Time
    hg = round(goals_exp / 2 + (0.5 * supremacy), 2)
    ag = round(goals_exp / 2 - (0.5 * supremacy), 2)

    # Calculate Home and Away Goals Expected 1H
    hg1h = round((hg / 100) * f_half_perc, 2)
    ag1h = round((ag / 100) * f_half_perc, 2)

    s_half_perc = 100 - f_half_perc

    # Calculate Home and Away Goals Expected 2H
    hg2h = round((hg / 100) * s_half_perc , 2)
    ag2h = round((ag / 100) * s_half_perc , 2)

    # Function to calculate Bivariate Poisson probability matrix
    def bivariate_poisson(lam0, lam1, lam2, max_goals):
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                s = 0
                for m in range(0, min(i, j) + 1):
                    term = (lam0**m) * (lam1**(i-m)) * (lam2**(j-m)) / (math.factorial(m) * math.factorial(i-m) * math.factorial(j-m))
                    s += term
                matrix[i, j] = np.exp(-(lam0 + lam1 + lam2)) * s
        matrix /= matrix.sum()  # Normalize
        return matrix

    # Set shared lambda (lam0) to induce correlation
    # You can adjust lam0 depending on how correlated you want the teams to be
    # so to increase draw scoreline percentages (make more likely) - increase lambda values and vice versa 
    lam0_ft = draw_lambda
    lam0_1h = lam0_ft / 2
    lam0_2h = lam0_ft / 2

    # Calculate lambda1 and lambda2 for each period
    lam1_ft = hg - lam0_ft
    lam2_ft = ag - lam0_ft

    lam1_1h = hg1h - lam0_1h
    lam2_1h = ag1h - lam0_1h

    lam1_2h = hg2h - lam0_2h
    lam2_2h = ag2h - lam0_2h

    # Calculate probability matrices
    prob_matrix_ft = bivariate_poisson(lam0_ft, lam1_ft, lam2_ft, max_goals)
    prob_matrix_1h = bivariate_poisson(lam0_1h, lam1_1h, lam2_1h, max_goals)
    prob_matrix_2h = bivariate_poisson(lam0_2h, lam1_2h, lam2_2h, max_goals)

    # Tiny manual adjustment just on the FT matrix
    prob_matrix_ft[1,1] *= 1.09  # Boost 1-1 % (decrease odds)
    prob_matrix_ft[0,0] *= 0.98 # Lower 0-0 
    prob_matrix_ft[2,2] *= 1.01 # Lower 2-2 

    # Then normalize the matrix again
    prob_matrix_ft /= prob_matrix_ft.sum()

    return prob_matrix_ft, prob_matrix_1h, prob_matrix_2h, hg, ag

# --------------------------

def calculate_expected_team_goals_from_1x2():

    def odds_to_probs(odds):
        raw_probs = np.array([1/o for o in odds])
        return raw_probs / raw_probs.sum()

    # --- Poisson Matrix Generator ---
    def poisson_matrix(lam_home, lam_away, max_goals=6):
        matrix = np.zeros((max_goals+1, max_goals+1))
        for i in range(max_goals+1):
            for j in range(max_goals+1):
                matrix[i, j] = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
        return matrix

    # --- Outcome Probability Calculator ---
    def outcome_probs(matrix):
        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        
        total_goals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        over_2_5 = matrix[total_goals > 2.5].sum()
        under_2_5 = matrix[total_goals <= 2.5].sum()
        
        return [home_win, draw, away_win, over_2_5, under_2_5]

    # --- Loss Function for Optimization ---
    def loss(params, target_probs):
        lam_home, lam_away = params
        matrix = poisson_matrix(lam_home, lam_away)
        model_probs = outcome_probs(matrix)
        return sum((m - t)**2 for m, t in zip(model_probs, target_probs))

    # Initialize session state if not already set
    if "match_suprem" not in st.session_state:
        st.session_state.match_suprem = 0.0
    if "total_match_gls" not in st.session_state:
        st.session_state.total_match_gls = 2.5

    # --- UI ---
    st.subheader("Generate Implied Home/Away Goal Expectation from Market Odds")

    st.markdown("Input market odds below:")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Home Win Odds", value=st.session_state.get("home_odds", 2.10), key="home_odds")
        st.number_input("Draw Odds", value=st.session_state.get("draw_odds", 3.20), key="draw_odds")
        st.number_input("Away Win Odds", value=st.session_state.get("away_odds", 3.60), key="away_odds")
    with col2:
        st.number_input("Over 2.5 Odds", value=st.session_state.get("over_2_5_odds", 1.90), key="over_2_5_odds")
        st.number_input("Under 2.5 Odds", value=st.session_state.get("under_2_5_odds", 1.90), key="under_2_5_odds")


    # --- Run Estimation ---
    if st.button("Estimate Goals", key="estimate_btn_1x2"):
        # Step 1: Implied probabilities from odds
        prob_1x2 = odds_to_probs([
            st.session_state.home_odds,
            st.session_state.draw_odds,
            st.session_state.away_odds
        ])
        prob_ou = odds_to_probs([
            st.session_state.over_2_5_odds,
            st.session_state.under_2_5_odds
        ])
        target_probs = list(prob_1x2) + list(prob_ou)

        # Step 2: Optimize
        initial_guess = [1.2, 1.2]
        bounds = [(0.1, 5), (0.1, 5)]
        res = minimize(loss, initial_guess, args=(target_probs,), bounds=bounds)
        lam_home, lam_away = res.x

        match_suprem = round(lam_home - lam_away, 2)
        total_match_gls = round(lam_home + lam_away, 2)

        # Store results in session state
        st.session_state.match_suprem = match_suprem
        st.session_state.total_match_gls = total_match_gls

        # st.session_state.expander_open = True  # Ensure expander stays open

        # Step 3: Display results
        # with st.expander('Expected Goals Estimator from 1X2', expanded=True):
        st.subheader("🎯 Estimated Expected Goals")
        st.write(f"**Home Team Expected Goals:** `{lam_home:.2f}`")
        st.write(f"**Away Team Expected Goals:** `{lam_away:.2f}`")
        st.write(f"**Match Supremacy:** `{match_suprem:.2f}`")
        st.write(f"**Total Goals:** `{total_match_gls:.2f}`")

    # Always return something (even if button not clicked)
    return st.session_state.get("match_suprem", 0.0), st.session_state.get("total_match_gls", 2.5)

        # # Optional: Show matrix
        # st.subheader("📊 Score Probability Matrix")
        # matrix = poisson_matrix(lam_home, lam_away)
        # df = pd.DataFrame(matrix, index=[f"{i} goals" for i in range(matrix.shape[0])],
        #                 columns=[f"{j} goals" for j in range(matrix.shape[1])])
        # st.dataframe(df.style.format("{:.3f}"))

# -------------------------------------------------------------------------------------------------------------------
# Same function as above but without Streamlit UI
# takes match and goals odds as args and returns hg and ag
def calculate_expected_team_goals_from_1x2_refined(home_odds, draw_odds, away_odds, over_2_5_odds, under_2_5_odds):
    def odds_to_probs(odds):
        raw_probs = np.array([1 / o for o in odds])
        return raw_probs / raw_probs.sum()

    def poisson_matrix(lam_home, lam_away, max_goals=7):
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                matrix[i, j] = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
        return matrix

    def outcome_probs(matrix):
        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        
        total_goals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        over_2_5 = matrix[total_goals > 2.5].sum()
        under_2_5 = matrix[total_goals <= 2.5].sum()
        
        return [home_win, draw, away_win, over_2_5, under_2_5]

    def loss(params, target_probs):
        lam_home, lam_away = params
        matrix = poisson_matrix(lam_home, lam_away)
        model_probs = outcome_probs(matrix)
        return sum((m - t) ** 2 for m, t in zip(model_probs, target_probs))

    # Step 1: Convert odds to target probabilities
    prob_1x2 = odds_to_probs([home_odds, draw_odds, away_odds])
    prob_ou = odds_to_probs([over_2_5_odds, under_2_5_odds])
    target_probs = list(prob_1x2) + list(prob_ou)

    # Step 2: Run optimization
    initial_guess = [1.2, 1.2]
    bounds = [(0.1, 5), (0.1, 5)]
    res = minimize(loss, initial_guess, args=(target_probs,), bounds=bounds)
    lam_home, lam_away = res.x

    lam_home, lam_away = round(lam_home, 2), round(lam_away, 2)

    return lam_home, lam_away


###################  FUNCTION: GENERATE_MARGINATED_ODDS_WITH_FAV_LOCK()  ############################
# import numpy as np

'''
Returns marginated odds given initial true odds and margin (eg 1.2 5.0). Function provides for short price handling conditions where we want less margin
        ascribed to the shorter price. so short prices, once marginated, are still close to the true - 
        so true 1.14 > marginated 1.11. remaining outcomes normalized
Args: 1.true odds - takes any number (eg wdw 2,3.5,3.75 or ou 1.8,1.9). 
      Input the first arg as an iterable eg - generate_marginated_odds_with_fav_lock([1.5, 3.5, 3.75], 1.08)
      2. margin - final margin to be offered (eg 1.08)
Returns: margined odds as numpy array (eg [1.17,4.51]). To unpack: fav, dog = generate_marginated_odds_with_fav_lock([1.2, 5.0], 1.08)
'''

# function to return marginated odds given initial true odds and margin UI variable (eg 1.08). Function provides for short price handling conditions
# so short prices, once marginated, are still close to the true - so true 1.14 > marginated 1.11. remaining outcomes normalized.
# takes any number of true odds outcomes as args eg, match odds (3) and ov/un (2). Input as iterable eg
# 
def generate_marginated_odds_with_fav_lock(true_odds, margin):
    true_odds = np.array(true_odds, dtype=float)
    true_probs = 1 / true_odds
    adjusted_probs = true_probs.copy()

    # Rules for adjustments based on short-priced favorite
    def get_adjustment(odd):
        if odd < 1.05:
            return 0.01
        elif odd < 1.1:
            return 0.015
        elif odd < 1.2:
            return 0.02
        elif odd < 1.3:
            return 0.025
        elif odd <= 1.4:
            return 0.03
        return 0.0

    # Identify the lowest odds
    min_odd_index = np.argmin(true_odds)
    min_odd = true_odds[min_odd_index]
    adj = get_adjustment(min_odd)

    if adj > 0:
        # Lock adjustment on the shortest-priced outcome
        adjusted_probs[min_odd_index] += adj
        locked_prob = adjusted_probs[min_odd_index]

        # Distribute remaining margin to other outcomes proportionally
        other_indices = [i for i in range(len(true_odds)) if i != min_odd_index]
        remaining_true_probs = true_probs[other_indices]
        remaining_margin = margin - locked_prob

        scaling_factor = remaining_margin / remaining_true_probs.sum()
        adjusted_probs[other_indices] = remaining_true_probs * scaling_factor
    else:
        # Standard normalization if no short-priced condition is triggered
        scaling_factor = margin / true_probs.sum()
        adjusted_probs = true_probs * scaling_factor

    # Convert back to odds
    margined_odds = 1 / adjusted_probs
    return margined_odds.round(2)


    # ------------------------ GENERATE FST FROM FILTERED_DF ---------------------------
@st.cache_data
def generate_team_fst(filtered_df, selected_year): # added selected_year argument to determine if current season or not for downgrade adjustments
    
    # Increase/decrease rating based on: 
    # The number of minutes played
    max_mins = filtered_df['Minutes'].max()
    # Proportion of goals scored
    total_goals = filtered_df['Goals'].sum()
    # Proportion of assists created
    total_assists = filtered_df['Assists'].sum()

    # Define a function to apply rating adjustments row-by-row - GOALKEEPERS
    def mins_played_rating_boost_goalkeeper(row):
        if row['Minutes'] > max_mins * 0.95:
            return row['Rating'] * 1.05
        elif row['Minutes'] > max_mins * 0.85:
            return row['Rating'] * 1.03
        elif row['Minutes'] < max_mins * 0.7:
            return row['Rating'] * 0.96
        elif row['Minutes'] < max_mins * 0.6:
            return row['Rating'] * 0.92
        else:
            return row['Rating']
        
        # Define a function to apply rating adjustments row-by-row - OUTFIELDERS
    def mins_played_rating_boost_outfield_def(row):
        if row['Minutes'] > max_mins * 0.95:
            return row['Rating'] * 1.03
        elif row['Minutes'] > max_mins * 0.85:
            return row['Rating'] * 1.015
        elif row['Minutes'] > max_mins * 0.7:
            return row['Rating'] * 0.985
        elif row['Minutes'] > max_mins * 0.6:
            return row['Rating'] * 0.97 
        elif row['Minutes'] > max_mins * 0.5:
            return row['Rating'] * 0.93 
        elif row['Minutes'] > max_mins * 0.40:
            return row['Rating'] * 0.91 
        elif row['Minutes'] > max_mins * 0.30:
            return row['Rating'] * 0.89 
        else:
            return row['Rating'] * 0.87
        
        # Define a function to apply rating adjustments row-by-row - OUTFIELDERS
    def mins_played_rating_boost_outfield_non_def(row):
        if row['Minutes'] > max_mins * 0.95:
            return row['Rating'] * 1.045
        elif row['Minutes'] > max_mins * 0.85:
            return row['Rating'] * 1.025
        elif row['Minutes'] > max_mins * 0.7:
            return row['Rating'] * 0.985
        elif row['Minutes'] > max_mins * 0.6:
            return row['Rating'] * 0.96 
        elif row['Minutes'] > max_mins * 0.5:
            return row['Rating'] * 0.93 
        elif row['Minutes'] > max_mins * 0.40:
            return row['Rating'] * 0.91 
        elif row['Minutes'] > max_mins * 0.30:
            return row['Rating'] * 0.89 
        else:
            return row['Rating'] * 0.87
        
        
    # Define a function to apply rating adjustments row-by-row - GOALS SCORED (increase player rating if they score ALOT of the team goals)
    def goals_scored_rating_boost(row):
        if row['Goals'] / total_goals > 0.6: # if player scored more than 60% of team goals
            return row['Rating'] * 1.06
        elif row['Goals'] / total_goals > 0.5:
            return row['Rating'] * 1.045
        elif row['Goals'] / total_goals > 0.4:
            return row['Rating'] * 1.03
        elif row['Goals'] / total_goals > 0.3:
            return row['Rating'] * 1.015
        else:
            return row['Rating']
        
    # Define a function to apply rating adjustments row-by-row - ASSISTS (increase player rating if they assist ALOT of the team goals)
    def assists_rating_boost(row):
        if row['Assists'] / total_assists > 0.6: # if player assists more than 60% of team goals
            return row['Rating'] * 1.04
        elif row['Assists'] / total_assists > 0.5:
            return row['Rating'] * 1.03
        elif row['Assists'] / total_assists > 0.4:
            return row['Rating'] * 1.02
        elif row['Assists'] / total_assists > 0.3:
            return row['Rating'] * 1.01
        else:
            return row['Rating']
        

    # Apply mins_played_rating_boost functions depending on position. Defenders were being made k+ too easily so outfield
    # non-defenders function created
    filtered_df['Rating'] = round(filtered_df.apply(
        lambda row: (
            mins_played_rating_boost_goalkeeper(row)
            if row['Position'] == 'Goalkeeper' else
            mins_played_rating_boost_outfield_def(row)
            if row['Position'] == 'Defender' else
            mins_played_rating_boost_outfield_non_def(row)
        ), axis=1
    ), 1)


    # goals_scored_rating_boost
    filtered_df['Rating'] = round(filtered_df.apply(
        lambda row: goals_scored_rating_boost(row), axis=1),1) 
    
    # assists_rating_boost
    filtered_df['Rating'] = round(filtered_df.apply(
        lambda row: assists_rating_boost(row), axis=1),1) 


    df_r = filtered_df[['Player', 'Rating', 'Position', 'Player_id']]

    # Calculate mean and standard deviation of 'Rating'
    mean_rating = df_r['Rating'].mean()
    std_rating = df_r['Rating'].std()

    # Add the 'Rating STD' column
    df_r = df_r.copy()
    df_r.loc[:, 'Rating St.Dev'] = (df_r['Rating'] - mean_rating) / std_rating

    # Define a function to classify the value based on 'Rating STD'
    def classify_rating(std_value):
        if std_value > 2.2:
            return 'Key +'
        elif std_value > 1.6:
            return 'Key'
        elif std_value > 1.15:
            return 'Imp +'
        elif std_value > 0.8:
            return 'Imp'
        elif std_value > 0.4:
            return 'Reg +'
        elif std_value > -0.2:
            return 'Reg'
        elif std_value > -1.0:
            return 'Sub +'
        else:
            return 'Sub'

    # Apply the classification function to create the 'Value' column
    df_r = df_r.copy()
    df_r.loc[:,'Classification'] = df_r['Rating St.Dev'].apply(classify_rating)
    df_r = df_r.drop(['Rating St.Dev'], axis=1)
    df_r.index = df_r.index +1


    # Assign variable downgrades depending on month, if july to oct, if nov to jan, if feb to june

    # Get the current month
    month = datetime.now().month

    # Define adjustment sets by season
    adjustments = {
        "winter": {  # Oct –January inclusive
            "Key +": -12, "Key": -9, "Imp +": -6, "Imp": -4, "Reg +": -2, "Reg": -1
        },
        "autumn": {  # Aug–Sept inclusive
            "Key +": -9, "Key": -6, "Imp +": -4, "Imp": -2, "Reg +": -1, "Reg": 0
        },
        "spring": {  # February–July inclusive
            "Key +": -15, "Key": -12, "Imp +": -9, "Imp": -6, "Reg +": -3, "Reg": -1
        }
    }

    # Determine which set to use based on month
    if month in [10, 11, 12, 1]:
        current_adj_dict = adjustments["winter"]
    elif month in range(8, 10):     # August–September
        current_adj_dict = adjustments["autumn"]
    else:                           # February–July 
        current_adj_dict = adjustments["spring"]

    # if selected_season is not current_year, then use spring adjustments
    if selected_year != CURRENT_SEASON_FST_FORMAT:
        current_adj_dict = adjustments["spring"]


    # Define a function to classify the value based on 'Rating STD'
    def classify_dg(classification):
        return current_adj_dict.get(classification, 0)   # Return the adjustment if found, otherwise 0

    # st.write('477', df_r)
    # Apply the classification function to create the 'Downgrade' column
    df_r = df_r.copy()
    df_r.loc[:, 'Downgrade'] = df_r['Classification'].apply(classify_dg)

    df_squad = df_r.copy()

    # get the top 1 GK
    df_goalkeepers = df_r[df_r['Position'] == 'Goalkeeper'].nlargest(1, 'Rating')

    # Get the top 4 Defenders
    df_defenders = df_r[df_r['Position'] == 'Defender'].nlargest(4, 'Rating')

    # Get the top 5 Midfielders
    df_midfielders = df_r[df_r['Position'] == 'Midfielder'].nlargest(5, 'Rating')

    # Get the top 3 Attackers
    df_attackers = df_r[df_r['Position'] == 'Attacker'].nlargest(3, 'Rating')

    # 13 players selected - now concat and filter best 11 - this is so teams with fewer attackers still return 11 players, 10 outfield
    # Concatenate the three results into a single dataframe
    df_fst_outfield = pd.concat([df_defenders, df_midfielders, df_attackers])
    df_fst_outfield = df_fst_outfield.nlargest(10, 'Rating')

    # now tag on gk
    df_fst = pd.concat([df_fst_outfield, df_goalkeepers], axis = 0)
                           
    df_fst['Rating'] = round(df_fst['Rating'], 2)

    # Custom sort order for Position: Defender > Midfielder > Attacker
    position_order = ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']
    # Apply the categorical type to Position for the final dataframe
    df_fst['Position'] = pd.Categorical(df_fst['Position'], categories=position_order, ordered=True)

    # Now sort by Position according to the custom order
    df_fst = df_fst[['Player', 'Downgrade', 'Position', 'Rating', 'Classification', 'Player_id']]
    df_fst = df_fst.sort_values(by=['Position', 'Rating'], ascending=[True, False])
    # df_fst.loc[:, 'Position'] = pd.Categorical(df_fst['Position'], categories=position_order, ordered=True)
    df_fst.set_index('Player', inplace=True)

    return df_fst, df_squad

# -----------------------------------------------------------------------------

# Function to return team injuries/absences df for last 2 weeks passing args team 'api_id' and current season
@st.cache_resource
def get_injuries_by_team(api_id, current_season):

    load_dotenv()
    API_KEY = os.getenv('API_KEY_FOOTBALL-API')

    url = "https://api-football-v1.p.rapidapi.com/v3/injuries"
    querystring = {"season": current_season, "team": api_id}

    headers = {
        "X-RapidAPI-Key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    try:
        data = response.json()
    except ValueError:
        return pd.DataFrame()  # Return empty DataFrame if JSON parsing fails
    
    # If response is empty or invalid, return empty DataFrame
    if not data or "response" not in data or not data["response"]:
        return pd.DataFrame(columns=["Player Name", "Reason", "Date"])

    # Get the current date and calculate the date two weeks ago
    two_weeks_ago = datetime.utcnow() - timedelta(weeks=2)

    # Dictionary to store the first occurrence of each player
    injury_dict = {}

    for injury in data.get('response', []):
        player_name = injury['player']['name']
        reason = injury['player']['reason']
        fixture_date = injury['fixture']['date']  # e.g., '2024-08-16T19:00:00+00:00'
        
        # Convert fixture date to datetime
        fixture_datetime = datetime.strptime(fixture_date, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        
        # Only include injuries in the last 2 weeks and avoid duplicates
        if fixture_datetime >= two_weeks_ago and player_name not in injury_dict:
            injury_dict[player_name] = {"Reason": reason, "Date": fixture_datetime.date()}  # Store date as well


    # Convert to DataFrame
    df_injuries = pd.DataFrame.from_dict(injury_dict, orient="index").reset_index()
    df_injuries.columns = ["Player Name", "Reason", "Date"]  # Rename columns properly

    return df_injuries

# -------------------------------------------------------------------

# Function to return transfers/loans in the last year
@st.cache_resource
def get_transfers_by_team(api_id, current_season):

    load_dotenv()
    API_KEY = os.getenv('API_KEY_FOOTBALL-API')

    url = "https://api-football-v1.p.rapidapi.com/v3/transfers"
    querystring = {"team": api_id}  # Replace "33" with your team API_ID if needed

    headers = {
        "X-RapidAPI-Key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    # Convert the response to JSON
    data = response.json()

    # Check if the response contains transfers data
    if 'response' in data and data['response']:
        # Get the current date and calculate the date one year ago and one year in the future
        one_year_ago = datetime.utcnow() - timedelta(days=365)
        one_year_in_future = datetime.utcnow() + timedelta(days=365)
        
        # List to store filtered transfers
        transfer_list = []
        
        # Iterate over the response and filter transfers by date
        for player_data in data['response']:
            player_name = player_data['player']['name']
            
            # Iterate over the player's transfers
            for transfer in player_data['transfers']:
                transfer_date = transfer['date']
                
                try:
                    # Convert the date string to datetime
                    transfer_datetime = datetime.strptime(transfer_date, "%Y-%m-%d")
                    
                    # Check if the year is within the acceptable range (CURRENT_SEASON ± 1 year)
                    transfer_year = str(transfer_datetime.year)  # Convert year to string for comparison
                    if current_season == transfer_year or current_season == str(int(transfer_year) + 1) or current_season == str(int(transfer_year) - 1):
                        # Filter the transfers that occurred in the last year or future one year window
                        if one_year_ago <= transfer_datetime <= one_year_in_future:
                            # Extract the relevant data
                            transfer_type = transfer['type']
                            team_in = transfer['teams']['in']['name']
                            team_out = transfer['teams']['out']['name']
                            # Add player name to the data
                            transfer_list.append([player_name, transfer_date, transfer_type, team_in, team_out])
                except ValueError:
                    # Handle the case of invalid date format or incorrect date
                    continue
        
        # Create DataFrame from the filtered transfer data
        df_transfers = pd.DataFrame(transfer_list, columns=["Player Name", "Date", "Type", "Team In", "Team Out"])

        return df_transfers
