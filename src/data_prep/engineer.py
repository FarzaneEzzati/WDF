"""
engineer_data.py

Applies parametric/non-parametric tests on the data to identify the homogeneity of variance.
Generates time-aware features to enrich the data with time-related feature.

Usage:
    from src.data.fetch_integrate_data import get_integrated_data
    from src.data.engineer_data import test_and_engineer
    
    df = get_integrated_data()
    df = test_and_engineer(df)
    df now goes to the prediction stage.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)

def test_and_engineer(
    table: pd.DataFrame, 
    p_value_thres: float = 0.05
) -> pd.DataFrame:
    logger.info(f"Variance Homogeneity and Analysis of The Data...")
    # Group-Wise Statistical Analysis
    # Check normality within each store type (p-value threshold = 0.05).
    ingroup_norm = True
    for _, group in table.groupby(by='Store'):
        _, p_value = stats.shapiro(group['Weekly_Sales'])
        if p_value < 0.05:
            ingroup_norm = False


    # Check for homogeneity of variance across stores (p-value threshold = 0.05)
    _, p_value = stats.levene(*[group['Weekly_Sales'].values for store, group in table.groupby('Store')])
    logger.info(f'Store Levene\'s p-value: {p_value:0.4f}')
    if p_value < p_value_thres:
        intergroup_homo = False
    else:
        intergroup_homo = True


    # Specify the type of analysis
    parametric = all([ingroup_norm, intergroup_homo])
    if parametric:
        logger.info('Normality + Homogeneity passed. Run parameteric test.')
        table = execute_parametric_test(table)
    else:
        logger.info('Normality + Homogeneity failed. Run non-parameteric test.')
        table = execute_nonparametric_test(table)


    # Generate time-aware features
    logger.info('Generating tim-aware features.')
    table = generate_time_features(table)
    return table
       


def execute_parametric_test(table: pd.DataFrame) -> pd.DataFrame:
    pass

def execute_nonparametric_test(table: pd.DataFrame) -> pd.DataFrame:
    # Check for homogeneity among groups
    h_stat, p_value = stats.kruskal(*[group['Weekly_Sales'] for _ , group in table.groupby('Store')])
    if p_value < 0.05:
        print(f'Kruskal p_value={p_value}')


    # Feature Correlation and Statistical Significance
    # Normalize sales per store (mean=0, std=1)
    table['Sales_Normalized'] = table.groupby('Store')['Weekly_Sales'].transform(lambda x: (x - x.mean()) / x.std())
    # Specify the columns to evaluate
    features_to_evaluate = ['Fuel_Price', 'household_dept_srvc_ratio',
                    'Temperature', 'Unemployment', 
                    'effective_federal_fund', 'personal_consum_expend', 
                    'producer_price_index', 'personal_saving_rate', 
                    'advanced_retail_sales', 'consumer_sentiment',
                    'Holiday_Flag'
                    ]
    # Run the correlation analysis for each pair of sales and numeric features.
    results = []
    alpha = 0.05
    for feature in features_to_evaluate:
        r_stat, p_stat = stats.spearmanr(table['Sales_Normalized'], table[feature])
        results.append({
            'feature': feature,
            'corr': r_stat,
            'p_value': p_stat,
            'is_significant': p_stat < alpha
        })

    corr_table = pd.DataFrame(results).sort_values(by='p_value', ascending=True)
    non_significant_features = corr_table[corr_table['is_significant'] == False]['feature'].to_list()
    logger.info(f'Correlated but non-significant numeric features: {non_significant_features}')

    table.drop('Sales_Normalized', axis=1, inplace=True)
    return table


def generate_time_features(table: pd.DataFrame) -> pd.DataFrame:
    # Seasonality
    # Covert date to datetime type, then get the week number of each data point
    table['Date'] = pd.to_datetime(table['Date'])
    table = table.sort_values(['Store', 'Date']).reset_index(drop=True)
    week_series = table['Date'].dt.isocalendar().week

    # Convert the week into a smooth 360-degree cyclical coordinates
    table['sin_week'] = np.sin(2 * np.pi * week_series / 52.1429)
    table['cos_week'] = np.cos(2 * np.pi * week_series / 52.1429)
    logger.info('Seasonal time features using sine/cosine completed.')

    # Rolling Window Aggregation (for 4 previous persiods, NaN filled with mean)
    table['rolling_4_mean'] = table.groupby(['Store'])['Weekly_Sales'].transform(lambda x: x.shift(1)).rolling(window=4).mean()
    # Backfilling the first 4 rows without the mean of previosu 4 weeks
    table['rolling_4_mean'] = table.groupby(['Store'])['rolling_4_mean'].bfill()
    logger.info('Rolling window aggregation for past average sales completed.')

    return table


