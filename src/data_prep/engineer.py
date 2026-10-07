"""
engineer.py

This script engineers the features in the integrated data.

generate_time_features: Generates time-aware features to enrich the data with time-related feature.

Usage:
    from src.data_prep import generate_time_features
    
    df = get_integrated_data()
    df = generate_time_features(df)
    
    df now goes to the prediction stage.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


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


