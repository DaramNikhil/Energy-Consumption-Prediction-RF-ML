import pandas as pd
import numpy as np


def data_cleaning(energy_data, weather_data):
    energy_data['time'] = pd.to_datetime(energy_data['time'])
    weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'])

    energy_data = energy_data.rename(columns={'time': 'datetime'})
    weather_data = weather_data.rename(columns={'dt_iso': 'datetime'})

    merged_data = pd.merge(energy_data, weather_data, on='datetime', how='left')

    selected_features = [
        'generation fossil gas',
        'generation nuclear',
        'generation wind onshore',
        'generation solar',
        'temp',
        'pressure',
        'humidity',
        'wind_speed',
        'clouds_all',
        'total_generation'
    ]

    merged_data['total_generation'] = merged_data[[col for col in merged_data.columns if 'generation' in col]].sum(axis=1)
    merged_data['energy_demand'] = merged_data['total load actual']
    final_data = merged_data[selected_features + ['energy_demand']].copy()
    final_data = final_data.fillna(final_data.mean())
    return final_data
