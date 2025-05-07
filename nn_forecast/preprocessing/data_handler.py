import pandas as pd
import numpy as np

from nn_forecast.utils.logging_custom import get_logger
from nn_forecast.consts import dirs


class DataHandler:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
    def concat_files(self):
        self.logger.info("Concatenating files")

        total_load = []
        for file in dirs.RAW_DATA_PATH.glob("Total*.csv"):
            df = pd.read_csv(file)
            total_load.append(df)

        combined_load = pd.concat(total_load, ignore_index=True)

        #Including only first 2 columns from data
        combined_load = combined_load.iloc[:,[0,2]]
        combined_load.columns = ['time', 'total_load']

        #Changing type to datetime
        combined_load['time'] = pd.to_datetime(combined_load['time'].str.split(' - ').str[0], format='%d.%m.%Y %H:%M')

        combined_load.set_index('time', inplace=True)
        combined_load = combined_load.resample('H').mean()
        combined_load.reset_index(inplace=True)

        temperature_data = pd.read_csv(dirs.RAW_DATA_PATH/ "temperature_by_location2.csv")

        temperature_data.columns = ['time', 'latitude', 'longitude', 'temperature']
        temperature_data['time'] = pd.to_datetime(temperature_data['time'], format='mixed')

        temperature_data.set_index('time', inplace=True)
        temperature_data = temperature_data.resample('H').mean()
        temperature_data.reset_index(inplace=True)

        final_data = pd.merge(combined_load, temperature_data, on='time', how='inner')

        final_data.to_csv(dirs.DATA_PATH/ 'combined_data.csv', index=False)

    def preprocess(self):
        self.logger.info("Preprocessing combined data")

        df = pd.read_csv(dirs.DATA_PATH / 'combined_data.csv')

        df['time'] = pd.to_datetime(df['time'])

        df = self.create_features(df)

        df = self.create_cyclicity(df)

        df = self.previous_load_and_temp(df)
        df.to_csv(dirs.DATA_PATH / 'preprocessed_data.csv', index=False)
        return df

    @staticmethod
    def create_features(df):
        df['date'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.day_name()
        return df

    @staticmethod
    def create_cyclicity(df):
        day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                       'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        # dni tygodnia
        df['day_of_week_num'] = df['day_of_week'].map(day_mapping)
        df['day_of_week_sin'] = np.sin(df['day_of_week_num'] * 2 * np.pi / 7)
        df['day_of_week_cos'] = np.cos(df['day_of_week_num'] * 2 * np.pi / 7)

        # godziny
        df['hour_sin'] = np.sin(df['hour'] * 2 * np.pi / 24)
        df['hour_cos'] = np.cos(df['hour'] * 2 * np.pi / 24)

        # dni roku
        df['day_of_year'] = df['time'].dt.dayofyear.astype(int)
        df['days_in_year'] = df['time'].dt.is_leap_year.map(lambda x: 366 if x else 365)

        df['day_of_year_sin'] = np.sin(df['day_of_year'] * 2 * np.pi / df['days_in_year'])
        df['day_of_year_cos'] = np.cos(df['day_of_year'] * 2 * np.pi / df['days_in_year'])

        return df

    @staticmethod
    def previous_load_and_temp(df):
        for i in [1, 2, 3, 22, 23, 24, 25, 26]:
            df[f'load-{i}'] = df['total_load'].shift(i)
            df[f'temp_t-{i}'] = df['temperature'].shift(i)
        df['mean_t_3'] = (df[['temp_t-1', 'temp_t-2', 'temp_t-3']].fillna(method='ffill')).mean(axis=1)
        df['mean_t_5'] = (
            df[['temp_t-22', 'temp_t-23', 'temp_t-24', 'temp_t-25', 'temp_t-26', ]].fillna(method='ffill')).mean(axis=1)

        for i in range(1, 24):
            df[f'next_load_{i}'] = df['total_load'].shift(-i)

        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        return df






