from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from nn_forecast.utils.logging_custom import get_logger
from nn_forecast.consts import dirs

class ModularSystem:
    def __init__(self,df)-> None:
        self.logger = get_logger(self.__class__.__name__)
        self.df = df

    def prepare_data(self):
        features = ['load-1', 'load-2', 'load-3', 'load-22', 'load-23', 'load-24', 'load-25', 'load-26',
                    'mean_t_3', 'mean_t_5',
                    'day_of_week_sin', 'day_of_week_cos',
                    'day_of_year_sin', 'day_of_year_cos']
        X = self.df[features].copy()
        y = self.df['total_load'].copy()
        return X, y

    def split_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def modular_network(self):
        X, y = self.prepare_data()

        # Normalization
        other_scaler = MinMaxScaler(feature_range=(0, 1))
        temp_scaler = MinMaxScaler(feature_range=(-1, 1))
        for temp in [f'mean_t_{t}' for t in [3, 5]]:
            X[temp] = temp_scaler.fit_transform(X[[temp]])

        for col in [f'load-{t}' for t in [1, 2, 3, 22, 23, 24, 25, 26]]:
            X[col] = other_scaler.fit_transform(X[[col]])

        X = np.array(X)
        y = np.array(y)

        models = {}
        results = []
        vector = []
        mape_vec = []
        temp = []
        y_t = []

        hours = self.df['hour'].values
        days = self.df['date'].values
        starting_hour = np.where(hours == 11)[0][0]

        L_i_1 = X[starting_hour][0]
        L_i_2 = X[starting_hour][1]
        L_i_3 = X[starting_hour][2]

        for hour in range(24):
            filter = hours == hour
            X_hour = X[filter].copy()
            y_hour = y[filter]
            days_hour = days[filter]

            X_hour[hour][0] = L_i_1
            X_hour[hour][1] = L_i_2
            X_hour[hour][2] = L_i_3

            X_train, X_test, y_train, y_test = self.split_data(X_hour, y_hour)

            days_test = days_hour[len(X_train):]

            models[hour] = Sequential([
                Dense(25, activation='sigmoid', input_dim=X_hour.shape[1]),  # zwraca nam liczbe cech czyli 16
                Dense(1, activation='linear')
            ])
            print(f"Dla godziny: {hour}")
            optimizer = keras.optimizers.SGD(learning_rate=0.001)
            models[hour].compile(optimizer=optimizer, loss='mse', metrics=['mape'])
            models[hour].fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
            y_pred = models[hour].predict(X_test)

            vector.append(f"{y_pred[0][0]:.2f}")
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mape_vec.append(f'{mape * 100:.2f}')
            print(f'MAPE: {mape * 100:.2f}%')

            for i in range(len(y_test)):
                results.append({
                    'date': days_test[i],
                    'hour': hour,
                    'y_true': y_test[i],
                    'y_pred': y_pred[i][0],
                    'mape': abs((y_test[i] - y_pred[i][0]) / y_test[i]) * 100
                })

            L_i_3 = L_i_2
            L_i_2 = L_i_1
            L_i_1 = y_pred[0][0]

        for i in range(len(vector)):
            print(f"Load for {i} hour:", vector[i], "MAPE:", mape_vec[i])
        results_df = pd.DataFrame(results)

        return results_df
    @staticmethod
    def print_result(result_df, day):
        results = result_df.copy()
        results['date'] = pd.to_datetime(results['date'])
        day_data = results[results['date'] == day]
        plt.figure(figsize=(12, 6))
        plt.plot(day_data['hour'], day_data['y_true'], label='Rzeczywiste')
        plt.plot(day_data['hour'], day_data['y_pred'], label='Predykcje')
        plt.title(f'Predykcje vs Rzeczywiste dla dnia {day}')
        plt.xlabel('Godzina')
        plt.ylabel('Obciążenie')
        plt.legend()
        plt.grid(True)
        plt.show()