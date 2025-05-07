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

class SinglePerceptron:
    def __init__(self, df) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.df = df
    def x_y(self, df):
        X = df[['load-1', 'load-2', 'load-3', 'load-22', 'load-23', 'load-24', 'load-25', 'load-26', 'mean_t_3',
                'mean_t_5',
                'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos']]
        lista = []
        lista.append('total_load')
        for i in range(1, 24):
            lista.append(f'next_load_{i}')

        y = df[lista]
        return X, y

    def split_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def network(df):
        X, y = x_y(df)
        # normalizacja
        other_scaler = MinMaxScaler(feature_range=(0, 1))
        temp_scaler = MinMaxScaler(feature_range=(-1, 1))
        for temp in [f'mean_t_{t}' for t in [3, 5]]:
            X[temp] = temp_scaler.fit_transform(X[[temp]])

        for col in [f'load-{t}' for t in [1, 2, 3, 22, 23, 24, 25, 26]]:
            X[col] = other_scaler.fit_transform(X[[col]])

        # na numpy array
        X = np.array(X)
        y = np.array(y)
        day_df = np.array(df['day_of_week_num'])
        # podzial danych
        X_train, X_test, y_train, y_test = split_data(X, y)
        day_train, day_test = train_test_split(day_df, test_size=0.2)

        model = Sequential([
            Dense(25, activation='sigmoid', input_dim=X.shape[1]),  # zwraca nam liczbe cech czyli 16
            Dense(24, activation='linear')
        ])

        optimizer = keras.optimizers.SGD(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])
        trained_model = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

        y_pred = model.predict(X_test)

        mape_by_day = {}

        for day in range(7):
            indices = np.where(day_test == day)
            y_true_day = y_test[indices]
            y_pred_day = y_pred[indices]

            if len(y_true_day) > 0:
                mape = mean_absolute_percentage_error(y_true_day, y_pred_day)
                mape_by_day[day] = mape * 100
            else:
                mape_by_day[day] = None

        week_days = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }

        for day, mape in mape_by_day.items():
            day_name = week_days.get(day, f"{day}")
            print(f"{day_name}: MAPE = {mape:.2f}%" if mape else f"Day {day}: No data")

        plt.plot(y_test[3, :], label='Rzeczywiste')
        plt.plot(y_pred[3, :], label='Przewidywane')

        return y_pred, y_test






