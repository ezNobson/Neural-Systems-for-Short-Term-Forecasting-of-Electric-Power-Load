from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from nn_forecast.consts.dirs import DATA_PATH
from nn_forecast.utils.logging_custom import get_logger
from nn_forecast.consts import dirs

class CommiteeSystem:
    def __init__(self,df):
        self.logger = get_logger(self.__class__.__name__)
        self.df = df
        dirs.VISUALIZATION_COMMITEE.mkdir(parents=True, exist_ok=True)
    def prepare_data(self):
        X = self.df[['load-1', 'load-2', 'load-3', 'load-22', 'load-23', 'load-24', 'load-25', 'load-26', 'mean_t_3',
                'mean_t_5',
                'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos']]

        lista = []
        lista.append('total_load')
        for i in range(1, 24):
            lista.append(f'next_load_{i}')

        y = self.df[lista]
        return X, y
    @staticmethod
    def split_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test
    def commitee_network(self,epochs =20, Neurons=25, K=5):


        X, y = self.prepare_data()
        other_scaler = MinMaxScaler(feature_range=(0, 1))
        temp_scaler = MinMaxScaler(feature_range=(-1, 1))
        for temp in [f'mean_t_{t}' for t in [3, 5]]:
            X[temp] = temp_scaler.fit_transform(X[[temp]])
        for col in [f'load-{t}' for t in [1, 2, 3, 22, 23, 24, 25, 26]]:
            X[col] = other_scaler.fit_transform(X[[col]])
        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train_all, y_test_all = self.split_data(X, y)


        commitee_preds = {hour: [] for hour in range(24)}

        for k in range(K):
            for hour in range(24):
                y_train = y_train_all[:, hour]
                y_test = y_test_all[:, hour]

                model = Sequential([
                    Dense(Neurons, activation='sigmoid', input_dim=X.shape[1]),
                    Dense(1, activation='linear')
                ])
                optimizer = keras.optimizers.SGD(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])
                print(f"Dla godizny: {hour}, komitet: {k+1}")
                model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
                y_pred = model.predict(X_test).flatten()

                commitee_preds[hour].append(y_pred)

        y_preds_matrix = np.column_stack([
            np.mean(np.stack(commitee_preds[hour], axis=1), axis=1) for hour in range(24)
        ])
        dates = self.df['time'].values
        _, dates_test = train_test_split(dates, test_size=0.2, shuffle=False)
        result_df = pd.DataFrame({'date': dates_test})

        for i in range(24):
            result_df[f'y_real_{i}'] = y_test_all[:, i]
            result_df[f'y_pred_{i}'] = y_preds_matrix[:, i]
            result_df[f'mape_{i}'] = np.abs(result_df[f'y_real_{i}'] - result_df[f'y_pred_{i}']) / result_df[
                f'y_real_{i}'] * 100

        result_df['mape_mean'] = result_df[[f'mape_{i}' for i in range(24)]].mean(axis=1)
        mape_per_hour = result_df[[f'mape_{i}' for i in range(24)]].mean(axis=0)
        for i, mape_hour in enumerate(mape_per_hour):
            print(f"Średnia MAPE (committee) dla godziny {i}: {mape_hour:.2f}%")

        result_df.to_csv(dirs.DATA_PATH / "committee_results.csv", index=False)
        return result_df

    @staticmethod
    def print_result(result_df, start_datetime, if_save=False,epochs=None, neurons = None):
        results = result_df.copy()
        results['date'] = pd.to_datetime(results['date'])
        start_datetime = pd.to_datetime(start_datetime)

        # Wiersz z predykcjami (dla startowej godziny)
        pred_row = results[results['date'] == start_datetime]
        if pred_row.empty:
            print(f"Brak predykcji dla {start_datetime}")
            return

        # Rzeczywiste wartości z kolejnych 24 godzin
        mask = (results['date'] >= start_datetime) & (results['date'] < start_datetime + pd.Timedelta(hours=24))
        real_rows = results[mask]
        if len(real_rows) < 24:
            print(f"Brak wystarczających danych rzeczywistych od {start_datetime} (znaleziono {len(real_rows)})")
            return

        y_true = real_rows['y_real_0'].values[:24]
        y_pred = [pred_row[f'y_pred_{i}'].values[0] for i in range(24)]
        hours = list(range(24))

        mape_val = pred_row['mape_mean'].values[0]
        print(f"MAPE (średnia z 24h) dla {start_datetime}: {mape_val:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(hours, y_true, label='Rzeczywiste')
        plt.plot(hours, y_pred, label='Predykcje')


        title = f'Predykcje vs Rzeczywiste od {start_datetime}'
        if epochs is not None and neurons is not None:
            title += f' (epochs={epochs}, neurons={neurons})'
        elif epochs is not None:
            title += f' (epochs={epochs})'
        elif neurons is not None:
            title += f' (neurons={neurons})'

        plt.title(title)
        plt.suptitle(f"MAPE (średnia z 24h): {mape_val:.2f}%", fontsize=12, y=0.94)
        plt.xlabel('Godzina od startu')
        plt.ylabel('Obciążenie')
        plt.legend()
        plt.grid(True)
        if if_save:
            start_datetime_safe = str(start_datetime).replace(':', '-').replace(' ', '_')

            plt.savefig(dirs.VISUALIZATION_COMMITEE / f"commitee_results_{start_datetime_safe}.png")
        plt.show()