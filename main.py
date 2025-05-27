from nn_forecast.preprocessing.data_handler import DataHandler
from nn_forecast.consts.dirs import DATA_PATH
import pandas as pd
from nn_forecast.modeling.modlar_system import ModularSystem


def main():
    df = pd.read_csv( DATA_PATH / 'preprocessed_data.csv')
    modular = ModularSystem(df)
    result_df = modular.modular_network(epochs=2)
    result_df.head(20)
    modular.print_result(result_df, '2024-05-01 22:00')

if __name__ == "__main__":
    main()