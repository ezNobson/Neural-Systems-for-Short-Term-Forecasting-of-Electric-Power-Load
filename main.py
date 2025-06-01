from nn_forecast.preprocessing.data_handler import DataHandler
from nn_forecast.consts.dirs import DATA_PATH
import pandas as pd
from nn_forecast.modeling.modular_system import ModularSystem
from nn_forecast.modeling.commitee_system import CommiteeSystem

def main():
    df = pd.read_csv( DATA_PATH / 'preprocessed_data.csv')
    modular = ModularSystem(df)
    result_df = modular.modular_network(epochs=15)
    result_df.head(20)
    modular.print_result(result_df, '2024-12-31 10:00', if_save=True)

def comitee():
    df = pd.read_csv(DATA_PATH / 'preprocessed_data.csv')
    commitee = CommiteeSystem(df)
    result_df = commitee.commitee_network(epochs=100,Neurons=50)
    result_df.head(20)
    commitee.print_result(result_df, '2024-03-10 10:00', if_save=True)


if __name__ == "__main__":
    comitee()