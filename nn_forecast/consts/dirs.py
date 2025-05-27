from pathlib import Path

ROOT_PATH = Path(__file__).parent.absolute()
DATA_PATH = ROOT_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw_data"

VISUALIZATION_DATA_PATH = DATA_PATH / "visualization_data"
VISUALIZATION_PERCEPTRON = VISUALIZATION_DATA_PATH / "perceptron"