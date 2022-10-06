RAW_DATASET_ROOT_FOLDER = "Data"

STATE_DICT_KEY = "model_state_dict"
OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"

# Column names in the dataset
USER_COLUMN_NAME = "user_id"
ITEM_COLUMN_NAME = "item_id"
RATING_COLUMN_NAME = "rating"
TIMESTAMP_COLUMN_NAME = "timestamp"

# Path
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"
LOG_PATH = ROOT_PATH / "logs"
CONFIG_PATH = ROOT_PATH / "configs"
