from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_PATH / "data"
LOG_PATH = ROOT_PATH / "logs"
CONFIG_PATH = ROOT_PATH / "configs"

# caches
CACHE_PATH = DATA_PATH / "cache"
NEGATIVE_SAMPLE_PATH = CACHE_PATH / "nsample"
DATACLASS_PATH = CACHE_PATH / "dataclass"
OUTPUT_PATH = ROOT_PATH / "out"
