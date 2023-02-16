from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_PATH / "data"
LOG_PATH = ROOT_PATH / "logs"
CONFIG_PATH = ROOT_PATH / "configs"

# caches
CACHE_PATH = ROOT_PATH / "cache"
NEGATIVE_SAMPLE_PATH = CACHE_PATH / "nsample"
DATACLASS_PATH = CACHE_PATH / "dataclass"

OUTPUT_PATH = ROOT_PATH / "out"

ROOT_PATH.mkdir(exist_ok=True)
DATA_PATH.mkdir(exist_ok=True)
LOG_PATH.mkdir(exist_ok=True)
CONFIG_PATH.mkdir(exist_ok=True)

CACHE_PATH.mkdir(exist_ok=True)
NEGATIVE_SAMPLE_PATH.mkdir(exist_ok=True)
DATACLASS_PATH.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True)
