"""Common code and variables."""

# Data
TARGET_COLUMN = "Log1pSalary"
CATEGORICAL_COLUMNS = ["Category", "ContractType", "ContractTime"]
# Directories
DATA_DIR = "data/"
PROCESSED_DATA_DIR = "data/processed/"
MODEL_DIR = "model/"
PLOTS_DIR = "plots/"
# Parameters
RANDOM_STATE = 42
MAX_LEN = 10
WORD_DROPOUT_RATE = 0
BATCH_SIZE = 512
EPOCHS = 30