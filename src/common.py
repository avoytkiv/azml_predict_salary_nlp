"""Common code and variables."""

TARGET_COLUMN = "Log1pSalary"
CATEGORICAL_COLUMNS = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]
DATA_DIR = "data/"
MODEL_DIR = "model/"
RANDOM_STATE = 42

MAX_LEN = 10
WORD_DROPOUT_RATE = 0
BATCH_SIZE = 512
EPOCHS = 5