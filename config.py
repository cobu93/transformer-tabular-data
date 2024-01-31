import logging 
import torch

LOGGING_LEVEL = logging.INFO

DATASETS_FILE = "datasets.csv"
DATA_BASE_DIR = "data"
SEED = 11
TEST_PARTITION = 0.2
K_FOLD = 10

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
PROJECT = "tabular-transformer"
RUNS = 1
DATASETS = ["jasmine", "anneal", "adult", "ldpa"]
MAX_EPOCHS = 100
BATCH_SIZE = 32
OPTIMIZER=torch.optim.AdamW

