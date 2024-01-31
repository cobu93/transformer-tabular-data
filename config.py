import logging 
import torch

LOGGING_LEVEL = logging.INFO
WORKER_JOBS = [{"dataset": "anneal", "aggregator": "max"}]
PROJECT_NAME = "tabular-transformer"
N_TRIALS = 2

DATASETS_FILE = "datasets.csv"
DATA_BASE_DIR = "data"
CHECKPOINT_BASE_DIR = "checkpoint"
SEED = 11
TEST_PARTITION = 0.2
K_FOLD = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 2
BATCH_SIZE = 32
OPTIMIZER = torch.optim.AdamW

