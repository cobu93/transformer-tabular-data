import logging 
import torch

LOGGING_LEVEL = logging.INFO
K_FOLD = 5

PROJECT_NAME = "tabular-transformer"
ENTITY_NAME = "tt-ndsl"
WORKER_JOBS = [{"dataset": "anneal", "aggregator": "rnn"}]
N_TRIALS = 64

DATASETS_FILE = "datasets.csv"
DATA_BASE_DIR = "data"
CHECKPOINT_BASE_DIR = "checkpoint"
SEED = 11
TEST_PARTITION = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 150
BATCH_SIZE = 32
OPTIMIZER = torch.optim.AdamW

