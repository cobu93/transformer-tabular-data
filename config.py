import logging 
import torch

LOGGING_LEVEL = logging.INFO
K_FOLD = 5

PROJECT_NAME = "tabular-transformer"
ENTITY_NAME = "tt-ndsl"
WORKER_JOBS = [
    {"dataset": "anneal", "aggregator": "cls"},
    {"dataset": "anneal", "aggregator": "concatenate"},
    {"dataset": "anneal", "aggregator": "max"},
    {"dataset": "anneal", "aggregator": "mean"},
    {"dataset": "anneal", "aggregator": "rnn"},
    {"dataset": "anneal", "aggregator": "sum"},

    {"dataset": "australian", "aggregator": "cls"},
    {"dataset": "australian", "aggregator": "concatenate"},
    {"dataset": "australian", "aggregator": "max"},
    {"dataset": "australian", "aggregator": "mean"},
    {"dataset": "australian", "aggregator": "rnn"},
    {"dataset": "australian", "aggregator": "sum"},
]

RUN_EXCEPTIONS = [
    {"dataset": "sylvine", "numerical_passthrough": True},
    {"dataset": "volkert", "numerical_passthrough": True}
]

DATASETS_FILE = "datasets.csv"
HYPERPARAMETERS_FILE = "hyperparameters.json"
DATA_BASE_DIR = "data"
CHECKPOINT_BASE_DIR = "checkpoint"
SEED = 11
TEST_PARTITION = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 150
BATCH_SIZE = 32
OPTIMIZER = torch.optim.AdamW

