import logging 
import torch

LOGGING_LEVEL = logging.INFO
K_FOLD = 5

PROJECT_NAME = "tabular-transformer"
ENTITY_NAME = "tt-ndsl"
WORKER_JOBS = [
    {"dataset": "jasmine", "aggregator": "cls"},
    {"dataset": "jasmine", "aggregator": "concatenate"},
    {"dataset": "jasmine", "aggregator": "max"},
    {"dataset": "jasmine", "aggregator": "mean"},
    {"dataset": "jasmine", "aggregator": "rnn"},
    {"dataset": "jasmine", "aggregator": "sum"},

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

    {"dataset": "kr-vs-kp", "aggregator": "cls"},
    {"dataset": "kr-vs-kp", "aggregator": "concatenate"},
    {"dataset": "kr-vs-kp", "aggregator": "max"},
    {"dataset": "kr-vs-kp", "aggregator": "mean"},
    {"dataset": "kr-vs-kp", "aggregator": "rnn"},
    {"dataset": "kr-vs-kp", "aggregator": "sum"},

    {"dataset": "sylvine", "aggregator": "cls"},
    {"dataset": "sylvine", "aggregator": "concatenate"},
    {"dataset": "sylvine", "aggregator": "max"},
    {"dataset": "sylvine", "aggregator": "mean"},
    {"dataset": "sylvine", "aggregator": "rnn"},
    {"dataset": "sylvine", "aggregator": "sum"},

    {"dataset": "nomao", "aggregator": "cls"},
    {"dataset": "nomao", "aggregator": "concatenate"},
    {"dataset": "nomao", "aggregator": "max"},
    {"dataset": "nomao", "aggregator": "mean"},
    {"dataset": "nomao", "aggregator": "rnn"},
    {"dataset": "nomao", "aggregator": "sum"},

    {"dataset": "volkert", "aggregator": "cls"},
    {"dataset": "volkert", "aggregator": "concatenate"},
    {"dataset": "volkert", "aggregator": "max"},
    {"dataset": "volkert", "aggregator": "mean"},
    {"dataset": "volkert", "aggregator": "rnn"},
    {"dataset": "volkert", "aggregator": "sum"},

    {"dataset": "adult", "aggregator": "cls"},
    {"dataset": "adult", "aggregator": "concatenate"},
    {"dataset": "adult", "aggregator": "max"},
    {"dataset": "adult", "aggregator": "mean"},
    {"dataset": "adult", "aggregator": "rnn"},
    {"dataset": "adult", "aggregator": "sum"},

    {"dataset": "ldpa", "aggregator": "cls"},
    {"dataset": "ldpa", "aggregator": "concatenate"},
    {"dataset": "ldpa", "aggregator": "max"},
    {"dataset": "ldpa", "aggregator": "mean"},
    {"dataset": "ldpa", "aggregator": "rnn"},
    {"dataset": "ldpa", "aggregator": "sum"},

    
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

REFIT_SELECTION_METRICS = [
                            {
                                "metric": "log_loss",
                                "mode": "min"
                            }
                        ]
"""
Reporting variables
"""

ASSETS_DIR = "assets"
TEST_TRAININGS = 5

FEATURE_SELECTION_N_CLUSTERS = {
                                "adult": 4,
                                "anneal": 4,
                                "australian": 4,
                                "jasmine": 4,
                                "kr-vs-kp": 4,
                                "sylvine": 4,
                                "nomao": 4,
                                "volkert": 4,
                            }

FEATURE_SELECTION_K_FOLD = 10

