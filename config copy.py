import torch 

DATASETS_INFO = {
    "jasmine": { "id": 41143, }, 
    "anneal": { "id": 2, }, 
    "australian": { "id": 40981, }, 
    "kr-vs-kp": { "id":  3, }, 
    "sylvine": { "id": 41146,}, 
    "nomao": { "id": 1486, }, 
    "volkert": { "id": 41166, }, 
    "adult": { "id": 1590, }, 
    "ldpa": { "id": 1483, }
}


DATA_BASE_DIR = "data"
PREPROCESSING_FILE = "preprocessing.csv"
SEED = 11

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

SS_PROJECT = "self-supervised-tab-transformer"
SS_RUNS = 1
SS_DATASETS = ["jasmine", "anneal", "adult", "ldpa"]
SS_MAX_EPOCHS = 500
SS_BATCH_SIZE = 32
SS_CHECKPOINT_DIR = "self_supervised"
SS_OPTIMIZER=torch.optim.AdamW
SS_CHECKPOINT_METRICS = ["valid_loss_best"]
SS_EARLY_STOPPING_PATIENCE = 30

FT_PROJECT = "fine-tuning-tab-transformer"
#FT_DATASETS = ["jasmine", "anneal", "australian", "kr-vs-kp", "sylvine", "nomao", "volkert", "adult", "ldpa"]
FT_DATASETS = ["jasmine", "anneal", "australian", "kr-vs-kp", "sylvine", "nomao", "volkert", "adult", "ldpa"]
FT_MAX_EPOCHS = 100
FT_BATCH_SIZE = 32
FT_CHECKPOINT_DIR= "finetuning"
FT_PRELOADED_METRIC = "valid_loss_best"
FT_PRELOADED_METRIC_SEARCH = "valid_loss_opt"
FT_PRELOADED_METRIC_PREFIX = "+"
FT_OPTIMIZER=torch.optim.AdamW
FT_CHECKPOINT_METRICS = ["valid_loss_best", "accuracy_best", "balanced_accuracy_best"]
FT_CHECKPOINT_PREFIXES = ["+", "-", "-"]
FT_EARLY_STOPPING_PATIENCE = 20
FT_HP_SAMPLES = 5