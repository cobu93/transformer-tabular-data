import nn_utils
import builders
import importlib

from ray import tune
import optuna
from ray.tune.suggest.optuna import OptunaSearch
import torch

from ray.tune.schedulers import AsyncHyperBandScheduler

import inspect
import argparse
import skorch
import os

from torch.utils import tensorboard

#####################################################
# Configuration
#####################################################

parser = argparse.ArgumentParser()
parser.add_argument("dataset", metavar="dataset", type=str, help="Dataset parameter search")
parser.add_argument("aggregator", metavar="aggregator", type=str, help="Aggregator type")

args = parser.parse_args()

dataset = args.dataset
aggregator_str = args.aggregator

print(f"Using -- Dataset:{dataset} Aggregator:{aggregator_str}")

#####################################################
# Configuration
#####################################################

MODULE = f"{dataset}.{aggregator_str}.config"
CHECKPOINT_DIR = f"./{dataset}/{aggregator_str}/checkpoint"
SEED = 11
N_SAMPLES = 30

BATCH_SIZE = 128
MAX_EPOCHS = 1000
EARLY_STOPPING = 30
MAX_CHECKPOINTS = 10
multiclass = False

#####################################################
# Util functions
#####################################################

def get_class_from_type(module, class_type):
    for attr in dir(module):
        clazz = getattr(module, attr)
        if callable(clazz) and inspect.isclass(clazz) and issubclass(clazz, class_type) and not str(clazz)==str(class_type):
            return clazz
        
    return None

def get_params_startswith(params, prefix):
    keys = [k for k in params.keys() if k.startswith(prefix)]
    extracted = {}

    for k in keys:
        extracted[k.replace(prefix, "")] = params.pop(k)

    return extracted


def trainable(config, checkpoint_dir=CHECKPOINT_DIR):
    
    embedding_size = config.pop("embedding_size")

    encoders_params = get_params_startswith(config, "encoders__")
    aggregator_params = get_params_startswith(config, "aggregator__")
    preprocessor_params = get_params_startswith(config, "preprocessor__")

    model_params = {
        **config,
        "encoders": transformer_config.get_encoders(embedding_size, **encoders_params),
        "aggregator": transformer_config.get_aggregator(embedding_size, **aggregator_params),
        "preprocessor": transformer_config.get_preprocessor(**preprocessor_params),
        "optimizer": torch.optim.SGD,
        "criterion": criterion,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "n_output": n_labels, # The number of output neurons
        "need_weights": False,
        "verbose": 1
        
    }

    def key_mapper_fn(key):
        return "best_model/" + key

    if os.path.exists(os.path.join(CHECKPOINT_DIR, "best_model/.fitted")):
        print("Fitted before! I'm not going to do anything")
        return

    
    checkpoint = skorch.callbacks.Checkpoint(monitor="balanced_accuracy_best", dirname=os.path.join(CHECKPOINT_DIR, "best_model"))

    model = nn_utils.build_transformer_model(
                train_indices,
                val_indices,                
                nn_utils.get_default_train_callbacks(seed=SEED, multiclass=multiclass) + [
                    ("early_stopping", skorch.callbacks.EarlyStopping(monitor="balanced_accuracy", patience=EARLY_STOPPING, lower_is_better=False)),
                    ("checkpoint", checkpoint),
                    ("load_init_state", skorch.callbacks.LoadInitState(checkpoint)),
                    ("tensorboard", skorch.callbacks.TensorBoard(tensorboard.writer.SummaryWriter(
                        log_dir=os.path.join(CHECKPOINT_DIR, "best_model", "tensorboard"), 
                        filename_suffix="best_model"
                    ), key_mapper=key_mapper_fn
                    )),
                    ("lr_scheduler", skorch.callbacks.LRScheduler(policy="ReduceLROnPlateau", monitor="balanced_accuracy", patience=EARLY_STOPPING // 3))
                ],
                **model_params
                )
        
    model = model.fit(X=all_features, y=all_labels)

    with open(os.path.join(CHECKPOINT_DIR, "best_model/.fitted"), "w") as f:
        f.write("Fitted before")

#####################################################
# Dataset and components
#####################################################

module = importlib.import_module(MODULE)

dataset = get_class_from_type(module, builders.DatasetConfig)
if dataset is not None:
    dataset = dataset()
else:
    raise ValueError("Dataset configuration not found")

transformer_config = get_class_from_type(module, builders.TransformerConfig)
if transformer_config is not None:
    transformer_config = transformer_config()
else:
    raise ValueError("Transformer configuration not found")

search_space_config = get_class_from_type(module, builders.SearchSpaceConfig)
if search_space_config is not None:
    search_space_config = search_space_config()
else:
    raise ValueError("Search space configuration not found")

#####################################################
# Configure dataset
#####################################################

if not dataset.exists():
    dataset.download()
    
dataset.load(seed=SEED)

preprocessor = nn_utils.get_default_preprocessing_pipeline(
                        dataset.get_categorical_columns(),
                        dataset.get_numerical_columns()
                    )

#####################################################
# Data preparation
#####################################################

train_features, train_labels = dataset.get_train_data()
val_features, val_labels = dataset.get_val_data()
test_features, test_labels = dataset.get_test_data()

preprocessor = preprocessor.fit(train_features, train_labels)

train_features = preprocessor.transform(train_features)
val_features = preprocessor.transform(val_features)
test_features = preprocessor.transform(test_features)

all_features, all_labels, indices = nn_utils.join_data([train_features, val_features], [train_labels, val_labels])
train_indices, val_indices = indices[0], indices[1]

if dataset.get_n_labels() <= 2:
    n_labels = 1
    criterion = torch.nn.BCEWithLogitsLoss
else:
    n_labels = dataset.get_n_labels()
    multiclass = True
    criterion = torch.nn.CrossEntropyLoss

#####################################################
# Hyperparameter search
#####################################################
analysis = tune.run(
    trainable,
    resume="AUTO",
    local_dir=CHECKPOINT_DIR, 
    name="param_search"    
)

best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
del analysis
print("Best config: ", best_config)

trainable(best_config)
