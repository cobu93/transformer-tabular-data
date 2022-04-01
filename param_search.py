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
MAX_EPOCHS = 500 
EARLY_STOPPING = 15
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
        "verbose": 0
        
    }
    
    model = nn_utils.build_transformer_model(
                train_indices,
                val_indices,
                nn_utils.get_default_callbacks(seed=SEED, multiclass=multiclass),
                **model_params
                )
    
    model = model.fit(X=all_features, y=all_labels)

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
resume_modes = ["AUTO", "ERRORED_ONLY"]


for try_cnt, resume_mode in enumerate(resume_modes):
    try:
        analysis = tune.run(
            trainable,
            config=search_space_config.get_search_space(),
            resources_per_trial={
                "gpu": 1,
                "cpu": 6
            },
            search_alg=OptunaSearch(
                metric="balanced_accuracy",
                mode="max",
                sampler=optuna.samplers.TPESampler()
            ),
            num_samples=N_SAMPLES,
            fail_fast=True,
            checkpoint_score_attr="max-balanced_accuracy",
            keep_checkpoints_num=MAX_CHECKPOINTS,
            resume=resume_mode,
            local_dir=CHECKPOINT_DIR, 
            name="param_search",
            scheduler=AsyncHyperBandScheduler(
                            time_attr="training_iteration",
                            metric="balanced_accuracy",
                            mode="max",
                            grace_period=EARLY_STOPPING
                        )
        )

        break
    except Exception as e:

        if try_cnt + 1 == len(resume_modes):
            raise(e)

        print(e)
        print("Retrying in second mode")

print("Best config: ", analysis.get_best_config(metric="balanced_accuracy", mode="max"))