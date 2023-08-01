import utils
import skorch

from sklearn import metrics
import torch
from ray.tune import ExperimentAnalysis
from ray.tune import register_trainable
import json

import pandas as pd

import os

import numpy as np

from ndsl.architecture.attention import TabularTransformer
import torch.nn as nn


#####################################################
# Utils
#####################################################
def count_parameters(model, only_trainable=True):
    total_params = 0    
    for name, parameter in model.module.named_parameters():
        if not parameter.requires_grad and only_trainable: 
            continue
        params = parameter.numel()
        total_params+=params
        
    return total_params

#####################################################
# Configuration
#####################################################

DATASETS = [
    "sylvine", "volkert",
    "adult", "australian",
    "anneal",  
    "jasmine", "kr-vs-kp", 
    "nomao", "ldpa"
]
AGGREGATORS = ["cls", "concatenate", "rnn", "sum", "mean", "max"]

results = []

for dataset in DATASETS:
    for aggregator in AGGREGATORS:
        print(f"Using -- Dataset:{dataset} Aggregator:{aggregator}")

        #####################################################
        # Configuration
        #####################################################
        SEP = "-" * 60
        BASE_DIR = f"./{dataset}/{aggregator}"
        CHECKPOINT_DIR = f"{BASE_DIR}/checkpoint"

        SEED = 11
        N_SAMPLES = 15
        N_STARTUP_TRIALS = 7

        BATCH_SIZE = 32
        MAX_EPOCHS = 150 
        EARLY_STOPPING = 15
        multiclass = False

        VAL_PARTITION=0.2
        TEST_PARTITION=0.2
        IMPUTER_N_NEIGHBORS=10


        OPTIMIZER=torch.optim.AdamW
        DEVICE="cuda" if torch.cuda.is_available() else "cpu"

        CHECKPOINT_METRIC = "valid_loss_best"
        HYPERPARAM_OPT_METRIC = "valid_loss_opt"
        HYPERPARAM_OPT_MODE = "min"

        #####################################################
        # Load dataset
        #####################################################
        ds = utils.read_dataset(dataset)

        multiclass = not (len(ds["labels"]) <= 2)

        if not multiclass:
            n_outputs = 1
            criterion = torch.nn.BCEWithLogitsLoss
        else:
            n_outputs = len(ds["labels"])
            criterion = torch.nn.CrossEntropyLoss

        #####################################################
        # Define trainable
        #####################################################

        def trainable(config, checkpoint_dir="."):

            module = TabularTransformer(
                n_categories=ds["n_categorical"], # List of number of categories
                n_numerical=ds["n_numerical"], # Number of numerical features
                n_head=config["n_head"], # Number of heads per layer
                n_hid=config["n_hid"], # Size of the MLP inside each transformer encoder layer
                n_layers=config["n_layers"], # Number of transformer encoder layers    
                n_output=n_outputs, # The number of output neurons
                embed_dim=config["embed_dim"],
                attn_dropout=config["attn_dropout"], 
                ff_dropout=config["ff_dropout"], 
                aggregator=aggregator, # The aggregator for output vectors before decoder
                rnn_aggregator_parameters={
                    "cell": config.get("aggregator__cell", None), 
                    "output_size": config.get("aggregator__hidden_size", None), 
                    "num_layers": config.get("aggregator__num_layers", None), 
                    "dropout": config.get("aggregator__dropout", None)
                },
                decoder_hidden_units=[128, 64],
                decoder_activation_fn=nn.ReLU(),
                need_weights=False,
                numerical_passthrough=config["numerical_passthrough"]
            )

            checkpoint = skorch.callbacks.Checkpoint(
                monitor=CHECKPOINT_METRIC, 
                dirname=os.path.join(checkpoint_dir, "best_model")
            )

            model = skorch.NeuralNetClassifier(
                    module=module,
                    criterion=criterion,
                    optimizer=OPTIMIZER,
                    device=DEVICE,
                    batch_size=BATCH_SIZE,
                    max_epochs=MAX_EPOCHS,
                    callbacks=[]
                )

            model.initialize()
            model.load_params(checkpoint=checkpoint)

            return model


        #####################################################
        # Hyperparameter search
        #####################################################
        register_trainable("trainable", trainable)

        print(SEP)

        analysis = ExperimentAnalysis(os.path.join(CHECKPOINT_DIR, "param_search"))

        for trial_idx, trial in enumerate(analysis.trials):
            model = trainable(
                trial.config, 
                checkpoint_dir=os.path.join(CHECKPOINT_DIR, "param_search", trial.custom_dirname)
            )

            # Recently added
            test_eval_file = os.path.join(BASE_DIR, "evaluation_{}.json".format(trial.custom_dirname))
            
            additional_metrics = {}
            if os.path.exists(test_eval_file):

                with open(test_eval_file, "r") as f:
                    trial_eval = json.load(f)

                for metric_mode in trial_eval["metrics"]:
                    additional_metrics["{}_bacc".format(metric_mode)] = trial_eval["metrics"][metric_mode]["balanced_acc"]
           
            results.append({
                "dataset": dataset,
                "aggregator": aggregator,
                "trial": trial.trial_id,
                **trial.config,
                HYPERPARAM_OPT_METRIC: trial.metric_analysis[HYPERPARAM_OPT_METRIC][HYPERPARAM_OPT_MODE],
                **additional_metrics,
                "non_trainable_params": count_parameters(model, only_trainable=False) - count_parameters(model, only_trainable=True),
                "trainable_params": count_parameters(model, only_trainable=True)
            })
            

results = pd.DataFrame(results)
datasets = pd.read_csv("selected_datasets.csv")[["name", "label"]]
results = results.merge(datasets, left_on="dataset", right_on="name").drop("name", axis=1)

dataset_agg_metric = results.groupby(["dataset", "aggregator"])[HYPERPARAM_OPT_METRIC].aggregate(HYPERPARAM_OPT_MODE)
dataset_metric = results.groupby(["dataset"])[HYPERPARAM_OPT_METRIC].aggregate(HYPERPARAM_OPT_MODE)

results = results.merge(
    dataset_agg_metric, 
    left_on=["dataset", "aggregator"],
    right_on=["dataset", "aggregator"],
    suffixes=("", "_ds_agg")
    )

results = results.merge(
    dataset_metric, 
    left_on=["dataset"],
    right_on=["dataset"],
    suffixes=("_trial", "_ds")
    )

# Balanced accuracy metrics
dataset_agg_metric = results.groupby(["dataset", "aggregator"])["test_bacc"].aggregate("max")
dataset_metric = results.groupby(["dataset"])["test_bacc"].aggregate("max")

results = results.merge(
    dataset_agg_metric, 
    left_on=["dataset", "aggregator"],
    right_on=["dataset", "aggregator"],
    suffixes=("", "_ds_agg")
    )

results = results.merge(
    dataset_metric, 
    left_on=["dataset"],
    right_on=["dataset"],
    suffixes=("_trial", "_ds")
    )


results.to_csv("trials_info.csv", index=False, encoding="utf-8")
