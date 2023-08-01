import utils
import skorch

from ray import tune
import optuna
from ray.tune.suggest.optuna import OptunaSearch
import torch

from ray.tune.schedulers import AsyncHyperBandScheduler

from ray.tune import ExperimentAnalysis
from ray.tune import register_trainable

import argparse

import os

import numpy as np

from ndsl.architecture.attention import TabularTransformer
import torch.nn as nn

import pandas as pd
from sklearn import model_selection, metrics

#####################################################
# Configuration
#####################################################
dataset = "adult"
aggregator = "concatenate"

print(f"Using -- Dataset:{dataset} Aggregator:{aggregator}. Replicating TabTransformer.")

#####################################################
# Configuration
#####################################################
SEP = "-" * 60
BASE_DIR = f"./tabtransformer/{dataset}/{aggregator}"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoint"
REPORT_FILENAME = f"{BASE_DIR}/report.txt"
SEED = 7

N_SAMPLES = 15
N_STARTUP_TRIALS = 7

BATCH_SIZE = 128
MAX_EPOCHS = 150
EARLY_STOPPING = 15
multiclass = False

VAL_PARTITION=0.15
IMPUTER_N_NEIGHBORS=10

OPTIMIZER=torch.optim.AdamW
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_METRIC = "roc_auc_best"
HYPERPARAM_OPT_METRIC = "roc_auc_opt"
HYPERPARAM_OPT_MODE = "max"

CV_SPLITS=5

#####################################################
# Define search space
#####################################################

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

search_space = {
    "n_layers": tune.randint(1, 5), # Number of transformer encoder layers    
    "optimizer__lr": tune.loguniform(1e-5, 1e-3),    
    "optimizer__weight_decay": tune.loguniform(1e-6, 1e-3),
    "n_head": tune.choice([1, 2, 4, 8, 16, 32]), # Number of heads per layer
    "n_hid": tune.choice([32, 64, 128, 256, 512, 1024]), # Size of the MLP inside each transformer encoder layer
    "attn_dropout": tune.uniform(0, 0.5), # Used dropout   
    "ff_dropout": tune.uniform(0, 0.5), # Used dropout   
    "embed_dim": tune.choice([32, 64, 128, 256, 512, 1024]),
    "numerical_passthrough": tune.choice([True])
}

if aggregator == "rnn":
    search_space = {
        **search_space,
        "aggregator__cell": tune.choice(["LSTM", "GRU"]),
        "aggregator__hidden_size": tune.choice([32, 64, 128, 256, 512, 1024]),
        "aggregator__num_layers": tune.randint(1, 3),
        "aggregator__dropout": tune.uniform(0, 0.5)
    }

#####################################################
# Load dataset
#####################################################
ds = utils.read_dataset(dataset)
df = pd.read_csv("./tabtransformer/income_evaluation.csv")
print(df.columns, df.shape)
ds["target"] = "income"
ds["features"] = df.drop(ds["target"], axis=1)
ds["outputs"] = df[ds["target"]].replace(ds["labels"])

#####################################################
# Configuring
#####################################################

multiclass = not (len(ds["labels"]) <= 2)

if not multiclass:
    n_outputs = 1
    criterion = torch.nn.BCEWithLogitsLoss
else:
    n_outputs = len(ds["labels"])
    criterion = torch.nn.CrossEntropyLoss

print(SEP)

print("N. Features: {}".format(ds["features"].shape))
print("N. Outputs: {}".format(ds["outputs"].shape))
print("Target: {}".format(ds["target"]))
print("Labels: {}".format(ds["labels"]))
print("N. Columns: {}".format(ds["columns"].shape))
print("N. Categorical: {}".format(ds["categorical"].shape))
print("N. Categories: {}".format(ds["n_categorical"]))
print("N. Numerical: {}".format(ds["numerical"].shape))
print("N. Numerical: {}".format(ds["n_numerical"]))
print("Classification type: {}".format("Multiclass" if multiclass else "Binary"))

#####################################################
# Split data
#####################################################

k_fold = model_selection.KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)

scores = np.zeros(CV_SPLITS)

for i, (train_index, test_index) in enumerate(k_fold.split(ds["features"])):

    base_dir = BASE_DIR + f"/cv_{i}"
    checkpoint_dir = f"{base_dir}/checkpoint"
    report_filename = f"{base_dir}/report.txt"
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    X_train, X_val, y_train, y_val = model_selection.train_test_split(
            ds["features"].iloc[train_index], 
            ds["outputs"].iloc[train_index], 
            test_size=VAL_PARTITION, 
            random_state=SEED
        )

    data = {
        "train": { "features": X_train, "outputs": y_train },
        "val": { "features": X_val, "outputs": y_val },
        "test": { "features": ds["features"].iloc[test_index], "outputs": ds["outputs"].iloc[test_index] },
    }

    print(SEP)
    print("N. Train examples: {}".format(data["train"]["outputs"].shape[0]))
    print("N. Validation examples: {}".format(data["val"]["outputs"].shape[0]))
    print("N. Test examples: {}".format(data["test"]["outputs"].shape[0]))

    #####################################################
    # Preprocessing
    #####################################################

    preprocessor = utils.get_preprocessor(
        ds["categorical"],
        ds["numerical"],
        ds["categories"],
        n_neighbors=IMPUTER_N_NEIGHBORS
    )

    print(SEP)
    print("Preprocessing...")

    data["train"]["features"] = preprocessor.fit_transform(data["train"]["features"])
    data["val"]["features"] = preprocessor.transform(data["val"]["features"])
    data["test"]["features"] = preprocessor.transform(data["test"]["features"])

    assert np.isnan(data["train"]["features"]).sum() == 0, "Something went wrong while preprocessing (train)"
    assert np.isnan(data["val"]["features"]).sum() == 0, "Something went wrong while preprocessing (val)"
    assert np.isnan(data["test"]["features"]).sum() == 0, "Something went wrong while preprocessing (test)"

    print("Done")

    #####################################################
    # Define trainable
    #####################################################

    all_features, all_outputs, (train_indices, val_indices) = utils.join_data(
        (data["train"]["features"], data["val"]["features"]),
        (data["train"]["outputs"], data["val"]["outputs"])
    ) 

    if multiclass:
        all_outputs = all_outputs.astype(np.int64)
    else:
        all_outputs = all_outputs.astype(np.float32)

    def trainable(config, checkpoint_dir=None):

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

        model = skorch.NeuralNetClassifier(
                module=module,
                criterion=criterion,
                optimizer=OPTIMIZER,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                callbacks=[]
            )

        checkpoint = skorch.callbacks.Checkpoint(
                        monitor=CHECKPOINT_METRIC, 
                        dirname=os.path.join(checkpoint_dir, "best_model")
                    )

        model.load_params(checkpoint=checkpoint)
        return model


    #####################################################
    # Hyperparameter search
    #####################################################
    print(SEP)
    

    register_trainable("trainable", trainable)
    analysis = ExperimentAnalysis(os.path.join(checkpoint_dir, "param_search"))
    best_config = analysis.get_best_config(metric=HYPERPARAM_OPT_METRIC, mode=HYPERPARAM_OPT_MODE)
    best_trial = analysis.get_best_trial(metric=HYPERPARAM_OPT_METRIC, mode=HYPERPARAM_OPT_MODE)
    model = trainable(
        best_config, 
        checkpoint_dir=os.path.join(checkpoint_dir, "param_search", utils.trial_dirname_creator(best_trial))
    )

    predictions = model.predict_proba({
            "x_numerical": data["test"]["features"][:, :ds["n_numerical"]].astype(np.float32),
            "x_categorical": data["test"]["features"][:, ds["n_numerical"]:].astype(np.int32)
            })

    scores[i] = metrics.roc_auc_score( data["test"]["outputs"], predictions[:, 1])

    print("Best configuration: {}".format(best_config))

    
print(SEP)
print("All scores: {}".format(scores))
print("Mean metric: {}".format(scores.mean()))
print("Standard deviation: {}".format(scores.std()))