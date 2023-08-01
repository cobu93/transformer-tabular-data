import utils
import skorch

from sklearn import metrics
import torch
from ray.tune import ExperimentAnalysis
from ray.tune import register_trainable
import json

import argparse

import os

import numpy as np

from ndsl.architecture.attention import TabularTransformer
import torch.nn as nn

#####################################################
# Configuration
#####################################################

parser = argparse.ArgumentParser()
parser.add_argument("dataset", metavar="dataset", type=str, help="Dataset parameter search")
parser.add_argument("aggregator", metavar="aggregator", type=str, help="Aggregator type")

args = parser.parse_args()

dataset = args.dataset
aggregator = args.aggregator

print(f"Using -- Dataset:{dataset} Aggregator:{aggregator}")

#####################################################
# Configuration
#####################################################
SEP = "-" * 60
BASE_DIR = f"./{dataset}/{aggregator}"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoint"
REPORT_FILENAME = f"{BASE_DIR}/report.txt"

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
data = utils.split_train_val_test(
    ds["features"], 
    ds["outputs"], 
    val_size=VAL_PARTITION,
    test_size=TEST_PARTITION,
    seed=SEED
)

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
print("Loading checkpoint")

analysis = ExperimentAnalysis(os.path.join(CHECKPOINT_DIR, "param_search"))

evaluators = [
    ("acc", metrics.accuracy_score, "outputs"),
    ("balanced_acc", metrics.balanced_accuracy_score, "outputs")
]

if not multiclass:
    evaluators += [
        ("roc_auc", metrics.roc_auc_score, "probas"),
        ("f1", metrics.f1_score, "outputs"),
        ("precision", metrics.precision_score, "outputs"),
        ("recall", metrics.recall_score, "outputs"),
    ]
    
predictions = {}


for trial in analysis.trials:

    model = trainable(
        trial.config,
        checkpoint_dir=os.path.join(CHECKPOINT_DIR, "param_search", trial.custom_dirname)
        )

    print(SEP)
    print("Computing predictions for:")

    for data_name in ["train", "val", "test"]:
        print(f"...... {data_name}")
        predictions[data_name] = {}

        predictions[data_name]["probas"] = model.predict_proba(
                X={
                    "x_numerical": data[data_name]["features"][:, :ds["n_numerical"]].astype(np.float32),
                    "x_categorical": data[data_name]["features"][:, ds["n_numerical"]:].astype(np.int32)
                    }
                )
                
        if not multiclass:
            predictions[data_name]["probas"] = predictions[data_name]["probas"][:, 1]
            predictions[data_name]["outputs"] = np.rint(predictions[data_name]["probas"])
        else:
            predictions[data_name]["outputs"] = np.argmax(predictions[data_name]["probas"], axis=1)
            
    prediction_metrics = {}

    print("Computing metrics for:")

    for data_name in ["train", "val", "test"]:
        print(f"...... {data_name}")
        prediction_metrics[data_name] = {}
        for metric_name, scorer, input_type  in evaluators:
            score = scorer(data[data_name]["outputs"], predictions[data_name][input_type])
            prediction_metrics[data_name][metric_name]= np.round(score * 100, decimals=3)


    eval_file = os.path.join(BASE_DIR, "evaluation_{}.json".format(trial.custom_dirname))
    print(SEP)
    print(f"Saving evaluation file at {eval_file}")

    with open(eval_file, "w") as f:
        evaluation_info = {
            "config": trial.config,
            "metrics": prediction_metrics
        }
        json.dump(evaluation_info, f, indent=6)
