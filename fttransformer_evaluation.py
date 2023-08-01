import utils
import skorch
import torch
import os
import numpy as np
from ndsl.architecture.attention import TabularTransformer

import torch.nn as nn
from sklearn import metrics

#####################################################
# Configuration
#####################################################

dataset = "adult"
aggregator = "cls"

print(f"Using -- Dataset:{dataset} Aggregator:{aggregator}")

#####################################################
# Configuration
#####################################################
SEP = "-" * 60
BASE_DIR = f"./fttransformer/{dataset}/{aggregator}"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoint"
REPORT_FILENAME = f"{BASE_DIR}/report.txt"

SEED = 7

BATCH_SIZE = 128
MAX_EPOCHS = 100 
EARLY_STOPPING = 15
multiclass = False

VAL_PARTITION=6513
TEST_PARTITION=16281
IMPUTER_N_NEIGHBORS=10
N_TRIALS=15

OPTIMIZER=torch.optim.AdamW
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

EARLY_STOPPING_METRIC = "accuracy"
EARLY_STOPPING_LOWER_IS_BETTER = False
CHECKPOINT_METRIC = "accuracy_best"
HYPERPARAM_OPT_METRIC = "accuracy_opt"
HYPERPARAM_OPT_MODE = "max"


#####################################################
# Define search space
#####################################################

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

search_space = {
    "n_layers": 3,
    "optimizer__lr": 1e-4,
    "optimizer__weight_decay": 1e-5,
    "n_head": 8,
    "n_hid": 256,
    "attn_dropout": 0.2,
    "ff_dropout": 0.1,
    "embed_dim": 192,
    "numerical_passthrough": False,
}

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

config = search_space

scores = np.zeros(N_TRIALS)

for i in range(N_TRIALS):
    
    #####################################################
    # Split data
    #####################################################
    data = utils.split_train_val_test(
        ds["features"], 
        ds["outputs"], 
        val_size=VAL_PARTITION,
        test_size=TEST_PARTITION,
        seed=SEED + i
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

    all_features, all_outputs, (train_indices, val_indices) = utils.join_data(
        (data["train"]["features"], data["val"]["features"]),
        (data["train"]["outputs"], data["val"]["outputs"])
    ) 

    if multiclass:
        all_outputs = all_outputs.astype(np.int64)
    else:
        all_outputs = all_outputs.astype(np.float32)

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
            train_split=skorch.dataset.CVSplit(((train_indices, val_indices),)),
            callbacks=[]
        )

    checkpoint = skorch.callbacks.Checkpoint(
            monitor=CHECKPOINT_METRIC, 
            dirname=os.path.join(CHECKPOINT_DIR, f"split_{i}/best_model")
        )

    model.load_params(checkpoint=checkpoint)

    print(SEP)
    print(f"Evaluating {i} trial")

    predictions = model.predict({
            "x_numerical": data["test"]["features"][:, :ds["n_numerical"]].astype(np.float32),
            "x_categorical": data["test"]["features"][:, ds["n_numerical"]:].astype(np.int32)
            })

    scores[i] = metrics.accuracy_score( data["test"]["outputs"], predictions)

print(SEP)
print("All scores: {}".format(scores))
print("Mean metric: {}".format(scores.mean()))
print("Standard deviation: {}".format(scores.std()))
    