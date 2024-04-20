import numpy as np
import os

import utils
import torch

from ray.tune import register_trainable
from ray.tune import ExperimentAnalysis

import skorch
from ndsl.architecture.attention import TabularTransformer

import torch.nn as nn
import pandas as pd

import pickle


SEP = "-" * 60

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

def compute_std_attentions(attn, aggregator):
    batch_size = attn.shape[1]
    n_layers = attn.shape[0]
    n_features = attn.shape[-1]

    # Sum heads
    # layers, batch, heads, o_features, i_features
    heads_attn = attn.mean(axis=2)

    # Initial: layers, batch, o_features, i_features
    # Final: batch, layers, i_features, o_features
    heads_attn = heads_attn.permute((1, 0, 3, 2))
    general_attn = None

    # For each layer
    single_attns = torch.zeros((batch_size, n_layers, n_features))
    cum_attns = torch.zeros((batch_size, n_layers, n_features))

    for layer_idx in range(n_layers):
        if layer_idx == n_layers - 1 and aggregator == "cls":
            single_attns[:, layer_idx] = heads_attn[:, layer_idx, :, 0]
        else:
            single_attns[:, layer_idx] = heads_attn[:, layer_idx].mean(axis=-1)
        
        if general_attn is None:
            general_attn = heads_attn[:, layer_idx]
        else:
            general_attn = torch.matmul(general_attn, heads_attn[:, layer_idx])

        if layer_idx == n_layers - 1 and aggregator == "cls":
            cum_attns[:, layer_idx] = general_attn[:, :, 0]
        else:
            cum_attns[:, layer_idx] = general_attn.mean(axis=-1)

    #assert np.allclose(single_attns.sum(axis=-1), 1), "There is a logistic problem: " + str(single_attns.sum(axis=-1))
    #assert np.allclose(cum_attns.sum(axis=-1), 1), "There is a logistic problem: " + str(cum_attns.sum(axis=-1))

    # Before: batch_size, n_layers, n_features
    # After: n_layers, batch_size, n_features
    return single_attns.permute((1, 0, 2)), cum_attns.permute((1, 0, 2))


results = pd.read_csv("trials_info.csv", encoding="utf-8")

top_k = 1
attn_df = results
attn_df = attn_df \
    .groupby(["dataset", "aggregator"], group_keys=False) \
    .apply(lambda g: g.sort_values(["valid_loss_opt_trial"], ascending=True).head(top_k)) \
    .groupby(["dataset"], group_keys=False) \
    .apply(lambda g: g.sort_values(["test_bacc_trial"], ascending=False).head(top_k)) 

for bconfig in attn_df.iloc:

    dataset = bconfig["dataset"]
    aggregator = bconfig["aggregator"]

    BASE_DIR = f"./{dataset}/{aggregator}"
    CHECKPOINT_DIR = f"{BASE_DIR}/checkpoint"
    REPORT_FILENAME = f"{BASE_DIR}/report.txt"

    ds = utils.read_dataset(bconfig["dataset"])

    multiclass = not (len(ds["labels"]) <= 2)

    if not multiclass:
        n_outputs = 1
        criterion = torch.nn.BCEWithLogitsLoss
    else:
        n_outputs = len(ds["labels"])
        criterion = torch.nn.CrossEntropyLoss

    data = utils.split_train_val_test(
        ds["features"], 
        ds["outputs"], 
        val_size=VAL_PARTITION,
        test_size=TEST_PARTITION,
        seed=SEED
    )

    preprocessor = utils.get_preprocessor(
        ds["categorical"],
        ds["numerical"],
        ds["categories"],
        n_neighbors=IMPUTER_N_NEIGHBORS
    )

    data["train"]["features"] = preprocessor.fit_transform(data["train"]["features"])
    data["val"]["features"] = preprocessor.transform(data["val"]["features"])
    data["test"]["features"] = preprocessor.transform(data["test"]["features"])

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
            need_weights=True,
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

    register_trainable("trainable", trainable)

    analysis = ExperimentAnalysis(os.path.join(CHECKPOINT_DIR, "param_search"))
    best_trial = analysis.get_best_trial(metric=HYPERPARAM_OPT_METRIC, mode=HYPERPARAM_OPT_MODE)
    
    assert best_trial.config["n_layers"]==bconfig["n_layers"], "Inconsistent n_layers"
    assert best_trial.config["n_head"]==bconfig["n_head"], "Inconsistent n_head"
    assert best_trial.config["n_hid"]==bconfig["n_hid"], "Inconsistent n_hid"
    assert best_trial.config["embed_dim"]==bconfig["embed_dim"], "Inconsistent embed_dim"
    assert best_trial.config["numerical_passthrough"]==bconfig["numerical_passthrough"], "Inconsistent numerical_passthrough"
    
    model = trainable(
        best_trial.config,
        checkpoint_dir=os.path.join(CHECKPOINT_DIR, "param_search", best_trial.custom_dirname)
        )
    
    for data_part in ["train", "val", "test"]:

        folder = f"./attention/{dataset}/{aggregator}/{data_part}/"
        attn_file = os.path.join(folder, "attention.npy")
        cubes_attn_file = os.path.join(folder, "cattention.pkl")

        if os.path.exists(attn_file):
            print("Skipping", attn_file) 
            continue

        results = model.forward_iter(
                X={
                    "x_numerical": data[data_part]["features"][:, :ds["n_numerical"]].astype(np.float32),
                    "x_categorical": data[data_part]["features"][:, ds["n_numerical"]:].astype(np.int32)
                    }
                )
        
        n_features = data[data_part]["features"].shape[1]
        n_instances = data[data_part]["features"].shape[0]
        n_layers = bconfig["n_layers"]
        n_heads = bconfig["n_head"]

        if aggregator == "cls":            
            n_features += 1

        attn_mem = np.zeros((n_instances, n_features + 1))
        attn_mem[:, -1] = data[data_part]["outputs"].__array__().flatten()
        idx = 0

        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(cubes_attn_file, "ab") as f:
        
            for _, _, attn in results:

                assert attn.shape[0]==n_layers, "Inconsistent n_layers"
                assert attn.shape[2]==bconfig["n_head"], "Inconsistent n_head"
                assert attn.shape[4]==attn.shape[3], "Inconsistent n_features"

                batch_size = attn.shape[1]

                if aggregator == "cls":  
                    # layers, batch, heads, o_features, i_features          
                    cls_attn = np.copy(attn)
                    cls_attn[-1, :, :, 1:] = np.zeros((n_features - 1, n_features))
                    pickle.dump(cls_attn, f)
                else:
                    pickle.dump(attn, f)
                
                _, cum_attns = compute_std_attentions(attn, aggregator)
                attn_mem[idx:idx+batch_size, :-1] = cum_attns[-1]

                idx += batch_size

        with open(attn_file, "wb") as f:
            np.save(f, attn_mem)
            
        print("Saved", attn_file, attn_mem.shape)    