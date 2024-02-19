import os
import numpy as np
import json
import pandas as pd
import joblib
import skorch
from utils import training, log
import torch

from config import (
    DATA_BASE_DIR,
    OPTIMIZER,
    BATCH_SIZE,
    DEVICE
)

logger = log.get_logger()

"""
Define the sweep configuration
"""
SCORING_GOAL = "minimize"
SCORING = "valid_loss"


"""
Training on split function
"""


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

    # assert np.allclose(single_attns.sum(axis=-1), 1), "There is a logistic problem: " + str(single_attns.sum(axis=-1))
    # assert np.allclose(cum_attns.sum(axis=-1), 1), "There is a logistic problem: " + str(cum_attns.sum(axis=-1))

    # Before: batch_size, n_layers, n_features
    # After: n_layers, batch_size, n_features
    return single_attns.permute((1, 0, 2)), cum_attns.permute((1, 0, 2))


def extract_attention(
        dataset,
        checkpoint_dir,
        aggregator,
        selection_metric,
        dataset_meta,
        config
    ):
    
    run_name = f"{dataset}-{selection_metric}"
    logger.info("+" * 40 + f" Extracting {run_name}")

    data_dir = os.path.join(DATA_BASE_DIR, dataset, "attention", selection_metric)
    logger.info("Saving attention at " + data_dir)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    multiclass = len(dataset_meta["labels"]) > 2
    
    logger.info(f"Loading preprocessor")
    preprocessor = joblib.load(os.path.join(checkpoint_dir, "preprocessor.jl"))

    logger.info("Reading data")
    target_column = dataset_meta["target"]
    n_numerical = dataset_meta["n_numerical"]
    
    train_dataset_file = os.path.join(DATA_BASE_DIR, dataset, "train.csv")
    train_data = pd.read_csv(train_dataset_file)
    logger.info(f"Training size: {train_data.shape}")

    test_dataset_file = os.path.join(DATA_BASE_DIR, dataset, "test.csv")
    test_data = pd.read_csv(test_dataset_file)
    logger.info(f"Test size: {test_data.shape}")

    data = pd.concat([train_data, test_data], axis=0)
    logger.info(f"Total size: {data.shape}")

    features = data.drop(target_column, axis=1)
    labels = data[target_column]
    
    logger.info("Preprocessing data")
    X = preprocessor.transform(features)
    y = labels.values

    if multiclass:
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float32)

    logger.info("Building model")
    model = training.build_default_model_from_configs(
        config, 
        dataset_meta,
        None,
        optimizer=OPTIMIZER,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        monitor_metric=SCORING, 
        max_epochs=1,
        checkpoint_dir=os.path.join(checkpoint_dir, "ignore"),
    )
    
    logger.info("Loading checkpoint")
    checkpoint = skorch.callbacks.TrainEndCheckpoint(
        dirname=os.path.join(checkpoint_dir, "model")
    )
    load_state = skorch.callbacks.LoadInitState(checkpoint)
    model.callbacks = [load_state]
    model.initialize()
    model.module_.need_weights = True

    preds_iter = model.forward_iter({
            "x_numerical": X[:, :n_numerical].astype(np.float32),
            "x_categorical": X[:, n_numerical:].astype(np.int32)
        })
    
    for preds in preds_iter:
        output, layer_outs, attn = preds
        _, cum_attn = compute_std_attentions(attn, aggregator)
        



"""
Defines the main data flow, it includes:

- Find or create a sweep
- Run trials until it reaches the number of desired trials per experiment
"""
def main():

    """
    Retrieves the best architectures for each dataset depending on the optimization metrics
    """
    # Exports best dataset configuration
    logger.info("Reading selected architectures")
    archs_file = "selected_architectures.csv"
    if not os.path.exists(archs_file):
        raise ValueError(f"File {archs_file} does not exists. Should run model_selection before.")
    
    best_archs_df = pd.read_csv(archs_file)

    archs_file = os.path.join(DATA_BASE_DIR, "architectures.json")
    logger.info(f"Reading architectures from {archs_file}")
    architectures = None
    with open(archs_file, "r") as f:
        architectures = json.load(f)

    
    for job in best_archs_df.iloc:

        dataset = job["dataset"]
        aggregator = job["aggregator"]
        arch_name = job["architecture_name"]
        selection_metric = job["selection_metric"]
        checkpoint_dir = job["checkpoint_dir"]

        logger.info("-" * 60 + f"Running worker {dataset}-{selection_metric}")
       
        search_architectures = architectures.get("rnn" if aggregator == "rnn" else "regular", None)
        
        if not search_architectures:
            logger.fatal("The architectures file is incorrect")
            raise ValueError("The architectures file is incorrect")

        logger.info(f"Running training")
        logger.info(f"Appending dataset and aggregator to configurations")

        arch = search_architectures[arch_name]
        arch["dataset"] = dataset
        arch["aggregator"] = aggregator

        logger.info(f"Reading {dataset} metadata")
        meta_file = os.path.join(DATA_BASE_DIR, dataset, "train.meta.json")
        dataset_meta = None
        with open(meta_file, "r") as f:
            dataset_meta = json.load(f)
            
        extract_attention(
            dataset,
            checkpoint_dir,
            aggregator,
            selection_metric,
            dataset_meta,
            config=arch
        )

        logger.info("-" * 60 + f"Worker finished {dataset}-{selection_metric}")

            


if __name__ == "__main__":
    main()


    
    