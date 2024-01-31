import wandb

#from skorch.callbacks import Checkpoint
#from utils.callback import get_default_callbacks
#from utils.training import train

import os
import numpy as np
import json
import pandas as pd
import joblib
from utils import training, processing

from config import (
    PROJECT_NAME,
    WORKER_JOBS,
    N_TRIALS,
    DATA_BASE_DIR,
    CHECKPOINT_BASE_DIR,
    OPTIMIZER,
    MAX_EPOCHS,
    BATCH_SIZE,
    DEVICE
)

from utils import log

logger = log.get_logger()


"""
Define the sweep configuration
"""
sweep_configuration = {
    "method": "random",
    #"metric": {"goal": "maximize", "name": "balanced_accuracy"},
    "metric": {"goal": "minimize", "name": "valid_loss"}, # "valid_loss_opt"
}

parameters = {
        "n_layers": {"values": [2, 4, 8, 16]},
        "optimizer__lr": {"values": [1e-4]},
        "optimizer__weight_decay": {"values": [1e-4]},
        "n_head": {"values": [8, 16, 32]},
        "n_hid": {"values": [128, 256]},
        "attn_dropout": {"values": [0.3]},
        "ff_dropout": {"values": [0.2]},
        "embed_dim": {"values": [128, 256]},
        "numerical_passthrough": {"values": [True, False]}
    }

rnn_parameters = {
        "aggregator__cell": {"values": ["LSTM", "GRU"]},
        "aggregator__hidden_size": {"values": [256]},
        "aggregator__num_layers":  {"values": [2]},
        "aggregator__dropout":  {"values": [0.2]}
}

"""
Reset the WandB environment variables
"""
def reset_wandb_env(exclude={
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }):
    
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]

"""
Training on split function
"""
#def train(fold_name, sweep_id, sweep_run_name, config):
def train(
        dataset, 
        aggregator, 
        dataset_meta,
        fold_name, 
        sweep_id, 
        sweep_run_name, 
        config
    ):
    
    checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, dataset, aggregator, sweep_run_name, fold_name)
    multiclass = len(dataset_meta["labels"]) > 2
    run_name = f"{sweep_run_name}/{fold_name}"

    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )

    logger.info("=" * 40 + f" Running {run_name}")
    train_indices = np.array(dataset_meta["splits"][fold_name]["train"])
    val_indices = np.array(dataset_meta["splits"][fold_name]["val"])
    logger.info(f"Train dataset size: {train_indices.shape[0]}")
    logger.info(f"Valid dataset size: {val_indices.shape[0]}")



    logger.info("Building preprocessing pipeline")
    preprocessor = processing.get_preprocessor(
        dataset_meta["categorical"],
        dataset_meta["numerical"],
        dataset_meta["categories"],
    )

    logger.info("Reading data")
    dataset_file = os.path.join(DATA_BASE_DIR, dataset, "train.csv")
    target_column = dataset_meta["target"]
    n_numerical = dataset_meta["n_numerical"]
    data = pd.read_csv(dataset_file)
    features = data.drop(target_column, axis=1)
    labels = data[target_column]
    
    logger.info("Preprocessing data")
    preprocessor = preprocessor.fit(features.iloc[train_indices])
    X = preprocessor.transform(features)
    y = labels.values

    if multiclass:
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float32)

    logger.info("Saving fitted preprocessing pipeline")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    joblib.dump(preprocessor, os.path.join(checkpoint_dir, "preprocessor.jl"))
    
    logger.info("Building model")
    model = training.build_default_model_from_configs(
        config, 
        dataset_meta,
        fold_name,
        optimizer=OPTIMIZER,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS
    )
    
    model = model.fit(X={
        "x_numerical": X[:, :n_numerical].astype(np.float32),
        "x_categorical": X[:, n_numerical:].astype(np.int32)
        }, 
        y=y
    )

    return model.score(X, y)




"""
Define the function performing the cross validation
"""
def cross_validate():
    
    logger.info("=" * 50 + " Sweep execution")
    
    # Sweep recovering and configuration
    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id
    dataset = sweep_run.config.dataset
    aggregator = sweep_run.config.aggregator

    if not sweep_id:
        raise ValueError("Sweep does not contains ID")
    
    sweep_run_name = sweep_run.name or sweep_run.id
    sweep_run_id = sweep_run.id

    if not sweep_id:
        raise ValueError("Sweep run does not contains ID")

    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)

    # Iterates every fold
    logger.info(f"Reading {dataset} metadata")
    meta_file = os.path.join(DATA_BASE_DIR, dataset, "train.meta.json")
    dataset_meta = None
    with open(meta_file, "r") as f:
        dataset_meta = json.load(f)

    splits = list(dataset_meta["splits"].keys())
    logger.info(f"There are {len(splits)} splits")
    

    metrics = []
    for split in splits:
        logger.info(f"Running split {split}")
        # reset_wandb_env()
        result = train(
            dataset,
            aggregator,
            dataset_meta,
            fold_name=split,
            sweep_id=sweep_id,
            sweep_run_name=sweep_run_name,
            config=dict(sweep_run.config),
        )

        metrics.append(result)

    # resume the sweep run
    #logger.info("Resuming run")
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    # log metric to sweep run
    sweep_run.log(dict(val_accuracy=sum(metrics) / len(metrics)))
    sweep_run.finish()
    logger.info("=" * 50 + " Finishing sweep execution")

"""
Defines the main data flow, it includes:

- Find or create a sweep
- Run trials until it reaches the number of desired trials per experiment
"""
def main():

    wandb.login()
    api = wandb.Api()

    for job in WORKER_JOBS:

        dataset = job["dataset"]
        aggregator = job["aggregator"]
        project = PROJECT_NAME
        n_trials = N_TRIALS

        logger.info("Recovering sweeps")
        logger.info(f"\tProject: {project}")

        sweeps = wandb.Api().project(project).sweeps()
        
        logger.info("Searching in sweeps")
        logger.info(f"\tDataset: {dataset}")
        logger.info(f"\tAggregator: {aggregator}")
        

        sweep_name = f"{dataset}/{aggregator}"
        sweep_id = None

        for s in sweeps:
            print(s.name, s.id)
            if s.name == sweep_name:
                sweep_id = s.id
                break
        
        if not sweep_id:
            logger.info("Sweep does not exists. Creating.")

            s_config = sweep_configuration.copy()
            s_config["name"] = sweep_name
            s_config["parameters"] = {
                **parameters, 
                "dataset": {"value": dataset}, 
                "aggregator": {"value": aggregator}
            }

            if aggregator == "rnn":
                logger.info("Appending RNN aggregator configuration")
                s_config["parameters"] = {**s_config["parameters"], **rnn_parameters}


            sweep_id = wandb.sweep(s_config, project=project)
            logger.info(f"Sweep created with id {sweep_id}")
        else:
            logger.info(f"Sweep found with id {sweep_id}")


        run_trial = True
        trials_count = 0
        while run_trial:
            logger.info(f"Recovering sweep information")
            sweep = api.sweep(f"{project}/sweeps/{sweep_id}")
            sweep.load(force=True)
            existing_trials = len(sweep.runs)
            trials_left = n_trials - existing_trials
            logger.info(f"Sweep has {existing_trials} runs. {trials_left} trials left.")
            
            if trials_left > 0:
                logger.info(f"Running trial")
                wandb.agent(sweep_id, function=cross_validate, project=project, count=1)
                wandb.finish()

            trials_count += 1

            run_trial = trials_left - trials_count > 0

if __name__ == "__main__":
    main()


    
    