import wandb

#from skorch.callbacks import Checkpoint
#from utils.callback import get_default_callbacks
#from utils.training import train

import os
import numpy as np
import json
import pandas as pd
import joblib
from utils import training, processing, evaluating

from config import (
    PROJECT_NAME,
    ENTITY_NAME,
    WORKER_JOBS,
    DATA_BASE_DIR,
    CHECKPOINT_BASE_DIR,
    OPTIMIZER,
    MAX_EPOCHS,
    BATCH_SIZE,
    RUN_EXCEPTIONS,
    DEVICE
)

from utils import log

logger = log.get_logger()

"""
Define the sweep configuration
"""
SCORING_GOAL = "minimize"
SCORING = "valid_loss"


"""
Training on split function
"""
#def train(fold_name, sweep_id, sweep_run_name, config):
def train(
        dataset, 
        aggregator, 
        trial_name,
        fold_name,
        dataset_meta,
        config
    ):
    
    run_name = f"{dataset}-{aggregator}-{trial_name}-{fold_name}"
    logger.info("+" * 40 + f" Running {run_name}")

    checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, dataset, aggregator, trial_name, fold_name)
    
    scores_file = os.path.join(checkpoint_dir, "scores.json")
    
    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            summary_scores = json.load(f)

        if SCORING in summary_scores.keys():
            logger.info("This run has been executed before. Skipping.")
            return summary_scores[SCORING]
        else: 
            summary_scores = None
        
    
    multiclass = len(dataset_meta["labels"]) > 2
    
    run = wandb.init(
        name=run_name,
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        group=f"{dataset}/{aggregator}/{trial_name}",
        config=config,
        reinit=True
    )

    
    train_indices = np.array(dataset_meta["splits"][fold_name]["train"])
    val_indices = np.array(dataset_meta["splits"][fold_name]["val"])
    logger.info(f"Train dataset size: {train_indices.shape[0]}")
    logger.info(f"Valid dataset size: {val_indices.shape[0]}")



    logger.info("Building preprocessing pipeline")
    preprocessor = processing.get_preprocessor(
        dataset_meta["categorical"],
        dataset_meta["numerical"],
        dataset_meta["categories"],
        categorical_unknown_value=-1
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
        monitor_metric=SCORING, 
        max_epochs=MAX_EPOCHS,
        checkpoint_dir=os.path.join(checkpoint_dir, "model"),
    )
    
    logger.info("Training model")
    model = model.fit(X={
        "x_numerical": X[:, :n_numerical].astype(np.float32),
        "x_categorical": X[:, n_numerical:].astype(np.int32)
        }, 
        y=y
    )

    logger.info("Computing checkpoints scores")
    final_scores = None
    summary_scores = {}
    checkpoints_list = evaluating.list_models(os.path.join(checkpoint_dir, "model"))
    for c in checkpoints_list:
        logger.info(f"\tCheckpoint {c}:" + " (scoring)" if c == SCORING else "")
        model = evaluating.load_model(model, os.path.join(checkpoint_dir, "model", c))

        preds = model.predict_proba({
                "x_numerical": X[val_indices, :n_numerical].astype(np.float32),
                "x_categorical": X[val_indices, n_numerical:].astype(np.int32)
            })
    
        scores = evaluating.get_default_scores(
            y[val_indices],
            preds,
            prefix="",
            multiclass=multiclass
        )

        summary_scores[c] = scores.copy()

        if c == SCORING:
            final_scores = scores.copy()

        for s_k, s_v in scores.items():
            logger.info(f"\t\t{s_k}: {s_v}")
        
    logger.info("Saving scores")

    with open(scores_file, "w") as f:
        json.dump(summary_scores, f, indent=4)

    logger.info("+" * 40 + f" Finishing {run_name}")
    run.finish()

    return final_scores


"""
Define the function performing the cross validation and its builder
"""

def cross_validate(dataset, aggregator, trial_name, config):

    sweep_run_name = f"{dataset}-{aggregator}-{trial_name}"
    
    logger.info("=" * 50 + f" Trial execution: {sweep_run_name}")  

    # Iterates every fold
    logger.info(f"Reading {dataset} metadata")
    meta_file = os.path.join(DATA_BASE_DIR, dataset, "train.meta.json")
    dataset_meta = None
    with open(meta_file, "r") as f:
        dataset_meta = json.load(f)

    splits = list(dataset_meta["splits"].keys())
    logger.info(f"There are {len(splits)} splits")
    

    metrics = {}
    for split in splits:
        logger.info(f"Running split {split}")
        
        train(
            dataset,
            aggregator,
            trial_name,
            split,
            dataset_meta,
            config=config
        )

        

    logger.info("=" * 50 + f" Finishing trial: {sweep_run_name}")

    
    

"""
Defines the main data flow, it includes:

- Find or create a sweep
- Run trials until it reaches the number of desired trials per experiment
"""
def main():

    logger.info("Logging in WandB")
    wandb.login()


    for job in WORKER_JOBS:

        dataset = job["dataset"]
        aggregator = job["aggregator"]

        logger.info("-" * 60 + f"Running worker {dataset}-{aggregator}")

        archs_file = os.path.join(DATA_BASE_DIR, "architectures.json")
        logger.info(f"Reading architectures from {archs_file}")
        architectures = None
        with open(archs_file, "r") as f:
            architectures = json.load(f)

        architectures = architectures.get("rnn" if aggregator == "rnn" else "regular", None)
        
        if not architectures:
            logger.fatal("The architectures file is incorrect")
            raise ValueError("The architectures file is incorrect")

        logger.info(f"There exists {len(architectures)} architectures")
        logger.info(f"Appending dataset and aggregator to configurations")

        for _, arch in architectures.items():
            arch["dataset"] = dataset
            arch["aggregator"] = aggregator

        for arch_name, arch in architectures.items():
            skip_run = True
            checked_items = 0
            
            for run_exception in RUN_EXCEPTIONS:
                for k, v in run_exception.items():
                    skip_run = skip_run and (arch.get(k, None) == v)
                    checked_items += 1

            if skip_run and checked_items > 0:
                logger.info(f"Skipping run because it is in exceptions: {dataset}-{aggregator}-{arch_name}")
            else:
                cross_validate(dataset, aggregator, arch_name, arch)
            

        logger.info("-" * 60 + f"Worker finished {dataset}-{aggregator}")

            


if __name__ == "__main__":
    main()


    
    