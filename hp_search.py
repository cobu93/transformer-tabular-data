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
SWEEP_SCORING_GOAL = "minimize"
SWEEP_SCORING = "valid_loss"

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": SWEEP_SCORING_GOAL, "name": f"{SWEEP_SCORING}_opt"},
}

parameters = {
        "n_layers": {"values": [2, 3, 4, 5]},
        "optimizer__lr": {"values": [1e-4]},
        "optimizer__weight_decay": {"values": [1e-4]},
        "n_head": {"values": [4, 8, 16, 32]},
        # "n_hid": {"values": embed_dim}, # Defined in code
        "attn_dropout": {"values": [0.3]},
        "ff_dropout": {"values": [0.1]},
        "embed_dim": {"values": [128, 256]},
        "numerical_passthrough": {"values": [True, False]}
    }


rnn_parameters = {
        "aggregator__cell": {"values": ["GRU"]},
        "aggregator__hidden_size": {"values": [256]},
        "aggregator__num_layers":  {"values": [1]},
        "aggregator__dropout":  {"values": [0.]}
}

"""
Reset the WandB environment variables
"""
def reset_wandb_env(exclude={
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }):

    backup = {}
    
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            backup[key] = os.environ[key]
            del os.environ[key]

    return backup

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
    multiclass = len(dataset_meta["labels"]) > 2
    

    run = wandb.init(
        name=run_name,
        group=f"{dataset}/{aggregator}/{trial_name}",
        job_type="fold_run",
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
        monitor_metric=SWEEP_SCORING, 
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
        logger.info(f"\tCheckpoint {c}:" + " (sweep scoring)" if c == SWEEP_SCORING else "")
        model = evaluating.load_model(model, os.path.join(checkpoint_dir, "model", c))

        preds = model.predict({
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

        if c == SWEEP_SCORING:
            final_scores = scores.copy()

        for s_k, s_v in scores.items():
            logger.info(f"\t\t{s_k}: {s_v}")

        
        logger.info("\tCompressing and saving model")
        joblib.dump(model, os.path.join(checkpoint_dir, f"model.{c}.jl"))

    logger.info("Saving scores")

    with open(os.path.join(checkpoint_dir, "scores.json"), "w") as f:
        json.dump(summary_scores, f, indent=4)

    logger.info("+" * 40 + f" Finishing {run_name}")
    run.finish()

    return final_scores


"""
Define the function performing the cross validation and its builder
"""

def cross_validate_builder(dataset, aggregator, trial_name):

    def cross_validate():

        sweep_run_name = f"{dataset}-{aggregator}-{trial_name}"
        
        logger.info("=" * 50 + f" Trial execution: {sweep_run_name}")  

        sweep_run = wandb.init(
            name=sweep_run_name,
            group=f"{dataset}/{aggregator}",
            job_type="trial_run",
            reinit=True
        )

        sweep_run_id = sweep_run.id

        if not sweep_run_id:
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
        

        metrics = {}
        for split in splits:
            logger.info(f"Running split {split}")
            reset_wandb_env()
            result = train(
                dataset,
                aggregator,
                trial_name,
                split,
                dataset_meta,
                config=dict(sweep_run.config)
            )

            metrics[split] = result

        logger.info("Computing trial metrics")
        metrics_df = pd.DataFrame(metrics)
        logger.info("\n" + str(metrics_df))
        # resume the sweep run
        sweep_run = wandb.init(id=sweep_run_id, resume="must")
        # log metric to sweep run
        sweep_run.log(metrics_df.mean(axis=1).to_dict())
        logger.info("=" * 50 + f" Finishing trial: {sweep_run_name}")
        sweep_run.finish()

    return cross_validate

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
        entity = ENTITY_NAME
        n_trials = N_TRIALS

        logger.info("Recovering sweeps")
        logger.info(f"\tProject: {project}")

        sweeps = wandb.Api().project(project, entity=entity).sweeps()
        
        logger.info("Searching in sweeps")
        logger.info(f"\tDataset: {dataset}")
        logger.info(f"\tAggregator: {aggregator}")
        

        sweep_name = f"{dataset}-{aggregator}"
        sweep_id = None

        for s in sweeps:
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


            sweep_id = wandb.sweep(s_config, project=project, entity=entity)
            logger.info(f"Sweep created with id {sweep_id}")
        else:
            logger.info(f"Sweep found with id {sweep_id}")


        run_trial = True
        sweep = api.sweep(f"{entity}/{project}/sweeps/{sweep_id}")
        sweep.load(force=True)
        trials_count = len(sweep.runs)
        while run_trial:
            logger.info(f"Recovering sweep information")
            sweep = api.sweep(f"{entity}/{project}/sweeps/{sweep_id}")
            sweep.load(force=True)
            existing_trials = len(sweep.runs)
            trials_left = n_trials - existing_trials
            logger.info(f"Sweep has {existing_trials} runs. {trials_left} trials left.")
            
            if trials_left > 0:
                logger.info(f"Running trial")
                wandb.agent(
                    sweep_id, 
                    function=cross_validate_builder(dataset, aggregator, f"T{trials_count + 1}"), 
                    entity=entity,
                    project=project, 
                    count=1
                )
                wandb.finish()

            trials_count += 1

            run_trial = trials_left - trials_count > 0

if __name__ == "__main__":
    main()


    
    