import wandb

#from skorch.callbacks import Checkpoint
#from utils.callback import get_default_callbacks
#from utils.training import train

import os
import numpy as np
import json
import pandas as pd
import joblib
import skorch
from utils import training, processing, evaluating, reporting, callback

from config import (
    PROJECT_NAME,
    ENTITY_NAME,
    DATA_BASE_DIR,
    CHECKPOINT_BASE_DIR,
    OPTIMIZER,
    REFIT_SELECTION_METRICS,
    BATCH_SIZE,
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
def refit(
        dataset,
        aggregator,
        architecture_name, 
        selection_metric,
        dataset_meta,
        config,
        best_tag="best",
        extra_epochs_part=0. # With 100% for training / 5 = 20%, then with 20% extra data, we train 20% more
    ):
    
    run_name = f"{dataset}-{best_tag}-{selection_metric}"
    logger.info("+" * 40 + f" Running {run_name}")

    checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, dataset, best_tag, selection_metric)
    logger.info("Checkpoint dir " + checkpoint_dir)
    
    scores_file = os.path.join(checkpoint_dir, "scores.json")
    
    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            scores = json.load(f)

        return scores, checkpoint_dir
    
    multiclass = len(dataset_meta["labels"]) > 2
    
    run = wandb.init(
        name=run_name,
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        group=f"{dataset}/{best_tag}/{selection_metric}",
        config=config,
        reinit=True
    )

    
    train_indices = np.array(dataset_meta["df_indices"])
    logger.info(f"Train dataset size: {train_indices.shape[0]}")

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
    preprocessor = preprocessor.fit(features)
    
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


    logger.info(f"Computing max epochs + {extra_epochs_part * 100}%")
    cv_folder = os.path.join(CHECKPOINT_BASE_DIR, dataset, aggregator, architecture_name)

    n_epochs = []

    for fold_name in os.listdir(cv_folder):
        with open(os.path.join(cv_folder, fold_name, "model", "valid_loss", "history.json")) as f:
            history = json.load(f)
        n_epochs.append(len(history))

    n_epochs = np.max(n_epochs)
    logger.info(f"Max epochs: {n_epochs}")
    n_epochs *= (1 + extra_epochs_part)
    n_epochs = int(np.ceil(n_epochs))
    logger.info(f"Final epochs: {n_epochs}")
    
    
    logger.info("Building model")
    model = training.build_default_model_from_configs(
        config, 
        dataset_meta,
        None,
        optimizer=OPTIMIZER,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        monitor_metric=SCORING, 
        max_epochs=n_epochs,
        checkpoint_dir=os.path.join(checkpoint_dir, "ignore"),
    )

    model_checkpoint_path = os.path.join(checkpoint_dir, "model")
    model_checkpoint = [(f"checkpoint", skorch.callbacks.TrainEndCheckpoint(
                    dirname=model_checkpoint_path
                    ))]

    model.callbacks = callback.get_default_callbacks(multiclass=multiclass) + model_checkpoint
    
    logger.info("Training model")
    model = model.fit(X={
        "x_numerical": X[:, :n_numerical].astype(np.float32),
        "x_categorical": X[:, n_numerical:].astype(np.int32)
        }, 
        y=y
    )

    logger.info("Computing checkpoints scores")
    logger.info(f"\tLoading test data")
    test_data = pd.read_csv(os.path.join(DATA_BASE_DIR, dataset, "test.csv"))
    test_features = test_data.drop(target_column, axis=1)
    test_labels = test_data[target_column]
    
    logger.info(f"\tPreprocessing test data")
    
    for c in test_features.columns:
        columns_uniques = test_features[c].unique()
        if len(columns_uniques) == 1 \
            and not isinstance(columns_uniques[0], str) \
            and np.isnan(columns_uniques[0]):
            test_features[c] = test_features[c].astype("object")

    X_test = preprocessor.transform(test_features)
    y_test = test_labels.values

    preds = model.predict_proba({
            "x_numerical": X_test[:, :n_numerical].astype(np.float32),
            "x_categorical": X_test[:, n_numerical:].astype(np.int32)
        })

    scores = evaluating.get_default_scores(
        y_test,
        preds,
        prefix="",
        multiclass=multiclass
    )

    for s_k, s_v in scores.items():
        logger.info(f"\t\t{s_k}: {s_v}")
    
    logger.info("Saving scores")

    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)

    logger.info("+" * 40 + f" Finishing {run_name}")
    run.finish()

    return scores, checkpoint_dir


"""
Defines the main data flow, it includes:

- Find or create a sweep
- Run trials until it reaches the number of desired trials per experiment
"""
def main():

    logger.info("Logging in WandB")
    wandb.login()

    """
    Retrieves the best architectures for each dataset depending on the optimization metrics
    """
    # Exports best dataset configuration
    executions_df = reporting.load_report_df()
    executions_cv_df = reporting.get_executions_cv_score(executions_df)

    executions_cv_df.to_csv("cross_validation_scores.csv", index=False)
    
    best_archs = []

    for selection_config in REFIT_SELECTION_METRICS:
        selection_metric = selection_config["metric"]
        selection_mode = selection_config["mode"]


        best_archs_df = executions_cv_df.loc[reporting.get_top_k_indices(
                                        executions_cv_df.groupby(["dataset"]),
                                        1,
                                        f"{selection_metric}_mean",
                                        selection_mode
                                        )
                                  ]
        
        best_archs_df["selection_metric"] = selection_metric
        best_archs_df["selection_mode"] = selection_mode
        
        best_archs.extend(best_archs_df.to_dict("records"))

    best_archs_df = pd.DataFrame(best_archs)
    logger.info(f"Total architectures: {len(best_archs_df)}")
    best_archs_df = best_archs_df.dropna(subset=[f"{m['metric']}_mean" for m in REFIT_SELECTION_METRICS])
    logger.info(f"After removing nan: {len(best_archs_df)}")
    best_archs_df_unique = best_archs_df \
                                .drop_duplicates(subset=["dataset", "aggregator", "architecture_name"]) \
                                .sort_values("dataset")
    logger.info(f"There are {len(best_archs_df)} unique architectures to train")

    archs_file = os.path.join(DATA_BASE_DIR, "architectures.json")
    logger.info(f"Reading architectures from {archs_file}")
    architectures = None
    with open(archs_file, "r") as f:
        architectures = json.load(f)

    logger.info("Thebest architectures are:")
    logger.info(str(best_archs_df_unique))

    scores = []
    for job in best_archs_df_unique.iloc:

        dataset = job["dataset"]
        aggregator = job["aggregator"]
        arch_name = job["architecture_name"]
        selection_metric = job["selection_metric"]

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
            
        results, checkpoint_dir = refit(
            dataset,
            aggregator,
            arch_name,
            selection_metric,
            dataset_meta,
            config=arch
        )

        scores.append({
            **{f"{k}_test": v for k, v in results.items()},
            "dataset": dataset,
            "aggregator": aggregator,
            "architecture_name": arch_name,
            "checkpoint_dir": checkpoint_dir
        })

        logger.info("-" * 60 + f"Worker finished {dataset}-{selection_metric}")

    best_archs_df = best_archs_df.merge(
                        pd.DataFrame(scores), 
                        on=["dataset", "aggregator", "architecture_name"]
                    ).sort_values("dataset")

    logger.info("Saving scored architectures")
    best_archs_df.to_csv("selected_architectures.csv", index=False)        


if __name__ == "__main__":
    main()


    
    