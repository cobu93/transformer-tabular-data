import os
import numpy as np
import json
import pandas as pd
import joblib
import skorch
from utils import training, log, attention, evaluating, processing

from sklearn import linear_model, pipeline, svm, neighbors, base

from config import (
    DATA_BASE_DIR,
    OPTIMIZER,
    BATCH_SIZE,
    CHECKPOINT_BASE_DIR,
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

def read_meta_csv(dirname, file_prefix):
    dataset_file = os.path.join(dirname, f"{file_prefix}.csv")
    meta_file = os.path.join(dirname, f"{file_prefix}.meta.json")
    data = pd.read_csv(dataset_file)

    with open(meta_file, "r") as f:
        meta = json.load(f)

    return data, meta

def processing_return_features(X, aggregator):
    if aggregator == "cls":
        return np.hstack((np.zeros((X.shape[0], 1)), X))
    return X


def extract_attention(
        dataset,
        checkpoint_dir,
        aggregator,
        selection_metric,
        config
    ):
    
    run_name = f"{dataset}-{selection_metric}"
    logger.info("+" * 40 + f" Extracting {run_name}")

    data_dir = os.path.join(DATA_BASE_DIR, dataset, "attention", selection_metric)
    attention_file = os.path.join(data_dir, "attention.npy")
        
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    logger.info(f"Loading preprocessor")
    preprocessor = joblib.load(os.path.join(checkpoint_dir, "preprocessor.jl"))

    logger.info("Reading data")
    
    datasets_dirname = os.path.join(DATA_BASE_DIR, dataset)
    train_data, dataset_meta = read_meta_csv(datasets_dirname, "train")
    train_indices = dataset_meta["df_indices"]
    target_column = dataset_meta["target"]
    n_numerical = dataset_meta["n_numerical"]
    logger.info(f"Training size: {train_data.shape}")

    test_data, test_dataset_meta = read_meta_csv(datasets_dirname, "test")
    test_indices = test_dataset_meta["df_indices"]
    logger.info(f"Test size: {test_data.shape}")

    data = pd.concat([train_data, test_data], axis=0)
    logger.info(f"Total size: {data.shape}")

    logger.info("Sorting dataset as original")
    indices = train_indices + test_indices
    data.index = indices
    data = data.sort_index()
    
    features = data.drop(target_column, axis=1)
    labels = data[target_column]
    
    logger.info("Preprocessing data")
    X = preprocessor.transform(features)
    y = labels.values

    multiclass = len(dataset_meta["labels"]) > 2
    
    if multiclass:
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float32)


    if os.path.exists(attention_file):
        logger.info("Skipping extraction. Attention file exists.")
        with open(attention_file, "rb") as f:
            cum_attns = np.load(f)
        return cum_attns

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
    

    n_instances, n_features = X.shape
    n_features -= n_numerical if config["numerical_passthrough"] else 0
    
    if aggregator == "cls":
        n_features += 1
    
    cum_attns = np.zeros((n_instances, n_features))

    for i, preds in enumerate(preds_iter):
        output, layer_outs, attn = preds
        _, batch_cum_attn = attention.compute_std_attentions(attn, aggregator)
        cum_attns[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = batch_cum_attn[-1]
        
    assert np.allclose(cum_attns.sum(axis=1), 1), "Something went wrong with attentions"
    
    logger.info("Saving attention at " + data_dir)
    
    with open(attention_file, "wb") as f:
        np.save(f, cum_attns)

    return cum_attns
    

def attention_percent_evaluation(
    dataset,
    aggregator,
    cum_attn,
    selection_metric,
    attn_selectors=[0.9],
    models=[{"name": "LR", "model": linear_model.LogisticRegression()}]
    ):

    logger.info("Reading data")
    
    datasets_dirname = os.path.join(DATA_BASE_DIR, dataset)
    train_data, dataset_meta = read_meta_csv(datasets_dirname, "train")
    splits = dataset_meta["splits"]
    target_column = dataset_meta["target"]
    features = train_data.drop(target_column, axis=1)
    labels = train_data[target_column]
    multiclass = len(dataset_meta["labels"]) > 2

    logger.info("=" * 30 + " Masking method")
    eval_data = []

    for s in attn_selectors:
        logger.info(f"Attention selector: {s}")
        attn_mask = attention.get_attention_mask(cum_attn, s)

        if aggregator == "cls":
            attn_mask = attn_mask[:, 1:]

        mask_df = pd.DataFrame(data=attn_mask, columns=features.columns)
        train_data = train_data[mask_df]

        for m_info in models:
            m_name = m_info["name"]
            m = m_info["model"]
            logger.info(f"Model: {m_name}")

            general_clf = pipeline.make_pipeline(
                processing.get_regular_preprocessor(
                    dataset_meta["categorical"],
                    dataset_meta["numerical"],
                    dataset_meta["categories"],
                ),
                m
            )

            for s_name, s_indices in splits.items():

                logger.info(f"Fold: {s_name}")

                clf_checkpoint = os.path.join(
                                    CHECKPOINT_BASE_DIR, 
                                    dataset,
                                    "attention",
                                    selection_metric,
                                    m_name,
                                    s_name
                                    )
                
                if not os.path.exists(clf_checkpoint):
                    os.makedirs(clf_checkpoint)

                scores_checkpoint = os.path.join(clf_checkpoint, "scores.json")

                if not os.path.exists(scores_checkpoint):
                
                    clf = base.clone(general_clf)

                    clf = clf.fit(
                        train_data.loc[s_indices["train"]],
                        labels.loc[s_indices["train"]]
                    )

                    joblib.dump(clf, os.path.join(clf_checkpoint, "model.jl"))

                    preds = clf.predict_proba(train_data.loc[s_indices["val"]])

                    scores = evaluating.get_default_scores(
                        labels[s_indices["val"]], preds, multiclass=multiclass
                    )

                    with open(scores_checkpoint, "w") as f:
                        json.dump(scores, f, indent=4)

                else:
                    logger.info("Skipping because it was trained before")
                    with open(scores_checkpoint, "r") as f:
                        scores = json.load(f)

                for k, v in scores.items():
                    logger.info(f"\t{k}: {v}")

                scores = {
                    "model": m_name,
                    "fold": s_name,
                    "attention_selection": s,
                    "selection_metric": selection_metric,
                    **scores
                }

                eval_data.append(scores)

    return eval_data
                


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

        cum_attn = extract_attention(
            dataset,
            checkpoint_dir,
            aggregator,
            selection_metric,
            config=arch
        )

        attention_percent_evaluation(
            dataset,
            aggregator,
            cum_attn,
            selection_metric,
            attn_selectors = np.arange(0.1, 1.1, .1),
            models=[
                {"name":"KNN", "model": neighbors.KNeighborsClassifier()},
                {"name":"SVC", "model": svm.SVC()},
                {"name":"LR", "model": linear_model.LogisticRegression()}
            ]
        )


        logger.info("-" * 60 + f"Worker finished {dataset}-{selection_metric}")

            


if __name__ == "__main__":
    main()


    
    