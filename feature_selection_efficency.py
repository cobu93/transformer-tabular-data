from config import (
    DATA_BASE_DIR,
    CHECKPOINT_BASE_DIR,
    OPTIMIZER,
    DEVICE,
    BATCH_SIZE,
    MAX_EPOCHS,
    SEED
)

import os
import json
import numpy as np
import pandas as pd
import skorch
import shutil
import xgboost
import time
import torch.nn as nn

from sklearn import (
                    linear_model,
                    tree, 
                    feature_selection, 
                    model_selection
                )

from attention_evaluation_cluster import (
        build_masker_from_model,
        build_masker_from_score,
)


from utils import log, processing, training, evaluating, reporting

logger = log.get_logger()
FS_EFFICENCY_CHECKPOINT_BASE_DIR = "fs_efficency"



def get_transformer_cv_stats(
        X_train,
        y_train,
        X_val,
        y_val,
        dataset,
        dataset_meta,
        fold_name,
        arch_name,
        arch,
        execution_name="cross_validation",
        mask=None
    ):

    f_scores_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, "transformer", execution_name, arch_name, fold_name)
    f_scores_filename = os.path.join(f_scores_dir, "scores.json")

    if not os.path.exists(f_scores_dir):
        os.makedirs(f_scores_dir)

    if os.path.exists(f_scores_filename):
        with open(f_scores_filename, "r") as f:
            scores = json.load(f)

        return scores

    with open(os.path.join(DATA_BASE_DIR, "architectures.json"), "r") as f:
        architectures = json.load(f)["regular"]

    original_arch_name = ""
    for o_arch_name, o_arch in architectures.items():
        is_same = True
        for hparam_name, hparam_value in o_arch.items():
            if(arch[hparam_name] != hparam_value):
                is_same = False
                break

        if is_same:
           original_arch_name = o_arch_name
           break

    aggregator = arch["aggregator"]
    
    checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, dataset, aggregator, 
                        original_arch_name, fold_name, "model", "valid_loss")

    multiclass = len(dataset_meta["labels"]) > 2
    n_numerical = dataset_meta["n_numerical"]

    model = training.build_default_model_from_configs(
        arch, 
        dataset_meta,
        fold_name,
        optimizer=OPTIMIZER,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        monitor_metric="valid_loss", 
        max_epochs=MAX_EPOCHS,
        checkpoint_dir=os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, "none"),
    )

    model.initialize()
    model.callbacks = None
    
    model.load_params(checkpoint=skorch.callbacks.Checkpoint(dirname=checkpoint_dir))
    shutil.rmtree(os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, "none"))

    if mask is not None:

        model.module_.numerical_encoder.weights = nn.Parameter(model.module_.numerical_encoder.weights[mask[:n_numerical]])
        model.module_.numerical_encoder.biases = nn.Parameter(model.module_.numerical_encoder.biases[mask[:n_numerical]])

        model.module_.register_buffer("categories_offset", model.module_.categories_offset[mask[n_numerical:]])
        n_numerical = int(mask[:n_numerical].sum())


    inf_start_time = time.time()
    preds = model.predict_proba({
            "x_numerical": X_val[:, :n_numerical].astype(np.float32),
            "x_categorical": X_val[:, n_numerical:].astype(np.int32)
        })
    inf_end_time = time.time()

    scores = evaluating.get_default_scores(
        y_val,
        preds,
        prefix="",
        multiclass=multiclass
    )

    scores["inference_time"] = inf_end_time - inf_start_time

    history_file = os.path.join(checkpoint_dir, "history.json")
    with open(history_file, "r") as f:
        history = json.load(f)

    epochs_times = [history[i]["dur"] for i in range(len(history))]
    mean_epochs_times = np.mean(epochs_times)
    total_epochs = int(min(len(history) + 0.2 * MAX_EPOCHS, MAX_EPOCHS))
    epochs_times = epochs_times + [mean_epochs_times] * (total_epochs - len(history))
    total_time = np.sum(epochs_times)
    
    scores["training_time"] = total_time

    with open(f_scores_filename, "w") as f:
        json.dump(scores, f, indent=4)

    return scores

def get_xgboost_cv_stats(
        X_train,
        y_train,
        X_val,
        y_val,
        dataset,
        dataset_meta,
        fold_name,
        arch_name,
        arch,
        execution_name="cross_validation",
        mask=None
    ):

    f_scores_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, "xgboost", execution_name, arch_name, fold_name)
    f_scores_filename = os.path.join(f_scores_dir, "scores.json")

    if not os.path.exists(f_scores_dir):
        os.makedirs(f_scores_dir)


    if os.path.exists(f_scores_filename):
        with open(f_scores_filename, "r") as f:
            scores = json.load(f)

        return scores

    multiclass = len(dataset_meta["labels"]) > 2

    opt_metric = "mlogloss" if multiclass else "logloss"
    
    model = xgboost.XGBClassifier(
                n_estimators=MAX_EPOCHS, 
                eval_metric=opt_metric,
                callbacks=[
                    xgboost.callback.EarlyStopping(
                        int(0.2 * MAX_EPOCHS), 
                        metric_name=opt_metric, 
                        data_name="validation_0",
                        maximize=False, 
                        save_best=True)
                    ],
                random_state=SEED, 
                device=DEVICE, 
                **arch
            )
    
    start_time = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=2)
    end_time = time.time()

    inf_start_time = time.time()
    preds = model.predict_proba(X_val)
    inf_end_time = time.time()
    
    scores = evaluating.get_default_scores(
        y_val,
        preds,
        prefix="",
        multiclass=multiclass
    )

    scores["training_time"] = end_time - start_time
    scores["inference_time"] = inf_end_time - inf_start_time

    with open(f_scores_filename, "w") as f:
        json.dump(scores, f, indent=4)

    return scores
    

models_info = {
    "transformer": {
        "cv_stats_fn": get_transformer_cv_stats,
        "masked_cv_stats_fn": get_transformer_cv_stats,
        "hp_space": {
            "aggregator": ["cls"],
            "attn_dropout": [0.3],
            "embed_dim": [128, 256],
            "ff_dropout": [0.1],
            "n_head": [4, 8, 16, 32],
            "n_layers": [2, 3, 4, 5],
            "numerical_passthrough": [False],
            "optimizer__lr": [0.0001],
            "optimizer__weight_decay": [0.0001]
        }
    },
    "xgboost": {
        "cv_stats_fn": get_xgboost_cv_stats,
        "masked_cv_stats_fn": get_xgboost_cv_stats,
        "hp_space": {
            "max_depth": (np.arange(20) + 1).tolist(),
            "learning_rate": [0.1]
        }
    }
}

def evaluate_feature_selection_cv(
        best_archs_df,
        feature_selectors,
        feature_percents,
        models_info=models_info
    ):


    if not os.path.exists(FS_EFFICENCY_CHECKPOINT_BASE_DIR):
        os.makedirs(FS_EFFICENCY_CHECKPOINT_BASE_DIR)

    datasets = best_archs_df["dataset"].unique().tolist()

    scores = []
    
    for dataset in datasets:
        checkpoint_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        logger.info(f"Reading {dataset} metadata")
        meta_file = os.path.join(DATA_BASE_DIR, dataset, "train.meta.json")
        dataset_meta = None

        with open(meta_file, "r") as f:
            dataset_meta = json.load(f)

        multiclass = len(dataset_meta["labels"]) > 2

        logger.info("Reading data")
        dataset_file = os.path.join(DATA_BASE_DIR, dataset, "train.csv")
        target_column = dataset_meta["target"]
        n_numerical = dataset_meta["n_numerical"]
        data = pd.read_csv(dataset_file)
        features = data.drop(target_column, axis=1)
        labels = data[target_column]

        splits = list(dataset_meta["splits"].keys())
        logger.info(f"There are {len(splits)} splits")

        for split in splits:
            for f_selector in feature_selectors:
                mask_fn_name = f_selector["name"]
                mask_fn = f_selector["mask_fn"]

                train_indices = dataset_meta["splits"][split]["train"]
                val_indices = dataset_meta["splits"][split]["val"]

                preprocessor = processing.get_preprocessor(
                    dataset_meta["categorical"],
                    dataset_meta["numerical"],
                    dataset_meta["categories"],
                    categorical_unknown_value=-1
                )

                preprocessor = preprocessor.fit(features.iloc[train_indices])
    
                X = preprocessor.transform(features)
                y = labels.values

                if multiclass:
                    y = y.astype(np.int64)
                else:
                    y = y.astype(np.float32)

                for f_percent in feature_percents:
                    n_features_selected = int(features.shape[1] * f_percent)
                    mask = mask_fn(
                                features.iloc[train_indices],
                                labels.iloc[train_indices],
                                None,
                                dataset_meta,
                                n_features_selected
                            )
                    
                    assert mask.sum() == n_features_selected, "Something went wrong generating the mask"
                    
                    current_features_order = dataset_meta["numerical"] + dataset_meta["categorical"]
                    mask_features_order = features.columns
                    mask_sort_indices = [current_features_order.index(f) for f in mask_features_order]
                    mask = mask[mask_sort_indices]
                    
                    for job in best_archs_df.query("dataset==@dataset").iloc:

                        m = job["model"]
                        arch_name = job["architecture_name"]

                        logger.info(f"Running {split}:{mask_fn_name}:{f_percent}:{m}:{arch_name}")
                        archs_filename = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, f"{m}_architectures.json")

                        with open(archs_filename, "r") as f:
                            architectures = json.load(f)

                        arch = architectures[arch_name]
                        stats_fn = models_info[m]["masked_cv_stats_fn"]

                        mf_scores = stats_fn(
                                X[train_indices][:, mask],
                                y[train_indices],
                                X[val_indices][:, mask],
                                y[val_indices],
                                dataset,
                                dataset_meta,
                                split,
                                arch_name,
                                arch,
                                execution_name=f"fs_evaluation/{mask_fn_name}/{f_percent}",
                                mask=mask
                            )
                        
                        mf_scores["dataset"] = dataset
                        mf_scores["model"] = m
                        mf_scores["fold"] = split
                        mf_scores["architecture_name"] = arch_name
                        mf_scores["fs_method"] = mask_fn_name
                        mf_scores["fs_percent"] = f_percent
                        mf_scores["fs_n_features"] = n_features_selected

                        scores.append(mf_scores)
    
    scores_df = pd.DataFrame(scores)
    
    mean_scores_df = scores_df.drop(["fold"], axis=1) \
            .groupby(["dataset", "model", "architecture_name", "fs_method", "fs_percent", "fs_n_features"], as_index=False, dropna=False) \
            .agg(["mean", "std"])
    
    mean_scores_df.columns = ["_".join(col) if col[1] else col[0] for col in mean_scores_df.columns]
    
    return mean_scores_df
                        
                        
def get_best_architectures(
        datasets,
        models,
        models_info=models_info
    ):

    if not os.path.exists(FS_EFFICENCY_CHECKPOINT_BASE_DIR):
        os.makedirs(FS_EFFICENCY_CHECKPOINT_BASE_DIR)

    scores = []

    for dataset in datasets:
        checkpoint_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        logger.info(f"Reading {dataset} metadata")
        meta_file = os.path.join(DATA_BASE_DIR, dataset, "train.meta.json")
        dataset_meta = None

        with open(meta_file, "r") as f:
            dataset_meta = json.load(f)

        multiclass = len(dataset_meta["labels"]) > 2

        logger.info("Reading data")
        dataset_file = os.path.join(DATA_BASE_DIR, dataset, "train.csv")
        target_column = dataset_meta["target"]
        n_numerical = dataset_meta["n_numerical"]
        data = pd.read_csv(dataset_file)
        features = data.drop(target_column, axis=1)
        labels = data[target_column]

        splits = list(dataset_meta["splits"].keys())
        logger.info(f"There are {len(splits)} splits")

        for split in splits:
            
            train_indices = dataset_meta["splits"][split]["train"]
            val_indices = dataset_meta["splits"][split]["val"]

            preprocessor = processing.get_preprocessor(
                dataset_meta["categorical"],
                dataset_meta["numerical"],
                dataset_meta["categories"],
                categorical_unknown_value=-1
            )

            preprocessor = preprocessor.fit(features.iloc[train_indices])
    
            X = preprocessor.transform(features)
            y = labels.values

            if multiclass:
                y = y.astype(np.int64)
            else:
                y = y.astype(np.float32)

            for m in models:
                archs_filename = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, f"{m}_architectures.json")

                models_cv_stats_fn = models_info[m]["cv_stats_fn"]

                with open(archs_filename, "r") as f:
                    architectures = json.load(f)

                for arch_name, arch in architectures.items():
                    logger.info(f"Running cross validation {split}:{m}:{arch_name}")

                    mf_scores = models_cv_stats_fn(
                        X[train_indices],
                        y[train_indices],
                        X[val_indices],
                        y[val_indices],
                        dataset,
                        dataset_meta,
                        split,
                        arch_name,
                        arch,
                        execution_name="cross_validation"
                    )

                    mf_scores["dataset"] = dataset
                    mf_scores["model"] = m
                    mf_scores["fold"] = split
                    mf_scores["architecture_name"] = arch_name
                    mf_scores["fs_method"] = "none"
                    mf_scores["fs_percent"] = 1.0
                    mf_scores["fs_n_features"] = features.shape[1]

                    scores.append(mf_scores)

    scores_df = pd.DataFrame(scores)
    
    mean_scores_df = scores_df.drop(["fold"], axis=1) \
            .groupby(["dataset", "model", "architecture_name", "fs_method", "fs_percent", "fs_n_features"], as_index=False, dropna=False) \
            .agg(["mean", "std"])
    
    mean_scores_df.columns = ["_".join(col) if col[1] else col[0] for col in mean_scores_df.columns]
    
    best_archs_dfs = []
    for m in models:
        executions_cv_df = mean_scores_df.query("model==@m")

        best_archs_dfs.append( executions_cv_df.loc[reporting.get_top_k_indices(
                                    executions_cv_df.groupby(["dataset"]),
                                    1,
                                    "log_loss_mean",
                                    "min"
                                    )
                                ])
        
    best_archs_df = pd.concat(best_archs_dfs, axis=0)
    return best_archs_df
                        
def setup(models):
    logger.info("=" * 50 + "Setup architectures")
    
    for model in models:
        architectures = {}
        parameters = models_info[model]["hp_space"]
        param_grid = list(model_selection.ParameterGrid(parameters))

        logger.info(f"There are {len(param_grid)} regular architectures for {model}")
        for i, arch in enumerate(param_grid):
            architectures[f"A{i}"] = arch
        
    
        archs_file = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, f"{model}_architectures.json")
        if os.path.exists(archs_file):
            with open(archs_file, "r") as f:
                old_architectures = json.load(f)
            
            if str(dict(sorted(old_architectures.items()))) != str(dict(sorted(architectures.items()))):
                logger.fatal("Previous an current architectures are not the same")
                raise ValueError("Previous an current architectures are not the same")
        else:
            with open(archs_file, "w") as f:
                json.dump(architectures, f, indent=4)

if __name__ == "__main__":

    models = ["transformer", "xgboost"]
    #datasets = ["kr-vs-kp", "sylvine", "nomao", "jasmine"]
    datasets = ["kr-vs-kp", "sylvine", "nomao", "adult", "jasmine"]

    if not os.path.exists(FS_EFFICENCY_CHECKPOINT_BASE_DIR):
        os.makedirs(FS_EFFICENCY_CHECKPOINT_BASE_DIR)

    
    scores_filename = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, "scores.csv")

    
    setup(models)
    
    best_archs_df = get_best_architectures(
        datasets = datasets,
        models = models
    )


    mean_scores_df = evaluate_feature_selection_cv(
        best_archs_df,
        feature_selectors = [
            {"name": "linear_model", "mask_fn": build_masker_from_model(linear_model.LogisticRegression(random_state=SEED)), "level": "fold"},
            {"name": "decision_tree", "mask_fn": build_masker_from_model(tree.DecisionTreeClassifier(random_state=SEED)), "level": "fold"},
            {"name": "f_classif", "mask_fn": build_masker_from_score(feature_selection.f_classif), "level": "fold"}       
        ],
        feature_percents = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    fs_scores_df = pd.concat([best_archs_df, mean_scores_df], axis=0)

    fs_scores_com_df = fs_scores_df.copy()
    fs_scores_com_df = fs_scores_com_df[fs_scores_com_df["fs_method"] != "none"]

    for fs_method in fs_scores_df.query("fs_method != 'none'")["fs_method"].unique():
        for ds in fs_scores_df["dataset"].unique():
            
            insertion_df = fs_scores_df.query(
                "fs_method=='none' "
                "and dataset==@ds"
            ) 
            
            insertion_df.loc[:, "fs_method"] = fs_method
            
            fs_scores_com_df = pd.concat([fs_scores_com_df, insertion_df])


    fs_scores_com_df.to_csv(scores_filename, index=False)
