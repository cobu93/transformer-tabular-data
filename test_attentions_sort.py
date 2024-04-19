from config import (
    DATA_BASE_DIR,
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
from skorch import toy
import shutil
import xgboost
import time
import torch.nn as nn
import joblib
import torch
import copy

from attention_evaluation_cluster import (
        build_masker_from_model,
        build_masker_from_score,
        get_random_mask
)

from sklearn import (
                    linear_model,
                    tree, 
                    feature_selection, 
                    model_selection,
                    preprocessing
                )


from utils import log, processing, training, evaluating, reporting, attention

logger = log.get_logger()


def evaluate_feature_selection_order(
        datasets,
        models,
        feature_selectors,
        feature_percents
    ):

    stats = []
    non_valid = 0
    non_valid_mean = 0
    total_exps = 0
    for dataset in datasets:
        
        logger.info(f"Reading {dataset} metadata")
        meta_file = os.path.join(DATA_BASE_DIR, dataset, "train.meta.json")
        test_meta_file = os.path.join(DATA_BASE_DIR, dataset, "test.meta.json")
        dataset_meta = None

        with open(meta_file, "r") as f:
            dataset_meta = json.load(f)

        with open(test_meta_file, "r") as f:
            test_dataset_meta = json.load(f)

        multiclass = len(dataset_meta["labels"]) > 2

        logger.info("Reading data")
        dataset_file = os.path.join(DATA_BASE_DIR, dataset, "train.csv")
        test_dataset_file = os.path.join(DATA_BASE_DIR, dataset, "test.csv")
        target_column = dataset_meta["target"]
        n_numerical = dataset_meta["n_numerical"]
        data = pd.read_csv(dataset_file)
        
        features = data.drop(target_column, axis=1)
        labels = data[target_column]

        mask_stats = []

        param_grid = list(model_selection.ParameterGrid(models))

        architectures = {}
        for i, arch in enumerate(param_grid):
            architectures[f"A{i}"] = arch

        # Iterates on architectures
        for arch_name, arch in architectures.items():
            
            multiclass = len(dataset_meta["labels"]) > 2
            n_numerical = dataset_meta["n_numerical"]

            # Fixes an architecture
            original_model = training.build_default_model_from_configs(
                arch, 
                dataset_meta,
                None,
                optimizer=OPTIMIZER,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                monitor_metric="valid_loss", 
                max_epochs=MAX_EPOCHS,
                checkpoint_dir=os.path.join("test/none"),
                need_weights=True
            )

            original_model.initialize()
            original_model.callbacks = None
            
            instances_indices = np.random.choice(features.shape[0], min(200, features.shape[0]), replace=False)

            # Preprocess data
            preprocessor = processing.get_preprocessor(
                dataset_meta["categorical"],
                dataset_meta["numerical"],
                dataset_meta["categories"],
                categorical_unknown_value=-1
            )

            preprocessor = preprocessor.fit(features.iloc[instances_indices])
            X = preprocessor.transform(features.iloc[instances_indices])
            y = labels.values[instances_indices]


            if multiclass:
                y = y.astype(np.int64)
            else:
                y = y.astype(np.float32)

            attn_indices = {}    
            # Evaluate all features selectors in the fixed architecture
            for f_selector in feature_selectors:
                for f_percent in feature_percents:

                    # If the full dataset was computed
                    if f_percent == 1 and X.shape[1] in attn_indices:
                        continue


                    n_numerical = dataset_meta["n_numerical"]
                    mask_fn_name = f_selector["name"]
                    mask_fn = f_selector["mask_fn"]
                    
                    n_features_selected = int(features.shape[1] * f_percent)
                    mask = mask_fn(
                                features.iloc[instances_indices],
                                labels[instances_indices],
                                None,
                                dataset_meta,
                                n_features_selected
                            )
                    
                    assert mask.sum() == n_features_selected, "Something went wrong generating the mask"
                    
                    current_features_order = dataset_meta["numerical"] + dataset_meta["categorical"]
                    mask_features_order = features.columns
                    mask_sort_indices = [current_features_order.index(f) for f in mask_features_order]
                    mask = mask[mask_sort_indices]

                    logger.info(f"Testing {dataset}:{mask_fn_name}:{f_percent}:{arch_name}")

                    # Clones the model to modify it
                    model = copy.deepcopy(original_model)

                    if mask is not None:

                        model.module_.numerical_encoder.weights = nn.Parameter(model.module_.numerical_encoder.weights[mask[:n_numerical]])
                        model.module_.numerical_encoder.biases = nn.Parameter(model.module_.numerical_encoder.biases[mask[:n_numerical]])

                        model.module_.register_buffer("categories_offset", model.module_.categories_offset[mask[n_numerical:]])
                        n_numerical = int(mask[:n_numerical].sum())

                    preds_iter = model.forward_iter({
                        "x_numerical": X[:, mask][:, :n_numerical].astype(np.float32),
                        "x_categorical": X[:, mask][:, n_numerical:].astype(np.int32)
                    })
                
                    preds_sorted = -np.ones((arch["n_layers"], X.shape[0], len(mask)), dtype=np.int32)

                    for i, preds in enumerate(preds_iter):
                        output, layer_outs, attn = preds
                        _, batch_cum_attn = attention.compute_std_attentions(attn, "cls")

                        assert np.allclose(batch_cum_attn.sum(axis=-1), 1), "Something went wrong with attentions"
                        
                        # Removes the CLS token
                        cum_attns = batch_cum_attn[:, :, 1:]
                        n_layers, n_instances, n_red_features = cum_attns.shape
                        
                        redim_attns = -np.ones((n_layers, n_instances, len(mask)))
                        redim_attns[:, :, mask] = cum_attns
                        sorted_indices = np.argsort(redim_attns)[:, :, ::-1]

                        preds_sorted[:, i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = sorted_indices
                    
                    if int(mask.sum()) not in attn_indices:
                        attn_indices[int(mask.sum())] = []

                    attn_indices[int(mask.sum())].append(preds_sorted)


            # Aqui codificarlas validaciones de 
            t_fts = X.shape[-1]
            assert len(attn_indices[t_fts]) == 1, "More than one processing for full features"
            full_features_indices = attn_indices[t_fts][0]

            for k in attn_indices.keys():
                for i_indices, indices in enumerate(attn_indices[k]):
                    logger.info(f"Comparing {dataset}:{arch_name}:{k}:{i_indices}")
                    full_positions = np.take_along_axis(np.argsort(full_features_indices, axis=-1), indices[:, :, :k], axis=-1)
                    # Verify full_positions are always incremental
                    kept_sorting = (full_positions[:, :, 1:] > full_positions[:, :, :-1])
                                        
                    stats.append({
                        "dataset": dataset,
                        "architecture_name": arch_name,
                        "n_features": int(k),
                        "n_comparison": int(i_indices),
                        "valid": bool(np.all(kept_sorting))
                    })

                    # Layers per instances
                    total_exps += kept_sorting.shape[0] * kept_sorting.shape[1]

                    if not np.all(kept_sorting):
                        non_valid_instances = (~kept_sorting).sum(axis=-1)
                        non_valid += (non_valid_instances > 0).sum()
                        non_valid_mean += non_valid_instances.sum()
                        
                        logger.warn(f"Invalid hypotheses found: {non_valid}/{total_exps}")
           
    logger.info(f"Invalid hypotheses found: {non_valid}/{total_exps}") 
    logger.info(f"Mean errors: {non_valid_mean/non_valid}") 
    return stats
        

if __name__ == "__main___":

    models = {
                "aggregator": ["cls"],
                "attn_dropout": [0.3],
                "embed_dim": [128],
                "ff_dropout": [0.1],
                "n_head": [4, 8],
                "n_layers": [2, 4],
                "numerical_passthrough": [False],
                "optimizer__lr": [0.0001],
                "optimizer__weight_decay": [0.0001]
            }
    
    datasets = ["volkert", "adult", "kr-vs-kp", "sylvine", "nomao", "jasmine", "australian"]
    datasets = ["jasmine"]
    
    stats = evaluate_feature_selection_order(
        datasets,
        models,
        feature_selectors = [
            {"name": "random_1", "mask_fn": get_random_mask},
            {"name": "random_2", "mask_fn": get_random_mask},
            {"name": "random_3", "mask_fn": get_random_mask},
            {"name": "random_4", "mask_fn": get_random_mask},
            {"name": "random_5", "mask_fn": get_random_mask},
            # {"name": "linear_model", "mask_fn": build_masker_from_model(linear_model.LogisticRegression(random_state=SEED))},
            # {"name": "decision_tree", "mask_fn": build_masker_from_model(tree.DecisionTreeClassifier(random_state=SEED))},
            # {"name": "f_classif", "mask_fn": build_masker_from_score(feature_selection.f_classif)}   

        ],
        #feature_percents = [0.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        feature_percents = [1.0, 0.8, 0.6, 0.4]
    )

    with open("fs_efficency/cum_attn_order.json", "w") as f:
        json.dump(f, stats, indent=4)

if __name__ == "__main__":


    def softmax(m):
        return np.exp(m) / np.exp(m).sum(axis=-1, keepdims=True)
        

    embed_size = 8
    n_features = 6
    f_selectors = [0.5]


    for i in range(1):
        w_k = np.random.normal(size=(embed_size, embed_size))
        w_q = np.random.normal(size=(embed_size, embed_size))
        w_v = np.random.normal(size=(embed_size, embed_size))
        embeddings = np.random.normal(size=(n_features, embed_size))

        attn = softmax(((embeddings @ w_q) @ (embeddings @ w_k).T) / np.sqrt(embed_size))
        attn_indices = np.argsort(attn, axis=-1)[:, ::-1]

        for f_s in f_selectors:
            mask = get_random_mask(
                np.ones((1, n_features)), 
                None, 
                None, 
                None, 
                int(f_s * n_features)
            )

            mask_attn = -np.ones((n_features, n_features))
            mask_attn[np.outer(mask, mask).T] = softmax(((embeddings[mask] @ w_q) @ (embeddings[mask] @ w_k).T) / np.sqrt(embed_size)).flatten()
            mask_attn_indices = np.argsort(mask_attn, axis=-1)[:, ::-1]

            print(attn_indices, mask_attn_indices, f_s * n_features)
