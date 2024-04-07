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
from skorch import toy
import shutil
import xgboost
import time
import torch.nn as nn
import joblib

from sklearn import (
                    linear_model,
                    tree, 
                    feature_selection, 
                    model_selection,
                    preprocessing
                )

from attention_evaluation_cluster import (
        build_masker_from_model,
        build_masker_from_score,
        get_random_mask
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
        model_info,
        fs_percent,
        fs_n_features,
        mask,
        mask_fn_name,
    ):

    model_name = model_info["name"]
    base_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, model_name)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Build the full architectures
    parameters = model_info["hp_space"]
    param_grid = list(model_selection.ParameterGrid(parameters))

    architectures = {}
    for i, arch in enumerate(param_grid):
        architectures[f"A{i}"] = arch
    
    archs_file = os.path.join(base_dir, "architectures.json")
        
    if os.path.exists(archs_file):
        with open(archs_file, "r") as f:
            old_architectures = json.load(f)
        
        if str(dict(sorted(old_architectures.items()))) != str(dict(sorted(architectures.items()))):
            logger.fatal("Previous an current architectures are not the same")
            raise ValueError("Previous an current architectures are not the same")
    else:
        with open(archs_file, "w") as f:
            json.dump(architectures, f, indent=4)


    # Reduce to only one architecture; the best.
    if fs_percent != 1:
        # Search the best architecture from the full model
        parameters = model_info["hp_space"]
        param_grid = list(model_selection.ParameterGrid(parameters))
        splits = list(dataset_meta["splits"].keys())

        full_model_scores = []

        for arch_name in [f"A{i}" for i in range(len(param_grid))]:
            for split in splits:
                full_model_scores_filename = os.path.join(base_dir, "none", "1.0", arch_name, split, "scores.json")

                if not os.path.exists(full_model_scores_filename):
                    raise ValueError("The full models are not scored yet")

                with open(full_model_scores_filename, "r") as f:
                    full_model_scores.append(json.load(f))

        full_model_scores = pd.DataFrame(full_model_scores)
        full_model_scores = full_model_scores.drop(["model", "fold", "fs_method"], axis=1)
        full_model_scores = full_model_scores.groupby(["dataset", "architecture_name"], as_index=False).agg("mean")
        best_arch = full_model_scores.loc[
                        reporting.get_top_k_indices(
                                        full_model_scores.groupby(["dataset"]),
                                        1,
                                        "balanced_accuracy",
                                        "max"
                                        )
                    ]["architecture_name"].values[0]
        
        best_arch = str(best_arch) 
        architectures = {best_arch: architectures[best_arch]}
    else:
        mask_fn_name = "none"
    
    
    archs_scores = []
    for arch_name, arch in architectures.items():
        f_scores_dir = os.path.join(base_dir, mask_fn_name, str(fs_percent), arch_name, fold_name)
        f_scores_filename = os.path.join(f_scores_dir, "scores.json")

        if not os.path.exists(f_scores_dir):
            os.makedirs(f_scores_dir)

        if os.path.exists(f_scores_filename):
            with open(f_scores_filename, "r") as f:
                scores = json.load(f)

            archs_scores.append(scores)
            continue

        with open(os.path.join(DATA_BASE_DIR, "architectures.json"), "r") as f:
            o_architectures = json.load(f)["regular"]

        original_arch_name = ""
        for o_arch_name, o_arch in o_architectures.items():
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
        scores["dataset"] = dataset
        scores["model"] = model_name
        scores["fold"] = fold_name
        scores["architecture_name"] = arch_name
        scores["fs_method"] = mask_fn_name
        scores["fs_percent"] = fs_percent
        scores["fs_n_features"] = fs_n_features

        with open(f_scores_filename, "w") as f:
            json.dump(scores, f, indent=4)

        archs_scores.append(scores)

    return archs_scores

def get_xgboost_cv_stats(        
        X_train,
        y_train,
        X_val,
        y_val,
        dataset,
        dataset_meta,
        fold_name,
        model_info,
        fs_percent,
        fs_n_features,
        mask,
        mask_fn_name,
    ):

    model_name = model_info["name"]
    base_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, model_name)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Build the full architectures
    parameters = model_info["hp_space"]
    param_grid = list(model_selection.ParameterGrid(parameters))

    architectures = {}
    for i, arch in enumerate(param_grid):
        architectures[f"A{i}"] = arch
    
    archs_file = os.path.join(base_dir, "architectures.json")
        
    if os.path.exists(archs_file):
        with open(archs_file, "r") as f:
            old_architectures = json.load(f)
        
        if str(dict(sorted(old_architectures.items()))) != str(dict(sorted(architectures.items()))):
            logger.fatal("Previous an current architectures are not the same")
            raise ValueError("Previous an current architectures are not the same")
    else:
        with open(archs_file, "w") as f:
            json.dump(architectures, f, indent=4)

    if fs_percent == 1:
        mask_fn_name = "none"

    
    multiclass = len(dataset_meta["labels"]) > 2
    opt_metric = "mlogloss" if multiclass else "logloss"
    
    archs_scores = []
    for arch_name, arch in architectures.items():
        f_scores_dir = os.path.join(base_dir, mask_fn_name, str(fs_percent), arch_name, fold_name)
        f_scores_filename = os.path.join(f_scores_dir, "scores.json")

        if not os.path.exists(f_scores_dir):
            os.makedirs(f_scores_dir)

        if os.path.exists(f_scores_filename):
            with open(f_scores_filename, "r") as f:
                scores = json.load(f)

            archs_scores.append(scores)
            continue
    
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
        

        le = preprocessing.LabelEncoder()
        le.fit(y_train)

        start_time = time.time()
        model.fit(X_train, le.transform(y_train), eval_set=[(X_val, le.transform(y_val))], verbose=2)
        end_time = time.time()
        
        inf_start_time = time.time()
        preds = model.predict_proba(X_val)
        inf_end_time = time.time()
        
        scores = evaluating.get_default_scores(
            le.transform(y_val),
            preds,
            prefix="",
            multiclass=multiclass
        )

        scores["training_time"] = end_time - start_time
        scores["inference_time"] = inf_end_time - inf_start_time
        scores["dataset"] = dataset
        scores["model"] = model_name
        scores["fold"] = fold_name
        scores["architecture_name"] = arch_name
        scores["fs_method"] = mask_fn_name
        scores["fs_percent"] = fs_percent
        scores["fs_n_features"] = fs_n_features

        with open(f_scores_filename, "w") as f:
            json.dump(scores, f, indent=4)

        archs_scores.append(scores)

    return archs_scores
    
def build_mlp_cv_stats(nonlinearity=nn.ReLU()):
    def get_mlp_cv_stats(        
            X_train,
            y_train,
            X_val,
            y_val,
            dataset,
            dataset_meta,
            fold_name,
            model_info,
            fs_percent,
            fs_n_features,
            mask,
            mask_fn_name,
        ):

        model_name = model_info["name"]
        base_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, model_name)

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Build the full architectures
        parameters = model_info["hp_space"]
        param_grid = list(model_selection.ParameterGrid(parameters))

        architectures = {}
        for i, arch in enumerate(param_grid):
            architectures[f"A{i}"] = arch
        
        archs_file = os.path.join(base_dir, "architectures.json")
            
        if os.path.exists(archs_file):
            with open(archs_file, "r") as f:
                old_architectures = json.load(f)
            
            if str(dict(sorted(old_architectures.items()))) != str(dict(sorted(architectures.items()))):
                logger.fatal("Previous an current architectures are not the same")
                raise ValueError("Previous an current architectures are not the same")
        else:
            with open(archs_file, "w") as f:
                json.dump(architectures, f, indent=4)

        if fs_percent == 1:
            mask_fn_name = "none"

        
        multiclass = len(dataset_meta["labels"]) > 2

        # Convert categorical to OHE and join data
        n_numerical = dataset_meta["n_numerical"]
        n_numerical = int(mask[:n_numerical].sum())

        # If all features are numerical
        ohe = preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if n_numerical == X_train.shape[1]:
            X_train_pre = X_train.copy().astype(np.float32)
            X_val_pre = X_val.copy().astype(np.float32)
        else:
            X_train_cat = ohe.fit_transform(X_train[:, n_numerical:])
            X_val_cat = ohe.transform(X_val[:, n_numerical:])

            X_train_pre = np.concatenate([X_train[:, :n_numerical], X_train_cat], axis=1).astype(np.float32)
            X_val_pre = np.concatenate([X_val[:, :n_numerical], X_val_cat], axis=1).astype(np.float32)

        X = np.concatenate([X_train_pre, X_val_pre], axis=0).astype(np.float32)
        y = np.concatenate([y_train, y_val], axis=0)
        

        archs_scores = []
        for arch_name, arch in architectures.items():
            f_scores_dir = os.path.join(base_dir, mask_fn_name, str(fs_percent), arch_name, fold_name)
            f_scores_filename = os.path.join(f_scores_dir, "scores.json")

            if not os.path.exists(f_scores_dir):
                os.makedirs(f_scores_dir)

            if fs_percent == 1:
                joblib.dump(ohe, os.path.join(f_scores_dir, "ohe.jl"))

            if os.path.exists(f_scores_filename):
                with open(f_scores_filename, "r") as f:
                    scores = json.load(f)

                archs_scores.append(scores)
                continue

            if not multiclass:
                n_outputs = 1
                criterion = nn.BCEWithLogitsLoss
            else:
                n_outputs = len(dataset_meta["labels"])
                criterion = nn.CrossEntropyLoss

            callbacks = [            
                    (f"checkpoint_valid_loss", skorch.callbacks.Checkpoint(
                        monitor=f"valid_loss_best",
                        dirname=os.path.join(f_scores_dir, "model")
                        )),
                    ("early_stopping", skorch.callbacks.EarlyStopping(
                        monitor="valid_loss", 
                        patience=int(0.2 * MAX_EPOCHS)
                    ))
                    ]
        

            train_split = skorch.dataset.ValidSplit(((
                np.arange(X_train.shape[0]), 
                np.arange(X_val.shape[0]) + X_train.shape[0] 
            ),))

            module = toy.MLPModule(
                input_units=X.shape[1], 
                output_units=n_outputs,
                hidden_units=arch["hidden_units"],
                num_hidden=arch["num_hidden"],
                nonlin=nonlinearity,
                dropout=arch["dropout"],
                squeeze_output=True
            )

            model = skorch.NeuralNetClassifier(
                module=module,
                criterion=criterion,
                optimizer=OPTIMIZER,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                train_split=train_split,
                callbacks=callbacks,
                optimizer__lr=arch["optimizer__lr"],
                optimizer__weight_decay=arch["optimizer__weight_decay"]
            )

            start_time = time.time()
            model.fit(X, y)
            end_time = time.time()

            model.initialize()
            model.load_params(checkpoint=skorch.callbacks.Checkpoint(dirname=os.path.join(f_scores_dir, "model")))
            

            inf_start_time = time.time()
            preds = model.predict_proba(X_val_pre)
            inf_end_time = time.time()
            
            scores = evaluating.get_default_scores(
                y_val,
                preds,
                prefix="",
                multiclass=multiclass
            )

            scores["training_time"] = end_time - start_time
            scores["inference_time"] = inf_end_time - inf_start_time
            scores["dataset"] = dataset
            scores["model"] = model_name
            scores["fold"] = fold_name
            scores["architecture_name"] = arch_name
            scores["fs_method"] = mask_fn_name
            scores["fs_percent"] = fs_percent
            scores["fs_n_features"] = fs_n_features

            with open(f_scores_filename, "w") as f:
                json.dump(scores, f, indent=4)

            archs_scores.append(scores)

        return archs_scores
    return get_mlp_cv_stats

def build_mlp_simple_cv_stats(base_model="mlp/full", nonlinearity=nn.ReLU()):
    def get_mlp_simple_cv_stats(
        X_train,
        y_train,
        X_val,
        y_val,
        dataset,
        dataset_meta,
        fold_name,
        model_info,
        fs_percent,
        fs_n_features,
        mask,
        mask_fn_name, 
        ):

        model_name = model_info["name"]
        base_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, model_name)

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Build the full architectures
        parameters = model_info["hp_space"]
        param_grid = list(model_selection.ParameterGrid(parameters))

        architectures = {}
        for i, arch in enumerate(param_grid):
            architectures[f"A{i}"] = arch
        
        archs_file = os.path.join(base_dir, "architectures.json")

        if os.path.exists(archs_file):
            with open(archs_file, "r") as f:
                old_architectures = json.load(f)
            
            if str(dict(sorted(old_architectures.items()))) != str(dict(sorted(architectures.items()))):
                logger.fatal("Previous an current architectures are not the same")
                raise ValueError("Previous an current architectures are not the same")
        else:
            with open(archs_file, "w") as f:
                json.dump(architectures, f, indent=4)

        # Reduce to only one architecture; the best.
        if fs_percent != 1:
            # Search the best architecture from the full model
            parameters = model_info["hp_space"]
            param_grid = list(model_selection.ParameterGrid(parameters))
            splits = list(dataset_meta["splits"].keys())

            full_model_scores = []

            for arch_name in [f"A{i}" for i in range(len(param_grid))]:
                for split in splits:
                    full_model_scores_filename = os.path.join(base_dir, "none", "1.0", arch_name, split, "scores.json")
                    
                    if not os.path.exists(full_model_scores_filename):
                        raise ValueError("The full models are not scored yet")

                    with open(full_model_scores_filename, "r") as f:
                        full_model_scores.append(json.load(f))

            full_model_scores = pd.DataFrame(full_model_scores)
            full_model_scores = full_model_scores.drop(["model", "fold", "fs_method"], axis=1)
            full_model_scores = full_model_scores.groupby(["dataset", "architecture_name"], as_index=False).agg("mean")
            
            best_arch = full_model_scores.loc[
                            reporting.get_top_k_indices(
                                            full_model_scores.groupby(["dataset"]),
                                            1,
                                            "balanced_accuracy",
                                            "max"
                                            )
                        ]["architecture_name"].values[0]
            
            best_arch = str(best_arch) 
            architectures = {best_arch: architectures[best_arch]}
        else:
            mask_fn_name = "none"
        

        multiclass = len(dataset_meta["labels"]) > 2
        # Convert categorical to OHE and join data
        n_numerical = dataset_meta["n_numerical"]

        archs_scores = []

        for arch_name, arch in architectures.items():
            f_scores_dir = os.path.join(base_dir, mask_fn_name, str(fs_percent), arch_name, fold_name)
            f_scores_filename = os.path.join(f_scores_dir, "scores.json")

            if not os.path.exists(f_scores_dir):
                os.makedirs(f_scores_dir)

            if os.path.exists(f_scores_filename):
                with open(f_scores_filename, "r") as f:
                    scores = json.load(f)

                archs_scores.append(scores)
                continue

            with open(os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, base_model, "architectures.json"), "r") as f:
                o_architectures = json.load(f)

            original_arch_name = ""
            for o_arch_name, o_arch in o_architectures.items():
                is_same = True
                for hparam_name, hparam_value in o_arch.items():
                    if(arch[hparam_name] != hparam_value):
                        is_same = False
                        break

                if is_same:
                    original_arch_name = o_arch_name
                    break

            base_checkpoint_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, base_model, "none", "1.0", 
                                original_arch_name, fold_name)
            checkpoint_dir = os.path.join(base_checkpoint_dir, "model")
            
            if not multiclass:
                n_outputs = 1
                criterion = nn.BCEWithLogitsLoss
            else:
                n_outputs = len(dataset_meta["labels"])
                criterion = nn.CrossEntropyLoss

            
            # If all features are numerical
            ohe = joblib.load(os.path.join(base_checkpoint_dir, "ohe.jl"))
            ohe_mask = mask[:n_numerical].tolist()

            for n_cat in range(mask[n_numerical:].shape[0]):
                for _ in ohe.categories_[n_cat]:
                    ohe_mask.append(mask[n_numerical + n_cat])

            ohe_mask = np.array(ohe_mask)
            masked_n_numerical = int(mask[:n_numerical].sum())

            if masked_n_numerical == X_val.shape[1]:
                X_val_pre = X_val.copy().astype(np.float32)
            else:
                X_val_cat = X_val[:, masked_n_numerical:].copy()
                om_indices = np.argwhere(mask[n_numerical:] == False).flatten()
                for om_idx in om_indices:
                    X_val_cat = np.insert(X_val_cat, om_idx, 0, axis=1)
                
                X_val_cat = ohe.transform(X_val_cat)
                
                X_val_pre = np.concatenate([
                    X_val[:, :masked_n_numerical],
                    X_val_cat[:, ohe_mask[n_numerical:]]

                ], axis=1).astype(np.float32)
                

            module = toy.MLPModule(
                input_units=ohe_mask.shape[0], 
                output_units=n_outputs,
                hidden_units=arch["hidden_units"],
                num_hidden=arch["num_hidden"],
                nonlin=nonlinearity,
                dropout=arch["dropout"],
                squeeze_output=True
            )

            model = skorch.NeuralNetClassifier(
                module=module,
                criterion=criterion,
                optimizer=OPTIMIZER,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                train_split=None,
                callbacks=None,
                optimizer__lr=arch["optimizer__lr"],
                optimizer__weight_decay=arch["optimizer__weight_decay"]
            )

            model.initialize()        
            model.load_params(checkpoint=skorch.callbacks.Checkpoint(dirname=checkpoint_dir))
            model.module_.sequential[0].weight = nn.Parameter(model.module_.sequential[0].weight[:, ohe_mask])
            
            inf_start_time = time.time()
            preds = model.predict_proba(X_val_pre)
            inf_end_time = time.time()

            scores = evaluating.get_default_scores(
                y_val,
                preds,
                prefix="",
                multiclass=multiclass
            )

            scores["inference_time"] = inf_end_time - inf_start_time
            scores["training_time"] = 0
            scores["dataset"] = dataset
            scores["model"] = model_name
            scores["fold"] = fold_name
            scores["architecture_name"] = arch_name
            scores["fs_method"] = mask_fn_name
            scores["fs_percent"] = fs_percent
            scores["fs_n_features"] = fs_n_features

            with open(f_scores_filename, "w") as f:
                json.dump(scores, f, indent=4)

            archs_scores.append(scores)
        return archs_scores
    
    return get_mlp_simple_cv_stats

def evaluate_feature_selection_cv(
        datasets,
        models,
        feature_selectors,
        feature_percents,
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

        for f_selector in feature_selectors:
            for f_percent in feature_percents:
                for split in splits:
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
                    
                    for m_info in models:

                        m_name = m_info["name"]
                        cv_stats_fn = m_info["cv_stats_fn"]

                        # Exceptions

                        if mask_fn_name.startswith("random_") and "mlp/full" in m_name:
                            logger.info(f"Skipping {dataset}:{split}:{mask_fn_name}:{f_percent}:{m_name}")
                            continue

                        if "mlp/full" in m_name and f_percent < 1:
                            logger.info(f"Skipping {dataset}:{split}:{mask_fn_name}:{f_percent}:{m_name}")
                            continue
                        # Exceptions

                        logger.info(f"Running {dataset}:{split}:{mask_fn_name}:{f_percent}:{m_name}")
                        

                        mf_scores = cv_stats_fn(
                                X[train_indices][:, mask],
                                y[train_indices],
                                X[val_indices][:, mask],
                                y[val_indices],
                                dataset,
                                dataset_meta,
                                split,
                                model_info=m_info,
                                fs_percent=f_percent,
                                fs_n_features=n_features_selected,
                                mask=mask,
                                mask_fn_name=mask_fn_name,
                                
                            )
                        
                        scores.extend(mf_scores)
    
    return scores
                        
if __name__ == "__main__":

    models = [
        {
            "name": "transformer/cls",
            "cv_stats_fn": get_transformer_cv_stats,
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
        {
            "name": "xgboost",
            "cv_stats_fn": get_xgboost_cv_stats,
            "hp_space": {
                "max_depth": [5, 10, 15, 20],
                "learning_rate": [0.1]
            }
        },
        {
            "name": "mlp/full",
            "cv_stats_fn": build_mlp_cv_stats(),
            "hp_space": {
                "hidden_units": [512],
                "num_hidden": [3, 6],
                "dropout": [0.2],
                "optimizer__lr": [0.001],
                "optimizer__weight_decay": [0.1]
            }
        },
        {
            "name": "mlp/full_nd_nn",
            "cv_stats_fn": build_mlp_cv_stats(nonlinearity=nn.Identity()),
            "hp_space": {
                "hidden_units": [512],
                "num_hidden": [3, 6],
                "dropout": [0.0],
                "optimizer__lr": [0.001],
                "optimizer__weight_decay": [0.1]
            }
        },
        {
           "name": "mlp/full_nd",
           "cv_stats_fn": build_mlp_cv_stats(),
           "hp_space": {
               "hidden_units": [512],
               "num_hidden": [3, 6],
               "dropout": [0.0],
               "optimizer__lr": [0.001],
               "optimizer__weight_decay": [0.1]
           }
        },
        {
            "name": "mlp/simple",
            "cv_stats_fn": build_mlp_simple_cv_stats(),
            "hp_space": {
                "hidden_units": [512],
                "num_hidden": [3, 6],
                "dropout": [0.2],
                "optimizer__lr": [0.001],
                "optimizer__weight_decay": [0.1]
            }
        },
        {
            "name": "mlp/simple_nd",
            "cv_stats_fn": build_mlp_simple_cv_stats(base_model="mlp/full_nd"),
            "hp_space": {
                "hidden_units": [512],
                "num_hidden": [3, 6],
                "dropout": [0.0],
                "optimizer__lr": [0.001],
                "optimizer__weight_decay": [0.1]
            }
        },
        {
            "name": "mlp/simple_nd_nn",
            "cv_stats_fn": build_mlp_simple_cv_stats(base_model="mlp/full_nd_nn", nonlinearity=nn.Identity()),
            "hp_space": {
                "hidden_units": [512],
                "num_hidden": [3, 6],
                "dropout": [0.0],
                "optimizer__lr": [0.001],
                "optimizer__weight_decay": [0.1]
            }
        }
    ]
    
    datasets = ["kr-vs-kp", "sylvine", "nomao", "adult", "jasmine", "australian", "volkert", "anneal", "ldpa"]
    datasets = ["volkert", "adult", "kr-vs-kp", "sylvine", "nomao", "jasmine", "australian"]
    
    if not os.path.exists(FS_EFFICENCY_CHECKPOINT_BASE_DIR):
        os.makedirs(FS_EFFICENCY_CHECKPOINT_BASE_DIR)
    
    scores_filename = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, "scores.csv")
    
    scores = evaluate_feature_selection_cv(
        datasets,
        models,
        feature_selectors = [
            {"name": "random_1", "mask_fn": get_random_mask, "level": "fold"},
            {"name": "random_2", "mask_fn": get_random_mask, "level": "fold"},
            {"name": "random_3", "mask_fn": get_random_mask, "level": "fold"},
            {"name": "random_4", "mask_fn": get_random_mask, "level": "fold"},
            {"name": "random_5", "mask_fn": get_random_mask, "level": "fold"},
            {"name": "linear_model", "mask_fn": build_masker_from_model(linear_model.LogisticRegression(random_state=SEED)), "level": "fold"},
            {"name": "decision_tree", "mask_fn": build_masker_from_model(tree.DecisionTreeClassifier(random_state=SEED)), "level": "fold"},
            {"name": "f_classif", "mask_fn": build_masker_from_score(feature_selection.f_classif), "level": "fold"}   

        ],
        feature_percents = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    )

    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.drop_duplicates()
    
    scores_df = scores_df.drop(["fold"], axis=1)
    scores_df = scores_df.groupby(["dataset", "model", "architecture_name", "fs_method", 
                                   "fs_percent", "fs_n_features"], as_index=False).agg(["mean", "std", "sum"])
    
    scores_df.columns = ["_".join(col) if col[1] else col[0] for col in scores_df.columns]


    # Replace the full methods in every feature selection method
    fs_scores_df = scores_df.copy()
    fs_scores_none_df = fs_scores_df.query("fs_method == 'none'")
    fs_scores_df = fs_scores_df.query("fs_method != 'none'")

    for fs_method in fs_scores_df["fs_method"].unique():
        fs_scores_none_df["fs_method"] = fs_method
        fs_scores_df = pd.concat([fs_scores_df, fs_scores_none_df])

    fs_scores_df.to_csv(scores_filename, index=False)
