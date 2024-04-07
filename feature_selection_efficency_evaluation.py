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


def get_transformer_test_stats(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset,
        dataset_meta,
        model_info,
        architecture,
        fs_percent,
        fs_n_features,
        mask,
        mask_fn_name,
    ):

    model_name = model_info["name"]
    base_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, model_name)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    archs_file = os.path.join(base_dir, "architectures.json")
        
    with open(archs_file, "r") as f:
        architectures = json.load(f)
        
    if fs_percent == 1:
        mask_fn_name = "none"
    
    
    arch_name = architecture
    arch = architectures[arch_name]

    f_scores_dir = os.path.join(base_dir, mask_fn_name, str(fs_percent), arch_name, "best")
    f_scores_filename = os.path.join(f_scores_dir, "scores.json")

    if not os.path.exists(f_scores_dir):
        os.makedirs(f_scores_dir)

    if os.path.exists(f_scores_filename):
        with open(f_scores_filename, "r") as f:
            scores = json.load(f)
            return [scores]

    multiclass = len(dataset_meta["labels"]) > 2
    n_numerical = dataset_meta["n_numerical"]

    model = training.build_default_model_from_configs(
        arch, 
        dataset_meta,
        None,
        optimizer=OPTIMIZER,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        monitor_metric="valid_loss", 
        max_epochs=40,
        checkpoint_dir=os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, "none"),
    )

    start_time = 0 
    end_time = 0
        
    if fs_percent == 1:
        model_checkpoint_path = os.path.join(f_scores_dir, "model")

        other_callbacks = [
            (f"checkpoint", skorch.callbacks.TrainEndCheckpoint(
                        dirname=model_checkpoint_path
            ))      
        ]

        if not os.path.exists(os.path.join(model_checkpoint_path, "train_end_history.json")):
            model.callbacks = other_callbacks

            start_time = time.time()
            model = model.fit(X={
                "x_numerical": X_train[:, :n_numerical].astype(np.float32),
                "x_categorical": X_train[:, n_numerical:].astype(np.int32)
                }, 
                y=y_train
            )
            end_time = time.time()
        else:
            other_callbacks[0][1].initialize()

        checkpoint = other_callbacks[0][1].checkpoint_
    else:
        model_checkpoint_path = os.path.join(base_dir, "none", "1.0", arch_name, "best", "model")
        checkpoint = skorch.callbacks.TrainEndCheckpoint(
                        dirname=model_checkpoint_path
            )
        checkpoint.initialize()
        checkpoint = checkpoint.checkpoint_
        
    model.initialize()
    model.callbacks = None
    model.load_params(checkpoint=checkpoint)
    if os.path.exists(os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, "none")):
        shutil.rmtree(os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, "none"))


    if mask is not None:

        model.module_.numerical_encoder.weights = nn.Parameter(model.module_.numerical_encoder.weights[mask[:n_numerical]])
        model.module_.numerical_encoder.biases = nn.Parameter(model.module_.numerical_encoder.biases[mask[:n_numerical]])

        model.module_.register_buffer("categories_offset", model.module_.categories_offset[mask[n_numerical:]])
        n_numerical = int(mask[:n_numerical].sum())


    inf_start_time = time.time()
    preds = model.predict_proba({
            "x_numerical": X_test[:, :n_numerical].astype(np.float32),
            "x_categorical": X_test[:, n_numerical:].astype(np.int32)
        })
    inf_end_time = time.time()

    scores = evaluating.get_default_scores(
        y_test,
        preds,
        prefix="",
        multiclass=multiclass
    )

    scores["inference_time"] = inf_end_time - inf_start_time
    scores["training_time"] = end_time - start_time
    scores["dataset"] = dataset
    scores["model"] = model_name
    scores["architecture_name"] = arch_name
    scores["fs_method"] = mask_fn_name
    scores["fs_percent"] = fs_percent
    scores["fs_n_features"] = fs_n_features

    with open(f_scores_filename, "w") as f:
        json.dump(scores, f, indent=4)

    return [scores]

def get_xgboost_test_stats(        
        X_train,
        y_train,
        X_test,
        y_test,
        dataset,
        dataset_meta,
        model_info,
        architecture,
        fs_percent,
        fs_n_features,
        mask,
        mask_fn_name,
    ):

    model_name = model_info["name"]
    base_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, model_name)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    archs_file = os.path.join(base_dir, "architectures.json")
    
    with open(archs_file, "r") as f:
        architectures = json.load(f)
        
    if fs_percent == 1:
        mask_fn_name = "none"

    
    multiclass = len(dataset_meta["labels"]) > 2
    opt_metric = "mlogloss" if multiclass else "logloss"
    
    
    arch_name = architecture
    arch = architectures[arch_name]


    f_scores_dir = os.path.join(base_dir, mask_fn_name, str(fs_percent), arch_name, "best")
    f_scores_filename = os.path.join(f_scores_dir, "scores.json")

    if not os.path.exists(f_scores_dir):
        os.makedirs(f_scores_dir)

    if os.path.exists(f_scores_filename):
        with open(f_scores_filename, "r") as f:
            scores = json.load(f)

        return [scores]
    
    model = xgboost.XGBClassifier(
                n_estimators=40, 
                random_state=SEED, 
                device=DEVICE, 
                **arch
            )
    
    le = preprocessing.LabelEncoder()
    le.fit(y_train)

    start_time = time.time()
    model.fit(X_train, le.transform(y_train), verbose=2)
    end_time = time.time()
    
    inf_start_time = time.time()
    preds = model.predict_proba(X_test)
    inf_end_time = time.time()
    
    scores = evaluating.get_default_scores(
        le.transform(y_test),
        preds,
        prefix="",
        multiclass=multiclass
    )

    scores["training_time"] = end_time - start_time
    scores["inference_time"] = inf_end_time - inf_start_time
    scores["dataset"] = dataset
    scores["model"] = model_name
    scores["architecture_name"] = arch_name
    scores["fs_method"] = mask_fn_name
    scores["fs_percent"] = fs_percent
    scores["fs_n_features"] = fs_n_features

    with open(f_scores_filename, "w") as f:
        json.dump(scores, f, indent=4)

    return [scores]

def build_mlp_test_stats(nonlinearity=nn.ReLU()):    
    def get_mlp_test_stats(        
            X_train,
            y_train,
            X_test,
            y_test,
            dataset,
            dataset_meta,
            model_info,
            architecture,
            fs_percent,
            fs_n_features,
            mask,
            mask_fn_name,
        ):

        model_name = model_info["name"]
        base_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, model_name)

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        archs_file = os.path.join(base_dir, "architectures.json")
            
        if not os.path.exists(archs_file):
            raise ValueError("There is not an architecture file")
        
        with open(archs_file, "r") as f:
            architectures = json.load(f)
            
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
            X_test_pre = X_test.copy().astype(np.float32)
        else:
            X_train_cat = ohe.fit_transform(X_train[:, n_numerical:])
            X_test_cat = ohe.transform(X_test[:, n_numerical:])

            X_train_pre = np.concatenate([X_train[:, :n_numerical], X_train_cat], axis=1).astype(np.float32)
            X_test_pre = np.concatenate([X_test[:, :n_numerical], X_test_cat], axis=1).astype(np.float32)

        arch_name = architecture
        arch = architectures[arch_name]

        f_scores_dir = os.path.join(base_dir, mask_fn_name, str(fs_percent), arch_name, "best")
        f_scores_filename = os.path.join(f_scores_dir, "scores.json")

        if not os.path.exists(f_scores_dir):
            os.makedirs(f_scores_dir)

        if fs_percent == 1:
            joblib.dump(ohe, os.path.join(f_scores_dir, "ohe.jl"))

        if os.path.exists(f_scores_filename):
            with open(f_scores_filename, "r") as f:
                scores = json.load(f)
                return [scores]

        if not multiclass:
            n_outputs = 1
            criterion = nn.BCEWithLogitsLoss
        else:
            n_outputs = len(dataset_meta["labels"])
            criterion = nn.CrossEntropyLoss

        callbacks = [            
                (f"checkpoint", skorch.callbacks.TrainEndCheckpoint(
                    dirname=os.path.join(f_scores_dir, "model")
                    ))
                ]

        module = toy.MLPModule(
            input_units=X_train_pre.shape[1], 
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
            max_epochs=40,
            train_split=None,
            callbacks=callbacks,
            optimizer__lr=arch["optimizer__lr"],
            optimizer__weight_decay=arch["optimizer__weight_decay"]
        )

        start_time = time.time()
        model.fit(X_train_pre, y_train)
        end_time = time.time()

        model.initialize()
        model.load_params(checkpoint=callbacks[0][1].checkpoint_)
        

        inf_start_time = time.time()
        preds = model.predict_proba(X_test_pre)
        inf_end_time = time.time()
        
        scores = evaluating.get_default_scores(
            y_test,
            preds,
            prefix="",
            multiclass=multiclass
        )

        scores["training_time"] = end_time - start_time
        scores["inference_time"] = inf_end_time - inf_start_time
        scores["dataset"] = dataset
        scores["model"] = model_name
        scores["architecture_name"] = arch_name
        scores["fs_method"] = mask_fn_name
        scores["fs_percent"] = fs_percent
        scores["fs_n_features"] = fs_n_features

        with open(f_scores_filename, "w") as f:
            json.dump(scores, f, indent=4)

        return [scores]
    return get_mlp_test_stats


def build_mlp_simple_test_stats(base_model="mlp/full", nonlinearity=nn.ReLU()):
    def get_mlp_simple_test_stats(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset,
        dataset_meta,
        model_info,
        architecture,
        fs_percent,
        fs_n_features,
        mask,
        mask_fn_name,
        ):

        model_name = model_info["name"]
        base_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset, model_name)

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        archs_file = os.path.join(base_dir, "architectures.json")

        if not os.path.exists(archs_file):
            raise ValueError("There is not an architecture file")
        
        with open(archs_file, "r") as f:
            architectures = json.load(f)
            
        if fs_percent == 1:
            mask_fn_name = "none"
        
        multiclass = len(dataset_meta["labels"]) > 2
        # Convert categorical to OHE and join data
        n_numerical = dataset_meta["n_numerical"]


        arch_name = architecture
        arch = architectures[arch_name]

        f_scores_dir = os.path.join(base_dir, mask_fn_name, str(fs_percent), arch_name, "best")
        f_scores_filename = os.path.join(f_scores_dir, "scores.json")

        if not os.path.exists(f_scores_dir):
            os.makedirs(f_scores_dir)

        
        if os.path.exists(f_scores_filename):
            with open(f_scores_filename, "r") as f:
                scores = json.load(f)
                return [scores]

        
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
                            original_arch_name, "best")
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

        if masked_n_numerical == X_test.shape[1]:
            X_test_pre = X_test.copy().astype(np.float32)
        else:
            X_test_cat = X_test[:, masked_n_numerical:].copy()
            om_indices = np.argwhere(mask[n_numerical:] == False).flatten()
            for om_idx in om_indices:
                X_test_cat = np.insert(X_test_cat, om_idx, 0, axis=1)
            
            X_test_cat = ohe.transform(X_test_cat)
            
            X_test_pre = np.concatenate([
                X_test[:, :masked_n_numerical],
                X_test_cat[:, ohe_mask[n_numerical:]]

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
            max_epochs=40,
            train_split=None,
            callbacks=None,
            optimizer__lr=arch["optimizer__lr"],
            optimizer__weight_decay=arch["optimizer__weight_decay"]
        )

        model.initialize()   
        checkpoint = skorch.callbacks.TrainEndCheckpoint(dirname=checkpoint_dir)
        checkpoint.initialize()
        model.load_params(checkpoint=checkpoint.checkpoint_)
        model.module_.sequential[0].weight = nn.Parameter(model.module_.sequential[0].weight[:, ohe_mask])
        
        inf_start_time = time.time()
        preds = model.predict_proba(X_test_pre)
        inf_end_time = time.time()

        scores = evaluating.get_default_scores(
            y_test,
            preds,
            prefix="",
            multiclass=multiclass
        )

        scores["inference_time"] = inf_end_time - inf_start_time
        scores["training_time"] = 0
        scores["dataset"] = dataset
        scores["model"] = model_name
        scores["architecture_name"] = arch_name
        scores["fs_method"] = mask_fn_name
        scores["fs_percent"] = fs_percent
        scores["fs_n_features"] = fs_n_features

        with open(f_scores_filename, "w") as f:
            json.dump(scores, f, indent=4)

        return [scores]
    return get_mlp_simple_test_stats

def evaluate_feature_selection(
        architectures,
        models,
        feature_selectors
    ):


    if not os.path.exists(FS_EFFICENCY_CHECKPOINT_BASE_DIR):
        os.makedirs(FS_EFFICENCY_CHECKPOINT_BASE_DIR)

    scores = []
    for dataset in architectures["dataset"].unique():
        
        checkpoint_dir = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        logger.info(f"Reading {dataset} metadata")
        meta_file = os.path.join(DATA_BASE_DIR, dataset, "train.meta.json")
        test_meta_file = os.path.join(DATA_BASE_DIR, dataset, "test.meta.json")
        dataset_meta = None
        test_dataset_meta = None

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
        test_data = pd.read_csv(test_dataset_file)
        
        features = data.drop(target_column, axis=1)
        labels = data[target_column]
        test_features = test_data.drop(target_column, axis=1)
        test_labels = test_data[target_column]

        for f_selector in architectures.query("dataset==@dataset")["fs_method"].unique():
            for f_percent in sorted(architectures.query("dataset==@dataset and fs_method==@f_selector")["fs_percent"].unique())[::-1]:

                mask_fn_name = f_selector
                mask_fn = None
                for fs in feature_selectors:
                    if fs["name"]==f_selector:
                        mask_fn = fs["mask_fn"]
                        break

                if mask_fn is None:
                    raise ValueError(f"Any feature selector match with {f_selector}")
                
                preprocessor = processing.get_preprocessor(
                    dataset_meta["categorical"],
                    dataset_meta["numerical"],
                    dataset_meta["categories"],
                    categorical_unknown_value=-1
                )

                preprocessor = preprocessor.fit(features)
    
                X = preprocessor.transform(features)
                y = labels.values

                X_test = preprocessor.transform(test_features)
                y_test = test_labels.values

                if multiclass:
                    y = y.astype(np.int64)
                else:
                    y = y.astype(np.float32)

                
                n_features_selected = int(features.shape[1] * f_percent)
                mask = mask_fn(
                            features,
                            labels,
                            None,
                            dataset_meta,
                            n_features_selected
                        )
                
                assert mask.sum() == n_features_selected, "Something went wrong generating the mask"
                
                current_features_order = dataset_meta["numerical"] + dataset_meta["categorical"]
                mask_features_order = features.columns
                mask_sort_indices = [current_features_order.index(f) for f in mask_features_order]
                mask = mask[mask_sort_indices]

                for m_name in architectures.query("dataset==@dataset and fs_method==@f_selector and fs_percent==@f_percent")["model"].unique():

                    # Exceptions

                    if mask_fn_name.startswith("random_") and "mlp/full" in m_name:
                        logger.info(f"Skipping {dataset}:{mask_fn_name}:{f_percent}:{m_name}")
                        continue

                    if "mlp/full" in m_name and f_percent < 1:
                        logger.info(f"Skipping {dataset}:{mask_fn_name}:{f_percent}:{m_name}")
                        continue
                    # Exceptions

                    test_stats_fn = None

                    for m in models:
                        if m["name"] == m_name:
                            test_stats_fn = m["test_stats_fn"]

                    if test_stats_fn is None:
                        print(f"Any test stats fn matches with {m_name}")
                        continue
                        #raise ValueError(f"Any  test stats fn matches with {m_name}")
                    
                    model_info = {"name": m_name, "test_stats_fn": test_stats_fn}
                    
                    for architecture_name in architectures.query("dataset==@dataset and fs_method==@f_selector and fs_percent==@f_percent and model==@m_name")["architecture_name"].values:
                                        
                        logger.info(f"Running {dataset}:{mask_fn_name}:{f_percent}:{m_name}:{architecture_name}")

                        mf_scores = test_stats_fn(
                                X[:, mask],
                                y,
                                X_test[:, mask],
                                y_test,
                                dataset,
                                dataset_meta,
                                model_info=model_info,
                                architecture=architecture_name,
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
            "test_stats_fn": get_transformer_test_stats,
        },
        {
            "name": "xgboost",
            "test_stats_fn": get_xgboost_test_stats,
        },
        {
            "name": "mlp/full",
            "test_stats_fn": build_mlp_test_stats(),
        },
        {
            "name": "mlp/full_nd",
            "test_stats_fn": build_mlp_test_stats(),
        },
        {
            "name": "mlp/full_nd_nn",
            "test_stats_fn": build_mlp_test_stats(nonlinearity=nn.Identity()),
        },
        {
            "name": "mlp/simple",
            "test_stats_fn": build_mlp_simple_test_stats(),
        },
        {
            "name": "mlp/simple_nd",
            "test_stats_fn": build_mlp_simple_test_stats(base_model="mlp/full_nd"),
        },
        {
            "name": "mlp/simple_nd_nn",
            "test_stats_fn": build_mlp_simple_test_stats(base_model="mlp/full_nd_nn", nonlinearity=nn.Identity()),
        }
    ]
    
    datasets = ["kr-vs-kp", "sylvine", "nomao", "adult", "jasmine", "australian", "volkert", "anneal", "ldpa"]
    datasets = ["kr-vs-kp", "sylvine", "nomao", "adult", "jasmine", "australian", "volkert"]

    if not os.path.exists(FS_EFFICENCY_CHECKPOINT_BASE_DIR):
        os.makedirs(FS_EFFICENCY_CHECKPOINT_BASE_DIR)
    
    scores_filename = os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, "test_scores.csv")
    scores_df = pd.read_csv(os.path.join(FS_EFFICENCY_CHECKPOINT_BASE_DIR, "scores.csv"))

    ################ Adding the extra cases considered
    best_archs_indices = scores_df.query(
        "fs_percent==1 "
        "and fs_method=='decision_tree' " # The method is not relevant
        "and model in ['mlp/full', 'mlp/full_nd', 'mlp/full_nd_nn', 'xgboost']" # This adds the best architecture instead of doing hyperparameter search
    ).groupby(["dataset", "model"])["balanced_accuracy_mean"].idxmax()

    best_archs_df = scores_df.loc[best_archs_indices, ["dataset", "model", "architecture_name"]]

    for r in best_archs_df.iloc:
        ds = r["dataset"]
        m = r["model"]
        a = r["architecture_name"]
        
        only_best_arch = scores_df.query("dataset==@ds and model==@m and architecture_name==@a").copy()
        only_best_arch["model"] = only_best_arch["model"] + "/best"
        
        scores_df = pd.concat([scores_df, only_best_arch], axis=0)

    scores_df = scores_df.reset_index(drop=True)
    
    # Retrieving the best architecture for each point in feature selection
    indices_best = scores_df.groupby([
        "dataset", "model", "fs_method", 
        "fs_percent", "fs_n_features"
        ])["balanced_accuracy_mean"].idxmax()


    best_scores_df = scores_df.loc[indices_best]
    architectures = best_scores_df[["dataset", "model", "architecture_name", "fs_method", "fs_percent"]]
    architectures = architectures.query("dataset not in ['anneal']")

    scores = evaluate_feature_selection(
        architectures.replace({
            "model": {
                "mlp/full/best": "mlp/full",
                "mlp/full_nd/best": "mlp/full_nd",
                "mlp/full_nd_nn/best": "mlp/full_nd_nn",
                "xgboost/best": "xgboost"
            }
        }).drop_duplicates(),
        models,
        feature_selectors = [
            {"name": "random_1", "mask_fn": get_random_mask},
            {"name": "random_2", "mask_fn": get_random_mask},
            {"name": "random_3", "mask_fn": get_random_mask},
            {"name": "random_4", "mask_fn": get_random_mask},
            {"name": "random_5", "mask_fn": get_random_mask},
            {"name": "linear_model", "mask_fn": build_masker_from_model(linear_model.LogisticRegression(random_state=SEED))},
            {"name": "decision_tree", "mask_fn": build_masker_from_model(tree.DecisionTreeClassifier(random_state=SEED))},
            {"name": "f_classif", "mask_fn": build_masker_from_score(feature_selection.f_classif)}       
        ]
    )

    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.drop_duplicates()
    
    # Replace the full methods in every feature selection method
    fs_scores_df = scores_df.copy()
    fs_scores_none_df = fs_scores_df.query("fs_method == 'none'")
    fs_scores_df = fs_scores_df.query("fs_method != 'none'")

    for fs_method in fs_scores_df["fs_method"].unique():
        fs_scores_none_df["fs_method"] = fs_method
        fs_scores_df = pd.concat([fs_scores_df, fs_scores_none_df])


    # Set the respective scores to the replaced models
    o_models = architectures["model"].values
    fs_scores_df = architectures.replace({
                "model": {
                    "mlp/full/best": "mlp/full",
                    "mlp/full_nd/best": "mlp/full_nd",
                    "mlp/full_nd_nn/best": "mlp/full_nd_nn",
                    "xgboost/best": "xgboost"   
                }
            }).merge(
        fs_scores_df,
        on=["dataset","model", "architecture_name", "fs_method", "fs_percent"],
        how="left"
    )

    fs_scores_df["model"] = o_models
    fs_scores_df.to_csv(scores_filename, index=False)
