import os
import numpy as np
import json
import pandas as pd
import joblib
import skorch
from utils import training, log, attention, evaluating, processing

from sklearn import linear_model, pipeline, neighbors, base, model_selection, tree, feature_selection

from config import (
    DATA_BASE_DIR,
    OPTIMIZER,
    BATCH_SIZE,
    CHECKPOINT_BASE_DIR,
    DEVICE,
    FEATURE_SELECTION_K_FOLD,
    SEED
)

logger = log.get_logger()


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

def read_dataset(dataset):

    datasets_dirname = os.path.join(DATA_BASE_DIR, dataset)
    train_data, dataset_meta = read_meta_csv(datasets_dirname, "train")
    train_indices = dataset_meta["df_indices"]
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
    
    return data, dataset_meta

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

    data, dataset_meta = read_dataset(dataset)
    target_column = dataset_meta["target"]
    n_numerical = dataset_meta["n_numerical"]
    
    features = data.drop(target_column, axis=1)
    original_order = features.columns.values.tolist()
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
        monitor_metric="valid_loss", 
        max_epochs=1,
        checkpoint_dir=os.path.join(checkpoint_dir, "ignore"),
    )
    
    logger.info("Loading checkpoint")
    checkpoint = skorch.callbacks.TrainEndCheckpoint(
        dirname=os.path.join(checkpoint_dir, "model")
    )
    model.callbacks = None
    checkpoint.initialize()
    model.initialize()
    model.load_params(checkpoint=checkpoint.checkpoint_)
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

    renormalize = False
    if aggregator == "cls":
        logger.info("The aggregator is CLS")
        cum_attns = cum_attns[:, 1:]
        renormalize=True

    numerical_passthrough = config["numerical_passthrough"]
    if numerical_passthrough:
        logger.info("Numerical passthrough is True")
        numerical_attention = np.ones((cum_attns.shape[0], n_numerical)) * cum_attns.max(axis=1, keepdims=True)
        cum_attns = np.concatenate([numerical_attention, cum_attns], axis=1)
        renormalize=True

    if renormalize:
        logger.info("Renormalizing")
        cum_attns = cum_attns / cum_attns.sum(axis=1, keepdims=True) 

    # At this point cum_attns has the numerical values first
    # For correctly masking we need to re sort features as originally
    
    current_order = dataset_meta["numerical"] + dataset_meta["categorical"]
    indices_sort = np.argsort(list(map(lambda x: original_order.index(x), current_order)))
    cum_attns = cum_attns[:, indices_sort]
    
    logger.info("Saving attention at " + data_dir)
    
    with open(attention_file, "wb") as f:
        np.save(f, cum_attns)

    return cum_attns
    
def get_fs_attention_mask(job, attn_selector): 

    archs_file = os.path.join(DATA_BASE_DIR, "architectures.json")
    logger.info(f"Reading architectures from {archs_file}")
    architectures = None
    with open(archs_file, "r") as f:
        architectures = json.load(f)

    dataset = job["dataset"]
    aggregator = job["aggregator"]
    arch_name = job["architecture_name"]
    selection_metric = job["selection_metric"]
    checkpoint_dir = job["checkpoint_dir"]
    numerical_passthrough = job["numerical_passthrough"]

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

    attn_mask = attention.get_attention_mask(cum_attn, attn_selector)

    return attn_mask

def get_random_mask(job, s_features): 

    dataset = job["dataset"]
    data, _ = read_dataset(dataset)
    n_instances, n_features = data.shape
    n_features -= 1
    features = np.random.choice(n_features, size=s_features, replace=False)
    assert len(features) == s_features, "Bad mask generation"
    mask = np.zeros((n_instances, n_features))
    mask[:, features] = 1
    mask = mask.astype(bool)

    return mask

def build_masker_from_model(model):

    def get_mask_from_model(job, s_features):
        dataset = job["dataset"]
        data, dataset_meta = read_dataset(dataset)
        n_instances, n_features = data.shape
        target = dataset_meta["target"]
        indices = dataset_meta["df_indices"]
        train_data = data.drop(target, axis=1).loc[indices]
        train_labels = data[target].loc[indices]
        
        preprocessor = processing.get_preprocessor(
            dataset_meta["categorical"],
            dataset_meta["numerical"],
            dataset_meta["categories"],
        )

        train_data = preprocessor.fit_transform(train_data)
        selector = feature_selection.SelectFromModel(model, max_features=s_features, threshold=-np.inf)
        selector = selector.fit(train_data, train_labels)
        mask = np.repeat(selector.get_support()[None, :], n_instances, axis=0)
        return mask

    return get_mask_from_model

def build_masker_from_score(scorer_fn):

    def get_mask_from_score(job, s_features):
        dataset = job["dataset"]
        data, dataset_meta = read_dataset(dataset)
        n_instances, n_features = data.shape
        target = dataset_meta["target"]
        indices = dataset_meta["df_indices"]
        train_data = data.drop(target, axis=1).loc[indices]
        train_labels = data[target].loc[indices]
        
        preprocessor = processing.get_preprocessor(
            dataset_meta["categorical"],
            dataset_meta["numerical"],
            dataset_meta["categories"],
        )

        train_data = preprocessor.fit_transform(train_data)
        selector = feature_selection.SelectKBest(scorer_fn, k=s_features)
        selector = selector.fit(train_data, train_labels)
        mask = np.repeat(selector.get_support()[None, :], n_instances, axis=0)
        return mask

    return get_mask_from_score

def feature_selection_evaluation(
    dataset,
    mask,
    ft_selection_name,
    attn_selector,
    model
    ):

    data_dir = os.path.join(DATA_BASE_DIR, dataset, "attention")
    checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, dataset, "feature_selection", ft_selection_name, str(attn_selector))

    logger.info("Reading data")
    data, dataset_meta = read_dataset(dataset)
    
    target_column = dataset_meta["target"]
    features = data.drop(target_column, axis=1)
    labels = data[target_column]
    multiclass = len(dataset_meta["labels"]) > 2

    cv_file = os.path.join(data_dir, "cross_validation.json")
    if os.path.exists(cv_file):
        logger.info("Reading splits")
        with open(cv_file, "r") as f:
            cv_info = json.load(f)
    else:
        logger.info("Creating splits")
        splitter = model_selection.StratifiedKFold(n_splits=FEATURE_SELECTION_K_FOLD, shuffle=True, random_state=SEED)
        k_splits = {}
        for i, (train_indices, val_indices) in enumerate(splitter.split(features, labels)):

            assert np.max(train_indices) < features.shape[0], "Indices are not correct"
            assert np.max(val_indices) < features.shape[0], "Indices are not correct"

            k_splits[f"F{i}"] = {
                "train": train_indices.tolist(),
                "val": val_indices.tolist(),
            }

        cv_info = {"splits": k_splits}
        with open(cv_file, "w") as f:
            json.dump(cv_info, f, indent=4)
    
        
    mask_df = pd.DataFrame(data=mask, columns=features.columns)
    features = features[mask_df]

    model_runner = model["model"]
    model_name = model["name"]
        
    general_clf = pipeline.make_pipeline(
        processing.get_regular_preprocessor(
            dataset_meta["categorical"],
            dataset_meta["numerical"],
            dataset_meta["categories"],
        ),
        model_runner
    )
    
    eval_data = []

    for s_name, s_indices in cv_info["splits"].items():
        logger.info(f"Fold: {s_name}")

        clf_checkpoint = os.path.join(
                            checkpoint_dir,
                            model_name,
                            s_name
                            )
        
        if not os.path.exists(clf_checkpoint):
            os.makedirs(clf_checkpoint)

        scores_checkpoint = os.path.join(clf_checkpoint, "scores.json")

        if not os.path.exists(scores_checkpoint):
        
            clf = base.clone(general_clf)

            clf = clf.fit(
                features.loc[s_indices["train"]],
                labels.loc[s_indices["train"]]
            )

            joblib.dump(clf, os.path.join(clf_checkpoint, "model.jl"))

            preds = clf.predict(features.loc[s_indices["val"]])

            scores = evaluating.get_default_feature_selection_scores(
                labels[s_indices["val"]], preds, multiclass=multiclass
            )

            with open(scores_checkpoint, "w") as f:
                json.dump(scores, f, indent=4)

        else:
            logger.info("Skipping because it was trained before")
            with open(scores_checkpoint, "r") as f:
                scores = json.load(f)

        # for k, v in scores.items():
        #    logger.info(f"\t{k}: {v}")

        scores = {
            "fold": s_name,
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

    feature_selectors = [
        {"name": "random", "mask_fn": get_random_mask},
        {"name": "linear_model", "mask_fn": build_masker_from_model(linear_model.LogisticRegression())},
        {"name": "decision_tree", "mask_fn": build_masker_from_model(tree.DecisionTreeClassifier())},
        {"name": "f_classif", "mask_fn": build_masker_from_score(feature_selection.f_classif)}       
    ]

    attn_selectors = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    models=[
        {"name":"KNN", "model": neighbors.KNeighborsClassifier()},
        {"name":"DT", "model": tree.DecisionTreeClassifier()},
        {"name":"LR", "model": linear_model.LogisticRegression()},
        
    ]

    """
    Retrieves the best architectures for each dataset depending on the optimization metrics
    """
    # Exports best dataset configuration
    logger.info("Reading selected architectures")
    archs_file = "selected_architectures.csv"
    if not os.path.exists(archs_file):
        raise ValueError(f"File {archs_file} does not exists. Should run model_selection before.")
    
    best_archs_df = pd.read_csv(archs_file)
    
    fs_scores = []

    for job in best_archs_df.iloc:
        
        dataset = job["dataset"]
        selection_metric = job["selection_metric"]

        logger.info("-" * 60 + f"Running worker {dataset}-{selection_metric}")

        for attn_s in attn_selectors:

            attn_mask_base_dir = os.path.join(
                CHECKPOINT_BASE_DIR, 
                dataset, 
                "feature_selection", 
                f"attention/{selection_metric}", 
                str(attn_s)
            )

            attn_mask_checkpoint = os.path.join(
                attn_mask_base_dir,
                "mask.npy"
            )

            # Executes attention masking
            logger.info(f"Attention selector: {attn_s}")
            if os.path.exists(attn_mask_checkpoint):
                logger.info("Loading existing mask")
                with open(attn_mask_checkpoint, "rb") as f:
                    attn_mask = np.load(f)
            else:
                logger.info("Creating and saving mask")
                if not os.path.exists(attn_mask_base_dir):
                    os.makedirs(attn_mask_base_dir)
                attn_mask = get_fs_attention_mask(job, attn_s)
                with open(attn_mask_checkpoint, "wb") as f:
                    np.save(f, attn_mask)

            n_instances, n_features = attn_mask.shape
            fraction_selected = attn_mask.sum() / (n_instances * n_features)
            logger.info(f"Fraction selected: {fraction_selected}")
            n_features_selected = n_features
            for i in range(1, n_features + 1):
                if np.abs(i / n_features - fraction_selected) \
                    < np.abs(n_features_selected / n_features - fraction_selected):
                    n_features_selected = i

            n_features_fraction_selected = n_features_selected / n_features

            logger.info(f"Equivalent features: {n_features_selected}")
            logger.info(f"Equivalent fraction: {n_features_fraction_selected}")

            for m in models:
                logger.info("=" * 40 + f" Feature selection [attention/{selection_metric}][selector: {attn_s}][Model: {m['name']}]")
                cv_scores = feature_selection_evaluation(
                    dataset,
                    attn_mask,
                    f"attention/{selection_metric}",
                    attn_s,
                    m
                )

                for s in cv_scores:
                    fs_scores.append({
                        "dataset": dataset,
                        "selection_method": "attention",
                        "selection_metric": selection_metric,
                        "attention_selector": attn_s,
                        "fraction_selected": fraction_selected,
                        "features_selected": n_features_selected,
                        "fraction_features_selected": n_features_fraction_selected,
                        "model": m["name"],
                        **s
                    })

                
            for ft_selector in feature_selectors:
                mask_base_dir = os.path.join(
                    CHECKPOINT_BASE_DIR, 
                    dataset, 
                    "feature_selection", 
                    ft_selector["name"], 
                    str(attn_s)
                )

                mask_checkpoint = os.path.join(
                    mask_base_dir,
                    "mask.npy"
                )

                if os.path.exists(mask_checkpoint):
                    logger.info("Loading existing mask")
                    with open(mask_checkpoint, "rb") as f:
                        mask = np.load(f)
                else:
                    logger.info("Creating and saving mask")
                    if not os.path.exists(mask_base_dir):
                        os.makedirs(mask_base_dir)
                    mask = ft_selector["mask_fn"](job, n_features_selected) 
                    with open(mask_checkpoint, "wb") as f:
                        np.save(f, mask)

                for m in models:
                    logger.info("=" * 40 + f" Feature selection [{ft_selector['name']}][selector: {n_features_selected}][Model: {m['name']}]")
                    cv_scores = feature_selection_evaluation(
                        dataset,
                        mask,
                        ft_selector["name"],
                        attn_s,
                        m
                    )

                    for s in cv_scores:
                        fs_scores.append({
                            "dataset": dataset,
                            "selection_method": ft_selector["name"],
                            "selection_metric": selection_metric,
                            "attention_selector": attn_s,
                            "fraction_selected": fraction_selected,
                            "features_selected": n_features_selected,
                            "fraction_features_selected": n_features_fraction_selected,
                            "model": m["name"],
                            **s
                        })

        logger.info("-" * 60 + f"Worker finished {dataset}-{selection_metric}")
        logger.info("Saving feature selection scores")

        fs_scores_df = pd.DataFrame(fs_scores)
        fs_scores_df = fs_scores_df.drop(["fold"], axis=1).groupby([
            "dataset", 
            "selection_method", 
            "selection_metric", 
            "attention_selector",
            "fraction_selected", 
            "features_selected", 
            "fraction_features_selected", 
            "model"], as_index=False).agg(["mean", "std"])
        fs_scores_df.columns = ["_".join(col) if col[1] else col[0] for col in fs_scores_df.columns]       
        fs_scores_df.to_csv("feature_selection_scores.csv", index=False)


    

            


if __name__ == "__main__":
    main()


    
    