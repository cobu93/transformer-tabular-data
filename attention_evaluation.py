import os
import numpy as np
import json
import pandas as pd
import joblib
import skorch
from utils import log, attention, evaluating, processing, data as datalib

from sklearn import (
                    linear_model, 
                    pipeline, 
                    neighbors, 
                    base, 
                    model_selection, 
                    tree, 
                    feature_selection, 
                    neural_network
                )

from config import (
    DATA_BASE_DIR,
    CHECKPOINT_BASE_DIR,
    FEATURE_SELECTION_K_FOLD,
    SEED
)

logger = log.get_logger()


"""
Training on split function
"""


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

    if os.path.exists(attention_file):
        logger.info("Skipping extraction. Attention file exists.")
        with open(attention_file, "rb") as f:
            cum_attns = np.load(f)
        return cum_attns

    cum_attns = attention.extract_attention(
        dataset,
        checkpoint_dir,
        aggregator,
        selection_metric,
        config,
        only_last=True
    )

    logger.info("Saving attention at " + data_dir)
    
    with open(attention_file, "wb") as f:
        np.save(f, cum_attns)

    return cum_attns

def caching_masking_wrapper(mask_fn, method_name, job, selector, attn_selector):

    dataset = job["dataset"]

    mask_base_dir = os.path.join(
                CHECKPOINT_BASE_DIR, 
                dataset, 
                "feature_selection", 
                method_name, 
                str(attn_selector)
            )

    mask_checkpoint = os.path.join(mask_base_dir, "mask.npy")

    if os.path.exists(mask_checkpoint):
        with open(mask_checkpoint, "rb") as f:
            mask = np.load(f)        
    else:
        mask = mask_fn(job, selector)
        if not os.path.exists(mask_base_dir):
            os.makedirs(mask_base_dir)

        with open(mask_checkpoint, "wb") as f:
            np.save(f, mask)

    return mask

def get_fs_attention_mask(job, attn_selector): 

    dataset = job["dataset"]
    aggregator = job["aggregator"]
    selection_metric = job["selection_metric"]
    checkpoint_dir = job["checkpoint_dir"]

    cum_attn = extract_attention(
        dataset,
        checkpoint_dir,
        aggregator,
        selection_metric,
        config=None
    )

    attn_mask = attention.get_attention_mask(cum_attn, attn_selector)

    return attn_mask

def get_random_mask(job, s_features): 

    dataset = job["dataset"]
    data, _ = datalib.read_dataset(dataset)
    n_instances, n_features = data.shape
    n_features -= 1
    features = np.random.choice(n_features, size=s_features, replace=False)
    assert len(features) == s_features, "Bad mask generation"
    mask = np.zeros((n_instances, n_features))
    mask[:, features] = 1
    mask = mask.astype(bool)

    return mask

def get_full_mask(job, s_features): 

    dataset = job["dataset"]
    data, _ = datalib.read_dataset(dataset)
    n_instances, n_features = data.shape
    n_features -= 1
    mask = np.ones((n_instances, n_features))
    mask = mask.astype(bool)

    return mask

def build_masker_from_model(model):

    def get_mask_from_model(job, s_features):
        dataset = job["dataset"]
        data, dataset_meta = datalib.read_dataset(dataset)
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
        data, dataset_meta = datalib.read_dataset(dataset)
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

def get_fraction_features(mask):

    n_instances, n_features = mask.shape
    fraction_selected = mask.sum() / (n_instances * n_features)

    n_features_selected = n_features
    n_features_fraction_selected = 1.0

    for i in range(1, n_features + 1):
        if np.abs(i / n_features - fraction_selected) \
            < np.abs(n_features_selected / n_features - fraction_selected):
    
            n_features_selected = i
            n_features_fraction_selected = n_features_selected / n_features

    # Return fraction, equivalent features, and real fraction
    return fraction_selected, n_features_selected, n_features_fraction_selected

def feature_selection_evaluation(
    dataset,
    mask,
    ft_selection_name,
    attn_selector,
    model
    ):

    data_dir = os.path.join(DATA_BASE_DIR, dataset, "attention")
    checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, dataset, "feature_selection", ft_selection_name, str(attn_selector))
    (real_fraction_masked, 
     approx_features, 
     approx_fraction_from_features) = get_fraction_features(mask)
    
    logger.info("Reading data")
    data, dataset_meta = datalib.read_dataset(dataset)
    
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
            "dataset": dataset,
            "fold": s_name,
            "selection_method": ft_selection_name,
            "attention_selector": attn_selector,
            "real_fraction_masked": real_fraction_masked, 
            "approx_features_masked": approx_features, 
            "fraction_from_features": approx_fraction_from_features,
            "model": model_name,
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

    # Do not include the 1.0
    attn_selectors = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    models=[
        {"name":"KNN", "model": neighbors.KNeighborsClassifier()},
        {"name":"DT", "model": tree.DecisionTreeClassifier()},
        {"name":"LR", "model": linear_model.LogisticRegression()},
        {"name":"MLP", "model": neural_network.MLPClassifier()},
    ]

    
    # Retrieves the best architectures for each dataset 
    # depending on the optimization metrics
    logger.info("Reading selected architectures")
    archs_file = "selected_architectures.csv"
    if not os.path.exists(archs_file):
        raise ValueError(f"File {archs_file} does not exists. Should run model_selection before.")
    
    best_archs_df = pd.read_csv(archs_file)
    
    # Extract the attention for required architectures
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

        search_architectures = architectures.get("rnn" if aggregator == "rnn" else "regular", None)
        
        if not search_architectures:
            logger.fatal("The architectures file is incorrect")
            raise ValueError("The architectures file is incorrect")
        
        arch = search_architectures[arch_name]
        arch["dataset"] = dataset
        arch["aggregator"] = aggregator

        extract_attention(
            dataset,
            checkpoint_dir,
            aggregator,
            selection_metric,
            config=arch
        )

    # Training full models, i.e., no feature selection applied
    logger.info("=" * 40 + f" Training non-masked models")
    fs_scores = []

    for job in best_archs_df.iloc:

        selection_metric = job["selection_metric"]
        dataset = job["dataset"]
        mask = get_full_mask(job, None)

        logger.info(f"Dataset: {dataset}")

        for m in models:
            
            cv_scores = feature_selection_evaluation(
                dataset,
                mask,
                "none",
                1.0,
                m
            )

            for s in cv_scores:
                fs_scores.append({
                    "selection_metric": selection_metric,
                    **s
                })
    
    logger.info("=" * 40 + f" Training non-masked models finished")




    logger.info("=" * 40 + f" Training masked models")
    for job in best_archs_df.iloc:

        selection_metric = job["selection_metric"]
        dataset = job["dataset"]
        logger.info(f"Dataset: {dataset}")

        for attn_s in attn_selectors:
            # From attention feature selection
            logger.info("Extracting attention mask")
            attn_mask = caching_masking_wrapper(get_fs_attention_mask, f"attention/{selection_metric}", job, attn_s, attn_s)
            
            (fraction_selected, 
             n_features_selected, 
             n_features_fraction_selected) = get_fraction_features(attn_mask)
            
            logger.info(f"Attention selector: {attn_s}")
            logger.info(f"Information masked: {fraction_selected}")
            logger.info(f"Equivalent features: {n_features_selected}")
            logger.info(f"Information masked from feeatures: {n_features_fraction_selected}")

            for m in models:
                cv_scores = feature_selection_evaluation(
                        dataset,
                        attn_mask,
                        f"attention/{selection_metric}",
                        attn_s,
                        m
                    )
                
                for s in cv_scores:
                    fs_scores.append({
                        "selection_metric": selection_metric,
                        **s
                    })
            
            # From other feature selection
            for ft_selector in feature_selectors:
                mask = caching_masking_wrapper(
                            ft_selector["mask_fn"], 
                            ft_selector["name"], 
                            job, 
                            n_features_selected,
                            attn_s
                        )
                
                for m in models:
                    cv_scores = feature_selection_evaluation(
                        dataset,
                        mask,
                        ft_selector["name"],
                        attn_s,
                        m
                    )

                    for s in cv_scores:
                        fs_scores.append({
                            "selection_metric": selection_metric,
                            **s
                        })

    logger.info("=" * 40 + f" Masked models finished")

    logger.info("Saving feature selection scores")

    fs_scores_df = pd.DataFrame(fs_scores)
    
    fs_scores_df = fs_scores_df.drop(["fold"], axis=1).groupby([
        "dataset", 
        "selection_method", 
        "selection_metric", 
        "attention_selector",
        "real_fraction_masked", 
        "approx_features_masked", 
        "fraction_from_features", 
        "model"], as_index=False).agg(["mean", "std"])
    fs_scores_df.columns = ["_".join(col) if col[1] else col[0] for col in fs_scores_df.columns]       
    fs_scores_df.to_csv("feature_selection_scores.csv", index=False)


    

            


if __name__ == "__main__":
    main()


    
    