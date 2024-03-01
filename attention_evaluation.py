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
                    neural_network,
                    cluster
                )

from config import (
    DATA_BASE_DIR,
    CHECKPOINT_BASE_DIR,
    FEATURE_SELECTION_N_CLUSTERS,
    FEATURE_SELECTION_K_FOLD,
    SEED
)

logger = log.get_logger()


def caching_masking_wrapper(mask_fn, method_name, job, selector):

    dataset = job["dataset"]

    mask_base_dir = os.path.join(
                CHECKPOINT_BASE_DIR, 
                dataset, 
                "feature_selection", 
                method_name, 
                str(selector)
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

def get_attention_mask(data, labels, attention, dataset_meta, feature_selector): 

    attn = attention.copy()
    is_nan_indices = np.isnan(attn)
    attn[is_nan_indices] = 0
    assert np.allclose(attn.sum(axis=1), 1), "The attention is incorrect"
    attn[is_nan_indices] = 1
    attn = attn.mean(axis=0)
    sorted_args = np.argsort(attn)[::-1]
    mask = np.zeros(attn.shape[0])
    mask[sorted_args[:feature_selector]] = 1
    mask = mask.astype(bool)

    return mask

def get_random_mask(data, labels, attention, dataset_meta, feature_selector): 

    n_instances, n_features = data.shape
    features = np.random.choice(n_features, size=feature_selector, replace=False)
    assert len(features) == feature_selector, "Bad mask generation"
    mask = np.zeros(n_features)
    mask[features] = 1
    mask = mask.astype(bool)

    return mask

def get_full_mask(data, labels, attention, dataset_meta, feature_selector): 

    n_instances, n_features = data.shape
    mask = np.ones(n_features)
    mask = mask.astype(bool)

    return mask

def build_masker_from_model(model):

    def get_mask_from_model(data, labels, attention, dataset_meta, feature_selector):
        
        preprocessor = processing.get_preprocessor(
            dataset_meta["categorical"],
            dataset_meta["numerical"],
            dataset_meta["categories"],
        )

        data = preprocessor.fit_transform(data)
        selector = feature_selection.SelectFromModel(model, max_features=feature_selector, threshold=-np.inf)
        selector = selector.fit(data, labels)
        mask = selector.get_support()
        return mask

    return get_mask_from_model

def build_masker_from_score(scorer_fn):

    def get_mask_from_score(data, labels, attention, dataset_meta, feature_selector):
        
        preprocessor = processing.get_preprocessor(
            dataset_meta["categorical"],
            dataset_meta["numerical"],
            dataset_meta["categories"],
        )

        data = preprocessor.fit_transform(data)
        selector = feature_selection.SelectKBest(scorer_fn, k=feature_selector)
        selector = selector.fit(data, labels)
        mask = selector.get_support()
        return mask

    return get_mask_from_score

def feature_selection_evaluation(
    dataset,
    mask_info,
    opt_metric,
    features_percent,
    model
    ):

    model_runner = model["model"]
    model_name = model["name"]

    ft_selection_name = mask_info["name"]
    mask_fn = mask_info["mask_fn"]

    data_dir = os.path.join(DATA_BASE_DIR, dataset, "feature_selection", opt_metric)
    checkpoint_dir = os.path.join(
                        CHECKPOINT_BASE_DIR, dataset, "feature_selection", opt_metric, 
                        ft_selection_name, str(features_percent), model_name
                    )
    
    logger.info("Reading data")
    
    # Reading feature info
    with open(os.path.join(data_dir, "feature_selection_info.json"), "r") as f:
        fs_info = json.load(f)

    # Reading data
    data, dataset_meta = datalib.read_dataset(dataset)
    data.index = np.argsort(fs_info["data_required_sort"])
    data = data.sort_index()
    target_column = dataset_meta["target"]
    features = data.drop(target_column, axis=1)
    labels = data[target_column]
    multiclass = len(dataset_meta["labels"]) > 2

    # Reading attention
    with open(os.path.join(data_dir, "attention.npy"), "rb") as f:
        attn = np.load(f)

    assert np.array_equal(labels, fs_info["labels"]), "Something went wrong sorting data"

    eval_data = []
    
    for c_name, c_process in fs_info["cluster_processes"].items():

        if not c_process["processable"]:
            continue

        s_idx = c_process["start_index"]
        e_idx = c_process["end_index"]
        splits = c_process["splits"]

        c_features = features.loc[s_idx:e_idx]
        c_labels = labels.loc[s_idx:e_idx]
        c_attn = attn[s_idx:e_idx]

        for f_name, f_indices in splits.items():

            clf_checkpoint = os.path.join(checkpoint_dir, model_name, c_name, f_name)
        
            if not os.path.exists(clf_checkpoint):
                os.makedirs(clf_checkpoint)

            scores_checkpoint = os.path.join(clf_checkpoint, "scores.json")
            n_features_selected = int(features_percent * c_features.shape[1])

            if not os.path.exists(scores_checkpoint):
                logger.info(f"Training {c_name}:{f_name}")
                
                train_indices = f_indices["train"] 
                val_indices = f_indices["val"] 

                mask = mask_fn(
                            c_features.iloc[train_indices],
                            c_labels.iloc[train_indices],
                            c_attn[train_indices],
                            dataset_meta,
                            n_features_selected
                        )
                
                assert mask.sum() == n_features_selected, "Something went wrong generating the mask"
                
                selected_columns = c_features.columns[mask]

                clf = pipeline.make_pipeline(
                    processing.get_regular_preprocessor(
                        [c for c in dataset_meta["categorical"] if c in selected_columns],
                        [c for c in dataset_meta["numerical"] if c in selected_columns],
                        { k: v for k, v in dataset_meta["categories"].items() if k in selected_columns}
                    ),
                    model_runner
                )

                clf = clf.fit(
                    c_features.iloc[train_indices][selected_columns],
                    c_labels.iloc[train_indices]
                )

                joblib.dump(clf, os.path.join(clf_checkpoint, "model.jl"))

                preds = clf.predict(c_features.iloc[val_indices][selected_columns])

                scores = evaluating.get_default_feature_selection_scores(
                    c_labels.iloc[val_indices], preds, multiclass=True
                )

                with open(scores_checkpoint, "w") as f:
                    json.dump(scores, f, indent=4)
                
            else:
                logger.info(f"Skipping {c_name}:{f_name}")
                with open(scores_checkpoint, "r") as f:
                    scores = json.load(f)

            
            scores = {
                "dataset": dataset,
                "cluster": c_name,
                "fold": f_name,
                "model": model_name,
                "selection_method": ft_selection_name,
                "opt_metric": opt_metric,
                "n_features_selected": n_features_selected,
                "features_percent": features_percent,
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

    mask_generators = [
        {"name": "attention", "mask_fn": get_attention_mask},
        {"name": "random", "mask_fn": get_random_mask},
        {"name": "linear_model", "mask_fn": build_masker_from_model(linear_model.LogisticRegression())},
        {"name": "decision_tree", "mask_fn": build_masker_from_model(tree.DecisionTreeClassifier())},
        {"name": "f_classif", "mask_fn": build_masker_from_score(feature_selection.f_classif)}       
    ]

    # Do not include the 1.0
    ft_percent_selectors = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
    
    # Extract the attention for required architectures and cluster it
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

        attention_dir = os.path.join(DATA_BASE_DIR, dataset, "feature_selection", selection_metric)
        attention_file = os.path.join(attention_dir, "attention.npy")
        attention_info_file = os.path.join(attention_dir, "feature_selection_info.json")

        if not os.path.exists(attention_dir):
            os.makedirs(attention_dir)

        if not (os.path.exists(attention_file) and os.path.exists(attention_info_file)):

            result = attention.extract_attention(
                dataset,
                checkpoint_dir,
                aggregator,
                selection_metric,
                config=arch,
                return_labels=True
            )

            attn = result["cumulated_attention"]
            labels = result["labels"]

            nan_indices = np.isnan(attn)
            attn[nan_indices] = -1

            cluster_algo = cluster.KMeans(n_clusters=FEATURE_SELECTION_N_CLUSTERS, random_state=SEED)
            cluster_labels = cluster_algo.fit_predict(attn)
            attn[nan_indices] = np.nan

            indices = np.lexsort((labels, cluster_labels))
            attn = attn[indices]
            labels = labels[indices]
            cluster_labels = cluster_labels[indices]

            attn_info = {
                "data_required_sort": indices.tolist(),
                "labels": labels.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "cluster_processes": {}
            }

            # Validate at least two classes per cluster
            cluster_uniques, clusters_indices = np.unique(cluster_labels, return_index=True)
            clusters_indices = np.array(clusters_indices.tolist() + [cluster_labels.shape[0]])
            
            for c_l, c_l_start, c_l_end in zip(cluster_uniques, clusters_indices[:-1], clusters_indices[1:]):
                existing_labels = np.unique(labels[c_l_start:c_l_end])
                processable = len(existing_labels) >= 2

                splits = {}

                if processable:
                    skf = model_selection.StratifiedKFold(
                        n_splits=FEATURE_SELECTION_K_FOLD, 
                        random_state=SEED, 
                        shuffle=True
                    )

                    for i, (train_index, val_index) in \
                        enumerate(skf.split(attn[c_l_start:c_l_end], labels[c_l_start:c_l_end])):

                        splits[f"F{i}"] = {
                            "train": train_index.tolist(),
                            "val": val_index.tolist()
                        }


                attn_info["cluster_processes"][f"C{c_l}"] = {
                    "processable": processable,
                    "start_index": int(c_l_start),
                    "end_index": int(c_l_end),
                    "splits": splits
                }

            with open(attention_file, "wb") as f:
                np.save(f, attn)
            
            with open(attention_info_file, "w") as f:
                json.dump(attn_info, f, indent=4)

    # Training full models, i.e., no feature selection applied
    logger.info("=" * 40 + f" Training non-masked models")
    fs_scores = []

    for job in best_archs_df.iloc:

        selection_metric = job["selection_metric"]
        dataset = job["dataset"]
        logger.info(f"Dataset: {dataset}")

        for m in models:
            
            cv_scores = feature_selection_evaluation(
                dataset,
                {"name": "none", "mask_fn": get_full_mask},
                selection_metric,
                1.0,
                m
            )

            fs_scores.extend(cv_scores)
    
    logger.info("=" * 40 + f" Training non-masked models finished")

    
    logger.info("=" * 40 + f" Training masked models")
    for job in best_archs_df.iloc:

        selection_metric = job["selection_metric"]
        dataset = job["dataset"]
        logger.info(f"Dataset: {dataset}")

        for ft_p_selector in ft_percent_selectors:
            for mask_gen in mask_generators:
                for m in models:
                    cv_scores = feature_selection_evaluation(
                            dataset,
                            mask_gen,
                            selection_metric,
                            ft_p_selector,
                            m
                        )
                    
                    fs_scores.extend(cv_scores)

    logger.info("=" * 40 + f" Masked models finished")

    logger.info("Saving feature selection scores")

    fs_scores_df = pd.DataFrame(fs_scores)

    fs_scores_df = fs_scores_df.drop(["fold"], axis=1).groupby([
        "dataset", 
        "cluster", 
        "model", 
        "selection_method",
        "opt_metric", 
        "n_features_selected",
        "features_percent"
        ], as_index=False).agg(["mean", "std"])

    fs_scores_df.columns = ["_".join(col) if col[1] else col[0] for col in fs_scores_df.columns]  

    fs_scores_com_df = fs_scores_df.copy()
    fs_scores_com_df = fs_scores_com_df[fs_scores_com_df["selection_method"] != "none"]

    for fs_method in fs_scores_df.query("selection_method != 'none'")["selection_method"].unique():
        for metric in fs_scores_df["opt_metric"].unique():
            for ds in fs_scores_df["dataset"].unique():
                
                insertion_df = fs_scores_df.query(
                    "selection_method=='none' "
                    "and opt_metric==@metric "
                    "and dataset==@ds"
                ) 
                
                insertion_df.loc[:, "selection_method"] = fs_method
                
                fs_scores_com_df = pd.concat([fs_scores_com_df, insertion_df])
                    
    fs_scores_com_df.to_csv("feature_selection_scores.csv", index=False)


    

            


if __name__ == "__main__":
    main()


    
    