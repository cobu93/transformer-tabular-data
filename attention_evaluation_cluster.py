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

def get_attention_summary(attn):
    summary = {}
    
    c_mean_attn = attn.mean(axis=0)
    summary["mean_attention"] = c_mean_attn.tolist()
    
    indices_sort = np.argsort(c_mean_attn)[::-1]
    sorted_attn = c_mean_attn[indices_sort]
    
    summary["sorted_attention"] = sorted_attn.tolist()
    
    cum_attn = np.insert(c_mean_attn[indices_sort].cumsum(), 0, 0)
    summary["cumulative_attention"] = cum_attn.tolist()
    
    slopes = sorted_attn[1:] - sorted_attn[:-1]
    summary["slopes"] = slopes.tolist()
    
    return summary

def get_attn_clusters_info(
    attn,
    labels,
    n_clusters=4
    ):    
    
    logger.info(f"Test using {n_clusters} clusters")

    #Clustering
    cluster_algo = cluster.KMeans(n_clusters=n_clusters, random_state=SEED)
    cluster_labels = cluster_algo.fit_predict(attn)

    indices = np.lexsort((labels, cluster_labels))
    t_attn = attn[indices]
    t_labels = labels[indices]
    t_cluster_labels = cluster_labels[indices]

    cluster_uniques, clusters_indices = np.unique(t_cluster_labels, return_index=True)
    clusters_indices = np.array(clusters_indices.tolist() + [cluster_labels.shape[0]])

    cluster_option_info = {
        "n_clusters": n_clusters,
        "cluster_labels": t_cluster_labels.tolist(),
        "data_required_sort": indices.tolist(),
        "clusters": []
    }

    # Test each cluster
    for c_l, c_l_start, c_l_end in zip(cluster_uniques, clusters_indices[:-1], clusters_indices[1:]):

        cluster_info = {
            "label": int(c_l),
            "start_index": int(c_l_start),
            "end_index": int(c_l_end),
        }

        cluster_info["attention_summary"] = get_attention_summary(t_attn[c_l_start:c_l_end])
        
        # Mean cluster entropy
        cluster_entropy = t_attn[c_l_start:c_l_end] * np.log(t_attn[c_l_start:c_l_end])
        cluster_entropy = -cluster_entropy.sum(axis=-1).mean()
        cluster_info["mean_entropy"] = cluster_entropy

        # Non predominant class
        c_labels = t_labels[c_l_start:c_l_end]
        existing_labels = np.unique(c_labels)
        # At least two classes
        cluster_info["classification_labels"] = []
        for e_c in existing_labels:
            fraction_in_cluster = c_labels[c_labels == e_c].shape[0] / c_labels.shape[0]
            cluster_info["classification_labels"].append({
                "label": int(e_c),
                "cluster_proportion": fraction_in_cluster
            })

        cluster_option_info["clusters"].append(cluster_info)

    return cluster_option_info      
         
def get_map_features_ohe(data, dataset_meta):

    original_order = data.columns.values.tolist()
    current_order = dataset_meta["numerical"] + dataset_meta["categorical"]
    indices_sort = np.argsort(list(map(lambda x: current_order.index(x), original_order)))
    
    map_features_order = []

    for c_i, c in enumerate(current_order):
        if c in dataset_meta["numerical"]:
            map_features_order.append(indices_sort[c_i]) 
        elif c in dataset_meta["categorical"]:
            for _ in dataset_meta["categories"][c]:
                map_features_order.append(indices_sort[c_i]) 
    
    map_features_order = np.array(map_features_order)
    return map_features_order

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
        
        n_instances, n_features = data.shape
        features_mapping = get_map_features_ohe(data, dataset_meta)

        preprocessor = processing.get_regular_preprocessor(
            dataset_meta["categorical"],
            dataset_meta["numerical"],
            dataset_meta["categories"],
        )

        data = preprocessor.fit_transform(data)

        assert data.shape[-1] == len(features_mapping), "Computing features mapping went wrong"
        mask = np.ones(n_features).astype(bool)

        model.fit(data, labels)

        for n_select_features in range(feature_selector, data.shape[-1]):

            if mask.sum() == feature_selector:
                break

            mask[:] = False
        
            selector = feature_selection.SelectFromModel(
                            model, 
                            prefit=True,
                            max_features=n_select_features, 
                            threshold=-np.inf
                        )
            
            importance_indices = selector.get_support(indices=True).astype(int)
            importance_indices = importance_indices[:n_select_features]
            unique_features_selection = list(set(features_mapping[importance_indices]))
            mask[unique_features_selection] = True
        
        return mask

    return get_mask_from_model

def build_masker_from_score(scorer_fn):

    def get_mask_from_score(data, labels, attention, dataset_meta, feature_selector):
        
        n_instances, n_features = data.shape
        features_mapping = get_map_features_ohe(data, dataset_meta)

        preprocessor = processing.get_regular_preprocessor(
            dataset_meta["categorical"],
            dataset_meta["numerical"],
            dataset_meta["categories"],
        )

        data = preprocessor.fit_transform(data)

        assert data.shape[-1] == len(features_mapping), "Computing features mapping went wrong"
        mask = np.ones(n_features).astype(bool)

        for n_select_features in range(feature_selector, data.shape[-1]):

            if mask.sum() == feature_selector:
                break

            mask[:] = False
        
            selector = feature_selection.SelectKBest(scorer_fn, k=n_select_features)
            
            selector = selector.fit(data, labels)
            importance_indices = selector.get_support(indices=True).astype(int)
            importance_indices = importance_indices[:n_select_features]
            unique_features_selection = list(set(features_mapping[importance_indices]))
            mask[unique_features_selection] = True

        return mask

    return get_mask_from_score

def feature_selection_evaluation(
    dataset,
    mask_info,
    opt_metric,
    features_percent,
    model,
    experiment_name="feature_selection"
    ):

    model_runner = model["model"]
    model_name = model["name"]

    ft_selection_name = mask_info["name"]
    mask_fn = mask_info["mask_fn"]
    mask_level = mask_info["level"].lower().strip()

    data_dir = os.path.join(DATA_BASE_DIR, dataset, experiment_name, opt_metric)
    checkpoint_dir = os.path.join(
                        CHECKPOINT_BASE_DIR, dataset, experiment_name, opt_metric, 
                        ft_selection_name, str(features_percent), model_name
                    )
    
    logger.info("Reading data")
    
    # Reading feature info
    with open(os.path.join(data_dir, "feature_selection_info.json"), "r") as f:
        fs_info = json.load(f)

    # Reading attention
    with open(os.path.join(data_dir, "attention.npy"), "rb") as f:
        attn = np.load(f)

    n_features_selected = max(1, int(features_percent * attn.shape[-1]))
    
    eval_data = []
    for c_process in fs_info["clusters"]:

        if not c_process["feasible"]:
            continue
        
        c_name = f"C{c_process['label']}"
        s_idx = c_process["start_index"]
        e_idx = c_process["end_index"]
        splits = c_process["splits"]

        # Validate if need to process something to avoid many data readings
        needs_processing = False
        for f_name, f_indices in splits.items():
            clf_checkpoint = os.path.join(checkpoint_dir, c_name, f_name)
            scores_checkpoint = os.path.join(clf_checkpoint, "scores.json")
            if os.path.exists(scores_checkpoint):
                logger.info(f"Skipping {dataset}:{opt_metric}:{ft_selection_name}:{features_percent}:{model_name}:{c_name}:{f_name}")
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
            else:
                needs_processing = True

        if not needs_processing:
            continue

        # Reading data
        data, dataset_meta = datalib.read_dataset(dataset)
        data_required_sort = np.argsort(fs_info["data_required_sort"])
        data.index = data_required_sort
        data = data.sort_index()
        target_column = dataset_meta["target"]
        features = data.drop(target_column, axis=1)
        labels = data[target_column]
        multiclass = len(dataset_meta["labels"]) > 2

        assert np.array_equal(labels, fs_info["labels"]), "Something went wrong sorting data"

        mask = None
        if  mask_level == "dataset":
            mask = mask_fn(
                        features.iloc[data_required_sort].iloc[dataset_meta["df_indices"]],
                        labels.iloc[data_required_sort].iloc[dataset_meta["df_indices"]],
                        attn[data_required_sort][dataset_meta["df_indices"]],
                        dataset_meta,
                        n_features_selected
                    )
    
        c_features = features.loc[s_idx:e_idx]
        c_labels = labels.loc[s_idx:e_idx]
        c_attn = attn[s_idx:e_idx]

        if  mask_level == "cluster":
            mask = mask_fn(
                        c_features,
                        c_labels,
                        c_attn,
                        dataset_meta,
                        n_features_selected
                    )

        for f_name, f_indices in splits.items():

            clf_checkpoint = os.path.join(checkpoint_dir, c_name, f_name)
        
            if not os.path.exists(clf_checkpoint):
                os.makedirs(clf_checkpoint)

            scores_checkpoint = os.path.join(clf_checkpoint, "scores.json")

            if not os.path.exists(scores_checkpoint):
                logger.info(f"Training {dataset}:{opt_metric}:{ft_selection_name}:{features_percent}:{model_name}:{c_name}:{f_name}")
                
                train_indices = f_indices["train"] 
                val_indices = f_indices["val"] 

                if  mask_level == "fold":
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
                logger.info(f"Skipping {dataset}:{opt_metric}:{ft_selection_name}:{features_percent}:{model_name}:{c_name}:{f_name}")
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

    experiment_name = "feature_selection_cluster_fmask"

    mask_generators = [
        {"name": "random", "mask_fn": get_random_mask, "level": "fold"},
        {"name": "random_1", "mask_fn": get_random_mask, "level": "fold"},
        {"name": "random_2", "mask_fn": get_random_mask, "level": "fold"},
        {"name": "random_3", "mask_fn": get_random_mask, "level": "fold"},
        {"name": "random_4", "mask_fn": get_random_mask, "level": "fold"},
        {"name": "attention", "mask_fn": get_attention_mask, "level": "fold"},
        {"name": "linear_model", "mask_fn": build_masker_from_model(linear_model.LogisticRegression(random_state=SEED)), "level": "fold"},
        {"name": "decision_tree", "mask_fn": build_masker_from_model(tree.DecisionTreeClassifier(random_state=SEED)), "level": "fold"},
        {"name": "f_classif", "mask_fn": build_masker_from_score(feature_selection.f_classif), "level": "fold"}       
    ]

    # Do not include the 1.0
    ft_percent_selectors = [0.1]
    models=[
        {"name":"KNN", "model": neighbors.KNeighborsClassifier()},
        #{"name":"DT", "model": tree.DecisionTreeClassifier(random_state=SEED)},
        #{"name":"LR", "model": linear_model.LogisticRegression(random_state=SEED)},
        {"name":"MLP", "model": neural_network.MLPClassifier(random_state=SEED)},
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

        attention_dir = os.path.join(DATA_BASE_DIR, dataset, experiment_name, selection_metric)
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

            attn_cluster_info = get_attn_clusters_info(
                attn.copy(),
                labels.copy(),
                n_clusters=FEATURE_SELECTION_N_CLUSTERS[dataset]
            )

            cluster_labels = attn_cluster_info["cluster_labels"]
            indices = attn_cluster_info["data_required_sort"]
            attn = attn[indices]
            labels = labels[indices]
            attn_cluster_info["labels"] = labels.tolist()

            for t_cluster in attn_cluster_info["clusters"]:
                feasible = len(t_cluster["classification_labels"]) >= 2
                t_cluster["feasible"] = bool(feasible)
                if not feasible:
                    logger.warning(f"Classification not feasible {dataset}:C{t_cluster['label']}")
                    continue

                c_l_start = t_cluster["start_index"]
                c_l_end = t_cluster["end_index"]
                splits = {}

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

                t_cluster["splits"] = splits

            with open(attention_file, "wb") as f:
                np.save(f, attn)
            
            with open(attention_info_file, "w") as f:
                json.dump(attn_cluster_info, f, indent=4)

    
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
                {"name": "none", "mask_fn": get_full_mask, "level": "dataset"},
                selection_metric,
                1.0,
                m,
                experiment_name=experiment_name
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
                            m,
                            experiment_name=experiment_name
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
                    
    fs_scores_com_df.to_csv(f"{experiment_name}_scores.csv", index=False)


if __name__ == "__main__":
    main()


    
    