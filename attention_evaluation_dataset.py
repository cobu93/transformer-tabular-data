import os
import numpy as np
import json
import pandas as pd
from utils import log, attention

from sklearn import (
                    linear_model,
                    neighbors, 
                    model_selection, 
                    tree, 
                    feature_selection, 
                    neural_network,
                )

from config import (
    DATA_BASE_DIR,
    FEATURE_SELECTION_K_FOLD,
    SEED
)

from attention_evaluation_cluster import (
        get_attention_mask,
        get_full_mask,
        build_masker_from_model,
        build_masker_from_score,
        get_random_mask,
        feature_selection_evaluation,
        get_attn_clusters_info
)

logger = log.get_logger()

"""
Defines the main data flow, it includes:

- Find or create a sweep
- Run trials until it reaches the number of desired trials per experiment
"""
def main():

    experiment_name = "feature_selection_dataset_fmask"

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
                n_clusters=1
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


    
    