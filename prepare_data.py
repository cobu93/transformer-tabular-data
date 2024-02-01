"""
This files includes:

1. Dataset selection (Cluster-based)
2. Data partition (80% train divided in 10 splits, 20% test)
"""

import pandas as pd
from sklearn import cluster, model_selection
import numpy as np
import openml
from config import SEED, DATASETS_FILE, DATA_BASE_DIR, TEST_PARTITION, K_FOLD
from utils import log, data
import os
import json

logger = log.get_logger()

np.random.seed(SEED)

# Read and validate datasets' information
logger.info("Validating dataset's information")
df = pd.read_csv(DATASETS_FILE)
datasets = openml.datasets.get_datasets(dataset_ids=df["openml_id"].values.tolist(), download_data=False)
assert np.all(df["n_categorical"].values == np.array([len(ds.get_features_by_type("nominal")) + len(ds.get_features_by_type("string")) for ds in datasets]))

# Compute the metadata for datasets clustering
logger.info("Computing datasets features")
df["features_instances"] = df["n_features"] / df["n_instances"]
df["percent_numerical"] = df["n_numerical"] / (df["n_features"] - 1)
df["percent_missing_values"] =  df["n_missing_values"] / ((df["n_numerical"] + df["n_categorical"] - 1) * df["n_instances"])

# Clustering datasets
logger.info("Clustering datasets")
vectors = df[["features_instances", "percent_numerical", "percent_missing_values"]].values
clustering = cluster.KMeans(n_clusters=5, n_init=10, random_state=SEED)
clustering = clustering.fit(vectors)

# Select the 2 centroid nearest datasets
selected = []
all_labels = set(clustering.labels_)
for label in all_labels:
    ds_indices = np.argwhere(clustering.labels_ == label).flatten() 
    cluster_center = clustering.cluster_centers_[label]
    distances = np.sqrt(np.sum((vectors[ds_indices] - cluster_center) ** 2, axis=-1))
    
    assert ds_indices.shape == distances.shape
        
    selected.extend(ds_indices[np.argsort(distances)[:2]])   
    
df["label"] = clustering.labels_
df_selected = df.iloc[selected].reset_index(drop=True)

logger.info("The selected datasets are:")
for ds in df_selected["name"]:
    logger.info("\t" + ds)



logger.info("Processing selection")
for ds in df_selected.iloc:

    logger.info("=" * 50 + f" Processing {ds['name']}")
    dataset_info = data.read_dataset_by_id(int(ds["openml_id"]))
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        dataset_info["features"], 
        dataset_info["outputs"], 
        test_size=TEST_PARTITION, 
        random_state=SEED
    )

    test_meta = {
        "target": dataset_info["target"],
        "labels": dataset_info["labels"],
        "columns": dataset_info["columns"].tolist(),
        "categorical": dataset_info["categorical"].tolist(),
        "categories": {k:v.tolist() for k, v in dataset_info["categories"].items()},
        "n_categorical": dataset_info["n_categorical"],
        "numerical": dataset_info["numerical"].tolist(),
        "n_numerical": dataset_info["n_numerical"],
        "df_indices": X_test.index.tolist()
    }

    test_meta_dir = os.path.join(DATA_BASE_DIR, ds["name"])
    test_meta_file = os.path.join(test_meta_dir, "test.meta.json")
    test_data_file = os.path.join(test_meta_dir, "test.csv")

    if os.path.exists(test_meta_file):
        logger.info("Comparing existing information for testing")
        with open(test_meta_file, "r") as f:
            old_test_meta = json.load(f)
        
        if str(dict(sorted(old_test_meta.items()))) != str(dict(sorted(test_meta.items()))):
            logger.fatal("Previous an current information is not the same")
            raise ValueError("Previous an current information is not the same")

    else:
        logger.info("Writing test meta")
        
        if not os.path.exists(test_meta_dir):
            os.makedirs(test_meta_dir)

        with open(test_meta_file, "w") as f:
            json.dump(test_meta, f, indent=4)

    logger.info("Writing test data")
    X_test[dataset_info["target"]] = y_test
    X_test.to_csv(test_data_file, index=False)

    # Replicating for training data

    splitter = model_selection.StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
    k_splits = {}
    for i, (train_indices, val_indices) in enumerate(splitter.split(X_train, y_train)):

        assert np.max(train_indices) < X_train.shape[0], "Indices are not correct"
        assert np.max(val_indices) < X_train.shape[0], "Indices are not correct"

        k_splits[f"F{i}"] = {
            "train": train_indices.tolist(),
            "val": val_indices.tolist(),
        }
    
    
    train_meta = {
        "target": dataset_info["target"],
        "labels": dataset_info["labels"],
        "columns": dataset_info["columns"].tolist(),
        "categorical": dataset_info["categorical"].tolist(),
        "categories": {k:v.tolist() for k, v in dataset_info["categories"].items()},
        "n_categorical": dataset_info["n_categorical"],
        "numerical": dataset_info["numerical"].tolist(),
        "n_numerical": dataset_info["n_numerical"],
        "df_indices": X_train.index.tolist(),
        "splits": k_splits
    }

    train_meta_dir = os.path.join(DATA_BASE_DIR, ds["name"])
    train_meta_file = os.path.join(train_meta_dir, "train.meta.json")
    train_data_file = os.path.join(train_meta_dir, "train.csv")

    if os.path.exists(train_meta_file):
        logger.info("Comparing existing information for training")
        with open(train_meta_file, "r") as f:
            old_train_meta = json.load(f)
        
        if str(dict(sorted(old_train_meta.items()))) != str(dict(sorted(train_meta.items()))):
            logger.fatal("Previous an current information is not the same")
            raise ValueError("Previous an current information is not the same")

    else:
        logger.info("Writing train meta")
        
        if not os.path.exists(train_meta_dir):
            os.makedirs(train_meta_dir)

        with open(train_meta_file, "w") as f:
            json.dump(train_meta, f, indent=4)

    logger.info("Writing train data")
    X_train[dataset_info["target"]] = y_train
    X_train.to_csv(train_data_file, index=False)




    #print(X_train)
    #print(y_train)

