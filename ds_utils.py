import os
import json
import pandas as pd
import os
import json
import openml

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

def get_default_config(columns, target, numerical_cols, df):
    
    recommended = []
    for column in columns:
        if column != target:
            if column in numerical_cols:
                recommended.append("NumericalEncoder(embedding_size)")
            else:
                n_column_labels = len(df[column].value_counts())
                recommended.append("CategoricalOneHotEncoder(embedding_size, {})".format(n_column_labels))
        
    return recommended

def download_data(
                dataset_did,
                dir_name
                ):
    
    print("Dataset did: {}".format(dataset_did))
    dataset = openml.datasets.get_dataset(dataset_did, download_data=False)
    print("Dataset info:")
    print(dataset)
    target = dataset.default_target_attribute

    print("Target label: {}".format(target))
    # Make dir
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Get data
    if not os.path.exists(os.path.join(dir_name, "dataset.csv")) or not os.path.exists(os.path.join(dir_name, "dataset.json")):
        print("Downloading data...")
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )

        X[target] = y
        X = X.dropna(axis=1, how="all")
        attribute_names = X.columns.to_list()
        attribute_names.remove(target)
        X.to_csv(os.path.join(dir_name, "dataset.csv"), index=False)

        labels = y.unique()

        ds_info = {
            "columns": attribute_names + [target],
            "categorical_columns": [column for column, is_categorical in zip(attribute_names, categorical_indicator) if is_categorical],
            "numerical_columns": [column for column, is_categorical in zip(attribute_names, categorical_indicator) if not is_categorical],
            "target_column": target,
            "target_mapping": {label: idx for idx, label in enumerate(labels)}
        }

        with open(os.path.join(dir_name, "dataset.json"), "w") as f:
            json.dump(ds_info, f)        
    else:
        print("Data exists!")
    
        ds_info = get_data_info(dir_name)
        X, _ = get_data(
            dir_name, 
            ds_info["columns"], 
            ds_info["target_column"],
            ds_info["target_mapping"]
        )


    recommended = get_default_config(
        ds_info["columns"], 
        ds_info["target_column"],
        ds_info["numerical_columns"],
        X
    )

    with open(os.path.join(dir_name, "recommended_conf.json"), "w") as f:
        json.dump(recommended, f, indent=4)
        

def get_data_info(dir_name):
    with open(os.path.join(dir_name, "dataset.json"), "r") as f:
        ds_info = json.load(f)

    columns = ds_info["columns"]
    numerical_cols = ds_info["numerical_columns"]
    categorical_cols = ds_info["categorical_columns"]
    target_col = ds_info["target_column"]
    target_mapping = ds_info["target_mapping"]

    try:
        target_mapping = {int(k): target_mapping[k] for k in target_mapping}
    except:
        pass

    print("Target mapping: {}".format(target_mapping))
    print("Numerical columns:", numerical_cols)
    print("Categorical columns:", categorical_cols)
    print("Columns:", columns)

    return columns, numerical_cols, categorical_cols, target_col, target_mapping

def get_data_2(dir_name, columns, target_col, target_mapping):
        
    # Open data
    df = pd.read_csv(os.path.join(dir_name, "dataset.csv"), header=0, names=columns, na_values=["?", "NA", "N/A", "nan", "NAN", "-", "NaN"])
    
    # Fill nan with mean of closest neighbors
    X = df.drop(target_col, axis=1)
    
    X = X.apply(lambda series: pd.Series(
        LabelEncoder().fit_transform(series[series.notnull()]),
        index=series[series.notnull()].index
    ))

    imputer = KNNImputer(n_neighbors=10)
    imputed_data = imputer.fit_transform(X)
    df.loc[:, X.columns] = imputed_data

    # Replace target mapping
    df[target_col] = df[target_col].replace(target_mapping)

    return df.drop(target_col, axis=1), df[target_col]

def get_data(dir_name, columns, target_col, target_mapping):
        
    # Open data
    df = pd.read_csv(os.path.join(dir_name, "dataset.csv"), header=0, names=columns, na_values=["?", "NA", "N/A", "nan", "NAN", "-", "NaN"])
    
    num_cols = df._get_numeric_data().columns
    for n_col in num_cols:
        df[n_col] = df[n_col].fillna(-1)

    for c_col in list(set(columns) - set(num_cols)):
        df[c_col] = df[c_col].fillna("NaN")

    # Fill nan with mean of closest neighbors
    X = df.drop(target_col, axis=1)

    # Replace target mapping
    df[target_col] = df[target_col].replace(target_mapping)

    return df.drop(target_col, axis=1), df[target_col]