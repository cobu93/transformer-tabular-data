import os
import shutil
import json
import wandb
import joblib
import numpy as np
from utils import processing
import pandas as pd
from config import ENTITY_NAME, PROJECT_NAME

WORK_ONLINE = False
FORCE_REFACTORING = False

## Delete checkpoints where numerical_passtrough = False

with open("data/architectures.json", "r") as f:
    architectures = json.load(f)

to_delete_regular_archs = []
to_delete_rnn_archs = []

for arch_name, arch in architectures["regular"].items():
    if arch["numerical_passthrough"] == False:
        to_delete_regular_archs.append(arch_name)

for arch_name, arch in architectures["rnn"].items():
    if arch["numerical_passthrough"] == False:
        to_delete_rnn_archs.append(arch_name)

print(f"There are {len(to_delete_regular_archs)} regular architectures to delete")
print(f"There are {len(to_delete_rnn_archs)} rnn architectures to delete")

for dataset in os.listdir("checkpoint"):
    for aggregator in os.listdir(f"checkpoint/{dataset}"):
        if aggregator == "rnn":
            for to_delete_arch in to_delete_rnn_archs:
                arch_dirname = f"checkpoint/{dataset}/{aggregator}/{to_delete_arch}"

                if os.path.exists(arch_dirname):
                    print(f"Removing {arch_dirname}")
                    shutil.rmtree(arch_dirname)
        else:
            for to_delete_arch in to_delete_regular_archs:
                arch_dirname = f"checkpoint/{dataset}/{aggregator}/{to_delete_arch}"

                if os.path.exists(arch_dirname):
                    print(f"Removing {arch_dirname}")
                    shutil.rmtree(arch_dirname)

# Delete runs where numerical_passtrough = "false"
# Update online runs where numerical_passtrough = "true"
                    
runs = os.listdir("wandb")

print(f"There are {len(runs)} runs")

to_delete_runs = []
to_update_runs = []

for run in runs:
    if os.path.isfile(f"wandb/{run}"):
        continue

    filename = f"wandb/{run}/logs/debug.log"

    with open(filename, "r") as file:
        info = file.read()

    if "'numerical_passthrough': 'false'" in info \
    	or "'dataset': 'volkert'" in info \
	or "'dataset': 'anneal'" in info :
        to_delete_runs.append(run)

    if "'numerical_passthrough': 'true'" in info:
        to_update_runs.append(run)


print(f"There are {len(to_delete_runs)} runs to delete")
print(f"There are {len(to_update_runs)} runs to update")

for r in to_delete_runs:
    dirname = f"wandb/{r}"
    print(f"Removing {dirname}")
    shutil.rmtree(dirname)

if WORK_ONLINE:
    api = wandb.Api()
    not_updated = []
    for r in to_update_runs:
        run_id = r.split("-")[-1]
        print(f"Updating {run_id}")
        
        try:
            run = api.run(f"{ENTITY_NAME}/{PROJECT_NAME}/{run_id}")
            run.config["numerical_passthrough"] = True
            run.update()
        except:
            not_updated.append(run_id)
        
    print(f"There were {len(not_updated)} not updated runs")



# Reconstruct preprocessors
# Delete joblib models

for dataset in os.listdir("checkpoint"):

    with open(f"data/{dataset}/train.meta.json") as f:
        dataset_meta = json.load(f)

    for aggregator in os.listdir(f"checkpoint/{dataset}"):    
        considered_archs = None
        if aggregator == "rnn":
            considered_archs = architectures["rnn"]
        else:
            considered_archs = architectures["regular"]

        for arch_name, arch in considered_archs.items():
            for split_name in dataset_meta["splits"].keys():
                checkpoint_name = f"checkpoint/{dataset}/{aggregator}/{arch_name}/{split_name}"

                if os.path.exists(f"{checkpoint_name}/model.jl"):
                    print(f"Removing {checkpoint_name}/model.jl")
                    os.remove(f"{checkpoint_name}/model.jl")

                if os.path.exists(f"{checkpoint_name}/model.valid_loss.jl"):
                    print(f"Removing {checkpoint_name}/model.valid_loss.jl")
                    os.remove(f"{checkpoint_name}/model.valid_loss.jl")

                if os.path.exists(f"{checkpoint_name}/preprocessor.jl"):
                    preprocessor = joblib.load(f"{checkpoint_name}/preprocessor.jl")

                    categorical_unk_val = preprocessor.steps[0][1].transformers_[1][1].steps[0][1].unknown_value
                    if categorical_unk_val != -1 or FORCE_REFACTORING:
                        print(f"Refactoring preprocessor. Categorical unknown value: {categorical_unk_val}")


                        train_indices = np.array(dataset_meta["splits"][split_name]["train"])
                        val_indices = np.array(dataset_meta["splits"][split_name]["val"])

                        new_preprocessor = processing.get_preprocessor(
                            dataset_meta["categorical"],
                            dataset_meta["numerical"],
                            dataset_meta["categories"],
                            categorical_unknown_value=-1
                        )

                        
                        dataset_file = os.path.join("data", dataset, "train.csv")
                        target_column = dataset_meta["target"]
                        n_numerical = dataset_meta["n_numerical"]
                        data = pd.read_csv(dataset_file)
                        features = data.drop(target_column, axis=1)
                        labels = data[target_column]
                        
                        new_preprocessor = new_preprocessor.fit(features.iloc[train_indices])
                        
                        X = preprocessor.transform(features)
                        new_X = new_preprocessor.transform(features)

                        if not np.array_equal(X, new_X):
                            raise ValueError(f"Preprocessing failed, not equal result: {checkpoint_name}")
                        
                        print(f"Saving refactored preprocessor {checkpoint_name}/preprocessor.jl")
                        joblib.dump(new_preprocessor, os.path.join(checkpoint_name, "preprocessor.jl"))







