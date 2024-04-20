import os
from config import CHECKPOINT_BASE_DIR, WORKER_JOBS, DATA_BASE_DIR, K_FOLD, RUN_EXCEPTIONS
import json
from utils import log
import pandas as pd
import numpy as np

logger = log.get_logger()

ARCHITECTURE_COLUMNS = ["architecture_name", 
                        "attn_dropout", 
                        "embed_dim", 
                        "ff_dropout", 
                        "n_head", 
                        "n_layers",
                        "numerical_passthrough", 
                        "optimizer__lr", 
                        "optimizer__weight_decay",
                        "aggregator__cell", 
                        "aggregator__dropout", 
                        "aggregator__hidden_size",
                        "aggregator__num_layers"
                    ]

def load_report_df(
        jobs=WORKER_JOBS,
        checkpoint_dirname=CHECKPOINT_BASE_DIR,
        architectures_filename=os.path.join(DATA_BASE_DIR, "architectures.json"),
        folds_names=[f"F{i}" for i in range(K_FOLD)],
        exception_scores=RUN_EXCEPTIONS,
        optimization_metric="valid_loss"
    ):
    
    executions = []
    datasets = set()
    aggregators = set()
    
    for job in jobs:
        datasets.add(job["dataset"])
        aggregators.add(job["aggregator"])

    with open(architectures_filename, "r") as f:
        architectures = json.load(f)


    for dataset in datasets:
        for aggregator in aggregators:

            if aggregator == "rnn":
                considered_architectures = architectures["rnn"]
            else:
                considered_architectures = architectures["regular"]

            for architecture_name in considered_architectures:
                for fold_name in folds_names:
                    scores_filename = os.path.join(checkpoint_dirname, dataset, aggregator, architecture_name, fold_name, "scores.json")
                    
                    has_score = True

                    if os.path.exists(scores_filename):
                        with open(scores_filename, "r") as f:
                            scores = json.load(f)
                    else:
                        has_score = False
                        scores = {}

                    execution = {
                        "dataset": dataset,
                        "aggregator": aggregator,
                        "architecture_name": architecture_name,
                        "fold_name": fold_name,
                        **considered_architectures[architecture_name],
                        **scores.get(optimization_metric, {})
                    }

                    include_execution = False

                    for ex in exception_scores:
                        include_execution = False

                        for k, v in ex.items():
                            if v != execution[k]:
                                include_execution = True
                                break

                        if not include_execution:
                            break
                    
                    if include_execution:
                        executions.append(execution)

                        if not has_score:
                            logger.warn(f"Execution without scoring: {scores_filename}")
                    

    executions_df = pd.DataFrame(executions)
    return executions_df

def get_executions_cv_score(
        executions_df, 
        arch_cols=ARCHITECTURE_COLUMNS
    ):
    best_archs = executions_df
    best_archs = best_archs.drop(["fold_name"], axis=1) \
                            .groupby(["dataset", "aggregator"] + arch_cols, as_index=False, dropna=False) \
                            .agg(["mean", "std"]) 

    best_archs.columns = ["_".join(col) if col[1] else col[0] for col in best_archs.columns]

    return best_archs

def build_indexer(k, sort_column, sort_mode):
    def indexer(x):

        fill_na_value = None

        if sort_mode == "max":
            fill_na_value = np.inf
        else:
            fill_na_value = -np.inf

        
        x_argsort = x[sort_column].fillna(fill_na_value).argsort().values
        if sort_mode == "max":
            x_argsort = x_argsort[::-1]
            
        return x.index[x_argsort[:k]].values
        
    return indexer

def get_top_k_indices(group, k, sort_column=f"balanced_accuracy_mean", sort_mode="max"):
    best_archs_indices = np.concatenate(
        group.apply(
            # Argsort goes from min to max
            build_indexer(k, sort_column, sort_mode), 
            include_groups=False
        ).values
    )
    
    return best_archs_indices
