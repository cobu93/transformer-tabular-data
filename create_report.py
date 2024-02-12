import pandas as pd
import os
from config import CHECKPOINT_BASE_DIR
import json
from utils import log

logger = log.get_logger()

def num_as_str(x):
    return "{0:.3f}".format(x)

dataset_map = {
    "adult": "ADUL",
    "australian": "AUST",
}

aggregator_map = {
    "cls": "CLS",
    "concatenate": "CAT",
    "max": "MAX",
    "mean": "AVG",
    "rnn": "RNN",
    "sum": "SUM",
}

executions = []

for dataset in os.listdir(CHECKPOINT_BASE_DIR):
    for aggregator in os.listdir(os.path.join(CHECKPOINT_BASE_DIR, dataset)):
        for architecture in os.listdir(os.path.join(CHECKPOINT_BASE_DIR, dataset, aggregator)):
            for fold in os.listdir(os.path.join(CHECKPOINT_BASE_DIR, dataset, aggregator, architecture)):
                scores_filename = os.path.join(CHECKPOINT_BASE_DIR, dataset, aggregator, architecture, fold, "scores.json")

                if os.path.exists(scores_filename):
                    with open(scores_filename, "r") as f:
                        scores = json.load(f)
                
                    for opt_metric in scores.keys():
                        execution = {
                            "dataset": dataset,
                            "aggregator": aggregator,
                            "architecture": architecture,
                            "fold": fold,
                            "optimization_metric": opt_metric,
                            **scores[opt_metric]
                        }

                        executions.append(execution)

                else:
                    logger.warn(f"Execution without scoring: {scores_filename}")

executions_df = pd.DataFrame(executions)
executions_df = executions_df.query("optimization_metric == 'valid_loss'")
executions_df = executions_df.drop("optimization_metric", axis=1)
print(executions_df)

best_archs = executions_df.drop(["fold"], axis=1) \
                        .groupby(["dataset", "aggregator", "architecture"], as_index=False) \
                        .agg(["mean", "std"]) 

best_archs.columns = ["_".join(col) if col[1] else col[0] for col in best_archs.columns]
print(best_archs)

best_archs = best_archs.loc[best_archs.groupby(["dataset", "aggregator"])["log_loss_mean"].idxmin()]
best_archs["reported_metric"] = "$" + best_archs["balanced_accuracy_mean"].apply(num_as_str) \
                                + " \pm " + best_archs["balanced_accuracy_std"].apply(num_as_str) + "$"
print(best_archs)

#best_archs = best_archs.set_index(["dataset", "aggregator"])["log_loss_mean"].transpose()
best_archs = best_archs[["dataset", "aggregator", "reported_metric"]] \
                        .pivot(index="dataset", columns="aggregator") \
                        .reset_index()
best_archs.columns = [col[1] for col in best_archs.columns]
print(best_archs.to_latex(index=False))
