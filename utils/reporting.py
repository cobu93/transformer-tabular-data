import os
from config import CHECKPOINT_BASE_DIR, WORKER_JOBS, DATA_BASE_DIR, K_FOLD, RUN_EXCEPTIONS
import json
from utils import log
import pandas as pd

logger = log.get_logger()

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
                    

    return pd.DataFrame(executions)
