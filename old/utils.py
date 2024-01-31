import pandas as pd
import numpy as np
from sklearn import model_selection, pipeline, preprocessing, impute, compose
import openml
import skorch
from ray import tune


class ReportTune(skorch.callbacks.Callback):    
    
    def __init__(self, *args, **kwargs):
        super(ReportTune, self).__init__(*args, **kwargs)

        self.metrics = {}

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        
        excluded = ["batches"]

        last_history = net.history[-1]
        
        for key in last_history.keys():

            if key in excluded:
                continue

            self.metrics[key] = last_history[key]

            if "_best" in key:
                if last_history[key]:
                    self.metrics[key.replace("_best", "_opt")] = last_history[key.replace("_best", "")]
        
        tune.report(**self.metrics)

def get_default_callbacks(multiclass=False, include_report_tune=True):

    callbacks = [
        ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
        ("accuracy", skorch.callbacks.EpochScoring("accuracy", lower_is_better=False)),
        ("duration", skorch.callbacks.EpochTimer())
        ]

    if not multiclass:
        callbacks.extend([
            ("roc_auc", skorch.callbacks.EpochScoring("roc_auc", lower_is_better=False)),
            ("f1", skorch.callbacks.EpochScoring("f1", lower_is_better=False)),
            ("precision", skorch.callbacks.EpochScoring("precision", lower_is_better=False)),
            ("recall", skorch.callbacks.EpochScoring("recall", lower_is_better=False))
        ])

    if include_report_tune:
        callbacks.append(
            ("report_tune", ReportTune())
        )

    return callbacks

def join_data(features, labels):
    cat_features = np.concatenate(features)
    cat_labels = np.concatenate(labels)
    cat_indices = []

    offset = 0
    for labels_group in labels:
        cat_indices.append(np.arange(offset, offset + labels_group.shape[0]))
        offset += labels_group.shape[0]
    
    return cat_features, cat_labels, tuple(cat_indices)

class FileReporter(tune.progress_reporter.CLIReporter):

    def __init__(self,
                filename,
                metric_columns=None,
                parameter_columns=None,
                max_progress_rows=20,
                max_error_rows=20,
                max_report_frequency=5):

        super(FileReporter, self).__init__(metric_columns, parameter_columns,
                                          max_progress_rows, max_error_rows,
                                          max_report_frequency)

        self.file = open(filename, "w")

    def report(self, trials, done, *sys_info):
        self.file.write(self._progress_str(trials, done, *sys_info))
        
        if not done:
            self.file.seek(0)
        else:
            self.file.close()


def trial_dirname_creator(trial):
    return "trial_{}".format(trial.trial_id)