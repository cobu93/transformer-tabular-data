from sklearn import metrics
import numpy as np
import skorch
import os

def get_default_scores(target, prediction_proba, prefix="", multiclass=False):

    prediction = np.argmax(prediction_proba, axis=1)
    labels = np.arange(prediction_proba.shape[-1])

    scores = {
        f"{prefix}balanced_accuracy": metrics.balanced_accuracy_score(target, prediction),
        f"{prefix}accuracy": metrics.accuracy_score(target, prediction),
        f"{prefix}log_loss": metrics.log_loss(target, prediction_proba, labels=labels)
    }

    if not multiclass:
        scores = {
            **scores,
            f"{prefix}roc_auc": metrics.roc_auc_score(target, prediction),
            f"{prefix}f1": metrics.f1_score(target, prediction),
            f"{prefix}precision": metrics.precision_score(target, prediction),
            f"{prefix}recall": metrics.recall_score(target, prediction)
        }

    return scores

def get_default_feature_selection_scores(target, prediction, prefix="", multiclass=False):

    scores = {
        f"{prefix}balanced_accuracy": metrics.balanced_accuracy_score(target, prediction),
        f"{prefix}accuracy": metrics.accuracy_score(target, prediction)
    }

    if not multiclass:
        scores = {
            **scores,
            f"{prefix}roc_auc": metrics.roc_auc_score(target, prediction),
            f"{prefix}f1": metrics.f1_score(target, prediction),
            f"{prefix}precision": metrics.precision_score(target, prediction),
            f"{prefix}recall": metrics.recall_score(target, prediction)
        }

    return scores

def list_models(dir):
    model_path = []
    for f in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, f)):
            model_path.append(f)

    return model_path

def load_model(model, checkpoint_dir):
    checkpoint = skorch.callbacks.Checkpoint(dirname=checkpoint_dir)
    model.initialize()
    model.load_params(checkpoint=checkpoint)
    return model
