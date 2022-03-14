from calendar import c
import torch
import skorch
from sklearn import base, pipeline, preprocessing, compose, metrics
import numpy as np
from ndsl.architecture.attention import TabularTransformer
from ray import tune


def join_data(features, labels):
    cat_features = np.concatenate(features)
    cat_labels = np.concatenate(labels)
    cat_indices = []

    offset = 0
    for labels_group in labels:
        cat_indices.append(np.arange(offset, offset + labels_group.shape[0]))
        offset += labels_group.shape[0]
    
    return cat_features, cat_labels, cat_indices

def build_transformer_model(
    train_indices,
    validation_indices,
    callbacks,
    n_head, # Number of heads per layer
    n_hid, # Size of the MLP inside each transformer encoder layer
    n_layers, # Number of transformer encoder layers    
    n_output, # The number of output neurons
    encoders, # List of features encoders
    dropout=0.1, # Used dropout
    aggregator=None, # The aggregator for output vectors before decoder
    preprocessor=None,
    need_weights=False,
    **kwargs
    ):

    # Define model
    module = TabularTransformer(
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_layers, # Number of transformer encoder layers    
        n_output, # The number of output neurons
        torch.nn.ModuleList(encoders), # List of features encoders
        dropout=dropout, # Used dropout
        aggregator=aggregator, # The aggregator for output vectors before decoder
        preprocessor=preprocessor,
        need_weights=need_weights
    )

    model = skorch.NeuralNetClassifier(
            module=module,
            train_split=skorch.dataset.CVSplit(((train_indices, validation_indices),)),
            callbacks=callbacks,
            **kwargs
        )

    return model

def load_best_params(model, callbacks):
    for cb in callbacks:
        if isinstance(cb[1], skorch.callbacks.Checkpoint):
            print("Restoring best params")
            model.initialize()
            model.load_params(checkpoint=cb[1])

    return model

class DTypeTransformer(base.BaseEstimator, base.TransformerMixin):
    #Class Constructor
    def __init__(self, dtype):
        super().__init__()
        self._dtype = dtype
        
    def get_params(self, deep=True):
        return {
            'dtype': self._dtype
        }
        
    #Return self, nothing else to do here
    def fit(self, X, y=None):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        return X.astype(self._dtype)


class FixRandomSeed(skorch.callbacks.Callback):
    
    def __init__(self, seed=42):
        self.seed = seed
    
    def initialize(self):
        print("Setting random seed to: ",self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        try:
            random.seed(self.seed)
        except NameError:
            import random
            random.seed(self.seed)

        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic=True


class ReportTune(skorch.callbacks.Callback):    

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        
        last_history = net.history[-1]
        
        metrics = {}

        for key in last_history.keys():
            if ("_score" in key and "_score_" not in key) \
                or ("_loss" in key and "_loss_" not in key):
                metrics[key.replace("_score", "")] = last_history[key]
        
        tune.report(**metrics)


def get_default_preprocessing_pipeline(categorical_cols, numerical_cols):
    categorical_transformer = pipeline.Pipeline(steps=[
    ('label', preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])

    numerical_transformer = pipeline.Pipeline(steps=[
        ('scaler', preprocessing.StandardScaler())
    ])

    preprocessing_pipe = pipeline.Pipeline([
        ('columns_transformer', compose.ColumnTransformer(
            remainder='passthrough', #passthough features not listed
            transformers=[
                ('categorical_transformer', categorical_transformer , categorical_cols),
                ('numerical_transformer', numerical_transformer , numerical_cols)
            ]),
        ),
        ('dtype_transform', DTypeTransformer(np.float32))
    ])

    return preprocessing_pipe

def get_default_callbacks(seed=11, multiclass=False):
    if multiclass:
        return [
            ("fix_random_seed", FixRandomSeed(seed=seed)),
            ("balanced_accuracy", skorch.callbacks.EpochScoring(metrics.balanced_accuracy_score, lower_is_better=False)),
            ("accuracy", skorch.callbacks.EpochScoring(metrics.accuracy_score, lower_is_better=False)),
            ("report_tune", ReportTune())
        ]

    return [
        ("fix_random_seed", FixRandomSeed(seed=seed)),
        ("balanced_accuracy", skorch.callbacks.EpochScoring(metrics.balanced_accuracy_score, lower_is_better=False)),
        ("accuracy", skorch.callbacks.EpochScoring(metrics.accuracy_score, lower_is_better=False)),
        ("roc_auc", skorch.callbacks.EpochScoring(metrics.roc_auc_score, lower_is_better=False)),
        ("f1", skorch.callbacks.EpochScoring(metrics.f1_score, lower_is_better=False)),
        ("precision", skorch.callbacks.EpochScoring(metrics.recall_score, lower_is_better=False)),
        ("recall", skorch.callbacks.EpochScoring(metrics.roc_auc_score, lower_is_better=False)),
        ("report_tune", ReportTune())
    ]



def get_default_train_callbacks(seed=11, multiclass=False):
    if multiclass:
        return [
            ("fix_random_seed", FixRandomSeed(seed=seed)),
            ("balanced_accuracy", skorch.callbacks.EpochScoring(metrics.balanced_accuracy_score, lower_is_better=False)),
            ("accuracy", skorch.callbacks.EpochScoring(metrics.accuracy_score, lower_is_better=False))
        ]

    return [
        ("fix_random_seed", FixRandomSeed(seed=seed)),
        ("balanced_accuracy", skorch.callbacks.EpochScoring(metrics.balanced_accuracy_score, lower_is_better=False)),
        ("accuracy", skorch.callbacks.EpochScoring(metrics.accuracy_score, lower_is_better=False)),
        ("roc_auc", skorch.callbacks.EpochScoring(metrics.roc_auc_score, lower_is_better=False)),
        ("f1", skorch.callbacks.EpochScoring(metrics.f1_score, lower_is_better=False)),
        ("precision", skorch.callbacks.EpochScoring(metrics.recall_score, lower_is_better=False)),
        ("recall", skorch.callbacks.EpochScoring(metrics.roc_auc_score, lower_is_better=False))
    ]