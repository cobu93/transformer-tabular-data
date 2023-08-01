import pandas as pd
import numpy as np
from sklearn import model_selection, pipeline, preprocessing, impute, compose
import openml
import skorch
from ray import tune

def read_dataset_by_id(
    id
    ):

    dataset_info = openml.datasets.get_dataset(id, download_data=False)
    target = dataset_info.default_target_attribute

    features, outputs, categorical_mask, columns = dataset_info.get_data(
            dataset_format="dataframe", target=target
        )

    # Remove rows with all nans
    features = features.dropna(axis=0, how="all")
    # Remove columns with all nans
    features = features.dropna(axis=1, how="all")

    removed_cols = set(columns) - set(columns).intersection(set(features.columns))

    removed_mask = np.isin(columns, list(removed_cols))
    columns = np.array(columns)
    columns = columns[~removed_mask]
    
    categorical_mask = np.array(categorical_mask)
    categorical_mask = categorical_mask[~removed_mask]

    assert features.shape[0] == outputs.shape[0], "Invalid features and predictions shapes"

    labels = {value: idx for idx, value in enumerate(outputs.unique().categories.values)}
    outputs = outputs.replace(labels).values

    categorical = columns[categorical_mask]
    numerical = columns[~categorical_mask]

    categories = {}

    for col in categorical:
        categories[col] = features[col].dropna().unique().categories.values

    return {
        "features": features,
        "outputs": outputs,
        "target": target,
        "labels": labels,
        "columns": columns,
        "categorical": categorical,
        "categories": categories,
        "n_categorical": [ len(categories[k] )for k in categories ],
        "numerical": numerical,
        "n_numerical": len(numerical)
    }

def read_dataset(
    name,
    datasets_file="selected_datasets.csv"
    ):

    datasets = pd.read_csv(datasets_file)
    dataset_dict = datasets.query(f"name == '{name}'").to_dict()
    dataset_dict = {k: list(dataset_dict[k].values())[0] for k in dataset_dict}

    return read_dataset_by_id(dataset_dict["id"])
        

def split_train_val_test(
        features,
        predictions,
        val_size=0.15,
        test_size=0.2,
        seed=7        
    ):

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
            features, 
            predictions, 
            test_size=test_size, 
            random_state=seed
        )

    if val_size < 1:
        val_size = features.shape[0] * val_size / X_train.shape[0]

    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train, 
        y_train, 
        test_size=val_size, 
        random_state=seed
    )    

    return {
        "train": { "features": X_train, "outputs": y_train },
        "val": { "features": X_val, "outputs": y_val },
        "test": { "features": X_test, "outputs": y_test },
    }

def get_preprocessor(
        categorical_columns,
        numerical_columns,
        categories,
        n_neighbors=10
    ):

    categories = [categories[k] for k in categories]
    n_categorical = len(categorical_columns)
    n_numerical = len(numerical_columns)
    n_total = n_categorical + n_numerical

    imputer = impute.KNNImputer(n_neighbors=n_neighbors)

    categorical_transformer = preprocessing.OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value", 
                    unknown_value=np.nan
                )

    numerical_transformer = preprocessing.StandardScaler()

    preprocessor = pipeline.Pipeline([
        ("categorical", compose.ColumnTransformer(
            remainder="passthrough", #passthough features not listed
            transformers=[
                ("categorical_transformer", categorical_transformer , categorical_columns)
            ])
        ),
        ("imputation", imputer),
        ("numerical", compose.ColumnTransformer(
            remainder="passthrough", #passthough features not listed
            transformers=[
                ('numerical_transformer', numerical_transformer , np.arange(n_categorical, n_total))
            ]),
        )
    ])

    return preprocessor


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