from builders import TransformerConfig, DatasetConfig, SearchSpaceConfig
from ds_utils import download_data, get_data_info, get_data
from typing import Dict, List
from ndsl.module.encoder import *
from ndsl.module.aggregator import *
from ndsl.module.preprocessor import *
import os
from sklearn.model_selection import train_test_split
from ray import tune

class KrVsKpTransformerConfig(TransformerConfig):
    def __init__(self):
        pass

    def get_encoders(self, embedding_size, *args, **kwargs) -> List[FeatureEncoder]:
        return [
                        CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 3),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 2),
            CategoricalOneHotEncoder(embedding_size, 3)
            ]
    
    def get_aggregator(self, embedding_size, hidden_size=128, **kwargs) -> BaseAggregator:

        kwargs = {
            "input_size": embedding_size, 
            "output_size": hidden_size,
            "hidden_size": hidden_size,
            "cell": kwargs["cell"],
            "num_layers": kwargs["num_layers"],
            "dropout": kwargs["dropout"]
        }

        return RNNAggregator(
            **kwargs
        )

    
    def get_preprocessor(self, *args, **kwargs) -> BasePreprocessor:
        return IdentityPreprocessor()


class KrVsKpSearchSpaceConfig(SearchSpaceConfig):
    def get_search_space(self):
        return {
            "n_layers": tune.randint(1, 5), # Number of transformer encoder layers    
            "optimizer__lr": tune.loguniform(1e-5, 1e-1),
            "n_head": tune.choice([1, 2, 4, 8, 16, 32]), # Number of heads per layer
            "n_hid": tune.choice([32, 64, 128, 256, 512, 1024]), # Size of the MLP inside each transformer encoder layer
            "dropout": tune.uniform(0, 0.5), # Used dropout
            "embedding_size": tune.choice([32, 64, 128, 256, 512, 1024]),
            "numerical_passthrough": tune.choice([False, True]),

            # Exclusive RNN param search
            "aggregator__cell": tune.choice(["LSTM", "GRU"]),
            "aggregator__hidden_size": tune.choice([32, 64, 128, 256, 512, 1024]),
            "aggregator__num_layers": tune.randint(1, 3),
            "aggregator__dropout": tune.uniform(0, 0.5)
        }

class KrVsKpDatasetConfig(DatasetConfig):

    def __init__(self):
        self.dir_name = "kr_vs_kp/data"
        self.columns = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.target_col = None
        self.target_mapping = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def download(self):
        download_data(
                3,
                self.dir_name 
                )

    def load(self, train_size=0.65, val_size=0.15, test_size=0.20, seed=11):
        self.columns, self.numerical_cols, self.categorical_cols, self.target_col, self.target_mapping = get_data_info(self.dir_name)
        X, y = get_data(self.dir_name, self.columns, self.target_col, self.target_mapping)

        y = y.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=test_size, 
            random_state=seed
        )

        val_size = X.shape[0] * val_size / X_train.shape[0]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, 
            y_train, 
            test_size=val_size, 
            random_state=seed
        )    

        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.X_test = X_test
        self.y_test = y_test

    def exists(self) -> bool:
        return os.path.exists(os.path.join(self.dir_name, "dataset.csv")) and os.path.exists(os.path.join(self.dir_name, "dataset.json"))
        
    def get_train_data(self):
        return self.X_train, self.y_train
    
    def get_val_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_numerical_columns(self) -> List[str]:
        return self.numerical_cols

    def get_categorical_columns(self) -> List[str]:
        return self.categorical_cols

    def get_label_columns(self) -> str:
        return self.target_col

    def get_n_labels(self):
        return len(self.target_mapping)
