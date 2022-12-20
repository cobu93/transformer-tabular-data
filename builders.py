from typing import List, Dict
from ndsl.module.encoder import FeatureEncoder
from ndsl.module.aggregator import BaseAggregator
from ndsl.module.preprocessor import BasePreprocessor

class TransformerConfig():
    def __init__(self):
        pass

    def get_decoder_hidden_units(self):
        raise NotImplementedError("This feature hasn't been implemented yet")

    def get_decoder_activation_fn(self):
        raise NotImplementedError("This feature hasn't been implemented yet")

    def get_n_categories(self):
        raise NotImplementedError("This feature hasn't been implemented yet")

    def get_n_numerical(self):
        raise NotImplementedError("This feature hasn't been implemented yet")
    
    def get_aggregator(self, *args, **kwargs) -> BaseAggregator:
        raise NotImplementedError("This feature hasn't been implemented yet")
    
    def get_preprocessor(self, *args, **kwargs) -> BasePreprocessor:
        raise NotImplementedError("This feature hasn't been implemented yet")


class SearchSpaceConfig():
    def get_search_space(self) -> Dict:
        raise NotImplementedError("This feature hasn't been implemented yet")



class DatasetConfig():

    def __init__(self):
        pass

    def download(self):
        raise NotImplementedError("This feature hasn't been implemented yet")

    def exists(self) -> bool:
        raise NotImplementedError("This feature hasn't been implemented yet")

    def get_train_data(self):
        raise NotImplementedError("This feature hasn't been implemented yet")
    
    def get_val_data(self):
        raise NotImplementedError("This feature hasn't been implemented yet")

    def get_test_data(self):
        raise NotImplementedError("This feature hasn't been implemented yet")

    def get_numerical_columns(self) -> List[str]:
        raise NotImplementedError("This feature hasn't been implemented yet")

    def get_categorical_columns(self) -> List[str]:
        raise NotImplementedError("This feature hasn't been implemented yet")

    def get_label_columns(self) -> str:
        raise NotImplementedError("This feature hasn't been implemented yet")

    def get_n_labels(self) -> int:
        raise NotImplementedError("This feature hasn't been implemented yet")