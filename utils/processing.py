import numpy as np
from sklearn import pipeline, preprocessing, impute, compose, base

import numpy as np

class OffsetTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, offset=1):
        self.offset = offset
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        return X + self.offset

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def get_preprocessor(
        categorical_columns,
        numerical_columns,
        categories,
        categorical_unknown_value=np.nan,
        n_neighbors=10
    ):

    categories = [categories[k] for k in categorical_columns]

    imputer = impute.KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)

    categorical_transformer = preprocessing.OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value", 
                    unknown_value=categorical_unknown_value
                )
    
    offset_transformer = OffsetTransformer()

    numerical_transformer = preprocessing.StandardScaler()

    preprocessor = pipeline.Pipeline([
        ("encodig_scaling", compose.ColumnTransformer(
            remainder="passthrough", #passthough features not listed
            transformers=[
                ('numerical_transformer', numerical_transformer , numerical_columns),
                ("categorical_transformer", pipeline.Pipeline([
                    ("encoder", categorical_transformer), 
                    ("offset", offset_transformer) # Added because KNN Imputer set empties to 0
                ]) , categorical_columns),
                
            ])
        ),
        ("imputation", imputer)
    ])

    return preprocessor



def get_regular_preprocessor(
        categorical_columns,
        numerical_columns,
        categories,
        n_neighbors=10
    ):

    categories = [categories[k] for k in categorical_columns]

    categorical_transformer = preprocessing.OneHotEncoder(
                    categories=categories,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False
                )
    
    numerical_transformer = preprocessing.StandardScaler()

    preprocessor = pipeline.Pipeline([
        ("encodig_scaling", compose.ColumnTransformer(
            remainder="passthrough", #passthough features not listed
            transformers=[
                ('numerical_transformer', pipeline.Pipeline([
                        ("encoding", numerical_transformer),
                        ("imputer", impute.SimpleImputer(strategy="constant"))                                               
                ]) , numerical_columns),
                ("categorical_transformer", pipeline.Pipeline([
                        ("imputer", impute.SimpleImputer(strategy="constant")),
                        ("encoding", categorical_transformer)                        
                ]), categorical_columns),
            ])
        )
    ])

    return preprocessor

