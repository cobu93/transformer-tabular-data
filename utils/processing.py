import numpy as np
from sklearn import pipeline, preprocessing, impute, compose, base

import numpy as np

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

