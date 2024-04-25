# Dynamic feature selection with Transformers for tabular data

This repository contains the experiments introduced in the "Dynamic feature selection with Transformers for tabular data" paper.

## Abstract

Tabular data is a widely used data format in machine learning because it can naturally describe many different phenomena. However, deep learning has not been as successful with tabular data as with other applications such as natural language processing or vision. In this paper, we demonstrate that Transformers can perform better than other methods for tabular data. We argue that this is because Transformers can function as an efficient dynamic feature selection algorithm.  We introduce a measure that captures the _cumulative_ attention that a feature gets across all the layers of the Transformer. Our experiments revealed that Transformers learn to give more _cumulative_ attention to relevant features, which is important for high-dimensional datasets where many features may be irrelevant or the relevant features may change depending on the input. Moreover, Transformers can handle a variable number of features with the same number of parameters. This contrasts with other machine learning models whose parameters increase with the number of features, requiring larger datasets to deduce relationships among features effectively. Our findings highlight the potential of Transformer-based models in addressing challenges associated with tabular data.


## Recommended setup

For practicality, we include a conda environment YML file ready to be imported. By simply running the following command the dependencies installation will be done (requires conda):

~~~
conda env create -f environment.yml
~~~

## Downloading data

To correctly replicate the results, download the full "data" directory containing the cumulative attention matrices and combine with the one existent.

- [Data directory](https://correoipn-my.sharepoint.com/:u:/g/personal/ucoronab_ipn_mx/ESjvdbUKeI9Dt8AcTU4ICIQB75_Al-wEZFRq-GKpKx7y4w?e=DmI9Hc) 

## Reproducing results

To rapidly reproduce the results shown in the paper, we include the notebook __Results.ipynb__. All information to execute the results' notebook is contained in this repository and in the downloaded data directory. The required files are:

- cross_validation_scores.csv: This files contains the cross-validation results for all the architectures explored in the hyperparameter search space for every dataset.

- selected_architectures.csv: Contains the architectures selected achieving the lowest cross-entropy loss.

- feature_selection_dataset_fmask_scores.csv: Contains the results achieved when applying the feature selection at dataset level.

- feature_selection_cluster_fmask_scores.csv: Contains the results achieved when applying the feature selection at cluster level.


If you would want to replicate the results from scratch (may take a several amount of time) we include the following scripts:

- setup.py: Prepare data and partitions used through the article experiments. 
- hp_search.py: This scripts executes the hyperparameter search proposed in the article.
- model_selection.csv: Selects and refits the architectures with the lowest cross-validation cross-entropy loss mean for each dataset.
- attention_evaluation_dataset.py: Evaluates the usage of cumulative attention and other feature selection algorithms at dataset level.
- attention_evaluation_cluster.py: Evaluates the usage of cumulative attention and other feature selection algorithms at cluster level.


## Training new datasets

Given that the included code could be a looks a little bit ad-hoc to our specific experiments, a notebook describing the process for training new datasets is included. The notebook is named __example.py__.

The notebook includes:
- A process of training, evaluating and testing a simulated dataset.
- The process of extracting attention cubes.
- The process for computing the cumulative attentions.
- The visualization of the cumulative attentions for the simulated data.

# Cumulative attention

The datafolder contains the cumulative attention vectors generated for each dataset. Nevertheless, since it could be of your interest to generate your own cumulative attention vectors, we provide the code for computing them.

Firstly, it is important to mention that we couldn't find a way to recover the cube attentions, whether directly in the Transformer encoder or through hooks using the current PyTorch's implementations. That is why we developed our own [Transformer implementation](https://github.com/cobu93/attn-fs-archs).

To compute the cumulative attention, we provide a function in the __utils.attention__ package. Once you have recovered the cube attention, you can compute the cumulative attention vectors by calling:

```python
from utils import attention

# Output dims: n_layers, batch_size, n_features
_, cumulative_attention = attention.compute_std_attentions(
    attention_cubes, # Dims: n_layers, batch_size, n_heads, n_features, n_features 
    aggregator, # {"cls", "other"}
)
```

If you simply want to test the function, copy and run the following code snippet: 


```python
import torch
import numpy as np
import torch.nn.functional as F
from utils import attention

n_layers = 4
batch_size = 16
n_heads = 4
n_features = 10

attention_cubes = torch.randn(
                        n_layers,
                        batch_size,
                        n_heads,
                        n_features,
                        n_features
                        )

# Normalizing as a probability mass function
attention_cubes = F.softmax(attention_cubes, dim=-1)

# Testing normalization
assert np.allclose(attention_cubes.sum(dim=-1), 1), "Bad normalization"

_, cumulative_attentions = attention.compute_std_attentions(
    attention_cubes,
    "other"
)

# Testing cum_attentions normality
assert np.allclose(cumulative_attentions.sum(dim=-1), 1), "Bad normalization" 
print("Cumulative attention shape:", cumulative_attentions.shape)
```

It is important to send "cls" to the function if you are using the FT-Transformer approach. Because the predictions are generated solely using the __CLS__ token for this architecture, the considered attention is the given through this embedding in the last layer, then, its computation is not as using any other method considering the whole set of the embeddings in the last layer.