# Transformers for high-dimensional tabular data

This repository contains the experiments introduced in the "Transformers for high-dimensional tabular data" paper.

## Downloading required data

To get started, the initial step is to clone or download the repository. However, please note that some additional files need to be downloaded as well since they are pretty large. These files are crucial to avoid the tedious process of reproducing results, training models, and extracting information from the datasets.


- [Replicating results files](https://correoipn-my.sharepoint.com/:u:/g/personal/ucoronab_ipn_mx/EaJg3TK85n5HkaDydbPpq7UBa1VcqcrhWDlzMP4ZJwe6Ww?e=QD130r) 
- [Hyperparameter search files](https://correoipn-my.sharepoint.com/:u:/g/personal/ucoronab_ipn_mx/EWN7ic04YQ1OldxVpWLzTgkBC5RlMRum45-ARjrr1rjIGg?e=605v7A)
- [Cumulative attention vectors](https://correoipn-my.sharepoint.com/:u:/g/personal/ucoronab_ipn_mx/ESekCil3dUlBk2am7wisHuYBjquzxgNDLueaLJbsf7j-wg?e=PaBPPS)

Below we explain how we used these files. If you don't require them all, continue reading and downloading those you need. You must extract the files in the repository's root path.

## Recommended setup

Our recommendation is to install conda, create a new environment and activate it running:

~~~
conda create -n tab-trans python==3.8.11
conda activate tab-trans
~~~

The requirements for executing any code are listed in the _requirements.txt_ file. If you are in the conda environment, you can install them by running:

~~~
pip install -r requirements.txt
~~~

## Procedures

### Tab and FT Transformers replication

To validate our implementation, we replicated the Tab and FT Transformers results. The replication of these results is included in files __tabtransformer_replicate.py__ and __fttransformer_replicate.py__. 

To avoid retraining the models, we provide the [fitted models](https://correoipn-my.sharepoint.com/:u:/g/personal/ucoronab_ipn_mx/EaJg3TK85n5HkaDydbPpq7UBa1VcqcrhWDlzMP4ZJwe6Ww?e=QD130r). If you go through this option, download the files and decompress them in the root path. To evaluate the models and get the same results reported for those architectures, we provide the files __tabtransformer_evaluation.py__ and __fttransformer_evaluation.py__. 

### Dataset selection

The described procedure of dataset selection is included in the __dataset_selection.ipynb__ file. The selected datasets and their properties are stored in a file named __selected_datasets.csv__.

### Hyperparameter search and models evaluations

The hyperparameter search was done using the script __full_param_search.py__. Internally, the script executes a tensorboard instance, the hyperparameter search, a cleansing files procedure, the best model evaluation, and each trial evaluation for each selected dataset.

The hyperparameter search is in the script __param_search.py__. It includes the procedure done by Optuna. The cleansing procedure is in the file __clean_files.py__. It moves the files of old Optuna executions to a secondary folder which can be deleted manually to free space. The best model evaluation is in the file __model_evaluation.py__. It automatically recovers the best Optuna trial and evaluates it. The evaluation metrics of the best trial are saved inside each dataset folder in the file _evaluation.json_. Finally, each trial evaluation is in the script __trials_evaluation.py__. As the best model evaluation, it saves a file inside each dataset folder named _evaluation_trial\_<trial_id>.json_.

To compile the information of all trials, we include the script __get_trials_info.py__, which generates a CSV file named _trials\_info.csv_.

To avoid the repetition of this process, we provide the [hyperparameter search files](https://correoipn-my.sharepoint.com/:u:/g/personal/ucoronab_ipn_mx/EWN7ic04YQ1OldxVpWLzTgkBC5RlMRum45-ARjrr1rjIGg?e=605v7A), including the fitted models and the Optuna files.

### Cumulative attention

To extract the cumulative attention vectors for each dataset's best result, we provide the script __attention_extraciton.py__. It generates a folder named attention. Inside the folder, there exists a folder for each dataset. Inside each dataset folder are the cumulative vectors for train, validation, and test, which were written using numpy. The last column of the array is the label assigned to each cumulative vector. To avoid the extractionprocess we provide the [cumulative attention vectors](https://correoipn-my.sharepoint.com/:u:/g/personal/ucoronab_ipn_mx/ESekCil3dUlBk2am7wisHuYBjquzxgNDLueaLJbsf7j-wg?e=PaBPPS) generated.

## Results

To reproduce the results in the paper, we include the notebook __Results.ipynb__.