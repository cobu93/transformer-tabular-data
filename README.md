# Transformers for high-dimensional tabular data
Proposed Transformer modification for tabular data

## Procedures

### Tab and Ft Transformers results replication

To validate our implementation we replicated the Tab and FT Transformers results. The replication of these results are included in files __tabtransformer_replicate.py__ and __fttransformer_replicate.py__. 

To avoid a retraining of the models we provide the fitted models. If you decide go through this option, download the files and decompress in the root path. To evaluate the models we provide the files __tabtransformer_evaluation.py__ and __fttransformer_evaluation.py__. 


### Dataset selection

The described procedure of dataset selection is included in the __dataset_selection.ipynb__ file. The selected datasets are stored in a file named __selected_datasets.csv__.

### Hyperparameter search and models evaluations

The hyperparameter search was done using the script __full_param_search.py__. Internally, the script executes a tensaorboard instance to check the search progress, the hyperparameter search, a cleansing files procedure, the best model evaluation and each trial evaluation, for each selected dataset.

The hyperparameter search is in the script __param_search.py__. It includes the procedure done by Optuna. The cleansing procedure is in the file __clean_files.py__. It move the files of old Otuna runs checkpoints to a secondary folder which could be deleted manually to free space. The best model evaluation is in the file __model_evaluation.py__. It automatically recovers the best Optuna trial and evaluate it. The evaluation metrics of the best trial is saved inside each dataset folder and it's named as _evaluation.json_. Finally, each trial evaluation is in the script __trials_evaluation.py__. As the best trial evaluation, it save a file inside each dataset folder named _evaluation_trial_<trial_id>.json_.

To compile the information of all trials, we include the script __get_trials_info.py__, which generates a CSV file named _trials_info.csv_.