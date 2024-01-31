import wandb

from skorch.callbacks import Checkpoint
from utils.callback import get_default_callbacks
from utils.training import train
import wandb


import os
import numpy as np

from config import (
    DEVICE,
)

from utils import log

logger = log.get_logger()


"""
Define the sweep configuration
"""
sweep_configuration = {
    "method": "random",
    "name": "hyperparameter-search",
    "metric": {"goal": "maximize", "name": "balanced_accuracy"},
    "parameters": {
        "dropout": {"values": [0., 0.2, 0.4]},
        "optimizer__lr": {"values": [1e-5, 5e-5, 1e-4, 2e-4]},
        "optimizer__weight_decay": {"values": [0.05, 0.1]},
        "decoder_hidden_sizes": {"values": [[128, 64]]},
        "decoder_activations": {"values": ["r-r-i"]},
        "aggregator": {"values": ["cls", "max", "mean", "sum", "rnn"]},
        "aggregator_params__output_size": {"values": [128, 256]},
        "aggregator_params__cell": {"values": ["LSTM", "GRU"]},
        "aggregator_params__num_layers": {"values": [2]},
        "aggregator_params__dropout": {"values": [0., 0.2, 0.4]},
    }
}


"""
Reset the WandB environment variables
"""
def reset_wandb_env(exclude={
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }):
    
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


"""
Train a single model here
"""
def train(fold_name, sweep_id, sweep_run_name, config):
    run_name = f"{sweep_run_name}-{fold_name}"
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )

    

    val_accuracy = random.random()

    run.log(dict(val_accuracy=val_accuracy))
    run.finish()
    return val_accuracy

"""
Define the function performing the cross validation
"""
def cross_validate():
    num_folds = 3

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = f"{project_url}/groups/{sweep_id}"
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
    sweep_run_id = sweep_run.id
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)

    metrics = []
    for num in range(num_folds):
        reset_wandb_env()
        result = train(
            sweep_id=sweep_id,
            num=num,
            sweep_run_name=sweep_run_name,
            config=dict(sweep_run.config),
        )
        metrics.append(result)

    # resume the sweep run
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    # log metric to sweep run
    sweep_run.log(dict(val_accuracy=sum(metrics) / len(metrics)))
    sweep_run.finish()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)

def main():
    wandb.login()
    sweep_id = wandb.sweep(sweep_configuration, project='sweep-test')
    wandb.agent(sweep_id, function=cross_validate)

    wandb.finish()

def finetuning_run(config=None):

    with wandb.init(config=config):

        config = wandb.config
        aggregator_params = {}

        if config.aggregator == "rnn":
            aggregator_params = {
                "output_size": config.aggregator_params__output_size,
                "cell": config.aggregator_params__cell,
                "num_layers": config.aggregator_params__num_layers,
                "dropout": config.aggregator_params__dropout
            }


        logging.info("Building callbacks.")        
        checkpoint_dirname = os.path.join(FT_CHECKPOINT_DIR, wandb.run.id)
        
        if os.path.exists(checkpoint_dirname):
            msg = f"Duplicated name {wandb.run.id}."
            logging.error(msg)
            raise ValueError(msg)
        
        callbacks = get_classification_callbacks(early_stopping_params={
            "monitor": "valid_loss", 
            "patience": FT_EARLY_STOPPING_PATIENCE, 
            "lower_is_better": True
        }) + [
            LoadPartialInit(
                Checkpoint(dirname=os.path.join(SS_CHECKPOINT_DIR, config.preload_id, FT_PRELOADED_METRIC)),
                restart_keys_regex=[
                            "categorical_embedding\.categories_means", 
                            "categorical_embedding\.categories_logvars"
                        ]
            )
        ]

        for checkpoint_metric in FT_CHECKPOINT_METRICS:
            checkpoint_dirname_metric = os.path.join(checkpoint_dirname, checkpoint_metric)
            os.makedirs(checkpoint_dirname_metric)
        
            callbacks += [
                        (f"checkpoint_{checkpoint_metric}", Checkpoint(
                            monitor=checkpoint_metric, 
                            dirname=checkpoint_dirname_metric 
                            ))
                    ]
        
        train(
            [config.dataset],
            # For module
            config.embed_dim,
            config.numerical_encoder_hidden_sizes,
            config.numerical_encoder_activations,
            config.n_head,
            config.n_hid,
            config.dropout,
            config.n_layers,
            # Only in finetuning
            aggregator=config.aggregator,
            aggregator_params=aggregator_params,
            decoder_hidden_sizes=config.decoder_hidden_sizes,
            decoder_activations=config.decoder_activations,
            # Other ooptions
            categorical_encoder_variational=False,
            need_weights=False,
            need_embeddings=False,
            # For training mode
            self_supervised=False,
            mask_proba=config.mask_proba,
            # For model
            callbacks=callbacks,
            optimizer=FT_OPTIMIZER,
            device=DEVICE,
            max_epochs=FT_MAX_EPOCHS,
            batch_size=FT_BATCH_SIZE,
            # Other model creation params       
            optimizer__lr=config.optimizer__lr,
            optimizer__weight_decay=config.optimizer__weight_decay,

        )


def finetuning_prepare(dataset, preload_id, pre_config=None):

    excluded_keys = [
        "criterion__equi_factor",
        "criterion__bounded_vars_factor",
        "criterion__non_zero_means_factor"
    ]

    current_config = sweep_configuration.copy()
    sweep_name = current_config["name"] + f"-{dataset}"

    current_config["name"] = sweep_name
    current_config["parameters"]["dataset"] = {"value": dataset}
    current_config["parameters"]["preload_id"] = {"value": preload_id}

    for k, v in pre_config.items():
        if k not in excluded_keys and k not in list(current_config["parameters"].keys()):
            logging.info(f"Adding {k} to sweep configuration")
            current_config["parameters"][k] = {"value": v}

    api = wandb.Api()
    sweeps = api.project(FT_PROJECT).sweeps()

    found = False
    for sweep in sweeps:
        if sweep.name == sweep_name:
            found = True
            sweep_id = sweep.id
            sweep.load(force=True)
            n_samples = FT_HP_SAMPLES - len(sweep.runs)
            logging.info(f"The sweep ID {sweep_id} name match. {n_samples} samples missing.")
            break

    if not found:
        sweep_id = wandb.sweep(sweep=current_config)
        n_samples = FT_HP_SAMPLES
        logging.info(f"The sweep ID is {sweep_id} was created.")
    
    return sweep_id, n_samples    
    

if __name__ == "__main__":

    logging.info("Recovering configurations from {}. Order: {}{}.".format(
        SS_PROJECT,
        FT_PRELOADED_METRIC_PREFIX, 
        FT_PRELOADED_METRIC_SEARCH
    ))

    api = wandb.Api()
    runs = api.runs(
                SS_PROJECT, 
                order=f"{FT_PRELOADED_METRIC_PREFIX}summary_metrics.{FT_PRELOADED_METRIC_SEARCH}"
            )
    
    best_run = runs[0]

    logging.info("The best configuration is:")
    for k, v in best_run.config.items():
        logging.info(f"\t{k}: {v}")

    for dataset in FT_DATASETS:

        logging.info(f"Finetuning dataset {dataset}")

        sweep_id, n_samples = finetuning_prepare(
            # Set the project where this run will be logged
            dataset,
            preload_id=best_run.id,
            pre_config=best_run.config
        )

        if n_samples > 0:
            wandb.agent(sweep_id, 
                        function=finetuning_run,
                        count=n_samples,
                        project=FT_PROJECT
                        )
        
