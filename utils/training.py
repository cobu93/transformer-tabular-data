import torch
import torch.nn as nn
import torch.optim as optim
import skorch
from ndsl.architecture.attention import TabularTransformer
import os
from utils import callback

def build_module(
    n_categories, # List of number of categories
    n_numerical, # Number of numerical features
    n_head, # Number of heads per layer
    n_hid, # Size of the MLP inside each transformer encoder layer
    n_layers, # Number of transformer encoder layers    
    n_output, # The number of output neurons
    embed_dim,
    attn_dropout, 
    ff_dropout, 
    aggregator, # The aggregator for output vectors before decoder
    rnn_aggregator_parameters=None,
    decoder_hidden_units=[128, 64],
    decoder_activation_fn=nn.ReLU(),
    need_weights=False,
    numerical_passthrough=False
    ):

    module = TabularTransformer(
        n_categories=n_categories, # List of number of categories
        n_numerical=n_numerical,
        n_head=n_head,
        n_hid=n_hid,
        n_layers=n_layers,
        n_output=n_output,
        embed_dim=embed_dim,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        aggregator=aggregator,
        rnn_aggregator_parameters=rnn_aggregator_parameters,
        decoder_hidden_units=decoder_hidden_units,
        decoder_activation_fn=decoder_activation_fn,
        need_weights=need_weights,
        numerical_passthrough=numerical_passthrough
    )

    return module

def build_model(
        n_categories, # List of number of categories
        n_numerical, # Number of numerical features
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_layers, # Number of transformer encoder layers    
        n_output, # The number of output neurons
        embed_dim,
        attn_dropout, 
        ff_dropout, 
        aggregator, # The aggregator for output vectors before decoder
        rnn_aggregator_parameters=None,
        decoder_hidden_units=[128, 64],
        decoder_activation_fn=nn.ReLU(),
        need_weights=False,
        numerical_passthrough=False,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        max_epochs=100,
        train_split=None,
        callbacks=None,
        learning_rate=1e-4,
        weight_decay=0.
    ):

    module = build_module(
        n_categories, # List of number of categories
        n_numerical, # Number of numerical features
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_layers, # Number of transformer encoder layers    
        n_output, # The number of output neurons
        embed_dim,
        attn_dropout, 
        ff_dropout, 
        aggregator, # The aggregator for output vectors before decoder
        rnn_aggregator_parameters,
        decoder_hidden_units,
        decoder_activation_fn,
        need_weights,
        numerical_passthrough
    )

    model = skorch.NeuralNetClassifier(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            max_epochs=max_epochs,
            train_split=train_split,
            callbacks=callbacks,
            optimizer__lr=learning_rate,
            optimizer__weight_decay=weight_decay
        )
    
    return model
    
def build_default_model_from_configs(
        config,          
        dataset_meta,
        fold_name,
        need_weights=False,
        optimizer=torch.optim.AdamW,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        max_epochs=100,
        monitor_metric="valid_loss",
        early_stopping_fraction=0.2,
        checkpoint_dir="."
    ):

    rnn_aggregator_parameters = None

    if config["aggregator"] == "rnn":
        rnn_aggregator_parameters = {
            "cell": config["aggregator__cell"],
            "output_size": config["aggregator__hidden_size"], 
            "num_layers": config["aggregator__num_layers"], 
            "dropout": config["aggregator__dropout"]
        }
         
    multiclass = len(dataset_meta["labels"]) > 2

    if not multiclass:
        n_outputs = 1
        criterion = torch.nn.BCEWithLogitsLoss
    else:
        n_outputs = len(dataset_meta["labels"])
        criterion = torch.nn.CrossEntropyLoss
    
    
    checkpoint = [(f"checkpoint_{monitor_metric}", skorch.callbacks.Checkpoint(
                    monitor=f"{monitor_metric}_best",
                    dirname=os.path.join(checkpoint_dir, monitor_metric)
                    ))]
    
    early_stopping = [
        skorch.callbacks.EarlyStopping(
            monitor=monitor_metric, 
            patience=int(early_stopping_fraction * max_epochs)
        )
    ]

    if fold_name:
        train_split = skorch.dataset.ValidSplit(((
            dataset_meta["splits"][fold_name]["train"], 
            dataset_meta["splits"][fold_name]["val"]
        ),))
    else:
        train_split = None


    model = build_model(
        dataset_meta["n_categorical"], # List of number of categories
        dataset_meta["n_numerical"], # Number of numerical features
        config["n_head"], # Number of heads per layer
        config["embed_dim"], # Size of the MLP inside each transformer encoder layer
        config["n_layers"], # Number of transformer encoder layers    
        n_outputs, # The number of output neurons
        config["embed_dim"],
        config["attn_dropout"], 
        config["ff_dropout"], 
        config["aggregator"], # The aggregator for output vectors before decoder
        rnn_aggregator_parameters=rnn_aggregator_parameters,
        decoder_hidden_units=[128, 64],
        decoder_activation_fn=nn.ReLU(),
        need_weights=need_weights,
        numerical_passthrough=config["numerical_passthrough"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        batch_size=batch_size,
        max_epochs=max_epochs,
        train_split=train_split,
        callbacks=callback.get_default_callbacks(multiclass=multiclass) + checkpoint + early_stopping,
        learning_rate=config["optimizer__lr"],
        weight_decay=config["optimizer__weight_decay"]
        )

    return model