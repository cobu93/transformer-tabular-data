import torch
import numpy as np
from config import DATA_BASE_DIR, OPTIMIZER, DEVICE, BATCH_SIZE
import os
import joblib
from . import data, training, log
import skorch

logger = log.get_logger()

def compute_std_attentions(attn, aggregator):
    batch_size = attn.shape[1]
    n_layers = attn.shape[0]
    n_features = attn.shape[-1]

    # Sum heads
    # layers, batch, heads, o_features, i_features
    heads_attn = attn.mean(axis=2)

    # Initial: layers, batch, o_features, i_features
    # Final: batch, layers, i_features, o_features
    heads_attn = heads_attn.permute((1, 0, 3, 2))
    general_attn = None

    # For each layer
    single_attns = torch.zeros((batch_size, n_layers, n_features))
    cum_attns = torch.zeros((batch_size, n_layers, n_features))

    for layer_idx in range(n_layers):
        if layer_idx == n_layers - 1 and aggregator == "cls":
            single_attns[:, layer_idx] = heads_attn[:, layer_idx, :, 0]
        else:
            single_attns[:, layer_idx] = heads_attn[:, layer_idx].mean(axis=-1)
        
        if general_attn is None:
            general_attn = heads_attn[:, layer_idx]
        else:
            general_attn = torch.matmul(general_attn, heads_attn[:, layer_idx])

        if layer_idx == n_layers - 1 and aggregator == "cls":
            cum_attns[:, layer_idx] = general_attn[:, :, 0]
        else:
            cum_attns[:, layer_idx] = general_attn.mean(axis=-1)

    # assert np.allclose(single_attns.sum(axis=-1), 1), "There is a logistic problem: " + str(single_attns.sum(axis=-1))
    # assert np.allclose(cum_attns.sum(axis=-1), 1), "There is a logistic problem: " + str(cum_attns.sum(axis=-1))

    # Before: batch_size, n_layers, n_features
    # After: n_layers, batch_size, n_features
    return single_attns.permute((1, 0, 2)), cum_attns.permute((1, 0, 2))

def extract_attention(
        dataset,
        checkpoint_dir,
        aggregator,
        selection_metric,
        config, 
        only_last=True,
        return_cubes=False,
        return_labels=False
    ):
    
    run_name = f"{dataset}-{selection_metric}"
    logger.info("+" * 40 + f" Extracting {run_name}")

    logger.info(f"Loading preprocessor")
    preprocessor = joblib.load(os.path.join(checkpoint_dir, "preprocessor.jl"))

    logger.info("Reading data")

    dataset_data, dataset_meta = data.read_dataset(dataset)
    target_column = dataset_meta["target"]
    n_numerical = dataset_meta["n_numerical"]
    
    features = dataset_data.drop(target_column, axis=1)
    original_order = features.columns.values.tolist()
    labels = dataset_data[target_column]
    
    logger.info("Preprocessing data")
    X = preprocessor.transform(features)
    y = labels.values

    multiclass = len(dataset_meta["labels"]) > 2
    
    if multiclass:
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float32)

    logger.info("Building model")
    model = training.build_default_model_from_configs(
        config, 
        dataset_meta,
        None,
        optimizer=OPTIMIZER,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        monitor_metric="valid_loss", 
        max_epochs=1,
        checkpoint_dir=os.path.join(checkpoint_dir, "ignore"),
    )
    
    logger.info("Loading checkpoint")
    checkpoint = skorch.callbacks.TrainEndCheckpoint(
        dirname=os.path.join(checkpoint_dir, "model")
    ).initialize().checkpoint_

    model.callbacks = None
    model.initialize()
    model.load_params(checkpoint=checkpoint)
    model.module_.need_weights = True

    preds_iter = model.forward_iter({
            "x_numerical": X[:, :n_numerical].astype(np.float32),
            "x_categorical": X[:, n_numerical:].astype(np.int32)
        })
    

    n_instances, n_features = X.shape
    n_features -= n_numerical if config["numerical_passthrough"] else 0
    n_heads = config["n_head"]

    if only_last:
        n_layers = 1
    else:
        n_layers = config["n_layers"]
    
    if aggregator == "cls":
        n_features += 1
    
    cum_attns = np.zeros((n_layers, n_instances, n_features))

    if return_cubes:
        cubes_attns = np.zeros((n_layers, n_instances, n_heads, n_features, n_features))

    for i, preds in enumerate(preds_iter):
        output, layer_outs, attn = preds

        if return_cubes:
            cubes_attns[:, i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = attn[-n_layers:]

        _, batch_cum_attn = compute_std_attentions(attn, aggregator)
        cum_attns[:, i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = batch_cum_attn[-n_layers:]
        
    assert np.allclose(cum_attns.sum(axis=-1), 1), "Something went wrong with attentions"

    if aggregator == "cls":
        logger.info("The aggregator is CLS")
        cum_attns = cum_attns[:, :, 1:]
        # Re-normalizing
        cum_attns = cum_attns / cum_attns.sum(axis=-1, keepdims=True)

        if return_cubes:
            raise NotImplementedError("CLS removal in attention cubes is not implemented")

    numerical_passthrough = config["numerical_passthrough"]
    if numerical_passthrough:
        logger.info("Numerical passthrough is True")
        numerical_attention = np.ones((n_layers, n_instances, n_numerical)) * np.nan
        cum_attns = np.concatenate([numerical_attention, cum_attns], axis=-1)

        if return_cubes:
            raise NotImplementedError("Numerical passthrough in attention cubes is not implemented")

    # At this point cum_attns has the numerical values first
    # For correctly masking we need to re sort features as originally
    
    current_order = dataset_meta["numerical"] + dataset_meta["categorical"]
    indices_sort = np.argsort(list(map(lambda x: original_order.index(x), current_order)))
    cum_attns = cum_attns[:, :, indices_sort].squeeze()

    ret_vals = {"cumulated_attention": cum_attns}

    if return_labels:
        ret_vals["labels"] = y

    if return_cubes:
        cubes_attns = cubes_attns[:, :, :, :, indices_sort]
        cubes_attns = cubes_attns[:, :, :, indices_sort]
        ret_vals["attention_cubes"] = cubes_attns
        
    
    return ret_vals

