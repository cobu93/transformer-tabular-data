import torch
import numpy as np

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


def get_attention_mask(attn, selector, as_binary=True):

    sorted_args = np.argsort(attn, axis=-1)[:, ::-1]
    inv_sorted_args = np.argsort(sorted_args, axis=-1)
    sorted_attn = np.take_along_axis(attn, sorted_args, axis=1)

    # Percent selection
    if selector <= 1:
        attn_cum_sum = sorted_attn.cumsum(axis=-1)
        sorted_attn[attn_cum_sum > selector] = 0
        sorted_attn = np.take_along_axis(sorted_attn, inv_sorted_args, axis=1)
    # Top k selection selection
    else:
        sorted_attn[int(selector):] = 0
        sorted_attn = np.take_along_axis(sorted_attn, inv_sorted_args, axis=1)
    
    if as_binary:
        sorted_attn[sorted_attn > 0] = 1
        sorted_attn = sorted_attn.astype(bool)

    return sorted_attn
