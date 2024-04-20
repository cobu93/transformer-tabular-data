import skorch
import torch
import torch.nn as nn
from ndsl.architecture.comspace import CommonSpaceTransformer
from ndsl.module.preprocessor import CLSPreprocessor
from ndsl.module.custom import TTransformerEncoderLayer, TTransformerEncoder
from ndsl.module.encoder import MLPEncoder, CategoricalEncoder
from ndsl.module.aggregator import (
    CLSAggregator, 
    MaxAggregator, 
    MeanAggregator, 
    SumAggregator, 
    RNNAggregator
)
from ndsl.utils.builder import (
    build_mlp
)

def trial_dirname_creator(trial):
    return "trial_{}".format(trial.trial_id)

def build_activations(activations_str):
    
    if activations_str is None:
        return None

    activations_list = activations_str.split("-")

    activations_map = {
        "r": nn.ReLU(),
        "i": nn.Identity(),
        "l": nn.Sigmoid(),
        "s": nn.Softmax()
    }

    activations = [ activations_map[a.strip().lower()] for a in activations_list]
    return activations

def build_module(
    embed_dim,
    numerical_encoder_hidden_sizes,
    numerical_encoder_activations,
    n_categories,
    # Will be the same for encoder and decoder
    n_head,
    n_hid,
    dropout,
    n_layers,
    # Only in finetuning
    aggregator=None,
    aggregator_params={},
    decoder_hidden_sizes=None,
    decoder_activations=None,
    n_outputs=None,
    # Other ooptions
    categorical_encoder_variational=True,
    need_weights=False,
    need_embeddings=True
    ):

    numerical_encoder = MLPEncoder(
                output_size=embed_dim,
                input_size=1,   
                hidden_sizes=numerical_encoder_hidden_sizes,
                activations=numerical_encoder_activations
                )
    
    categorical_encoder = CategoricalEncoder(
                output_size=embed_dim,
                n_categories=n_categories,
                variational=categorical_encoder_variational
                )
    
    embedding_processor = None
    if aggregator == "cls":
        embedding_processor=CLSPreprocessor(embed_dim)


    encoder_layers = TTransformerEncoderLayer(
                        embed_dim, 
                        n_head, 
                        n_hid, 
                        attn_dropout=dropout, 
                        ff_dropout=dropout
                    )
    
    encoder = TTransformerEncoder(
                    encoder_layers, 
                    n_layers, 
                    need_weights=need_weights, 
                    enable_nested_tensor=False
                )


    encoder_decoder_mid = None

    if aggregator == "cls":
        encoder_decoder_mid = CLSAggregator(embed_dim, **aggregator_params)
    elif aggregator == "max": 
        encoder_decoder_mid = MaxAggregator(embed_dim, **aggregator_params)
    elif aggregator == "mean": 
        encoder_decoder_mid = MeanAggregator(embed_dim, **aggregator_params)
    elif aggregator == "sum": 
        encoder_decoder_mid = SumAggregator(embed_dim, **aggregator_params)
    elif aggregator == "rnn": 
        encoder_decoder_mid = RNNAggregator(input_size=embed_dim, **aggregator_params)

    decoder = None

    if encoder_decoder_mid and decoder_hidden_sizes and decoder_activations and n_outputs:
        decoder = build_mlp(
                encoder_decoder_mid.output_size,
                n_outputs,
                decoder_hidden_sizes,
                decoder_activations
            ) 
    

    return CommonSpaceTransformer(
            numerical_embedding=numerical_encoder,
            categorical_embedding=categorical_encoder,
            embedding_processor=embedding_processor,
            encoder=encoder,
            encoder_decoder_mid=encoder_decoder_mid,
            decoder=decoder,
            need_embeddings=need_embeddings
        )