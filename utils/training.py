import os
import pandas as pd
import logging

import skorch
import torch

from skorch.helper import predefined_split

from .processing import preprocess_data
from .data import JointlyTabularDataset, VariableLengthSampler
from .builder import build_module, build_activations
from .loss import CommonSpacesLoss

from config import (
    SS_DATASETS,
    FT_DATASETS,
    PREPROCESSING_FILE
)

def get_preprocessing_summary(
        datasets = list(set(SS_DATASETS + FT_DATASETS)),
        preprocessing_file=PREPROCESSING_FILE,
        
    ):

    if not os.path.exists(preprocessing_file):
        logging.info("Preprocessing file not found")
        logging.info(f"Starting preprocessing for {datasets}")
        preprocessing_summary = preprocess_data(datasets)
        preprocessing_summary.to_csv(preprocessing_file, index=False)
        logging.info(f"Preprocessing file saved at {preprocessing_file}")
    else:
        logging.info(f"Preprocessing file found at {preprocessing_file}")
        preprocessing_summary = pd.read_csv(preprocessing_file)

    return preprocessing_summary


def prepare_model(
        datasets,
        # For module
        embed_dim,
        numerical_encoder_hidden_sizes,
        numerical_encoder_activations,
        n_head,
        n_hid,
        dropout,
        n_layers,
        # Only in finetuning
        aggregator=None,
        aggregator_params={},
        decoder_hidden_sizes=None,
        decoder_activations=None,
        # Other ooptions
        categorical_encoder_variational=False,
        need_weights=False,
        need_embeddings=False,
        # For training mode
        self_supervised=True,
        mask_proba=None,
        # For model
        callbacks=None,
        optimizer=torch.optim.AdamW,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=100,
        batch_size=32,
        shuffle_data=True,
        # Other model creation params
        **kwargs
    ):

    other_nn_params = kwargs
    fine_tuning = not self_supervised

    preprocessing_summary = get_preprocessing_summary(preprocessing_file=PREPROCESSING_FILE)
    n_classes = preprocessing_summary.query("dataset in @datasets")["n_classes"].sum()
    datasets_files = preprocessing_summary.query("dataset in @datasets")["file"].to_list()

    if len(datasets) != len(datasets_files):
        msg = f"Not all required datasets are preprocessed. Delete {PREPROCESSING_FILE} file."
        logging.error(msg)
        raise ValueError(msg)

    logging.info(f"Training for {datasets}")
    

    if fine_tuning:
        logging.info(f"Finetuning mode. Classifing {n_classes} classes")

        if n_classes <= 1:
            msg = f"Cannot perform classification task with {n_classes} class."
            logging.error(msg)
            raise ValueError(msg)
        
        if False:
            logging.info("Binary classification")
            n_outputs = 1
            multiclass = False
            criterion = torch.nn.BCEWithLogitsLoss
            clazz = skorch.NeuralNetBinaryClassifier
            predict_nonlinearity = torch.nn.Sigmoid()
        else:
            logging.info("Multiclass classification")
            n_outputs = n_classes
            multiclass = True
            criterion = torch.nn.CrossEntropyLoss
            clazz = skorch.NeuralNetClassifier
            predict_nonlinearity = torch.nn.Softmax(dim=-1)

        
        return_mask=False

    else:
        logging.info("Self supervised mode")
        n_outputs = None
        multiclass = True
        criterion = CommonSpacesLoss
        clazz = skorch.NeuralNet
        predict_nonlinearity = None
        return_mask=True
            
    
    logging.info("Loading datasets. Using {:2.4f} probability for masking".format(mask_proba if mask_proba else 0))
    train_joint_dataset = JointlyTabularDataset(datasets_files, key="train", bernoulli_p=mask_proba, multiclass=multiclass, return_mask=return_mask)
    val_joint_dataset = JointlyTabularDataset(datasets_files, key="val", bernoulli_p=mask_proba, multiclass=multiclass, return_mask=return_mask)
    test_joint_dataset = JointlyTabularDataset(datasets_files, key="test", bernoulli_p=mask_proba, multiclass=multiclass, return_mask=return_mask)

    n_categories = max(train_joint_dataset.max_category, val_joint_dataset.max_category)

    logging.info("Building module. The parameters are:")
    logging.info(f"\t embed_dim: {embed_dim}")
    logging.info(f"\t numerical_encoder_hidden_sizes: {numerical_encoder_hidden_sizes}")
    logging.info(f"\t numerical_encoder_activations: {numerical_encoder_activations}")
    logging.info(f"\t n_categories: {n_categories}")
    logging.info(f"\t n_head: {n_head}")
    logging.info(f"\t n_hid: {n_hid}")
    logging.info(f"\t dropout: {dropout}")
    logging.info(f"\t n_layers: {n_layers}")
    logging.info(f"\t aggregator: {aggregator}")
    logging.info(f"\t aggregator_params: {aggregator_params}")
    logging.info(f"\t decoder_hidden_sizes: {decoder_hidden_sizes}")
    logging.info(f"\t decoder_activations: {decoder_activations}")
    logging.info(f"\t n_outputs: {n_outputs}")
    logging.info(f"\t categorical_encoder_variational: {categorical_encoder_variational}")
    logging.info(f"\t need_weights: {need_weights}")
    logging.info(f"\t need_embeddings: {need_embeddings}")

    module = build_module(
                    embed_dim,
                    numerical_encoder_hidden_sizes,
                    build_activations(numerical_encoder_activations),
                    n_categories,
                    # Will be the same for encoder and decoder
                    n_head,
                    n_hid,
                    dropout,
                    n_layers,
                    # Only in finetuning
                    aggregator=aggregator,
                    aggregator_params=aggregator_params,
                    decoder_hidden_sizes=decoder_hidden_sizes,
                    decoder_activations=build_activations(decoder_activations),
                    n_outputs=n_outputs,
                    # Other ooptions
                    categorical_encoder_variational=categorical_encoder_variational,
                    need_weights=need_weights,
                    need_embeddings=need_embeddings
                    )


    if not fine_tuning:
        other_nn_params["criterion__cs_module"] = module

    logging.info("Building model. The parameters are:")
    logging.info(f"\t class: {clazz}")
    logging.info(f"\t criterion: {criterion}")
    logging.info(f"\t optimizer: {optimizer}")
    logging.info(f"\t device: {device}")
    logging.info(f"\t max_epochs: {max_epochs}")
    logging.info(f"\t batch_size: {batch_size}")
    logging.info(f"\t callbacks: {callbacks}")
    logging.info(f"\t predict_nonlinearity: {predict_nonlinearity}")

    for k, v in other_nn_params.items():
        logging.info(f"\t {k}: {v}")


    model = clazz(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_epochs=max_epochs,
            callbacks=callbacks,
            train_split=predefined_split(val_joint_dataset),
            iterator_train__batch_sampler=VariableLengthSampler(train_joint_dataset, batch_size, shuffle=shuffle_data),
            iterator_valid__batch_sampler=VariableLengthSampler(val_joint_dataset, batch_size, shuffle=shuffle_data),
            iterator_train__batch_size=1,
            iterator_valid__batch_size=1,
            predict_nonlinearity=predict_nonlinearity,
            **other_nn_params
        )

    if fine_tuning:
        logging.info("Setting expected classes")
        #model.classes = [i for i in range(n_outputs)]

    return model, train_joint_dataset, val_joint_dataset, test_joint_dataset


def train(
        datasets,
        # For module
        embed_dim,
        numerical_encoder_hidden_sizes,
        numerical_encoder_activations,
        n_head,
        n_hid,
        dropout,
        n_layers,
        # Only in finetuning
        aggregator=None,
        aggregator_params={},
        decoder_hidden_sizes=None,
        decoder_activations=None,
        # Other ooptions
        categorical_encoder_variational=False,
        need_weights=False,
        need_embeddings=False,
        # For training mode
        self_supervised=True,
        mask_proba=None,
        # For model
        callbacks=None,
        optimizer=torch.optim.AdamW,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=100,
        batch_size=32,
        # Other model creation params
        **kwargs
    ):

    model, train_joint_dataset, _, _ = prepare_model(
        datasets,
        # For module
        embed_dim,
        numerical_encoder_hidden_sizes,
        numerical_encoder_activations,
        n_head,
        n_hid,
        dropout,
        n_layers,
        # Only in finetuning
        aggregator=aggregator,
        aggregator_params=aggregator_params,
        decoder_hidden_sizes=decoder_hidden_sizes,
        decoder_activations=decoder_activations,
        # Other ooptions
        categorical_encoder_variational=categorical_encoder_variational,
        need_weights=need_weights,
        need_embeddings=need_embeddings,
        # For training mode
        self_supervised=self_supervised,
        mask_proba=mask_proba,
        # For model
        callbacks=callbacks,
        optimizer=optimizer,
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        # Other model creation params
        **kwargs
    )   

    logging.info("Starting train...")
    model.fit(train_joint_dataset, None)

