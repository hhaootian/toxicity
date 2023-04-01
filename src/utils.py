#!/usr/bin/env python
# -*- coding: utf-8 -*-

import deepchem as dc
from deepchem.models import (
    AttentiveFPModel, GCNModel, GATModel, PagtnModel
)
from deepchem.models.torch_models import MPNNModel
from copy import deepcopy


def get_featurizer(model_name: str):
    """Featurizer

    Args:
        model_name (str): model name

    Raises:
        ValueError: model name not found

    Returns:
        featurizer: dc.feat class
    """
    if model_name == "PagtnModel":
        featurizer = dc.feat.PagtnMolGraphFeaturizer()
    elif model_name in ["GATModel", "GCNModel"]:
        featurizer = dc.feat.MolGraphConvFeaturizer()
    elif model_name in ["AttentiveFPModel", "MPNNModel"]:
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    else:
        raise ValueError(f"Model name ({model_name}) not found.")

    return featurizer


class GraphModel():
    """GraphModels class
    """
    def __init__(
        self, name, batch_size, learning_rate, model_dir, **kwargs
    ):
        """Initialization function

        Args:
            name (str): model name

        Raises:
            ValueError: Mode is neither 'regression' nor 'classification'
            ValueError: Model name is not supported.
        """
        self.model = None

        if name == "AttentiveFPModel":
            self.model = AttentiveFPModel(
                n_tasks=1, batch_size=batch_size,
                learning_rate=learning_rate, model_dir=model_dir,
                **kwargs
            )
        elif name == "GCNModel":
            self.model = GCNModel(
                n_tasks=1, batch_size=batch_size,
                learning_rate=learning_rate, model_dir=model_dir,
            )
        elif name == "GATModel":
            self.model = GATModel(
                n_tasks=1, batch_size=batch_size,
                learning_rate=learning_rate, model_dir=model_dir,
            )
        elif name == "PagtnModel":
            self.model = PagtnModel(
                n_tasks=1, batch_size=batch_size,
                learning_rate=learning_rate, model_dir=model_dir,
            )
        elif name == "MPNNModel":
            self.model = MPNNModel(
                n_tasks=1, batch_size=batch_size,
                learning_rate=learning_rate, model_dir=model_dir,
            )
        else:
            raise ValueError(f"Model {name} is not supported!")

    def fit(
        self, X_train, y_train, epoch, validation=False,
        patience=20, interval=10, X_val=None, y_val=None,
        greater_is_better=False, metric=None, store_best=False, verbose=0
    ):
        """Train a model.

        Args:
            X_train (np.array): Dataset used for training.
            y_train (np.array): Dataset used for training.
            epoch (int): Number of epochs.
            validation (bool, optional): Defaults to False.
            patience (int, optional): Early stopping. Defaults to 20.
            interval (int, optional): Number of epochs. Defaults to 10.
            X_val (np.array): Dataset used for validation.
            y_val (np.array): Dataset used for validation.
            greater_is_better (bool, optional): Training direction.
            metric (dc.metrics, optional): Metric function.
            store_best (bool, optional): Store model of best score.
            verbose (int, optional): Defaults to 0 (silent).
        """
        if validation:
            assert X_val is not None, "Please provide validation data X."
            assert y_val is not None, "Please provide validation data y."
            assert metric is not None, "Please provide a metric function."
            dataset_val = dc.data.NumpyDataset(X=X_val, y=y_val)

        dataset_train = dc.data.NumpyDataset(X=X_train, y=y_train)
        dataset_train_X_only = dc.data.NumpyDataset(X=X_train)
        cnt = 0
        best_score = -float('inf') if greater_is_better else float('inf')
        best_model = None

        for i in range(epoch // interval):
            loss = self.model.fit(dataset_train, nb_epoch=interval)
            y_pred = self.model.predict(dataset_train_X_only)

            metric_train = metric(y_train, y_pred)
            epoch_message = (
                f"Epoch: {(i + 1) * interval}"
                f"\t loss: {loss:.3f}"
                f"\t training: {metric_train:.3f}"
            )
            if not validation:
                if verbose:
                    print(epoch_message)
                continue

            y_pred = self.model.predict(dataset_val)
            metric_val = metric(y_val, y_pred)

            epoch_message += f"\t validation: {metric_val:.3f}"
            if verbose:
                print(epoch_message)

            if greater_is_better:
                if metric_val > best_score:
                    best_score = metric_val
                    best_model = deepcopy(self.model)
                    cnt = 0
                else:
                    cnt += 1
            else:
                if metric_val < best_score:
                    best_score = metric_val
                    best_model = deepcopy(self.model)
                    cnt = 0
                else:
                    cnt += 1

            # if need early stopping
            if cnt == patience:
                break

        if store_best and best_model:
            self.model = best_model

    def predict(self, X_test):
        """Prediction

        Args:
            X_test (np.array): Prediction dataset.

        Returns:
            np.array: A np.array of predictions.
        """
        dataset_test = dc.data.NumpyDataset(X=X_test)
        return self.model.predict(dataset_test)
