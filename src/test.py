#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import pandas as pd
import numpy as np
import pubchempy as pcp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_percentage_error, r2_score
)
from utils import get_featurizer, GraphModel


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--sheet-dir", help="sheet dir", default="../data/toxicity_clean.xlsx"
)
parser.add_argument("--task-name", help="task name", required=True)
parser.add_argument("--model-name", help="graph model name", required=True)
parser.add_argument(
    "--model-dir", help="dir to save model", default="../models"
)
parser.add_argument(
    "--param-dir", help="dir to .json file with best param sets",
    default="../data/best.json"
)
parser.add_argument(
    "--output-dir", help="dir to write results",
    default="../results"
)
args = parser.parse_args()

sheet_dir = args.sheet_dir
task_name = args.task_name
model_name = args.model_name
model_dir = args.model_dir.rstrip("/") + f"/{task_name}_{model_name}_test"
param_dir = args.param_dir
output_dir = args.output_dir.rstrip("/")

# read params
with open(param_dir) as param_file:
    params = json.load(param_file)[task_name][model_name]

# read data sheet
df = pd.read_excel(sheet_dir, task_name)

# extract feature
featurizer = get_featurizer(model_name)

feature = []
toxicity = []
for cas, smiles, tox in tqdm(df.values[:, [1, 2, -1]]):
    feat = featurizer.featurize(smiles)[0]
    feature.append(feat)
    toxicity.append(tox)

# split train test
X_train_total, X_test, y_train_total, y_test = train_test_split(
    feature, toxicity, random_state=42, test_size=0.2
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_total, y_train_total, random_state=42, test_size=0.2
)

performance = []
for i in range(5):
    # model training
    model = GraphModel(name=model_name, model_dir=model_dir, **params)
    model.fit(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        epoch=1000, patience=30, interval=1, validation=True,
        metric=mean_squared_error, store_best=True,
        greater_is_better=False, verbose=0
    )
    # prediction
    y_test_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    performance.append([mse, mape, r2])

    # delete model to free memory
    del model


performance = np.array(performance)
mean = performance.mean(axis=0)
std = performance.std(axis=0)

msg = (
    f"{task_name}\n"
    f"{model_name}\n"
    f"MSE: {mean[0]:.4f} +/- {std[0]:.4f}\n"
    f"MAPE: {mean[1]:.4f} +/- {std[1]:.4f}\n"
    f"R2: {mean[2]:.4f} +/- {std[2]:.4f}\n"
)
print(msg)

# write to file
with open(f"{output_dir}/test.txt", "a") as file:
    file.write(msg)
