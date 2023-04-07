#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import itertools
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
    "--sheet-dir", help="sheet dir", default="../data/toxicity.xlsx"
)
parser.add_argument("--task-name", help="sheet name", required=True)
parser.add_argument("--model-name", help="graph model name", required=True)
parser.add_argument(
    "--model-dir", help="dir to save model", default="../models"
)
parser.add_argument(
    "--param-dir", help="dir to hyperparams.json file",
    default="../data/params.json"
)
parser.add_argument(
    "--output-dir", help="dir to write results",
    default="../results"
)
args = parser.parse_args()

sheet_dir = args.sheet_dir
task_name = args.task_name
model_name = args.model_name
model_dir = args.model_dir.rstrip("/") + f"/{task_name}_{model_name}"
param_dir = args.param_dir
output_dir = args.output_dir.rstrip("/")

# read params
with open(param_dir) as param_file:
    params = json.load(param_file)[model_name]

# read data sheet
df = pd.read_excel(sheet_dir, task_name)

# extract feature
featurizer = get_featurizer(model_name)

feature = []
toxicity = []

cnt = 0
for cas, smiles, tox in tqdm(df.values[:, [1, 2, -1]]):
    feat = featurizer.featurize(smiles)[0]
    # search for valid PubChem SMILES
    if type(feat) == np.ndarray:
        try:
            smiles = pcp.get_compounds(
                smiles.lstrip("0"), "name"
            )[0].canonical_smiles
            feat = featurizer.featurize(smiles)
        except:
            cnt += 1
    else:
        feature.append(feat)
        toxicity.append(tox)

# split 60/20/20
X_train, X_test, y_train, y_test = train_test_split(
    feature, toxicity, random_state=42, test_size=0.2
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, random_state=42, test_size=0.25
)

# full permutation of params
keys, values = zip(*params.items())
permutations = [
    dict(zip(keys, v)) for v in itertools.product(*values)
]

for perm in permutations:
    # model training
    model = GraphModel(name=model_name, model_dir=model_dir, **perm)

    model.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        epoch=500, patience=50, interval=1, validation=True,
        metric=mean_squared_error, store_best=True,
        greater_is_better=False, verbose=0
    )

    # prediction
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    msg = (
        f"model: {model_name}\n"
        f"setting: {perm}\n"
        "validation: "
        f"MSE: {mean_squared_error(y_val, y_val_pred):.4f} "
        f"MAPE: {mean_absolute_percentage_error(y_val, y_val_pred):.4f} "
        f"R2: {r2_score(y_val, y_val_pred):.4f}\n"
        "test:       "
        f"MSE: {mean_squared_error(y_test, y_test_pred):.4f} "
        f"MAPE: {mean_absolute_percentage_error(y_test, y_test_pred):.4f} "
        f"R2: {r2_score(y_test, y_test_pred):.4f}\n"
    )

    print(msg)

    # write to file
    with open(f"{output_dir}/{task_name}_{model_name}.txt", "a") as file:
        file.write(msg)
