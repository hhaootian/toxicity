#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_percentage_error, r2_score
)
import dgllife
import dgl
import matplotlib
import matplotlib.cm as cm
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
from utils import get_featurizer, GraphModel


sheet_dir = "../data/toxicity_clean.xlsx"
task_name = "DM"
model_name = "AttentiveFPModel"
model_dir = "../models".rstrip("/") + f"/{task_name}_{model_name}_test"

# read params
param = {
    'learning_rate': 0.001, 'batch_size': 32,
    'num_layers': 1, 'dropout': 0.2
}

# read data sheet
df = pd.read_excel(sheet_dir, task_name)

# extract feature
featurizer = get_featurizer(model_name)

feature = []
toxicity = []
cnt = 0
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

# model training
model = GraphModel(name=model_name, model_dir=model_dir, **param)
model.fit(
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    epoch=1000, patience=30, interval=1, validation=True,
    metric=mean_squared_error, store_best=True,
    greater_is_better=False, verbose=1
)
# prediction
y_test_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
mape = mean_absolute_percentage_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

msg = (
    f"{model_name}\n"
    f"MSE: {mse:.4f}\n"
    f"MAPE: {mape:.4f}\n"
    f"R2: {r2:.4f}\n"
)
print(msg)

# transfer DeepChem to DGL
device = 'cuda:0'

# extract DeepChem model weight
trained_model = model.model.model
trained_model_state_dict = trained_model.state_dict()
rename_key = {}
for key in trained_model_state_dict:
    # DeepChem key has additional "model:"
    rename_key[key[6:]] = trained_model_state_dict[key]

# assign weight to DGL model
dgl_model = dgllife.model.AttentiveFPPredictor(
    node_feat_size=30, edge_feat_size=11,
    num_layers=1, dropout=0.2
)
dgl_model.load_state_dict(rename_key)
dgl_model.eval()
dgl_model.double()
dgl_model.to(torch.device(device))


# define molecule
idx = 0
smiles = "C1(C(CC(O)=O)C(O)=O)CCCCC1"

g = X_train[idx].to_dgl_graph()
g = dgl.batch([g]).to(device)
node_features = torch.from_numpy(X_train[idx].node_features).to(device)
edge_features = torch.from_numpy(X_train[idx].edge_features).to(device)

_, atom_weights = dgl_model(
    g, node_features, edge_features,
    get_node_weight=True
)

timestep = 0
atom_weights = atom_weights[timestep]
min_value = torch.min(atom_weights)
max_value = torch.max(atom_weights)
atom_weights = (atom_weights - min_value) / (max_value - min_value)

# Conver the weights to atom colors
norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
cmap = cm.get_cmap('Oranges')
plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
atom_colors = {
    i: plt_colors.to_rgba(atom_weights[i].data.item())
    for i in range(g.number_of_nodes())
}

mol = Chem.MolFromSmiles(smiles)
rdDepictor.Compute2DCoords(mol)
drawer = rdMolDraw2D.MolDraw2DSVG(280,280)
drawer.SetFontSize(1)
op = drawer.drawOptions()

mol = rdMolDraw2D.PrepareMolForDrawing(mol)
drawer.DrawMolecule(
    mol,highlightAtoms=range(g.number_of_nodes()),
    highlightBonds=[],
    highlightAtomColors=atom_colors
)
drawer.FinishDrawing()
svg = drawer.GetDrawingText()
with open('output.svg', 'w') as f:
    f.write(svg)
