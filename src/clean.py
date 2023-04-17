#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import numpy as np
from utils import featurizer_map


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--sheet-dir", help="sheet dir", default="../data/toxicity.xlsx"
)
args = parser.parse_args()

sheet_dir = args.sheet_dir
task_names = ["TP", "DM", "VF", "Fish"]

# define clean sheet dir
clean_sheet_dir = sheet_dir.rsplit(".", 1)[0] + "_clean.xlsx"
writer = pd.ExcelWriter(clean_sheet_dir)

for task_name in task_names:
    df = pd.read_excel(sheet_dir, task_name)
    drop_idx = []
    
    for idx, smiles in enumerate(df.values[:, [2]]):
        # if any of the featurizer cannot do featurize, drop this entry
        for featurizer in featurizer_map.values():
            feat = featurizer.featurize(smiles)[0]
            if type(feat) == np.ndarray:
                drop_idx.append(idx)
                break
    
    print(
        f"sheet name: {task_name}\n"
        f"total entry: {len(df)}\n"
        f"drop entry: {len(drop_idx)}"
    )
    
    # save clean df
    df.drop(drop_idx).to_excel(writer, sheet_name=task_name, index=False)

writer.close()
