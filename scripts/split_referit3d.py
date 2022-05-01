import os
import sys
import json

import pandas as pd

from tqdm import tqdm
from ast import literal_eval

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF

RAW_REFERIT3D = os.path.join(CONF.PATH.DATA, "nr3d.csv")
PARSED_REFERIT3D_train = os.path.join(CONF.PATH.DATA, "nr3d_train.json") # split
PARSED_REFERIT3D_val = os.path.join(CONF.PATH.DATA, "nr3d_val.json") # split

val_scene_list = os.path.join(CONF.PATH.SCANNET_META, 'scannetv2_val.txt')
train_scene_list = os.path.join(CONF.PATH.SCANNET_META, 'scannetv2_train.txt')

with open(val_scene_list, 'r') as f:
    val_scene_list = f.readlines()
val_scene_list = [x.strip() for x in val_scene_list]

with open(train_scene_list, 'r') as f:
    train_scene_list = f.readlines()
train_scene_list = [x.strip() for x in train_scene_list]

print('val list', len(val_scene_list), 'train list', len(train_scene_list))
# assert len(val_scene_list) ==  141 and len(train_scene_list) == 562

print("parsing...")
organized = {}
df = pd.read_csv(RAW_REFERIT3D)
df.tokens = df["tokens"].apply(literal_eval)
nr3d_train = []
nr3d_val = []

for _, row in tqdm(df.iterrows()):
    entry = {}
    entry["scene_id"] = row["scan_id"]
    entry["object_id"] = str(row["target_id"])
    entry["object_name"] = row["instance_type"]
    entry["ann_id"] = str(row["assignmentid"])
    entry["description"] = row["utterance"].lower()
    entry["token"] = row["tokens"]

    if entry["scene_id"] in val_scene_list:
        nr3d_val.append(entry)
    elif entry["scene_id"] in train_scene_list:
        nr3d_train.append(entry)
    else:
        print('scene', entry["scene_id"], 'not in train/val split')

with open(PARSED_REFERIT3D_train, "w") as f:
    json.dump(nr3d_train, f, indent=4)
with open(PARSED_REFERIT3D_val, "w") as f:
    json.dump(nr3d_val, f, indent=4)
print("done!")
print('Saving', PARSED_REFERIT3D_train)
print('Saving', PARSED_REFERIT3D_val)

train = json.load(open(PARSED_REFERIT3D_train))
val = json.load(open(PARSED_REFERIT3D_val))
print('ReferIt3D-Nr3D: train val split length:', len(train), len(val))