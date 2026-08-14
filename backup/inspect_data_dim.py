#!/usr/bin/env python3

import sys
import os
import json
import numpy as np

ucn_data_path = "~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/output/inference_results/segmentation_000000"
cgn_data_path = "~/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/test_data"

# Inspect outputs of unseen object clustering of one scene

# 1. sample.npz
data = np.load(f"{ucn_data_path}/sample.npz")
print("Keys:", data.files)
for k in data.files:
    arr = data[k]
    print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")

'''
Keys: ['image_color', 'depth']
image_color: shape=(1, 3, 480, 640), dtype=float32
depth: shape=(1, 3, 480, 640), dtype=float32

'''

# 2. im_label.npy
arr = np.load(f"{ucn_data_path}/im_label.npy", allow_pickle=True)
print("im_label.npy → shape:", arr.shape, "dtype:", arr.dtype)
print("Sample values:", np.unique(arr)[:10])

'''
im_label.npy → shape: (480, 640, 3) dtype: uint8
Sample values: [0 1 2 3 4 5 6 7 8 9]

'''

# 3. segmentation.json
with open(f"{ucn_data_path}/segmentation.json") as f:
    data = json.load(f)
print("Top-level keys:", list(data.keys()))
if "label" in data:
    lbl = data["label"]
    if isinstance(lbl, list):
        import numpy as np
        lbl_arr = np.array(lbl)
        print("label shape:", lbl_arr.shape, "dtype:", lbl_arr.dtype)
        print("Unique values:", np.unique(lbl_arr)[:10])
    else:
        print("label type:", type(lbl))
else:
    print("'label' key not found.")

'''
Top-level keys: ['num_objects', 'label', 'label_refined', 'paths']
label shape: (480, 640) dtype: float64
Unique values: [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]

'''

# Inspect input to contact graspnet (i.e., <scene-id>.npy)
arr = np.load(f"{cgn_data_path}/0.npy", allow_pickle=True)
print("Loaded:", type(arr))
print("dtype:", arr.dtype, "shape:", arr.shape)

# If it's an object array (common for pickled point clouds)
if arr.dtype == object:
    content = arr.item()
    if isinstance(content, dict):
        print("Keys in 0.npy:", content.keys())
        for k, v in content.items():
            print(f"{k}: shape={np.shape(v)}, dtype={np.array(v).dtype}")
    elif isinstance(content, np.ndarray):
        print("Contained array shape:", content.shape, "dtype:", content.dtype)
    else:
        print("Contained object type:", type(content))
else:
    print("Array shape:", arr.shape, "dtype:", arr.dtype)
    print("First few rows:\n", arr[:5])

'''
Keys in 0.npy: dict_keys(['rgb', 'depth', 'K', 'seg'])
rgb: shape=(720, 1280, 3), dtype=uint8
depth: shape=(720, 1280), dtype=float32
K: shape=(3, 3), dtype=float64
seg: shape=(720, 1280), dtype=float32

'''