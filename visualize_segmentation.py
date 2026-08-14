import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

cmap = plt.cm.get_cmap("tab20")

# paths
# color_path = "/home/csrobot/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/input/from_rgbd-color.png"
# labels_path = "/home/csrobot/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/output/segmentation_from_rgbd/im_label.npy"

color_path = "/home/csrobot/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/results/segmentation_rgbd/input/from_rgbd-color.png"
labels_path = "/home/csrobot/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/results/segmentation_rgbd/output/im_label.npy"

# load data
im_color = np.array(Image.open(color_path))
labels = np.load(labels_path)

min_obj_point_number = 500 

print("im_color shape:", im_color.shape)
print("labels shape:", labels.shape)
uniq_labels = np.unique(labels)
print("unique labels:", uniq_labels)
print("pixel counts per label:")
for u in uniq_labels[uniq_labels != 0]:
    if (labels == u).sum() > min_obj_point_number:
        print(f"  label {u}: {(labels == u).sum()} pixels")




# build overlay
overlay = im_color.copy()
alpha = 0.8

# simple color palette
colors = plt.cm.get_cmap("tab20", len(uniq_labels))

for i, lbl in enumerate(uniq_labels):
    if lbl == 0:
        continue  # skip background
    mask = labels == lbl
    if not np.any(mask):
        continue
    c = (np.array(colors(i)[:3]) * 255).astype(np.uint8)
    overlay[mask] = (
        alpha * c + (1.0 - alpha) * overlay[mask]
    ).astype(np.uint8)

# show side-by-side
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("RGB")
plt.imshow(im_color)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("RGB + instances")
plt.imshow(overlay)
plt.axis("off")


plt_labels = uniq_labels[uniq_labels != 0]
legend_handles = []

colors = plt.cm.get_cmap("tab20", len(uniq_labels))

for i, label in enumerate(uniq_labels):
    if label == 0:
        continue

    if (labels == label).sum() > min_obj_point_number:
        color = colors(i)
        legend_handles.append(
            mpatches.Patch(color=color, label=f"L{label}")
        )

if legend_handles:
    plt.legend(handles=legend_handles, loc="upper right")

plt.tight_layout()
plt.savefig('/home/csrobot/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/results/segmentation_rgbd/output/segmentation_result.png')
plt.show()
