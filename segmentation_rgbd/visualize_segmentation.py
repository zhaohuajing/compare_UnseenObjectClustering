import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# paths
color_path = "./input/from_rgbd-color.png"
labels_path = "./output/segmentation_from_rgbd/im_label.npy"

# load data
im_color = np.array(Image.open(color_path))
labels = np.load(labels_path)

print("im_color shape:", im_color.shape)
print("labels shape:", labels.shape)
uniq = np.unique(labels)
print("unique labels:", uniq)
print("pixel counts per label:")
for u in uniq:
    print(f"  label {u}: {(labels == u).sum()} pixels")

# build overlay
overlay = im_color.copy()
alpha = 0.5

# simple color palette
colors = plt.cm.get_cmap("tab20", len(uniq))

for i, lbl in enumerate(uniq):
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

plt.tight_layout()
plt.savefig('segmentation_result.png')
# plt.show()
