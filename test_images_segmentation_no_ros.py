#!/usr/bin/env python3
"""
Test UnseenObjectClustering (UCN) segmentation without ROS dependencies.
Adapted from the original test_images_segmentation.py (ROS version).
"""
# Add project 'lib' folder to Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import torch
import torch.backends.cudnn as cudnn
import cv2
import argparse
import pprint
import os, sys
import numpy as np
import scipy.io
import networks
from utils.blob import pad_im
from fcn.config import cfg, cfg_from_file
from fcn.test_dataset import test_sample
from utils.mask import visualize_segmentation
import json


def make_sample_from_rgbd(im_color_bgr, depth_m, fx, fy, px, py):
    H, W = im_color_bgr.shape[:2]
    xyz_img = compute_xyz(depth_m, fx, fy, px, py, H, W)            # ← builds dense XYZ from depth
    im = im_color_bgr.astype(np.float32)
    im_tensor = torch.from_numpy(im) / 255.0
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
    im_tensor -= pixel_mean
    image_blob = im_tensor.permute(2, 0, 1)

    depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
    return {'image_color': image_blob.unsqueeze(0),
            'depth':       depth_blob.unsqueeze(0)}

# def make_sample_from_cloud(xyzrgb, H, W, K, have_rgb=True):
#     """
#     xyzrgb: (N, 6) or (N, 3) numpy array in meters; if unorganized, it will be rasterized.
#     H, W: output image size
#     K: 3x3 intrinsics (fx, 0, px; 0, fy, py; 0, 0, 1)
#     """
#     import numpy as np
#     # initialize empty buffers
#     depth_m = np.zeros((H, W), dtype=np.float32)
#     color   = np.zeros((H, W, 3), dtype=np.uint8)

#     # Project points to the image plane
#     X, Y, Z = xyzrgb[:, 0], xyzrgb[:, 1], xyzrgb[:, 2]
#     u = (K[0, 0] * (X / Z) + K[0, 2]).round().astype(int)
#     v = (K[1, 1] * (Y / Z) + K[1, 2]).round().astype(int)
#     valid = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

#     # z-buffer update (nearest wins)
#     # start with +inf so any valid depth will win
#     depth_m[:, :] = np.inf
#     for i in np.where(valid)[0]:
#         if Z[i] < depth_m[v[i], u[i]]:
#             depth_m[v[i], u[i]] = Z[i]
#             if have_rgb:
#                 color[v[i], u[i], :] = xyzrgb[i, 3:6].astype(np.uint8)

#     depth_m[np.isinf(depth_m)] = 0.0  # holes → 0
#     return make_sample_from_rgbd(color, depth_m, K[0,0], K[1,1], K[0,2], K[1,2])


def make_sample_from_cloud(xyzrgb, H, W, K, have_rgb=True):
    """
    xyzrgb: (N, 6) or (N, 3) numpy array in meters; if unorganized, it will be rasterized.
    H, W: output image size
    K: 3x3 intrinsics (fx, 0, px; 0, fy, py; 0, 0, 1)

    This mirrors the rasterization used in the ROS2 server:
    - z-buffer projection to depth
    - if RGB is missing or effectively empty, synthesize grayscale from depth
      and use that as pseudo-RGB.
    """
    import numpy as np

    depth_m = np.zeros((H, W), dtype=np.float32)
    color   = np.zeros((H, W, 3), dtype=np.uint8)

    # Light-weight debug
    print(f"[DEBUG cloud] xyzrgb shape: {xyzrgb.shape}")
    print(f"[DEBUG cloud] H,W: {H},{W}")

    # Project points
    X, Y, Z = xyzrgb[:, 0], xyzrgb[:, 1], xyzrgb[:, 2]
    u = (K[0, 0] * (X / Z) + K[0, 2]).round().astype(int)
    v = (K[1, 1] * (Y / Z) + K[1, 2]).round().astype(int)
    valid = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    print("[DEBUG cloud] valid count:", int(np.count_nonzero(valid)))

    # z-buffer in meters
    depth_m[:, :] = np.inf
    for i in np.where(valid)[0]:
        if Z[i] < depth_m[v[i], u[i]]:
            depth_m[v[i], u[i]] = Z[i]
            if have_rgb and xyzrgb.shape[1] >= 6:
                color[v[i], u[i], :] = xyzrgb[i, 3:6].astype(np.uint8)

    # fill holes with 0 depth
    depth_m[np.isinf(depth_m)] = 0.0

    # ---- synthesize grayscale RGB if no real color ----
    valid_depth_mask = depth_m > 0
    num_valid_depth = int(np.count_nonzero(valid_depth_mask))
    num_colored_pixels = int(np.count_nonzero(color)) // 3

    print("[DEBUG cloud] depth>0 count:", num_valid_depth)
    print("[DEBUG cloud] colored pixels:", num_colored_pixels)

    if num_valid_depth > 0 and (not have_rgb or num_colored_pixels < 100):
        d = depth_m.copy()
        nz = d[valid_depth_mask]

        lo = np.percentile(nz, 2.0)
        hi = np.percentile(nz, 98.0)
        if hi <= lo:
            hi = lo + 1.0

        norm = np.zeros_like(d, dtype=np.float32)
        norm[valid_depth_mask] = np.clip((d[valid_depth_mask] - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        color = np.repeat(gray[:, :, None], 3, axis=2)

    # Reuse the standard RGBD → sample path
    return make_sample_from_rgbd(color, depth_m, K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def build_networks_from_args(args):
    """
    Returns: network, network_crop (or None), device
    Mirrors how test_images_segmentation_no_ros.py loads its networks.
    """
    cfg_from_file(args.cfg)
    cfg.MODE = 'TEST'
    cfg.TEST.VISUALIZE = True
    np.random.seed(cfg.RNG_SEED)

    # setup GPU device
    cfg.gpu_id = args.gpu
    cfg.device = torch.device(f'cuda:{args.gpu}')
    torch.backends.cudnn.benchmark = True

    # === main segmentation network ===
    # print(f"[INFO] Loading main network: {args.network}")
    print(f"[INFO] Loading main network")
    network_data = torch.load(args.pretrained, map_location='cpu')
    num_classes = 2
    network = networks.__dict__[args.network](
        num_classes, cfg.TRAIN.NUM_UNITS, network_data
    )
    network = torch.nn.DataParallel(network, device_ids=[args.gpu]).cuda(cfg.device)
    network.eval()

    # === optional crop-refinement network ===
    network_crop = None
    if args.pretrained_crop:
        print("[INFO] Loading crop refinement network")
        network_data_crop = torch.load(args.pretrained_crop, map_location='cpu')
        network_crop = networks.__dict__[args.network](
            num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop
        )
        network_crop = torch.nn.DataParallel(network_crop, device_ids=[args.gpu]).cuda(cfg.device)
        network_crop.eval()

    return network, network_crop, cfg.device


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)
    return xyz_img


def parse_args():
    parser = argparse.ArgumentParser(description='Test UCN segmentation without ROS')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--pretrained_crop', type=str, default=None)
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--fx', type=float, default=615.0)
    parser.add_argument('--fy', type=float, default=615.0)
    parser.add_argument('--px', type=float, default=320.0)
    parser.add_argument('--py', type=float, default=240.0)
    parser.add_argument('--input_dir', type=str, default='data/demo')
    parser.add_argument('--output_dir', type=str, default='output/inference_results')
    parser.add_argument('--im_name', type=str, required=True, help='Path to RGB and depth images (in meters, npy or 16-bit png)')

     # NEW: optional point-cloud input
    parser.add_argument('--cloud_npy', type=str, default=None,
                        help='Optional path to xyzrgb.npy; if set, use point cloud instead of RGBD images.')
    # NEW: image dimensions for cloud rasterization
    parser.add_argument('--height', type=int, default=480,
                        help='Target image height when rasterizing a point cloud.')
    parser.add_argument('--width', type=int, default=640,
                        help='Target image width when rasterizing a point cloud.')

    return parser.parse_args()

def load_rgbd_from_args(args):
    """
    Loads color (BGR) and depth from disk, converts depth to meters if needed,
    and applies the same resize+pad behavior as the older script.

    Returns:
        im_color_bgr : np.ndarray (H, W, 3), BGR
        depth_m      : np.ndarray (H, W), meters
        fx, fy, px, py : floats (from args)
    """
    import cv2
    import numpy as np
    from utils.blob import pad_im
    from fcn.config import cfg

    # --- depth: .npy or 16-bit png (mm -> m if necessary) ---
    if args.im_name.endswith('.npy'):
        depth_m = np.load(f"{args.input_dir}/{args.im_name}-depth.npy").astype(np.float32)
    else:
        depth_raw = cv2.imread(f"{args.input_dir}/{args.im_name}-depth.png",
                               cv2.IMREAD_UNCHANGED).astype(np.float32)
        # If looks like millimeters, convert to meters (same as old script)
        depth_m = depth_raw / 1000.0 if depth_raw.max() > 100 else depth_raw

    # --- color (BGR) ---
    im_color_bgr = cv2.imread(f"{args.input_dir}/{args.im_name}-color.png")
    if im_color_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {args.im_name}-color.png")

    # --- resize + pad (mirror old behavior) ---
    if cfg.TEST.SCALES_BASE[0] != 1:
        scale = cfg.TEST.SCALES_BASE[0]
        im_color_bgr = pad_im(cv2.resize(im_color_bgr, None, fx=scale, fy=scale), 16)
        depth_m      = pad_im(cv2.resize(depth_m,      None, fx=scale, fy=scale), 16)

    return im_color_bgr, depth_m, args.fx, args.fy, args.px, args.py


def run_inference(sample, im_name, output_dir, network, network_crop, device):
    """
    sample: dict from make_sample_* helpers.
    Writes the same artifacts as the original script under:
       {output_dir}/segmentation_{im_name}/
    """
    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, f"segmentation_{im_name}")
    os.makedirs(result_dir, exist_ok=True)

    # HOOK: the exact forward you already do for a single sample
    with torch.no_grad():
        # If you already have a helper like test_sample(sample, network, network_crop), use it:
        label, label_refined = test_sample(sample, network, network_crop)  # <- replace with your actual call
        seg = label_refined if label_refined is not None else label

    seg_np = seg[0].detach().cpu().numpy().astype(np.int32)  # (H,W) ints
    num_objects = len(np.unique(seg_np)) - 1

    print(f"np.unique(seg_np) = {np.unique(seg_np)}")

    # HOOK: your existing visualization + saving code (kept identical)
    # 1) save im_label.npy
    np.save(os.path.join(result_dir, "im_label.npy"), seg_np)

    # 2) save sample for debugging (optional; mirrors your current sample.npz)
    np.savez_compressed(
        os.path.join(result_dir, "sample.npz"),
        image_color=sample['image_color'].cpu().numpy(),
        depth=sample['depth'].cpu().numpy()
    )

    # 3) write a JSON summary (add whatever fields you already output)
    seg_json = {
        "result_dir": result_dir,
        "base_output_dir": output_dir,
        "num_objects": num_objects,
        "instance_ids": seg_np.tolist(), #sorted(list(set(seg_np.flatten().tolist())))
    }
    with open(os.path.join(result_dir, "segmentation.json"), "w") as f:
        json.dump(seg_json, f, indent=2)

    # 4) (optional) save colorized mask PNGs exactly as your script does now
    # im_color = sample['image_color']
    # im_label = visualize_segmentation(im_color[:, :, ::-1], seg_np, return_rgb=True)

    return {"result_dir": result_dir, "json": seg_json}


def main():
    args = parse_args()

    # build networks
    network, network_crop, device = build_networks_from_args(args)

    # # load color & depth image as before
    # im_color_bgr, depth_m, fx, fy, px, py = load_rgbd_from_args(args)  # your existing loader

    # # construct the model input sample
    # sample = make_sample_from_rgbd(im_color_bgr, depth_m, fx, fy, px, py)

    # Decide between RGBD-from-disk and cloud-from-npy
    if args.cloud_npy is not None:
        # ---- CLOUD MODE ----
        print(f"[INFO] Using point cloud from {args.cloud_npy}")
        xyzrgb = np.load(args.cloud_npy).astype(np.float32)  # (N,3) or (N,6)

        H, W = args.height, args.width
        K = np.array([[args.fx, 0.0,     args.px],
                      [0.0,     args.fy, args.py],
                      [0.0,     0.0,     1.0    ]],
                     dtype=np.float32)
        have_rgb = (xyzrgb.shape[1] >= 6)

        sample = make_sample_from_cloud(xyzrgb, H, W, K, have_rgb=have_rgb)

    else:
        # ---- RGBD IMAGE MODE (existing behavior) ----
        im_color_bgr, depth_m, fx, fy, px, py = load_rgbd_from_args(args)
        sample = make_sample_from_rgbd(im_color_bgr, depth_m, fx, fy, px, py)

    # unified inference call
    run_inference(
        sample=sample,
        im_name=args.im_name,
        output_dir=args.output_dir,
        network=network,
        network_crop=network_crop,
        device=device
    )

    

if __name__ == '__main__':
    main()
