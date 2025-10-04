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
    parser.add_argument('--input_dir', type=str, default='./data/demo')
    parser.add_argument('--output_dir', type=str, default='output/inference_results')
    parser.add_argument('--rgb', type=str, required=True, help='Path to RGB image')
    parser.add_argument('--depth', type=str, required=True, help='Path to depth image (in meters, npy or 16-bit png)')
    return parser.parse_args()


def main():
    args = parse_args()

    print('Called with args:')
    pprint.pprint(vars(args))

    # Load config
    cfg_from_file(args.cfg)
    cfg.MODE = 'TEST'
    cfg.TEST.VISUALIZE = True
    np.random.seed(cfg.RNG_SEED)

    # Setup device
    cfg.gpu_id = args.gpu
    cfg.device = torch.device(f'cuda:{args.gpu}')
    cudnn.benchmark = True

    # Prepare network
    print(f"Loading network: {args.network}")
    network_data = torch.load(args.pretrained, map_location='cpu')
    num_classes = 2
    network = networks.__dict__[args.network](num_classes, cfg.TRAIN.NUM_UNITS, network_data)
    network = torch.nn.DataParallel(network, device_ids=[args.gpu]).cuda(cfg.device)
    network.eval()

    network_crop = None
    if args.pretrained_crop:
        network_data_crop = torch.load(args.pretrained_crop, map_location='cpu')
        network_crop = networks.__dict__[args.network](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop)
        network_crop = torch.nn.DataParallel(network_crop, device_ids=[args.gpu]).cuda(cfg.device)
        network_crop.eval()

    # Load RGB and Depth
    if args.depth.endswith('.npy'):
        depth_img = np.load(f"{args.input_dir}/{args.depth}.npy").astype(np.float32)
    else:
        depth_img = cv2.imread(f"{args.input_dir}/{args.depth}-depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth_img.max() > 100:  # assume depth in mm
            depth_img /= 1000.0

    im_color = cv2.imread(f"{args.input_dir}/{args.rgb}-color.png")
    if im_color is None:
        raise FileNotFoundError(f"RGB image not found: {args.rgb}-color.png")

    # Resize and pad
    if cfg.TEST.SCALES_BASE[0] != 1:
        scale = cfg.TEST.SCALES_BASE[0]
        im_color = pad_im(cv2.resize(im_color, None, fx=scale, fy=scale), 16)
        depth_img = pad_im(cv2.resize(depth_img, None, fx=scale, fy=scale), 16)

    # Prepare input
    im = im_color.astype(np.float32)
    im_tensor = torch.from_numpy(im) / 255.0
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
    im_tensor -= pixel_mean
    image_blob = im_tensor.permute(2, 0, 1)
    sample = {'image_color': image_blob.unsqueeze(0)}

    height, width = im_color.shape[:2]
    xyz_img = compute_xyz(depth_img, args.fx, args.fy, args.px, args.py, height, width)
    depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
    sample['depth'] = depth_blob.unsqueeze(0)

    # Run network
    out_label, out_label_refined = test_sample(sample, network, network_crop)
    label = out_label[0].cpu().numpy()
    num_objects = len(np.unique(label)) - 1
    print(f"Detected {num_objects} objects.")

    # Visualization
    im_label = visualize_segmentation(im_color[:, :, ::-1], label, return_rgb=True)
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, f'segmentation_{args.rgb}.png'), im_label[:, :, ::-1])
    scipy.io.savemat(os.path.join(args.output_dir, f'segmentation_{args.rgb}.mat'),
                     {'rgb': im_color, 'label': label})

    result = {
        "num_objects": int(num_objects),
        "label": label.tolist()
    }
    with open(f"{args.output_dir}/segmentation_{args.rgb}.json", 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Saved results to {args.output_dir}")


if __name__ == '__main__':
    main()
