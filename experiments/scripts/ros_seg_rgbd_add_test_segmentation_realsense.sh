#!/bin/bash
source ~/graspnet_ws/src/UnseenObjectClustering/ucn_env_py39_venv/bin/activate
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python ros/test_images_segmentation.py --gpu 0

	
set -x
set -e

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 25/07/31: added to resolve cuda out of memory issue # 25/08/27: removed, as it brings "RuntimeError: Unrecognized CachingAllocator option: expandable_segments" 

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

outdir="data/checkpoints"

./ros/test_images_segmentation.py --gpu $1 \
  --network seg_resnet34_8s_embedding \
  --pretrained $outdir/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth \
  --pretrained_crop $outdir/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
