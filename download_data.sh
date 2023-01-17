#!/bin/bash

# Download original ShapeNetPart dataset (around 1GB)
# wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip --no-check-certificate
# unzip shapenetcore_partanno_v0.zip
# rm shapenetcore_partanno_v0.zip
mkdir data
cd data
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
rm shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

# Download HDF5 for ShapeNet Part segmentation (around 346MB)
# wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip --no-check-certificate
# unzip shapenet_part_seg_hdf5_data.zip
# rm shapenet_part_seg_hdf5_data.zip