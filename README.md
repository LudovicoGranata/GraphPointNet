![alt text](https://github.com/LudovicoGranata/GraphPointNet/blob/main/plane.png)
# Graph PointNet

This study aims to investigate the efficacy of integrating Graph Neural Networks into PointNet and PointNet++ for the task of 3D part segmentation. We perform experiments using Graph Convolutional Networks and Graph Attention Networks, and evaluate two methods for constructing the graph: Nearest Neighbor and Ball Query. Our findings suggest that utilizing Graph Attention Networks and the Ball Query method can yield slightly improved results compared to using PointNet and PointNet++ alone.

## Getting Started
To download the Dataset
```
sh download_data.sh
```
To build the python virtual environment and download requirements

```
sh build-venv.sh
```

### Running the Project

The files to training and testing the models are in the folder `.\GraphPointNet` .

## Built With

This project make use of PyTorch and PyTorch Geometric

## Contributing

PR request are welcome


## Authors

Ludovico Granata


## Acknowledgments

PointNet and PointNet++ implementation from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
