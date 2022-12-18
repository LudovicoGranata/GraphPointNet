#https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
from itertools import combinations
import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
import torch_geometric.nn as pyg_nn


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.GCN = GCN(128, 128)
        self.relu = nn.ReLU()
        # self.edge_index_batch = torch.randint(0, 2500-1, (2, 1000)).to("cuda:0") # edge index shape (B, 2, num_edges)


        # add fc layers
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, xyz, cls_label, point_graph):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        # TODO here put a GCN
        #build the adjacency matrix (ball query?)
        # adjacency_matrix = torch.ones([l0_points.shape[2], l0_points.shape[2]], dtype=torch.bool)
        # Create a tensor of sequential integers from 0 to num_nodes-1
        # num_nodes = 2500
        # how do I build the edge index?

        x = self.GCN(l0_points.permute(0,2,1), point_graph).permute(0,2,1)# l0_points shape (B, N, F)
        # x = self.fc1(self.relu(x))
        # x = self.fc2(self.relu(x))
        # x = F.log_softmax(x, dim=2)

        # feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points

class GCN (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv_1 = pyg_nn.GCNConv(in_channels, in_channels)
        self.conv_2 = pyg_nn.GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, batch_edge_index):
        result = []
        for x_i, edge_index in zip(x, batch_edge_index):
            x_i = self.conv_1(x_i, edge_index)
            x_i = self.relu(x_i)
            x_i = self.conv_2(x_i, edge_index)
            result.append(x_i)
        
        result = torch.stack(result, dim=0)
        return result


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # The negative log likelihood loss
        total_loss = F.nll_loss(pred, target)

        return total_loss