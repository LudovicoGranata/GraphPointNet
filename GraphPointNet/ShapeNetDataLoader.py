import itertools
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
import torch
from scipy.linalg import block_diag

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False, config=None, debug=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.config = config


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:

            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            if debug:
                train_ids_debug = set()
                for i in range(500):
                    train_ids_debug.add(train_ids.pop())
                train_ids = train_ids_debug
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            if debug:
                val_ids_debug = set()
                for i in range(100):
                    val_ids_debug.add(val_ids.pop())
                val_ids = val_ids_debug
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        # build graph index
        graph = self.build_edge_index (point_set)

        return {"point_set":point_set, "cls":cls, "seg":seg, "graph":graph}

    def __len__(self):
        return len(self.datapath)

    
    def build_edge_index(self, points):

        num_connections = self.config.DATASET.NUM_CONNECTIONS

        # Flatten the points array to shape [n_points, n_dimensions]
        # points = points.reshape(points.shape[-1], -1)

        if num_connections >= points.shape[0]:
            num_connections = points.shape[0]-1

        # Build a KD-tree from the points
        tree = cKDTree(points)

        # Find the indices and distances of the k nearest neighbors for each point
        edges = tree.query(points, k=num_connections+1)

        # Extract the indices of the nearest neighbors, ignoring the point itself
        nearest_neighbors = edges[1][:, 1:]
        adjacency_matrix = np.zeros((points.shape[0], points.shape[0]))
        for i in range(points.shape[0]):
            for j in range(num_connections):
                adjacency_matrix[i, nearest_neighbors[i,j]] = 1
                adjacency_matrix[nearest_neighbors[i,j], i] = 1
               


        # Flatten the nearest neighbors array to shape [2, n_edges]
        # edge_index = nearest_neighbors.reshape(2, -1)

        # return edge_index
        return adjacency_matrix

    def my_collate(self, batch):
        
        point_set = [item["point_set"] for item in batch]
        cls = [item["cls"] for item in batch]
        seg = [item["seg"] for item in batch]
        graph = [item["graph"] for item in batch]

        point_set = torch.tensor(point_set, dtype=torch.float)
        cls = torch.tensor(cls, dtype=torch.long)
        seg = torch.tensor(seg, dtype=torch.long)
        graph = block_diag(*graph)
        #!too slow
        # edge_list = torch.tensor([graph[i, j] for i in range(graph.shape[0]) for j in range(graph.shape[1]) if graph[i, j] == 1])

        rows, cols = np.where(graph == 1)
        edge_list = torch.tensor(list(zip(rows, cols)), dtype=torch.long).permute(1,0)
        return {"points":point_set, "label":cls, "target":seg, "edge_list":edge_list}