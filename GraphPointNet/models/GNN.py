import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GNN (nn.Module):
    def __init__(self, dim, n_layers=3, heads=4, dropout=0.5, graph_type = "GCN"):
        super(GNN, self).__init__()
        self.graph_type = graph_type
        self.batch_norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        if graph_type == "GCN" :
            layers = []
            for _ in range(n_layers):
                layers.append(pyg_nn.GCNConv(dim, dim))
            self.graph_conv = nn.ModuleList(layers)

        if graph_type == "GAT" :
            layers = []
            for _ in range(n_layers ):
                layers.append(pyg_nn.GATConv(dim, dim, heads=heads, concat=False, dropout=dropout))
            self.graph_conv = nn.ModuleList(layers)

    def forward(self, x, batch_edge_index):

        #reshape x to (B*N, F)
        n_batch = x.shape[0]
        n_points = x.shape[1]
        x = x.reshape(-1, x.shape[-1])

        for i, layer in enumerate(self.graph_conv):
            graph_out = layer(x, batch_edge_index)
            #add residual connection
            x = x + graph_out
            if i != len(self.graph_conv) - 1 and self.graph_type == "GCN":
                x = self.dropout(x)
            x = self.relu(x)
            
        x = x.reshape(n_batch, n_points, x.shape[-1])
        return x