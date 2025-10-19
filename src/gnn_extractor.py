import torch
from torch.nn import Module, Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNNExtractor(Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GNNExtractor, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x