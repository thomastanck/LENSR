import sys
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nhidlayers=2, dropout=0, indep_weights=True):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(nhidlayers + 1):
            in_features, out_features = nhid, nhid
            if i == 0:
                in_features = nfeat
            if i == nhidlayers:
                out_features = nclass
            self.layers.append(GraphConvolution(in_features, out_features, indep_weights=indep_weights))

        self.dropout = dropout

    def forward(self, x, adj, labels):
        # Apply dropout before every message passing round
        # *except* before the first round (don't dropout directly on the input data)
        # and before the last round (i don't know why, but the original code did this)
        x = F.relu(self.layers[0](x, adj, labels))
        for layer in self.layers[1:-1]:
            x = F.dropout(x, self.dropout)
            x = F.relu(layer(x, adj, labels))
        x = self.layers[-1](x, adj, labels)
        return x


class MLP(nn.Module):
    def __init__(self, ninput=200, nhidden=150, nclass=2, dropout=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(ninput, nhidden)
        self.fc2 = nn.Linear(nhidden, nclass)
        self.dropout = dropout

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.dropout(out, self.dropout)
        out = self.fc2(out)
        return out
