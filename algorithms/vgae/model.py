import torch
import torch.nn as nn
import numpy as np
from .layers import GraphConvSparse

class VGAE(nn.Module):
    def __init__(self, adj, input_dim, hidden1_dim, hidden2_dim):
        super(VGAE, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred




def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


class GAE(nn.Module):
    def __init__(self, adj):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(self.input_dim, self.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(self.hidden1_dim, self.hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred

# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out