import torch
import torch.nn.functional as F
import sklearn
from .utils import cluster_kl_loss
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

EPS = 1e-15


class Encoder(torch.nn.Module):
    def __init__(self, dims, dropout):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            # self.layers.append(torch.nn.Conv1d(dims[i], dims[i + 1], 1))
            _init_weights(self.layers[-1])

    def forward(self, x):
        num_layers = len(self.layers)
        # x = x.view((x.shape[0], 1, -1))
        # x = x.transpose(2, 1)

        for idx, layer in enumerate(self.layers):
            if idx < num_layers - 1:
                x = layer(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x)

        # x = x.transpose(2, 1).contiguous()
        # x = x.view((x.shape[0], -1))

        return x


class Decoder(torch.nn.Module):
    def __init__(self, dims, dropout):
        super(Decoder, self).__init__()
        self.dropout = dropout
        dims = dims[::-1]
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            _init_weights(self.layers[-1])

    def forward(self, x):
        num_layers = len(self.layers)

        for idx, layer in enumerate(self.layers):
            if idx < num_layers - 1:
                x = layer(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x)
        return x


class InnerProductDecoder(torch.nn.Module):

    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class PointSpectrum(torch.nn.Module):

    def __init__(self, dims, num_centers, dropout, temperature):
        super(PointSpectrum, self).__init__()
        embedding_dim = dims[-1]
        self.data_dim = dims[0]
        # self.encoder = Encoder(dims, dropout)
        self.encoder = PointNetST(dims, dropout, False)
        self.decoder = InnerProductDecoder()

        self.clusterNet = ClusterNet(num_centers, temperature, embedding_dim)
        self.mu = None  # the cluster centers
        self.r = None  # the soft assignments to clusters

    def encode(self, x):
        z = self.encoder(x)
        return z

    def rec_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()  # for inner product decoder

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()  # for inner product decoder
        return pos_loss + neg_loss

    def cluster_loss(self, z):
        self.mu, self.r = self.clusterNet(z, 10)  # get centers, soft assignments and distribution
        loss = cluster_kl_loss(self.r)

        return loss

    def loss(self, z, alpha, beta, pos_edge_index, neg_edge_index=None):
        r_loss = self.rec_loss(z, pos_edge_index, neg_edge_index=neg_edge_index)
        c_loss = self.cluster_loss(z)

        loss = alpha * r_loss + beta * c_loss
        return loss, r_loss, c_loss

    def assign_clusters(self, z):
        _, r = self.clusterNet(z)
        return torch.argmax(r, dim=1)


class ClusterNet(torch.nn.Module):
    def __init__(self, k, beta, out_dim, device=torch.device('cuda')):
        super(ClusterNet, self).__init__()
        self.k = k
        self.beta = beta
        self.out = out_dim
        self.init = torch.rand(self.k, self.out).to(device)

    def forward(self, x, num_iter=1):
        mu_init, _ = self.cluster(x, self.k, num_iter, self.init, self.beta)
        mu, r = self.cluster(x, self.k, 1, mu_init.detach().clone(), self.beta)
        return mu, r

    def cluster(self, embeds, k, num_iter=1, init=None, beta=5):
        """
        pytorch (differentiable) implementation of soft k-means clustering.
        :param embeds: the actual embeddings to cluster
        :param k: number of clusters
        :param num_iter: iterations to run the process
        :param init: the initialization
        :param beta: cluster temperature parameter
        :return: means μ and soft assignments r
        """
        # normalize x so it lies on the unit sphere
        embeds = torch.diag(1. / torch.norm(embeds, p=2, dim=1)) @ embeds

        # use kmeans++ initialization if nothing is provided
        if init is None:
            data_np = embeds.cpu().detach().numpy()
            norm = (data_np ** 2).sum(axis=1)
            init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
            init = torch.tensor(init, requires_grad=True)
            if num_iter == 0:
                return init
        mu = init
        # n = embeds.shape[0]
        # d = embeds.shape[1]
        embeds = torch.diag(1./torch.norm(embeds, dim=1, p=2))@embeds
        for t in range(num_iter):
            # dist = torch.cosine_similarity(embeds[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
            dist = embeds @ mu.t()  # get distances between all data points and cluster centers
            r = torch.softmax(beta * dist, 1)  # cluster responsibilities via softmax
            cluster_r = r.sum(dim=0)  # total responsibility of each cluster
            cluster_mean = (r.t().unsqueeze(1) @ embeds.expand(k, *embeds.shape)).squeeze(1)  # mean of points in each cluster weighted by responsibility
            mu = torch.diag(1/cluster_r) @ cluster_mean  # update cluster means
        dist = embeds @ mu.t()
        r = torch.softmax(beta * dist, 1)
        return mu, r


class PointNetST(torch.nn.Module):
    """
    As proposed in "On Universal Equivariant Set Networks" paper. It is a PointNet with a single DeepSet layer.
    """
    def __init__(self, dims, dropout, regression=False):
        """

        :param dims: should be [initial_dim, width, ..., width, output_dim]
        :param regression:
        """
        super(PointNetST, self).__init__()
        self.out_features = dims[-1]
        self.regression = regression
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            if i == int(len(dims) / 2):
                self.layers.append(DeepSetLayer(dims[i], dims[i + 1]))
            else:
                self.layers.append(torch.nn.Conv1d(dims[i], dims[i + 1], 1))
            _init_weights(self.layers[-1])

    def forward(self, x):
        x = x.view((x.shape[0], 1, -1))
        x = x.transpose(2, 1)
        # the dims should now be BxFxn
        for idx, layer in enumerate(self.layers):
            if idx == len(self.layers) - 1:
                x = layer(x)
            else:
                x = F.relu(layer(x))
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(2, 1).contiguous()
        x = x.view((x.shape[0], -1))
        if self.regression:
            return x
        else:
            # x = F.log_softmax(x.view(-1, self.out_features), dim=-1)
            return x


class DeepSetLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(DeepSetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer1 = torch.nn.Conv1d(in_features, out_features, 1)
        self.layer2 = torch.nn.Conv1d(in_features, out_features, 1, bias=False)

    def forward(self, x):
        return self.layer1(x) + self.layer2(x - x.mean(dim=2, keepdim=True))


def _init_weights(layer):
    """
    He initialization for layer weights. Not Xavier as it was found that it has problems when used with ReLU
    """
    if isinstance(layer, DeepSetLayer):
        torch.nn.init.kaiming_uniform_(layer.layer1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(layer.layer2.weight, nonlinearity='relu')
    else:
        torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
