import sklearn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.nn.inits import reset
from torch_geometric.nn import GCNConv, BatchNorm
from .utils import cluster_kl_loss

EPS = 1e-15
MAX_LOGSTD = 10


class GCNEncoder(torch.nn.Module):
    def __init__(self, dims, dropout):
        super(GCNEncoder, self).__init__()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            conv = GCNConv(dims[i], dims[i + 1], cached=True)  # cached only for transductive
            self.layers.append(conv)
        # self.norm = BatchNorm(dims[1])

    def forward(self, x, edge_index):
        num_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            if idx < num_layers - 1:
                x = layer(x, edge_index)
                # x = self.norm(F.relu(x))
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x, edge_index)

        return x


class InnerProductDecoder(torch.nn.Module):

    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class FeatureDecoder(torch.nn.Module):
    def __init__(self, dims, dropout):
        super(FeatureDecoder, self).__init__()
        self.dropout = dropout

        dims = dims[::-1]  # reverse dimensions for the decoder
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(dims) - 1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x, sigmoid=True):
        num_layers = len(self.layers)

        for idx, layer in enumerate(self.layers):
            if idx < num_layers - 1:
                x = layer(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x)
        return torch.sigmoid(x) if sigmoid else x


class FullReconstructionDecoder(torch.nn.Module):
    def __init__(self, dims, dropout):
        super(FullReconstructionDecoder, self).__init__()
        self.f_dec = FeatureDecoder(dims, dropout)
        self.ip_dec = InnerProductDecoder()

    def forward(self, z, edge_index, sigmoid=True):
        rec_a = self.ip_dec(z, edge_index, sigmoid=sigmoid)
        rec_feats = self.f_dec(z)

        return rec_a, rec_feats


class GAE(torch.nn.Module):

    def __init__(self, dims, num_centers, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        embedding_dim = dims[-1]
        self.cl_module = torch.nn.Linear(embedding_dim, num_centers)
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        reset(self.cl_module)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """Given latent variables, computes the binary cross entropy loss for positive edges and negative sampled edges.

        Args:
            z (Tensor): The latent space representations.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train against.
                If not given, uses negative sampling to calculate negative edges.
        """

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def complex_loss(self, z, alpha, pos_edge_index, neg_edge_index=None):
        rec_loss = self.recon_loss(z, pos_edge_index, neg_edge_index=neg_edge_index)

        q = F.softmax(self.cl_module(z))
        c_loss = cluster_kl_loss(q)
        correction_factor = torch.floor(torch.log10(rec_loss) - torch.log10(c_loss))
        c_loss = c_loss * (10 ** correction_factor)
        loss = rec_loss + alpha * c_loss

        return loss, rec_loss, c_loss

    def pretrain_cluster_module(self, z, pred, device, epochs=50, lr=0.001):
        """
            Pretrain neural network for calculating centers by using kmeans clusters
            Args:
                z (Tensor): The latent space representations.
                pred (numpy array): The predictions of KMeans for zs
        """

        pred_one_hot = torch.zeros((pred.size, pred.max() + 1))
        pred_one_hot[torch.arange(pred.size), pred] = 1
        pred_one_hot = pred_one_hot.to(device)

        # create batches for data
        dataset = TensorDataset(z, pred_one_hot)
        dataloader = DataLoader(dataset, batch_size=100)
        data_len = len(list(dataloader))

        opt = torch.optim.Adam(self.cl_module.parameters(), lr=lr)

        for i in range(epochs):
            loss = 0
            for x_batch, y_batch in dataloader:
                x_batch.detach_()
                y_batch.detach_()
                opt.zero_grad()

                q = F.softmax(self.cl_module(x_batch))
                l = F.binary_cross_entropy(q, y_batch)
                l.backward()
                opt.step()
                loss += float(l)
            loss = loss / data_len
            print('Epoch: {}, Pretrain cluster_loss: {:.4f}'.format(i, loss))

    def assign_clusters(self, z):
        q = F.softmax(self.cl_module(z))
        return torch.argmax(q, dim=1)


class ClusterGAE(torch.nn.Module):
    """
    Best parameters lr=0.001, dim=32, a-max=1, a=linear, e=2000, t=30, dropout=0.2, BatchNorm after ReLu
    """

    def __init__(self, dims, num_centers, dropout, temperature, decoder=None):
        super(ClusterGAE, self).__init__()
        self.encoder = GCNEncoder(dims, dropout)
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        embedding_dim = dims[-1]
        self.clusterNet = ClusterNet(num_centers, temperature, embedding_dim)
        self.mu = None  # the cluster centers
        self.r = None  # the soft assignments to clusters
        ClusterGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        reset(self.clusterNet)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """Given latent variables, computes the binary cross entropy loss for positive edges and negative sampled edges.

        Args:
            z (Tensor): The latent space representations.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train against.
                If not given, uses negative sampling to calculate negative edges.
        """

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def complex_loss(self, z, alpha, pos_edge_index, neg_edge_index=None):
        r_loss = self.recon_loss(z, pos_edge_index, neg_edge_index=neg_edge_index)

        self.mu, self.r = self.clusterNet(z, 10)  # get centers, soft assignments and distribution

        c_loss = cluster_kl_loss(self.r)
        # correction_factor = torch.floor(torch.log10(rec_loss) - torch.log10(c_loss))
        # c_loss = c_loss * (10 ** correction_factor)
        loss = (1-alpha) * r_loss + alpha * c_loss

        return loss, r_loss, c_loss

    def assign_clusters(self, z):
        _, r = self.clusterNet(z)
        return torch.argmax(r, dim=1)


class PointCluster(torch.nn.Module):

    def __init__(self, dims, num_centers, dropout, temperature, decoder=None):
        super(PointCluster, self).__init__()
        embedding_dim = dims[-1]
        self.pointNetST = PointNetST(dims, False)
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.clusterNet = ClusterNet(num_centers, temperature, embedding_dim)
        self.mu = None  # the cluster centers
        self.r = None  # the soft assignments to clusters

    def encode(self, x):
        x = self.pointNetST(x)
        return x

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """Given latent variables, computes the binary cross entropy loss for positive edges and negative sampled edges.

        Args:
            z (Tensor): The latent space representations.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train against.
                If not given, uses negative sampling to calculate negative edges.
        """

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()  # for inner product decoder

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()  # for inner product decoder
        return pos_loss + neg_loss

    def complex_loss(self, z, alpha, pos_edge_index, neg_edge_index=None):
        rec_loss = self.recon_loss(z, pos_edge_index, neg_edge_index=neg_edge_index)

        self.mu, self.r = self.clusterNet(z, 10)  # get centers, soft assignments and distribution

        c_loss = cluster_kl_loss(self.r)
        # correction_factor = torch.floor(torch.log10(rec_loss) - torch.log10(c_loss))
        # c_loss = c_loss * (10 ** correction_factor)
        loss = (1-alpha) * rec_loss + alpha * c_loss

        return loss, rec_loss, c_loss

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
        n = embeds.shape[0]
        d = embeds.shape[1]
        embeds = torch.diag(1./torch.norm(embeds, dim=1, p=2))@embeds
        for t in range(num_iter):
            dist = torch.cosine_similarity(embeds[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
            # dist = embeds @ mu.t()  # get distances between all data points and cluster centers
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
    def __init__(self, dims, regression=False):
        """

        :param dims: should be [initial_dim, width, ..., width, output_dim]
        :param regression:
        """
        super(PointNetST, self).__init__()
        self.out_features = dims[-1]
        self.regression = regression

        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            if i == int(len(dims) / 2):
                self.layers.append(DeepSetLayer(dims[i], dims[i + 1]))
            else:
                self.layers.append(torch.nn.Conv1d(dims[i], dims[i + 1], 1))

    def forward(self, x):
        x = x.view((x.shape[0], 1, -1))
        x = x.transpose(2, 1)
        # the dims should now be BxFxn
        for idx, layer in enumerate(self.layers):
            if idx == len(self.layers) - 1:
                x = layer(x)
            else:
                x = F.relu(layer(x))
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
