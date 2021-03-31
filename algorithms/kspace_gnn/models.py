import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.nn.inits import reset
from torch_geometric.nn import GCNConv
from torch.autograd import Variable
from collections import OrderedDict
from .utils import cluster_kl_loss

EPS = 1e-15
MAX_LOGSTD = 10

class GCNEncoder(torch.nn.Module):
    def __init__(self, dims, dropout):
        super(GCNEncoder, self).__init__()
        self.dropout = dropout
        encoder_layers = OrderedDict()
        for i in range(len(dims) - 1):
            encoder_layers['gcn_{}'.format(i)] = GCNConv(dims[i], dims[i+1], cached=True) # cached only for transductive

        self.encoder = torch.nn.Sequential(encoder_layers)

    def forward(self, x, edge_index):
        layers = list(self.encoder.children())
        num_layers = len(layers)
        for idx, layer in enumerate(layers):
            if idx < num_layers - 1:
                x = layer(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = layer(x, edge_index)

        return x

class InnerProductDecoder(torch.nn.Module):

    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj



class GAE(torch.nn.Module):

    def __init__(self, dims, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder

        embedding_dim = dims[-1]
        centers = torch.randn(7, embedding_dim, requires_grad=True)
        self.centers = torch.nn.Parameter(centers)
        GAE.reset_parameters(self)


    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        reset(self.centers)


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
        c_loss = cluster_kl_loss(z, self.centers)

        correction_factor = abs(rec_loss / c_loss)
        c_loss = c_loss * correction_factor
        loss = rec_loss + alpha * c_loss

        return loss

    def test(self, z, pos_edge_index, neg_edge_index):
        """Given latent variables, positive edges and negative edges, computes area under the ROC curve (AUC)
            and average precision (AP) scores.

            Y contains 1 for existing edges and 0 for non-existing. Decoder produces probs. Metrics such as f1 do not work.
        Args:
            z (Tensor): The latent space representations.
            pos_edge_index (LongTensor): The positive edges to evaluate against.
            neg_edge_index (LongTensor): The negative edges to evaluate against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)



class VGAE(GAE):
    def __init__(self, encoder, decoder=None):
        super(VGAE, self).__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z


    def kl_loss(self, mu=None, logstd=None):
        """Computes the KL loss, either for the passed arguments mu and logstd, or based on latent variables
            from last encoding.

        Args:
            mu (Tensor, optional): The latent space mean. If set to None, uses the last computation.
            logstd (Tensor, optional): The latent space logvar.  If set to None, uses the last computation.
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
