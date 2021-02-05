# iSpine-framework
A framework for GNN-based graph embeddings

**Parameters**

- input: Input graph dataset. Options: ['cora', 'citeseer', 'pubmed', 'wiki']. Required

## Implemented Algorithms
- GCN [[1]](#1)
- VGAE [[2]](#2)
- DANE [[3]](#3)
- GAT [[4]](#4)
- DGI [[5]](#5)
- AGC [[6]](#6)
- AGE [[7]](#7)


## Graph Convolutional Network (GCN)

GCN[[1]](#1) introduces graph convolution as a network layer. By stacking many layers, deeper node dependencies can be found.

### Usage

**Parameters**

- type: Type of adjacency matrix. 'gcn' or 'cheby'. Default: 'gcn'
- epochs:  Number of epochs. Default: 200
- dimension: Number of latent dimensions to learn for each node. Default: 128
- batch-size: Size of the batch used for training. Default: 128
- max-degree: Maximum Chebyshev polynomial degree. Default: 3
- learning-rate: Initial learning rate. Default: 0.001
- dropout: Dropout rate (1 - keep probability). Default: 0.5
- weight-decay: Weight for L2 loss on embedding matrix. E.g. 0.008. Default: 0.0
- early-stopping: Tolerance for early stopping (# of epochs). E.g. 10. Default: None

**Example Usage**
    ``$python main.py cora gcn --type gcn --epochs 500 --dimension 128 --batch-size 100 --learning-rate 0.01 --dropout 0.2 ``


## Variational Graph AutoEncoder (VGAE)

VGAE[[2]](#2) uses gcn layers in a variational autoencoder, enabling data generation as well.

### Usage

**Parameters**

- type: Type of adjacency matrix. 'normal' or 'cheby'. Default: 'normal'
- model: Type of model, that is simple or variational graph autoencoder. 'gae' or 'vgae'. Default: 'gae'
- epochs:  Number of epochs. Default: 200
- iter: Number of iterations for the whole process. Default: 15
- dimension: Number of latent dimensions to learn for each node. Default: 128
- hidden: Neurons in hidden layer. Default: 512
- batch-size: Size of the batch used for training. It uses the whole adjacency at once by definition. Default: 1
- learning-rate: Initial learning rate. Default: 0.001
- dropout: Dropout rate (1 - keep probability). Default: 0.0
- weight-decay: Weight for L2 loss on embedding matrix. E.g. 0.008. Default: 0.0
- early-stopping: Tolerance for early stopping (# of epochs). E.g. 10. Default: 20
- sparse: If given, use sparse form of arrays

**Example Usage**
    ``$python main.py cora vgae --model gae --dimension 100 --hidden 200 --sparse ``

## Deep Attributed Network Embedding (DANE)

DANE[[3]](#3) captures the high non-linearity and preserve the proximity both in the topological structure and node attributes.

### Usage

**Parameters**

- num-walks: Number of walks. Default: 10
- walk-length: The length of the walk. Default: 80
- window-size: The size of the window. Default: 10
- dimension: Embedding dimension. Default: 100
- net-hidden: Hidden layer dimension of connectivity network. Default: 200
- att-hidden: Hidden layer dimension of attribute network. Default: 200
- dropout: Dropout rate (1 - keep probability). Default: 0.2
- learning-rate: Initial learning rate. Default: 0.00001
- batch-size: Size of the batch used for training. Default: 100
- epochs: Number of epochs. Default: 500
- early-stopping: Tolerance for early stopping (# of epochs). E.g. 10. Default: 20
- alpha: default=50., Initial learning rate. Default: 50.
- beta: default=100., Initial learning rate. Default: 100.
- gamma: default=500., Initial learning rate. Default: 500.

**Example Usage**
    ``$python main.py cora dane --num-walks 15 --walk-length 20 --alpha 10. --beta 10. --gamma 100 ``


## Graph Attention Networks (GAT)

GAT[[4]](#4) implicitly enables specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront, by using the attention mechanism.

### Usage

**Parameters**
- epochs:  Number of epochs. Default: 200
- hidden: Neurons in hidden layer. Default: 8
- batch-size: Size of the batch used for training. It uses the whole adjacency at once by definition. Default: 1
- learning-rate: Initial learning rate. Default: 0.001
- weight-decay: Weight for L2 loss on embedding matrix. E.g. 0.008. Default: 0.0
- early-stopping: Tolerance for early stopping (# of epochs). E.g. 10. Default: 20
- sparse: If given, use sparse form of arrays
- residual: If given, determines whether to add seq to ret
- ffd-drop: Dropout rate (1 - keep probability) for features. Default: 0.6
- attn-drop: Dropout rate (1 - keep probability) for attention. Default: 0.6

**Example Usage**
    ``$python main.py cora gat --epochs 100 --hidden 100 --residual --ffd-drop 0.2 --attn-drop 0.3 ``


## Deep Graph Infomax (DGI)

DGI[[5]](#5) - in contrast to most prior approaches to unsupervised learning with GCNs - does not rely on random walk objectives, and is readily applicable to both transductive and inductive learning setups

### Usage

**Parameters**

- epochs:  Number of epochs. Default: 200
- dimension: Neurons in hidden layer. Default: 512
- batch-size: Size of the batch used for training. It uses the whole adjacency at once by definition. Default: 1
- learning-rate: Initial learning rate. Default: 0.001
- weight-decay: Weight for L2 loss on embedding matrix. E.g. 0.008. Default: 0.0
- dropout: Dropout rate (1 - keep probability). Default: 0.0
- early-stopping: Tolerance for early stopping (# of epochs). E.g. 10. Default: 20
- sparse: If given, use sparse form of arrays

**Example Usage**
    ``$python main.py cora dgi --epochs 100 --dimension 100 --sparse ``

## Attributed graph clustering via adaptive graph convolution (AGC)

AGC[[6]](#6) proposes an adaptive graph convolution method for attributed graph clustering that exploits high-order graph convolution to capture global cluster structure and adaptively selects the appropriate order for different graphs

### Usage

**Parameters**

- max-iter: Number of max iterations if there is no conversion in intra_C. Default: 60

**Example Usage**
    ``$python main.py cora agc --max-iter 50 ``

## Adaptive graph encoder for attributed graph embedding (AGE)

AGE[[7]](#7) consists of two modules: (1) To better alleviate the high-frequency noises in the node features, AGE first applies a carefully-designed Laplacian smoothing filter. (2) AGE employs an adaptive encoder that iteratively strengthens the filtered features for better node embeddings

### Usage

**Parameters**

- gnnlayers: Number of gnn layers. Default: 3
- linlayers: Number of hidden linear layers. Default: 1
- epochs: Number of epochs to train. Default: 400
- dims: Number of units in hidden layers. Default: [500]
- lr: Initial learning rate. Default: 0.001
- upth_st: Upper Threshold start. Default: 0.0015
- lowth_st: Lower Threshold start. Default: 0.1
- upth_ed: Upper Threshold end. Default: 0.001
- lowth_ed: Lower Threshold end. Default: 0.5
- upd: Update epoch. Default: 10
- bs: Batchsize. Default: 10000
- dataset: Name of dataset. Used for saving results. Default: 'citeseer'
- no-cuda: If given, disables CUDA training

**Example Usage**
    ``$python main.py cora age --gnnlayers 8 --upth_st 0.011 --lowth_st 0.1 --upth_ed 0.001 --lowth_ed 0.5 ``

The example reproduces the node clustering experiment results from the original paper

## K-order Spectral Attributed Clustering Embedding (kSPACE)

kSPACE is this work's proposed method

### Usage

**Parameters**

- repeats: How many times to repeat the experiment. Default: 1
- model: Type of autoencoder. Simple, variational or denoising. Default: ae
- dims: Number of units in hidden layers. Example --dims 500 200. Default: [200, 100]
- dropout: Dropout rate (1 - keep probability). Default: 0.2
- learning-rate: Initial learning rate. Default: 0.001
- batch-size: Size of the batch used for training. Default: 100
- epochs: Number of epochs. Default: 500
- early-stopping: Tolerance for early stopping (# of epochs). E.g. 10. Default: 20
- power: The upper bound of convolution order to search. Default: 8
- a-max: The upper bound of alpha rate. Default: 5
- alpha: How to calculate alpha for every training epoch. [linear, exp]. Default: linear
- save: If given, it saves the embedding on disk

**Example Usage**
    ``$python main.py cora kspace --model ae --dims 500 100 --power 5 --a-max 5 --alpha linear --save ``

# Benchmark Datasets
- Citeseer [[8]](#8)
- Cora [[8]](#8)
- Pubmed [[8]](#8)
- Wiki (Reference missing)


# References
<a id="1">[1]</a> 
Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.

<a id="2">[2]</a> 
Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. arXiv preprint arXiv:1611.07308.

<a id="3">[3]</a> 
Gao, H., & Huang, H. (2018, July). Deep attributed network embedding. In Twenty-Seventh International Joint Conference on Artificial Intelligence (IJCAI)).

<a id="4">[4]</a> 
Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.

<a id="5">[5]</a> 
Velickovic, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2019, May). Deep Graph Infomax. In ICLR (Poster).

<a id="6">[6]</a> 
Zhang, X., Liu, H., Li, Q., & Wu, X. M. (2019). Attributed graph clustering via adaptive graph convolution. arXiv preprint arXiv:1906.01210.

<a id="7">[7]</a> 
Cui, G., Zhou, J., Yang, C., & Liu, Z. (2020, August). Adaptive Graph Encoder for Attributed Graph Embedding. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 976-985).

<a id="8">[8]</a> 
https://linqs.soe.ucsc.edu/data

