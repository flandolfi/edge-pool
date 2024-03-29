from typing import Union, Callable, Optional

import torch

from torch_geometric.nn import Sequential, conv, pool
from torch_geometric.nn.glob import global_add_pool, global_max_pool
from torch_geometric.nn.pool import TopKPooling, SAGPooling, ASAPooling, PANPooling
from torch_geometric.nn.models import MLP
from torch_geometric.data import InMemoryDataset, Data

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pool import EdgePooling


class Baseline(LightningModule):
    known_signatures = {
        'GCNConv': 'x, e_i, e_w -> x',
        'GATConv': 'x, e_i, e_w -> x',
        'GATv2Conv': 'x, e_i, e_w -> x',
        'SAGEConv': 'x, e_i -> x',
        'GraphConv': 'x, e_i, e_w -> x',
        'SGConv': 'x, e_i, e_w -> x',
        'ChebConv': 'x, e_i, e_w -> x',
        'LEConv': 'x, e_i, e_w -> x',
        'GINConv': 'x, e_i -> x',
        'PANConv': 'x, e_i -> x, M',
        'TopKPooling': 'x, e_i, e_w, b -> x, e_i, e_w, b, perm, score',
        'SAGPooling': 'x, e_i, e_w, b -> x, e_i, e_w, b, perm, score',
        'ASAPooling': 'x, e_i, e_w, b -> x, e_i, e_w, b, perm',
        'PANPooling': 'x, M, b -> x, e_i, e_w, b, perm, score',
        'EdgePooling': 'x, e_i, e_w, b -> x, e_i, e_w, b, perm, match, score',
    }
    
    requires_nn = {'GINConv', 'GINEConv'}
    requires_edge_dim = {'GATConv', 'GATv2Conv'}
    
    def __init__(self, dataset: InMemoryDataset,
                 lr: float = 0.001,
                 patience: int = 30,
                 channels: int = 64,
                 channel_multiplier: int = 2,
                 num_layers: int = 3,
                 gnn_class: Union[str, Callable] = 'GCNConv',
                 gnn_signature: Optional[str] = None,
                 gnn_kwargs: Optional[dict] = None,
                 pool_class: Optional[Union[str, Callable]] = None,
                 pool_signature: Optional[str] = None,
                 pool_kwargs: Optional[dict] = None):
        super(Baseline, self).__init__()
        
        if isinstance(gnn_class, str):
            gnn_class = getattr(conv, gnn_class)
            
        if gnn_class.__name__ in self.requires_nn:
            _gnn_cls = gnn_class
            
            def gnn_class(in_channels, out_channels, **kwargs):
                return _gnn_cls(nn=MLP([in_channels, in_channels, out_channels],
                                       batch_norm=False), **kwargs)

        if gnn_kwargs is None:
            gnn_kwargs = {}
            
        if gnn_signature is None:
            gnn_signature = self.known_signatures.get(gnn_class.__name__,
                                                      'x, e_i -> x')
            
        if isinstance(pool_class, str):
            pool_class = getattr(pool, pool_class)
        
        if pool_kwargs is None:
            pool_kwargs = {}
            
        if pool_class is not None and pool_signature is None:
            pool_signature = self.known_signatures.get(pool_class.__name__,
                                                       'x, e_i, e_w, b -> x, e_i, e_w, b')
        
        in_channels = dataset.num_node_features or 1
        edge_channels = dataset.num_edge_features or 1
        out_channels = dataset.num_classes
        
        if gnn_class.__name__ in self.requires_edge_dim:
            gnn_kwargs.setdefault('edge_dim', edge_channels)
        
        layers = []
        
        for l_id in range(num_layers):
            layers.append((gnn_class(in_channels=in_channels, out_channels=channels, **gnn_kwargs), gnn_signature))
            
            if l_id == num_layers - 1:
                layers.append((global_max_pool, 'x, b -> x_m'))
                layers.append((global_add_pool, 'x, b -> x_a'))
                layers.append((lambda x_m, x_a: torch.cat([x_m, x_a], dim=-1),
                               'x_m, x_a -> x'))
            elif pool_class is not None:
                layers.append((pool_class(in_channels=channels, **pool_kwargs), pool_signature))
                
            layers.append((torch.nn.ReLU(), 'x -> x'))
            
            in_channels = channels
            channels *= channel_multiplier
            
        layers.append((MLP([2*in_channels, in_channels//4, out_channels],
                           batch_norm=False, dropout=0.3), 'x -> x'))
        self.model = Sequential('x, e_i, e_w, b', layers)
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.patience = patience
        self.lr = lr
        
    @staticmethod
    def accuracy(y_pred, y_true):
        y_class = torch.argmax(y_pred, dim=-1)
        return torch.mean(torch.eq(y_class, y_true).float())
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if edge_attr is None:
            edge_attr = torch.ones_like(edge_index[0], dtype=torch.float)
        
        return self.model(x, edge_index, edge_attr, batch)

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = self.loss(y_hat, data.y)
        self.log('train_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = self.loss(y_hat, data.y)
        acc = self.accuracy(y_hat, data.y)
        self.log('val_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        self.log('val_acc', acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        return {
            'val_loss': loss,
            'val_acc': acc,
        }

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.edge_attr, data.batch)
        acc = self.accuracy(y_hat, data.y)
        self.log('test_acc', acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min",
                                   patience=self.patience)
        checkpoint = ModelCheckpoint(monitor="val_acc", mode="max")
        return [early_stop, checkpoint]


class TopKPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, ratio: float, **kwargs):
        kwargs['pool_class'] = TopKPooling
        kwargs['pool_kwargs'] = {'ratio': ratio}
        super(TopKPool, self).__init__(dataset=dataset, **kwargs)


class SAGPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, ratio: float, **kwargs):
        # Simulate the "augmentation" variant of
        # SAGPool paper with a 2-hop SGConv
        def _gnn_wrap(*gnn_args, **gnn_kwargs):
            return conv.SGConv(*gnn_args, K=2, **gnn_kwargs)

        kwargs['pool_class'] = SAGPooling
        kwargs['pool_kwargs'] = {
            'ratio': ratio,
            'GNN': _gnn_wrap,
        }

        super(SAGPool, self).__init__(dataset=dataset, **kwargs)


class ASAPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, ratio: float, **kwargs):
        kwargs['pool_class'] = ASAPooling
        kwargs['pool_kwargs'] = {'ratio': ratio}
        super(ASAPool, self).__init__(dataset=dataset, **kwargs)


class PANPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, ratio: float, filter_size: int, **kwargs):
        kwargs['gnn_class'] = conv.PANConv
        kwargs['gnn_kwargs'] = {'filter_size': filter_size}
        kwargs['pool_class'] = PANPooling
        kwargs['pool_kwargs'] = {'ratio': ratio}
        super(PANPool, self).__init__(dataset=dataset, **kwargs)


class GraclusPool(Baseline):
    def __init__(self, dataset: InMemoryDataset, **kwargs):
        def _graclus_wrap(in_channels=None):
            def _graclus(x, e_i, e_w, b):
                cluster = pool.graclus(edge_index=e_i, weight=e_w, num_nodes=x.size(0))
                data = pool.avg_pool(cluster, Data(x=x, edge_index=e_i, edge_attr=e_w, batch=b))
                return data.x, data.edge_index, data.edge_weight, data.batch

            return _graclus

        kwargs['pool_class'] = _graclus_wrap
        kwargs['pool_signature'] = 'x, e_i, e_w, b -> x, e_i, e_w, b'

        super(GraclusPool, self).__init__(dataset=dataset, **kwargs)


class EdgePool(Baseline):
    def __init__(self, dataset: InMemoryDataset,
                 score: str = 'linear',
                 score_nodes: bool = False,
                 score_bias: bool = True,
                 score_activation: str = 'tanh',
                 score_descending: bool = True,
                 reduce: str = 'sum',
                 reduce_with_pseudoinverse: bool = False,
                 remove_self_loops: bool = True,
                 **kwargs):
        kwargs['pool_class'] = EdgePooling
        kwargs['pool_kwargs'] = {
            'score': score,
            'score_nodes': score_nodes,
            'score_bias': score_bias,
            'score_activation': score_activation,
            'score_descending': score_descending,
            'reduce_x': reduce,
            'reduce_with_pseudoinverse': reduce_with_pseudoinverse,
            'remove_self_loops': remove_self_loops,
        }
        
        kwargs['pool_signature'] = self.known_signatures['EdgePooling']

        super(EdgePool, self).__init__(dataset=dataset, **kwargs)


class EdgePoolSoftmax(EdgePool):
    def __init__(self, dataset: InMemoryDataset, **kwargs):
        super(EdgePoolSoftmax, self).__init__(dataset=dataset, score_activation='softmax', **kwargs)


class EdgePoolV2(EdgePool):
    def __init__(self, dataset: InMemoryDataset, score_activation='sigmoid', **kwargs):
        super(EdgePoolV2, self).__init__(dataset=dataset, score_nodes=True,
                                         score_bias=False,
                                         score_activation=score_activation,
                                         score_descending=False,
                                         reduce_with_pseudoinverse=True,
                                         remove_self_loops=False, **kwargs)


class EdgePoolV2Random(EdgePoolV2):
    def __init__(self, dataset: InMemoryDataset, **kwargs):
        super(EdgePoolV2Random, self).__init__(dataset=dataset, score='random',
                                               score_activation='linear', **kwargs)


class EdgePoolV2Normal(EdgePoolV2):
    def __init__(self, dataset: InMemoryDataset, **kwargs):
        super(EdgePoolV2Normal, self).__init__(dataset=dataset, score='normal', **kwargs)


class EdgePoolV2Norm(EdgePoolV2):
    def __init__(self, dataset: InMemoryDataset, **kwargs):
        super(EdgePoolV2Norm, self).__init__(dataset=dataset, score='norm', **kwargs)

