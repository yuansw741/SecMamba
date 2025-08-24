import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
import os
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data, download_url, DataLoader
import pandas as pd
import torch.nn as nn
import argparse
import os.path as osp
import os
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
import random

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, split_ratios=(0.7, 0.2, 0.1)):
        self.split_ratios = split_ratios
        super().__init__(root, transform, pre_transform)
        self.prepare_data()

    def prepare_data(self):
        if not all(os.path.exists(os.path.join(self.processed_dir, name)) for name in self.processed_file_names):
            print("Processed files do not exist, processing now...")
            self.process()
        else:
            print("Processed files exist, loading...")

    @property
    def raw_file_names(self):
        return ['USTC.csv']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        G = nx.DiGraph()

        for index, row in df.iterrows():
            edge_features = torch.tensor(row.iloc[5:25].values, dtype=torch.float)
            G.add_edge(row['src_ip'], row['dst_ip'], edge_attr=edge_features, label=row['label'])
            # G.add_edge(row['dst_ip'], row['src_ip'], edge_attr=edge_features, label=row['label'])

        # print(G.edges.label)
        components = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        data_list = []

        for component in components:
            data = from_networkx(component)
            # Create node feature based on edges
            node_to_idx = {node: i + 1 for i, node in enumerate(component.nodes())}
            features = []
            for u, v in component.edges():
                features.append([node_to_idx[u]])
                features.append([node_to_idx[v]])

            node_ids = torch.arange(1, data.num_nodes + 1).unsqueeze(
                1).long()  # Use float for neural network processing
            data.x = node_ids
            data.edge_attr = torch.stack([attr['edge_attr'] for _, _, attr in component.edges(data=True)])
            # data.y = torch.tensor([attr['label'] for _, _, attr in component.edges(data=True)], dtype=torch.long)

            edge_labels = [attr['label'] for _, _, attr in component.edges(data=True)]
            data.y = torch.tensor(edge_labels, dtype=torch.long)

            # 打印每个边的标签，确保与data.y的内容一致
            # print("Edge labels:", edge_labels)
            # print("Data.y tensor:", data.y.tolist())

            # for _, _, attr in component.edges(data=True):
            #      print(attr)
            # print(data.x)
            # print(data.edge_attr)
            # print(data.y)

            if self.pre_transform:
                data = self.pre_transform(data)
            # print("@@@@@@@1")
            # print(data)
            # print(data.x)
            # print(data.edge_attr)
            # print(data.y)
            # print("@@@@@@@2")
            data_list.append(data)

        # for data in data_list:
        #     print(data.x)
        # train_data, temp_data = train_test_split(data_list, train_size=self.split_ratios[0], random_state=42)
        # val_data, test_data = train_test_split(temp_data, train_size=self.split_ratios[1] / (1 - self.split_ratios[0]),
        #                                        random_state=42)
        # random.shuffle(data_list)
        n = len(data_list)
        end_train = int(n * self.split_ratios[0])
        end_val = end_train + int(n * self.split_ratios[1])

        train_data = data_list[:end_train]

        val_data = data_list[end_train:end_val]
        test_data = data_list[end_val:]

        torch.save(train_data, os.path.join(self.processed_dir, 'train.pt'))
        torch.save(val_data, os.path.join(self.processed_dir, 'val.pt'))
        torch.save(test_data, os.path.join(self.processed_dir, 'test.pt'))

    def load_data(self):
        self.train_data = torch.load(os.path.join(self.processed_dir, 'train.pt'))
        self.val_data = torch.load(os.path.join(self.processed_dir, 'val.pt'))
        self.test_data = torch.load(os.path.join(self.processed_dir, 'test.pt'))


# transform = T.Compose([
#     T.AddRandomWalkPE(walk_length=20, attr_name='pe'),
#     T.NormalizeFeatures()
# ])
transform = T.AddRandomWalkPE(walk_length=20,attr_name='pe')

dataset_path = './dataset'
dataset = CustomGraphDataset(root=dataset_path, pre_transform=transform)
dataset.load_data()

train_loader = DataLoader(dataset.train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset.val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)



def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices


class GPSConv(torch.nn.Module):

    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            act: str = 'relu',
            att_type: str = 'transformer',
            order_by_degree: bool = False,
            shuffle_ind: int = 0,
            d_state: int = 16,
            d_conv: int = 4,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        assert (self.order_by_degree == True and self.shuffle_ind == 0) or (
                    self.order_by_degree == False), f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'

        if self.att_type == 'transformer':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                dropout=attn_dropout,
                batch_first=True,
            )
        if self.att_type == 'mamba.txt':
            self.self_attn = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=1
            )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []

        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        ### Global attention transformer-style model.
        if self.att_type == 'transformer':
            h, mask = to_dense_batch(x, batch)
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
            h = h[mask]

        if self.att_type == 'mamba.txt':

            if self.order_by_degree:
                deg = degree(edge_index[0], x.shape[0]).to(torch.long)
                order_tensor = torch.stack([batch, deg], 1).T
                _, x = sort_edge_index(order_tensor, edge_attr=x)
            # print(self.shuffle_ind)
            if self.shuffle_ind == 0:
                # print("dense check!")
                # print(x.shape)
                # print(batch.shape)
                # print(x)
                # print(batch)
                h, mask = to_dense_batch(x, batch)
                # print("dense OK!")
                h = self.self_attn(h)[mask]
            else:
                mamba_arr = []
                for _ in range(self.shuffle_ind):
                    h_ind_perm = permute_within_batch(x, batch)
                    h_i, mask = to_dense_batch(x[h_ind_perm], batch)
                    h_i = self.self_attn(h_i)[mask][h_ind_perm]
                    mamba_arr.append(h_i)
                h = sum(mamba_arr) / self.shuffle_ind
        ###

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')

import torch.nn.init as init

class EdgeFeatureCompressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EdgeFeatureCompressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)  # 30 -> 16
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)  # 16 -> 1

        # # 初始化权重
        # init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        # init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GraphModel(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, model_type: str, shuffle_ind: int, d_state: int,
                 d_conv: int, order_by_degree: False):
        super().__init__()

        self.node_emb = Embedding(20000, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)
        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        self.edge_feature_compressor = EdgeFeatureCompressor(6, 64)


        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            if self.model_type == 'gine':
                conv = GINEConv(nn)

            if self.model_type == 'mamba.txt':
                conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5,
                               att_type='mamba.txt',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)

            if self.model_type == 'transformer':
                conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5, att_type='transformer')

            # conv = GINEConv(nn)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(256, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 2),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        # print("Cat Error!")
        # print(x_pe.shape)
        # print(self.pe_lin)
        # print(self.pe_lin(x_pe))
        # print(x)
        # print("node",self.node_emb(x.squeeze(-1)))
        # print("batch",batch)
        x = torch.cat((self.pe_lin(x_pe),self.node_emb(x.squeeze(-1))), 1)
        # print("Cat OK!")
        # edge_attr = self.edge_emb(edge_attr)
        # print(edge_attr.shape)
        # print("@@@@@@before")
        # print(edge_attr)
        # print("@@@@@@before mid")
        if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
            # print("Input contains NaN or Inf.")
            edge_attr[torch.isinf(edge_attr)] = 0  # 用0替换无限大
            edge_attr[torch.isnan(edge_attr)] = 0  # 用0替换NaN
        edge_attr = self.edge_feature_compressor(edge_attr)
        # print(edge_attr)
        # print("@@@@@@before end!")
        # print(edge_attr.shape)
        # print("OUT BATCH B!!!")
        # print(x)
        # print(batch)
        # print("OUT BATCH E!!!")
        for conv in self.convs:
            if self.model_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)

        edge_features = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)
        combined_tensor = torch.cat((edge_attr, edge_attr), dim=1)
        # print(edge_index)
        # print(x[edge_index[0]], x[edge_index[1]])
        # print(edge_attr)
        # print("test@@@@@@@begin")
        # print(edge_features.shape)
        # print(edge_features)
        # print(combined_tensor.shape)
        # print(combined_tensor)
        combined_tensor = torch.cat((combined_tensor, edge_features), dim=1)
        # print(combined_tensor.shape)
        # print(combined_tensor)
        # print("test@@@@@@@end!!!")
        # x = global_add_pool(x, batch)
        logits = self.mlp(combined_tensor)
        # return self.mlp(x)
        # print(logits.shape)
        return logits

loss_func = BCEWithLogitsLoss()

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # print("batch train1!!!")
        # print(data)
        # print(data.x)
        # print(data.batch)
        # print("batch train2!!!")
        # print("before@@@@@@@@")
        # print(data.edge_attr)
        # print("before@@@@@@@@end!")
        logits = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        preds = logits.argmax(dim=1)
        # print("Predict:",preds)
        # print("Data.y:",data.y)
        loss = F.cross_entropy(logits, data.y)
        # loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
            loss = F.cross_entropy(logits, data.y)
            # loss = loss_func(logits.squeeze(), data.y.float())
            total_loss += loss.item() * data.num_graphs

            # 计算准确率
            preds = logits.argmax(dim=1)  # 获取概率最大的类别
            # print(preds)
            # print(data.y)
            correct += (preds == data.y).sum().item()  # 计算预测正确的数量
            total += data.y.size(0)  # 总样本数量

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


import os

def delete_all_files(folder_path):
    """删除文件夹中的所有文件（不删除子文件夹）"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 只删除文件，不删除文件夹
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"已删除文件: {file_path}")

# 使用示例
folder_path = "./dataset/processed/"
delete_all_files(folder_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num = 15
model = GraphModel(channels=64, pe_dim=8, num_layers=num,
                   model_type='gine', #transformer
                   shuffle_ind=0, order_by_degree=True,
                   d_conv=4, d_state=16,
                  ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)

# loss = train()

num_epochs = 100
with open(f'../../code/pythonProject1/超参数选择/USTC/{num}', 'w', encoding='utf-8') as f:
    for epoch in range(1, num_epochs + 1):
        train_loss = train()
        val_loss, val_accuracy = test(val_loader)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}, Val Accuracy: {val_accuracy:.5f}")
        print(f"Epoch {epoch}: Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}, Val Accuracy: {val_accuracy:.5f}", file=f)

# 最终测试
test_loss, test_accuracy = test(test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")