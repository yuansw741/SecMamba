import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
import os
import torch_geometric.transforms as T
from torch_geometric.data import download_url
import torch.nn as nn
import argparse
import os.path as osp
from typing import Any, Dict, Optional
import inspect

import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
    Dropout,
    BCEWithLogitsLoss
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch, degree, sort_edge_index

from mamba_ssm import Mamba
import numpy as np
from tqdm import tqdm
import random

# --- New Imports for Visualization and Metrics ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.manifold import TSNE


# -------------------------------------------------


class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, split_ratios=(0.7, 0.2, 0.1)):
        self.split_ratios = split_ratios
        super().__init__(root, transform, pre_transform)
        # The 'process' method will be called implicitly if processed files are not found.
        # We load the data explicitly to have it available on the dataset object.
        self.load_data()

    def load_data(self):
        # This method is to ensure data is loaded into the object attributes
        if not all(os.path.exists(path) for path in self.processed_paths):
            print("Processed files not found. Running processing...")
            self.process()

        print("Loading processed files...")
        self.train_data = torch.load(self.processed_paths[0])
        self.val_data = torch.load(self.processed_paths[1])
        self.test_data = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        # Assume 'USTC.csv' is in the self.raw_dir folder.
        # download_url('https://path-to-your/USTC.csv', self.raw_dir) # If it needs to be downloaded
        return ['USTC.csv']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        # This method is called if raw files are not found.
        # You can place your data downloading logic here.
        # For local files, you can just ensure the file is in the `raw_dir`.
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        # A more robust way to create a node mapping for all IPs
        all_ips = pd.concat([df['src_ip'], df['dst_ip']]).unique()
        node_to_idx = {ip: i for i, ip in enumerate(all_ips)}
        num_nodes = len(all_ips)

        G = nx.DiGraph()
        # Add all nodes to the graph first
        for i in range(num_nodes):
            G.add_node(i)

        for _, row in df.iterrows():
            src_idx = node_to_idx[row['src_ip']]
            dst_idx = node_to_idx[row['dst_ip']]
            edge_features = torch.tensor(row.iloc[5:11].values,
                                         dtype=torch.float)  # Modified to 6 features for compressor
            G.add_edge(src_idx, dst_idx, edge_attr=edge_features, label=row['label'])

        components = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        data_list = []

        for component in components:
            if component.number_of_edges() == 0:
                continue

            # Remap nodes to be contiguous from 0 for each subgraph
            node_mapping = {node: i for i, node in enumerate(component.nodes())}
            component = nx.relabel_nodes(component, node_mapping)

            data = from_networkx(component)
            data.x = torch.arange(data.num_nodes, dtype=torch.long).unsqueeze(1)
            data.edge_attr = torch.stack([attr['edge_attr'] for _, _, attr in component.edges(data=True)])
            data.y = torch.tensor([attr['label'] for _, _, attr in component.edges(data=True)], dtype=torch.long)

            if self.pre_transform:
                data = self.pre_transform(data)
            data_list.append(data)

        # Shuffle before splitting to ensure randomness
        random.seed(42)
        random.shuffle(data_list)
        n = len(data_list)
        end_train = int(n * self.split_ratios[0])
        end_val = end_train + int(n * self.split_ratios[1])

        train_data = data_list[:end_train]
        val_data = data_list[end_train:end_val]
        test_data = data_list[end_val:]

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])


# NOTE: If your raw data is not present, create a dummy file for the code to run.
# For example, create './dataset/raw/USTC.csv' with the correct headers.
if not os.path.exists('./dataset/raw/USTC.csv'):
    os.makedirs('./dataset/raw', exist_ok=True)
    dummy_data = {
        'src_ip': ['1.1.1.1', '2.2.2.2', '3.3.3.3'], 'dst_ip': ['2.2.2.2', '3.3.3.3', '1.1.1.1'],
        'label': [0, 1, 0]
    }
    for i in range(22):
        dummy_data[f'feat_{i}'] = np.random.rand(3)
    pd.DataFrame(dummy_data).to_csv('./dataset/raw/USTC.csv', index=False)

transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
dataset_path = './dataset'
dataset = CustomGraphDataset(root=dataset_path, pre_transform=transform)

train_loader = DataLoader(dataset.train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset.val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)


def permute_within_batch(x, batch):
    unique_batches = torch.unique(batch)
    permuted_indices = []
    for batch_index in unique_batches:
        indices_in_batch = (batch == batch_index).nonzero().squeeze(-1)
        permuted_indices.append(indices_in_batch[torch.randperm(len(indices_in_batch))])
    return torch.cat(permuted_indices)


class GPSConv(torch.nn.Module):
    def __init__(
            self, channels: int, conv: Optional[MessagePassing], heads: int = 1, dropout: float = 0.0,
            attn_dropout: float = 0.0, act: str = 'relu', att_type: str = 'transformer',
            order_by_degree: bool = False, shuffle_ind: int = 0, d_state: int = 16, d_conv: int = 4,
            act_kwargs: Optional[Dict[str, Any]] = None, norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        if self.att_type == 'transformer':
            self.attn = torch.nn.MultiheadAttention(channels, heads, dropout=attn_dropout, batch_first=True)
        if self.att_type == 'mamba.txt':
            self.self_attn = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        self.mlp = Sequential(Linear(channels, channels * 2), activation_resolver(act, **(act_kwargs or {})),
                              Dropout(dropout), Linear(channels * 2, channels), Dropout(dropout))
        self.norm1 = normalization_resolver(norm, channels, **(norm_kwargs or {}))
        self.norm2 = normalization_resolver(norm, channels, **(norm_kwargs or {}))
        self.norm3 = normalization_resolver(norm, channels, **(norm_kwargs or {}))
        self.norm_with_batch = 'batch' in inspect.signature(self.norm1.forward).parameters if self.norm1 else False

    def forward(self, x: Tensor, edge_index: Adj, batch: Optional[torch.Tensor] = None, **kwargs) -> Tensor:
        hs = []
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1:
                h = self.norm1(h, batch=batch) if self.norm_with_batch else self.norm1(h)
            hs.append(h)
        if self.att_type in ['transformer', 'mamba.txt']:
            h_attn, mask = to_dense_batch(x, batch)
            if self.att_type == 'transformer':
                h_attn, _ = self.attn(h_attn, h_attn, h_attn, key_padding_mask=~mask, need_weights=False)
            else:  # mamba.txt
                h_attn = self.self_attn(h_attn)
            h_attn = h_attn[mask]
            h_attn = F.dropout(h_attn, p=self.dropout, training=self.training)
            h_attn = h_attn + x
            if self.norm2:
                h_attn = self.norm2(h_attn, batch=batch) if self.norm_with_batch else self.norm2(h_attn)
            hs.append(h_attn)
        out = sum(hs)
        out = out + self.mlp(out)
        if self.norm3:
            out = self.norm3(out, batch=batch) if self.norm_with_batch else self.norm3(out)
        return out


class EdgeFeatureCompressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class GraphModel(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, model_type: str, shuffle_ind: int, d_state: int,
                 d_conv: int, order_by_degree: bool, num_nodes_total: int):
        super().__init__()
        # Use a larger embedding size based on total unique nodes
        self.node_emb = Embedding(num_nodes_total + 1, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_feature_compressor = EdgeFeatureCompressor(6, 32)  # Input: 6, Output: 32
        self.model_type = model_type
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn_seq = Sequential(Linear(channels, channels), ReLU(), Linear(channels, channels))
            if model_type == 'gine':
                conv = GINEConv(nn_seq)
            elif model_type in ['mamba.txt', 'transformer']:
                conv = GPSConv(channels, GINEConv(nn_seq), heads=4, attn_dropout=0.5,
                               att_type=model_type, shuffle_ind=shuffle_ind,
                               order_by_degree=order_by_degree, d_state=d_state, d_conv=d_conv)
            self.convs.append(conv)
        # Adjust MLP input dimension
        self.mlp = Sequential(
            Linear(channels * 2 + 32, channels),  # 2*node_embed + edge_feature_embed
            ReLU(),
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, 2),  # Output 2 for binary classification
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        # Squeeze dim 1 if it exists
        if x.dim() > 1 and x.shape[1] == 1:
            x = x.squeeze(1)
        node_features = self.node_emb(x)
        x = torch.cat((self.pe_lin(x_pe), node_features), 1)

        edge_attr_processed = self.edge_feature_compressor(edge_attr)

        for conv in self.convs:
            if self.model_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr_processed)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr_processed)

        # Edge-level prediction
        edge_node_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        combined_edge_features = torch.cat([edge_node_features, edge_attr_processed], dim=1)

        logits = self.mlp(combined_edge_features)
        return logits, combined_edge_features


def train(model, loader, optimizer, device):
    model.train()
    total_loss, total_examples = 0, 0
    all_preds, all_labels = [], []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits, _ = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        loss = F.cross_entropy(logits, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(data.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean()

    return total_loss / len(loader.dataset), accuracy.item()


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    total_loss, total_correct, total_edges = 0, 0, 0
    all_preds, all_labels, all_scores = [], [], []
    all_embeddings = []

    for data in loader:
        data = data.to(device)
        logits, edge_embeddings = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        loss = F.cross_entropy(logits, data.y)

        total_loss += loss.item() * data.num_graphs
        preds = logits.argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(data.y.cpu())
        all_scores.append(F.softmax(logits, dim=1)[:, 1].cpu())  # Probability of positive class
        all_embeddings.append(edge_embeddings.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_scores = torch.cat(all_scores).numpy()
    all_embeddings = torch.cat(all_embeddings).numpy()

    accuracy = (all_preds == all_labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    try:
        roc_auc = auc(*roc_curve(all_labels, all_scores)[:2])
    except ValueError:
        roc_auc = 0.5  # Handle case where only one class is present

    metrics = {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'labels': all_labels,
        'preds': all_preds,
        'scores': all_scores,
        'embeddings': all_embeddings
    }
    return metrics


# --- Visualization Functions ---

def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Plot Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Plot F1-Score
    axes[2].plot(history['val_f1'], label='Validation F1-Score')
    axes[2].set_title('F1-Score Over Epochs')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('F1-Score')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_and_roc(test_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion Matrix
    cm = confusion_matrix(test_metrics['labels'], test_metrics['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # ROC Curve
    fpr, tpr, _ = roc_curve(test_metrics['labels'], test_metrics['scores'])
    roc_auc = test_metrics['auc']
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[1].legend(loc="lower right")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_tsne_embeddings(test_metrics):
    print("\nCalculating and plotting t-SNE embeddings... (This may take a moment)")
    # To avoid long computation times, let's use a sample of the embeddings
    n_samples = min(2000, len(test_metrics['embeddings']))
    indices = np.random.choice(len(test_metrics['embeddings']), n_samples, replace=False)

    embeddings_sample = test_metrics['embeddings'][indices]
    labels_sample = test_metrics['labels'][indices]

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings_sample)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_sample, cmap='coolwarm', alpha=0.7)
    plt.title('t-SNE Visualization of Edge Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Benign', 'Malicious'])
    plt.grid(True)
    plt.show()


# --- Main Execution ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Determine total number of unique nodes for embedding layer size
all_ips = set()
df = pd.read_csv(dataset.raw_paths[0])
for ip in pd.concat([df['src_ip'], df['dst_ip']]).unique():
    all_ips.add(ip)
total_num_nodes = len(all_ips)

model = GraphModel(channels=64, pe_dim=8, num_layers=4,  # Reduced layers for faster demo
                   model_type='gine',
                   shuffle_ind=0, order_by_degree=False,
                   d_conv=4, d_state=16,
                   num_nodes_total=total_num_nodes
                   ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

num_epochs = 50  # Reduced for faster demo
history = {
    'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
    'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
}

print("\n--- Starting Training ---")
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, device)
    val_metrics = test(model, val_loader, device)
    scheduler.step(val_metrics['loss'])

    # Log history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_metrics['loss'])
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_precision'].append(val_metrics['precision'])
    history['val_recall'].append(val_metrics['recall'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_auc'].append(val_metrics['auc'])

    print(f"Epoch {epoch:02d}: "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

print("\n--- Training Finished ---")

# --- Final Evaluation and Visualization ---
print("\n--- Evaluating on Test Set ---")
test_metrics = test(model, test_loader, device)
print(f"Test Loss: {test_metrics['loss']:4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")
print(f"Test F1-Score: {test_metrics['f1']:.4f}")
print(f"Test AUC: {test_metrics['auc']:.4f}")

print("\n--- Generating Visualizations ---")
plot_training_history(history)
plot_confusion_matrix_and_roc(test_metrics)
plot_tsne_embeddings(test_metrics)