import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphSAGE
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pandas as pd
from torch_geometric.data import Data
import time

def load_data(path):
    df = pd.read_csv(path)

    # IP转为索引
    ip_list = pd.concat([df['src_ip'], df['dst_ip']]).unique()
    ip2idx = {ip: idx for idx, ip in enumerate(ip_list)}

    src = df['src_ip'].map(ip2idx).to_numpy()
    dst = df['dst_ip'].map(ip2idx).to_numpy()

    edge_features = torch.tensor(df.iloc[:, 3:33].values, dtype=torch.float32)
    edge_labels = torch.tensor(df['label'].values, dtype=torch.long)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    num_nodes = len(ip2idx)
    node_features = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for s in src:
        node_features[s, 0] += 1  # 出度
    for d in dst:
        node_features[d, 1] += 1  # 入度

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        edge_labels=edge_labels
    )

    return data

class GNNTransformerEdgeClassifier(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, num_classes):
        super(GNNTransformerEdgeClassifier, self).__init__()
        self.gnn_encoder = GraphSAGE(node_feat_dim, hidden_dim, num_layers=2)
        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=4, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)
        self.edge_classifier = torch.nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_pairs):
        x = self.gnn_encoder(x, edge_index)
        x_transformed = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)
        edge_repr = torch.cat([x_transformed[edge_pairs[0]], x_transformed[edge_pairs[1]]], dim=1)
        return self.edge_classifier(edge_repr)

data = load_data('./dataset/raw/Ton.csv')

# 划分数据集
num_edges = data.edge_index.shape[1]
indices = torch.randperm(num_edges)
train_idx = indices[:int(0.7 * num_edges)]
val_idx = indices[int(0.7 * num_edges):int(0.85 * num_edges)]
test_idx = indices[int(0.85 * num_edges):]

model = GNNTransformerEdgeClassifier(
    node_feat_dim=data.x.shape[1],
    hidden_dim=64,
    num_classes=int(data.edge_labels.max().item()) + 1
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, data, idx):
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_index[:, idx])
    labels = data.edge_labels[idx]
    pred = logits.argmax(dim=1)
    acc = (pred == labels).sum().item() / labels.size(0)
    return acc

total_time = 0

# 训练并评估
for epoch in range(1, 21):
    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_index[:, train_idx])
    loss = F.cross_entropy(logits, data.edge_labels[train_idx])
    loss.backward()
    optimizer.step()
    end_time = time.time()
    epoch_time = end_time - start_time
    total_time += epoch_time
    train_acc = evaluate(model, data, train_idx)
    val_acc = evaluate(model, data, val_idx)
    test_acc = evaluate(model, data, test_idx)
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {epoch_time:.4f} seconds'")

print(f'Total training time for 20 epochs {total_time:.4f} seconds')