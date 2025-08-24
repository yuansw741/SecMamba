import torch
import torch.nn as nn
import torch.nn.functional as F
# import dgl
from dgl import graph as dgl_graph
import dgl.nn as dglnn
import pandas as pd
import numpy as np
import time  # 引入time模块

# 加载数据
def load_data(path):
    df = pd.read_csv(path)
    src = df['src_ip'].to_numpy()
    dst = df['dst_ip'].to_numpy()
    edge_features = torch.tensor(df.iloc[:, 4:34].values, dtype=torch.float32)  # 取特征列
    edge_labels = torch.tensor(df['label'].values, dtype=torch.float32)  # 取标签列
    g = dgl_graph.graph((src, dst))
    g.edata['feature'] = edge_features
    g.edata['label'] = edge_labels
    return g

# 定义GraphSAGE模型
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, 'mean')
        self.conv2 = dglnn.SAGEConv(hid_feats, out_feats, 'mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

# 定义边预测模块
class EdgePredictor(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.linear = nn.Linear(2 * in_feats, 1)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.linear(x)
        return {'score': y}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

# 完整模型
class CompleteModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = GraphSAGE(in_features, hidden_features, out_features)
        self.pred = EdgePredictor(out_features)

    def forward(self, g, x):
        node_h = self.sage(g, x)
        return self.pred(g, node_h)

# 训练函数
def train(model, g, features, labels, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    total_start_time = time.time()  # 记录整个训练过程的开始时间

    for epoch in range(epochs):
        epoch_start_time = time.time()  # 记录单个epoch的开始时间
        logits = model(g, features)
        loss = loss_fn(logits.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_end_time = time.time()  # 记录单个epoch的结束时间
        epoch_time = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Epoch Time: {epoch_time:.2f}s')

    total_end_time = time.time()  # 记录整个训练过程的结束时间
    total_time = total_end_time - total_start_time
    print(f'Total training time: {total_time:.2f}s')

# 加载数据并训练模型
g = load_data(r'../dataset/raw/Ton2.csv')
features = g.edata['feature']
labels = g.edata['label']
model = CompleteModel(30, 16, 8)  # Adjust these parameters as needed
train(model, g, features, labels)
