import torch
import torch.nn as nn
import os
import json
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid, Reddit, QM9, PPI
from torch_geometric.utils import subgraph, mask_to_index
from sklearn.metrics import roc_auc_score
import numpy as np

# ---------------------- Model Definitions ----------------------
class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index, batch=None):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
    def forward(self, x, edge_index, batch=None):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class SAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    def forward(self, x, edge_index, batch=None):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# ---------------------- Helper for Link Prediction ----------------------
def predict_edges(out, edge_index):
    src, dst = edge_index
    edge_scores = (out[src] * out[dst]).sum(dim=1)
    return torch.sigmoid(edge_scores)

# ---------------------- Dataset Loader ----------------------
def load_dataset(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(root='./data', name='Cora')
        return dataset[0]
    elif dataset_name == 'Reddit':
        dataset = Reddit(root='./data/Reddit')
        return dataset[0]
    elif dataset_name == 'QM9':
        dataset = QM9(root='./data/QM9')
        return dataset
    elif dataset_name == 'Facebook':
        dataset = PPI(root='./data/Facebook', split='train')
        return dataset[0]

# ---------------------- Data Splitter ----------------------
def split_data(data, train_ratio=0.8, task='classification'):
    if task in ['classification', 'link']:
        num_nodes = data.x.size(0)
        perm = torch.randperm(num_nodes)
        train_size = int(num_nodes * train_ratio)
        train_mask = perm[:train_size]
        test_mask = perm[train_size:]

        train_nodes = mask_to_index(train_mask)
        test_nodes = mask_to_index(test_mask)

        train_edge_index, _ = subgraph(train_nodes, data.edge_index, relabel_nodes=True)
        test_edge_index, _ = subgraph(test_nodes, data.edge_index, relabel_nodes=True)

        train_data = Data(x=data.x[train_nodes], edge_index=train_edge_index, y=data.y[train_nodes])
        test_data = Data(x=data.x[test_nodes], edge_index=test_edge_index, y=data.y[test_nodes])
    else:
        num_graphs = len(data)
        perm = torch.randperm(num_graphs)
        train_size = int(num_graphs * train_ratio)
        train_data = [data[i] for i in perm[:train_size]]
        test_data = [data[i] for i in perm[train_size:]]
    return train_data, test_data

# ---------------------- Training Function ----------------------
def train_gnn(dataset_name, model_type, n, task, num_epochs=200, lr=0.01):
    data = load_dataset(dataset_name)
    train_data, test_data = split_data(data, task=task)

    if task == 'regression':
        train_loader = DataLoader(train_data[:min(n, len(train_data))], batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)
    else:
        idx = torch.randperm(train_data.x.size(0))[:n]
        edge_index, _ = subgraph(idx, train_data.edge_index, relabel_nodes=True)
        train_data = Data(x=train_data.x[idx], edge_index=edge_index, y=train_data.y[idx])

        train_loader = DataLoader([train_data], batch_size=32)
        test_loader = DataLoader([test_data], batch_size=32)

    in_channels = data.num_features
    if task != 'regression':
        out_channels = int(data.y.max().item()) + 1
    else:
        out_channels = 1

    if model_type == 'GCN':
        model = GCNModel(in_channels, 16, out_channels)
    elif model_type == 'GAT':
        model = GATModel(in_channels, 8, 8, out_channels)
    else:
        model = SAGEModel(in_channels, 16, out_channels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()

    # Training
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch_attr = getattr(batch, 'batch', None)
            out = model(batch.x, batch.edge_index, batch_attr if task == 'regression' else None)
            y = batch.edge_index[1] if task == 'link' else batch.y
            if task == 'link':
                out = predict_edges(out, batch.edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    test_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_attr = getattr(batch, 'batch', None)
            out = model(batch.x, batch.edge_index, batch_attr if task == 'regression' else None)
            if task == 'link':
                out = predict_edges(out, batch.edge_index)
                y = batch.edge_index[1]
                try:
                    test_error += roc_auc_score(y.cpu().numpy(), out.cpu().numpy())
                except ValueError:
                    test_error += 0.5
            elif task == 'regression':
                test_error += criterion(out, batch.y).item()
            else:
                pred = out.argmax(dim=1)
                test_error += (pred != batch.y).float().mean().item()
    test_error /= len(test_loader)
    return test_error

# ---------------------- Load or Initialize Results ----------------------
RESULTS_FILE = "results.json"
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
else:
    results = {}

# ---------------------- Experiment Loop ----------------------
for dataset_name, task in [('Cora', 'classification'), ('Reddit', 'classification'),
                           ('QM9', 'regression'), ('Facebook', 'link')]:
    if dataset_name not in results:
        results[dataset_name] = {}

    for model_type in ['GCN', 'GAT', 'GraphSAGE']:
        if model_type not in results[dataset_name]:
            results[dataset_name][model_type] = {}

        n_values = [100, 500, 1000, 5000, 10000, 50000]
        for n in n_values:
            if (dataset_name == 'Cora' and n > 2708) or (dataset_name == 'Facebook' and n > 4039):
                continue

            if str(n) in results[dataset_name][model_type]:
                print(f"Skipping {dataset_name}, {model_type}, n={n} (already computed)")
                continue

            errors = []
            for _ in range(5):
                error = train_gnn(dataset_name, model_type, n, task)
                errors.append(error)
            mean_error = np.mean(errors)
            std_error = np.std(errors)

            results[dataset_name][model_type][str(n)] = [mean_error, std_error]
            print(f'{dataset_name}, {model_type}, n={n}, error={mean_error:.4f} ± {std_error:.4f}')

            # Save after each config
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)
