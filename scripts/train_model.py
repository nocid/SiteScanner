# This script is a summary of what was used to train the model
# It is not meant to be run as a script, but rather to be used as a reference

import os
import torch_geometric
from torch_geometric.data import Data
from Bio import PDB
import torch
import numpy as np
from scipy.spatial import distance_matrix, KDTree
from Bio.PDB import PDBParser, PPBuilder, SASA, Polypeptide, ShrakeRupley, is_aa
import numpy as np
import math
from Bio.PDB import PICIO, PDBIO
import esm
from typing import TypedDict, Dict, Tuple
import time
import re
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, radius_graph
from e3nn import o3
from e3nn.nn import FullyConnectedNet
# import pymol
from e3nn.nn.models.gate_points_2101 import Network as GatePointsNetwork
from joblib import Parallel, delayed
from torch_geometric.nn import radius
from multiprocessing import Pool
import torch_geometric.nn as pyg_nn

class GlobalAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalAttentionLayer, self).__init__()
        self.attention = nn.Linear(in_channels, 1) # Learnable attention weights

    def forward(self, x, batch):
        # x: Node features, shape [num_nodes, in_channels]
        # batch: Batch vector, shape [num_nodes]

        # Compute attention weights for each node
        attn_weights = torch.sigmoid(self.attention(x))  # Sigmoid for 0-1 range
        # attn_weights = F.softmax(self.attention(x), dim=0) # Or softmax

        # Aggregate node features to form a global context vector
        global_context = pyg_nn.global_mean_pool(x * attn_weights, batch)  # or max_pool, add_pool

        # Apply attention, e.g., by concatenating with original node features
        x_attended = torch.cat([x, global_context[batch]], dim=-1)

        return x_attended

class MultiScaleEquivariantResidualNet(nn.Module):
    def __init__(self, node_dim=23, edge_dim=19, hidden_dim=64, num_layers=4):
        super().__init__()
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.edge_projection = nn.Linear(edge_dim, hidden_dim)

        self.equiv_transformer = GatePointsNetwork(
            irreps_in=o3.Irreps(f"{node_dim}x0e") if node_dim else None,
            irreps_hidden=o3.Irreps([(hidden_dim, (0, 1))]),
            irreps_out=o3.Irreps(f"{hidden_dim}x0e"),
            irreps_node_attr=o3.Irreps("0e"),
            irreps_edge_attr=o3.Irreps(f"{hidden_dim}x0e"),
            layers=num_layers,
            max_radius=5.0,
            number_of_basis=16,
            radial_layers=2,
            radial_neurons=hidden_dim,
            num_neighbors=12.0,
            num_nodes=100.0,
            reduce_output=False
        )

        self.graph_transformer_local = TransformerConv(hidden_dim, hidden_dim, heads=8, concat=False, dropout=0.1)
        self.norm_local = nn.LayerNorm(hidden_dim)

        # **Global Attention Layer for Global Branch (GLOBAL)**
        self.global_attention = GlobalAttentionLayer(hidden_dim, hidden_dim * 2)
        self.norm_global = nn.LayerNorm(hidden_dim * 2) # Changed due to concatenation in GlobalAttentionLayer

        self.projection_head = FullyConnectedNet([hidden_dim * 2 + hidden_dim, hidden_dim, 128], act=torch.nn.ReLU())  # Adjusted input size
        self.classifier = nn.Linear(128, 1)

    def forward(self, data):
        edge_attr_proj = self.edge_projection(data.edge_attr)
        # Equivariant transformation to integrate structural features
        x = self.equiv_transformer({
            'pos': data.pos,
            'x': data.x,
            'edge_attr': edge_attr_proj,
            'batch': data.batch if hasattr(data, 'batch') else None
        })

        x_local = self.graph_transformer_local(x, data.edge_index)
        x_local = self.norm_local(x_local + x)

        # Global branch with global attention
        x_global = self.global_attention(x, data.batch)
        x_global = self.norm_global(x_global)

        # Concatenate both scales and pass through projection head
        x_concat = torch.cat([x_local, x_global], dim=-1)
        proj = self.projection_head(x_concat)
        out = self.classifier(proj)
        return out, proj



# Unpack results for different uses
dataset = [complex[0] for complex in all_complexes]  # Labeled graphs
binding_info = [complex[1] for complex in all_complexes]  # Original binding residues
ligand_files = [(complex[2], complex[3]) for complex in all_complexes]  # Mol2/SDF paths


dataset = [g for g in dataset if g is not None]  # Remove None entries

# 4. Split dataset
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# 5. Create data loaders
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, follow_batch=['x'])
test_loader = DataLoader(test_data, batch_size=4, follow_batch=['x'])

from sklearn.metrics import f1_score, roc_auc_score

# 7. Training setup
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(5))  # Binary classification, is a given residue part of a binding site or not?
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, val_accuracies = [], []

# 8. Training loop
for epoch in range(150):
    model.train()
    total_loss = 0
    for batch in train_loader:

        optimizer.zero_grad()
        # print(f"Batch shapes: x={batch.x.shape}, edge_index={batch.edge_index.shape}, edge_attr={batch.edge_attr.shape}")
        # print(f"Max node index: {batch.x.size(0)-1}, Max edge index: {batch.edge_index.max().item()}")
        assert batch.edge_index.max().item() < batch.x.size(0), "Invalid edge index!"
        try:
            out, _ = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        except RuntimeError as e:
            print(f"Error occurred: {e}")
            print(f"Batch data: {batch}")
            raise e
        # out, _ = model(batch)  # We only need the classification output
        # loss = criterion(out, batch.y)
        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            out, _ = model(batch)
            preds = torch.sigmoid(out) > 0.5
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)

    train_losses.append(total_loss/len(train_loader))
    val_accuracies.append(correct/total)

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, "
          f"Val Acc: {correct/total:.4f}")

torch.save(model.state_dict(), BASE_FOLDER + 'model_weights_3.pth')


