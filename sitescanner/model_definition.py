# --- Contents for sitescanner/model_definition.py ---


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.nn.models.gate_points_2101 import Network as GatePointsNetwork

class GlobalAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalAttentionLayer, self).__init__()
        # Learnable attention weights
        self.attention = nn.Linear(in_channels, 1)

    def forward(self, x, batch):
        # x: Node features, shape [num_nodes, in_channels]
        # batch: Batch vector, shape [num_nodes]

        # Compute attention weights for each node
        # Sigmoid for 0-1 range
        attn_weights = torch.sigmoid(self.attention(x))
        # attn_weights = F.softmax(self.attention(x), dim=0) # Or softmax

        # Aggregate node features to form a global context vector
        # or max_pool, add_pool
        global_context = pyg_nn.global_mean_pool(x * attn_weights, batch)

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
        ) #

        self.graph_transformer_local = pyg_nn.TransformerConv(hidden_dim, hidden_dim, heads=8, concat=False, dropout=0.1) #
        self.norm_local = nn.LayerNorm(hidden_dim) #

        # **Global Attention Layer for Global Branch (GLOBAL)**
        self.global_attention = GlobalAttentionLayer(hidden_dim, hidden_dim * 2) #
         # Changed due to concatenation in GlobalAttentionLayer
        self.norm_global = nn.LayerNorm(hidden_dim * 2)

        # Adjusted input size
        self.projection_head = FullyConnectedNet([hidden_dim * 2 + hidden_dim, hidden_dim, 128], act=torch.nn.ReLU())
        self.classifier = nn.Linear(128, 1) #

    def forward(self, data):
        edge_attr_proj = self.edge_projection(data.edge_attr) #
        # Equivariant transformation to integrate structural features
        x = self.equiv_transformer({
            'pos': data.pos,
            'x': data.x,
            'edge_attr': edge_attr_proj,
            'batch': data.batch if hasattr(data, 'batch') else None
        }) #

        x_local = self.graph_transformer_local(x, data.edge_index) #
        x_local = self.norm_local(x_local + x) #

        # Global branch with global attention
        x_global = self.global_attention(x, data.batch) #
        x_global = self.norm_global(x_global) #

        # Concatenate both scales and pass through projection head
        x_concat = torch.cat([x_local, x_global], dim=-1) #
        proj = self.projection_head(x_concat) #
        out = self.classifier(proj) #
        return out # Return only the classification output for inference

# --- End of sitescanner/model_definition.py ---