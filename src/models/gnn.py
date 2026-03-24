"""
gnn.py
Graph Autoencoder (GAE) y Variational Graph Autoencoder (VGAE)
para detección de anomalías mediante reconstrucción de aristas.

Requiere: torch, torch_geometric
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE


class GCNEncoder(torch.nn.Module):
    """Encoder GCN para GAE/VGAE."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    """Encoder variacional para VGAE (produce mu y log_std)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def build_gae(in_channels: int, out_channels: int = 64) -> GAE:
    """Instancia un Graph Autoencoder."""
    encoder = GCNEncoder(in_channels, out_channels)
    return GAE(encoder)


def build_vgae(in_channels: int, out_channels: int = 64) -> VGAE:
    """Instancia un Variational Graph Autoencoder."""
    encoder = VariationalGCNEncoder(in_channels, out_channels)
    return VGAE(encoder)


def train_epoch(model, optimizer, data):
    """Entrena un epoch del GAE/VGAE."""
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    if hasattr(model, "kl_loss"):
        loss += (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def get_anomaly_scores(model, data) -> torch.Tensor:
    """
    Calcula el score de anomalía de cada nodo como el error de reconstrucción.
    Un score alto indica mayor probabilidad de ser anómalo.
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        # Error de reconstrucción por nodo (norma del embedding)
        recon = torch.sigmoid(torch.mm(z, z.t()))
        scores = (recon.sum(dim=1) - recon.diag())  # activación media de vecinos
    return scores
