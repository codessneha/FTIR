"""
STEP 3 — model.py
==================
GNN model: molecular graph -> real FTIR spectrum

No torch-scatter or torch-sparse required.
Uses GATConv from torch_geometric (built-in).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class FTIRNet(nn.Module):
    """
    Input  : molecular graph (atoms + bonds + charges)
    Output : real FTIR absorbance spectrum  [B, spectrum_bins]
    """

    def __init__(self,
                 node_dim   = 10,
                 edge_dim   = 6,
                 hidden_dim = 128,
                 num_layers = 3,
                 out_dim    = 500,
                 dropout    = 0.2):
        super().__init__()

        # Project atom features -> hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Graph attention layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        heads = 4
        for _ in range(num_layers):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // heads,
                        heads=heads, edge_dim=edge_dim,
                        dropout=dropout, add_self_loops=True, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Decoder: molecular embedding -> spectrum
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(512, out_dim),
            nn.Sigmoid(),   # output in [0,1] — same as normalised absorbance
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.silu(norm(conv(h, edge_index, edge_attr=edge_attr)))
        h_mol = torch.cat([global_mean_pool(h, batch),
                           global_max_pool(h, batch)], dim=-1)
        return self.decoder(h_mol)


class FTIRLoss(nn.Module):
    """
    Loss function for real FTIR spectrum prediction.

    Combines:
      1. MSE           — absolute intensity accuracy
      2. Cosine loss   — spectral shape accuracy (peak positions)

    For NH3/graphene systems, no region weighting needed
    since the Excel already covers the correct range.
    """

    def __init__(self, cosine_w=0.4):
        super().__init__()
        self.cosine_w = cosine_w

    def forward(self, pred, target):
        mse      = F.mse_loss(pred, target)
        cos_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        return mse + self.cosine_w * cos_loss


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = FTIRNet(node_dim=10, edge_dim=6, hidden_dim=64,
                    num_layers=2, out_dim=500)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x          = torch.randn(15, 10)
    edge_index = torch.randint(0, 15, (2, 30))
    edge_attr  = torch.randn(30, 6)
    batch      = torch.tensor([0]*8 + [1]*7)

    out = model(x, edge_index, edge_attr, batch)
    print(f"Output shape  : {out.shape}")       # [2, 500]
    print(f"Output range  : [{out.min():.3f}, {out.max():.3f}]")
    print(f"Parameters    : {n_p:,}")
    print("Model OK!")