"""
STEP 3 — model.py  (improved)
================================
Changes vs original:
  1. GlobalFeatureNet: HOMO/LUMO/dipole/ZPE injected into the decoder
     via FiLM-style conditioning (scale + shift) — lets the model use
     electronic structure to modulate spectral predictions.
  2. Deeper architecture: 3 GAT layers, hidden_dim=128 by default.
  3. FTIRLoss gains a spectral-gradient term (derivative of spectrum) that
     sharpens peak positions without over-smoothing.
  4. Batch-level gradient clipping remains; no other training changes needed here.
  5. Consistent node_dim=15, edge_dim=8 defaults.

No new dependencies — pure torch_geometric.

NOTE: fixed two bugs from the original draft:
  - `mol_dim` was used in the decoder but never defined -> set to 2*hidden_dim
    (mean_pool + max_pool concatenation).
  - `self.global_cond` was referenced in forward() but never created in
    __init__ -> now built as a GlobalConditioner when global_dim > 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d


class GlobalConditioner(nn.Module):
    """
    Takes a global feature vector u (e.g. HOMO/LUMO gap, dipole)
    and produces (scale, shift) to apply FiLM conditioning on an
    intermediate representation.
    """
    def __init__(self, global_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # -> scale + shift
        )

    def forward(self, u):
        out = self.net(u)
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift


class FTIRNet(nn.Module):
    """
    Input  : molecular graph (atoms + bonds) + global molecular descriptors
    Output : real FTIR absorbance spectrum  [B, spectrum_bins]

    Architecture:
      1. Atom feature projection -> hidden_dim
      2. num_layers x GATConv (TransformerConv) with residual + LayerNorm/BatchNorm
      3. Readout: cat(mean_pool, max_pool) -> 2*hidden_dim
      4. FiLM conditioning with global descriptors (HOMO/LUMO/dipole/…)
      5. MLP decoder -> spectrum
    """

    def __init__(self,
                 node_dim    = 15,
                 edge_dim    = 8,
                 global_dim  = 9,
                 hidden_dim  = 256,
                 num_layers  = 3,
                 out_dim     = 500,
                 dropout     = 0.5):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        heads = 4
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(hidden_dim, hidden_dim // heads,
                                heads=heads, edge_dim=edge_dim,
                                dropout=dropout, concat=True)
            )
            self.norms.append(BatchNorm1d(hidden_dim))

        # Readout = cat(mean_pool, max_pool) -> 2 * hidden_dim
        mol_dim = hidden_dim * 2

        # Global descriptor conditioning (FiLM). Only built if we actually
        # have global features to condition on.
        if global_dim and global_dim > 0:
            self.global_cond = GlobalConditioner(global_dim, mol_dim)
        else:
            self.global_cond = None

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(512, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr, batch, u=None):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.silu(norm(conv(h, edge_index, edge_attr=edge_attr)))

        h_mol = torch.cat([global_mean_pool(h, batch),
                           global_max_pool(h, batch)], dim=-1)  # [B, 2*hidden]

        # FiLM conditioning with global descriptors.
        # u shape after Batch.from_data_list: [B, global_dim] (stored as [1,9] per graph)
        # Defensive reshape in case an older .pt file stored u as [global_dim].
        if self.global_cond is not None and u is not None:
            if u.dim() == 1:
                u = u.view(h_mol.shape[0], -1)
            scale, shift = self.global_cond(u)
            h_mol = h_mol * (1.0 + scale) + shift

        return self.decoder(h_mol)


class FTIRLoss(nn.Module):
    """
    Combined loss for real FTIR spectrum prediction:

      L = MSE + cosine_w * (1 - cosine_sim) + grad_w * gradient_MSE

    - MSE           : absolute intensity accuracy
    - Cosine loss   : overall spectral shape / peak ratios
    - Gradient loss : first-difference of spectrum — penalises peak
                      position errors and encourages sharp features
                      (particularly useful for organometallic spectra
                      with narrow carbonyl / aromatic peaks)
    """

    def __init__(self, cosine_w=0.4, grad_w=0.35):
        super().__init__()
        self.cosine_w = cosine_w
        self.grad_w   = grad_w

    def forward(self, pred, target):
        mse      = F.mse_loss(pred, target)
        cos_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()

        # Spectral gradient (finite difference along wavenumber axis)
        pred_grad   = pred[:, 1:] - pred[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]
        grad_loss   = F.mse_loss(pred_grad, target_grad)

        return mse + self.cosine_w * cos_loss + self.grad_w * grad_loss


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = FTIRNet(node_dim=15, edge_dim=8, global_dim=9,
                    hidden_dim=256, num_layers=3, out_dim=500)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x          = torch.randn(20, 15)
    edge_index = torch.randint(0, 20, (2, 40))
    edge_attr  = torch.randn(40, 8)
    batch      = torch.tensor([0]*10 + [1]*10)
    u          = torch.randn(2, 9)       # global features per molecule

    out = model(x, edge_index, edge_attr, batch, u=u)
    print(f"Output shape  : {out.shape}")       # [2, 500]
    print(f"Output range  : [{out.min():.3f}, {out.max():.3f}]")
    print(f"Parameters    : {n_p:,}")
    print("Model OK!")