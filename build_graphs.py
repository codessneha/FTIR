"""
STEP 2 — build_graphs.py
=========================
Converts dataset.json into PyTorch Geometric graph files.

Each graph file contains:
  - x          : atom features      [n_atoms, 10]
  - edge_index : bond connectivity   [2, n_edges]
  - edge_attr  : bond features       [n_edges, 6]
  - y          : real FTIR spectrum  [1, N_BINS]  ← target to predict

Usage:
    python build_graphs.py --input data/dataset.json --output_dir data/graphs
"""

import json, argparse, os
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path


def build_graph(entry):
    x          = torch.tensor(entry['node_features'], dtype=torch.float)
    if not entry['edge_index']:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(entry['edge_index'],    dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(entry['edge_features'], dtype=torch.float)

    # Target = real FTIR spectrum from Excel
    y = torch.tensor(entry['ftir_spectrum'], dtype=torch.float).unsqueeze(0)  # [1, N_BINS]

    return Data(
        x         = x,
        edge_index= edge_index,
        edge_attr = edge_attr,
        y         = y,
        name      = entry['name'],
        n_atoms   = entry['n_atoms'],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',      default='data/dataset.json')
    ap.add_argument('--output_dir', default='data/graphs')
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    raw = json.load(open(args.input))

    print(f"\nBuilding {len(raw)} graphs ...")

    for i, entry in enumerate(raw):
        graph = build_graph(entry)
        torch.save(graph, os.path.join(args.output_dir, f'graph_{i:04d}.pt'))

    # Save config for training script
    config = {
        'n_samples':     len(raw),
        'node_feat_dim': len(raw[0]['node_features'][0]),   # 10
        'edge_feat_dim': len(raw[0]['edge_features'][0]),   # 6
        'spectrum_bins': len(raw[0]['ftir_spectrum']),      # 500
        'freq_min':      raw[0]['freq_min'],
        'freq_max':      raw[0]['freq_max'],
        'names':         [e['name'] for e in raw],
    }
    json.dump(config, open(os.path.join(args.output_dir,'config.json'),'w'), indent=2)

    print(f"Saved {len(raw)} graphs -> {args.output_dir}/")
    print(f"\nConfig:")
    print(f"  Node features  : {config['node_feat_dim']}")
    print(f"  Edge features  : {config['edge_feat_dim']}")
    print(f"  Spectrum bins  : {config['spectrum_bins']}  (real FTIR target)")
    print(f"  Freq range     : {config['freq_min']} - {config['freq_max']} cm-1\n")


if __name__ == '__main__':
    main()