"""
STEP 4 — train.py
==================
K-Fold cross-validation training for small datasets.

Usage:
    python train.py --graph_dir data/graphs --out_dir checkpoints --epochs 500
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from pathlib import Path
import csv

from model import FTIRNet, FTIRLoss


def load_graphs(graph_dir):
    graph_dir = Path(graph_dir)
    files = sorted(graph_dir.glob('graph_*.pt'))
    if not files:
        raise FileNotFoundError(f"No graph_*.pt files found in {graph_dir}")
    data = [torch.load(f, weights_only=False) for f in files]
    cfg  = json.load(open(graph_dir / 'config.json'))
    print(f"Loaded {len(data)} graphs")
    print(f"Structures: {cfg['names']}")
    return data, cfg


def augment(data, noise=0.02):
    """Add small noise to continuous atom features only."""
    d = data.clone()
    n = torch.randn_like(d.x) * noise
    n[:, 3:9] = 0.0      # don't perturb binary/flag features
    d.x = d.x + n
    return d


def expand(data_list, copies=4):
    out = list(data_list)
    for c in range(copies):
        out += [augment(d, noise=0.01*(c+1)) for d in data_list]
    print(f"  Augmented: {len(data_list)} -> {len(out)} samples")
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for b in loader:
        b   = b.to(device)
        out = model(b.x, b.edge_index, b.edge_attr, b.batch)
        preds.append(out.cpu().numpy())
        targets.append(b.y.squeeze(1).cpu().numpy())

    P = np.concatenate(preds)
    T = np.concatenate(targets)

    rmse = float(np.sqrt(np.mean((P-T)**2)))
    mae  = float(np.mean(np.abs(P-T)))
    rs   = [np.corrcoef(p,t)[0,1] for p,t in zip(P,T) if np.std(t)>0]
    r    = float(np.mean(rs)) if rs else 0.0
    dot  = np.sum(P*T, axis=1)
    norm = np.linalg.norm(P,axis=1)*np.linalg.norm(T,axis=1)+1e-8
    sam  = float(np.degrees(np.mean(np.arccos(np.clip(dot/norm,-1,1)))))

    return {'rmse':rmse, 'mae':mae, 'pearson_r':r, 'sam_deg':sam}


def train_fold(fold, train_data, val_data, cfg, args, device):
    print(f"\n  Fold {fold+1}  |  train={len(train_data)}  val={len(val_data)}")
    print(f"  Val structures: {[d.name for d in val_data]}")

    aug_train    = expand(train_data, copies=args.aug_copies)
    train_loader = DataLoader(aug_train, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,  batch_size=args.batch_size, shuffle=False)

    model = FTIRNet(
        node_dim   = cfg['node_feat_dim'],
        edge_dim   = cfg['edge_feat_dim'],
        hidden_dim = 64,
        num_layers = 2,
        out_dim    = cfg['spectrum_bins'],
        dropout    = 0.30,
    ).to(device)

    loss_fn   = FTIRLoss(cosine_w=0.4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-6
    )

    ckpt_path = os.path.join(args.out_dir, f'fold{fold+1}_best.pt')
    log_path  = os.path.join(args.out_dir, f'fold{fold+1}_log.csv')

    with open(log_path,'w',newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','val_loss','val_mae'])

    best_val = float('inf')
    patience = 0

    for epoch in range(1, args.epochs+1):

        # train
        model.train()
        total = 0.0
        for b in train_loader:
            b = b.to(device)
            optimizer.zero_grad()
            out  = model(b.x, b.edge_index, b.edge_attr, b.batch)
            loss = loss_fn(out, b.y.squeeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)

        # validate
        model.eval()
        vtotal = 0.0
        vmae   = 0.0
        with torch.no_grad():
            for b in val_loader:
                b   = b.to(device)
                out = model(b.x, b.edge_index, b.edge_attr, b.batch)
                vtotal += loss_fn(out, b.y.squeeze(1)).item()
                vmae   += torch.mean(torch.abs(out - b.y.squeeze(1))).item()
        val_loss = vtotal / len(val_loader)
        val_mae  = vmae   / len(val_loader)

        scheduler.step(epoch)

        with open(log_path,'a',newline='') as f:
            csv.writer(f).writerow([epoch,f'{train_loss:.5f}',
                                    f'{val_loss:.5f}',f'{val_mae:.5f}'])

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({'epoch':epoch,'state':model.state_dict(),
                        'val_loss':val_loss,'cfg':cfg}, ckpt_path)
            marker = ' <-- saved'
        else:
            patience += 1
            marker = ''

        if epoch % 50 == 0 or epoch <= 3:
            print(f"    ep {epoch:4d}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  mae={val_mae:.4f}{marker}")

        if patience >= args.patience:
            print(f"    Early stop at epoch {epoch}")
            break

    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['state'])
    m = evaluate(model, val_loader, device)
    print(f"    Best -> RMSE={m['rmse']:.4f}  r={m['pearson_r']:.3f}  SAM={m['sam_deg']:.1f}deg")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph_dir',  default='data/graphs')
    ap.add_argument('--out_dir',    default='checkpoints')
    ap.add_argument('--epochs',     type=int,   default=500)
    ap.add_argument('--folds',      type=int,   default=5)
    ap.add_argument('--batch_size', type=int,   default=8)
    ap.add_argument('--lr',         type=float, default=1e-4)
    ap.add_argument('--patience',   type=int,   default=80)
    ap.add_argument('--aug_copies', type=int,   default=4)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    data, cfg = load_graphs(args.graph_dir)

    n = len(data)
    # For very small datasets use leave-one-out CV
    if n <= args.folds:
        args.folds = n
        print(f"Small dataset -> using {n}-fold (leave-one-out) CV")
    elif n < 30:
        args.folds = min(args.folds, 3)
        print(f"Small dataset -> using {args.folds}-fold CV")

    print(f"\n{args.folds}-Fold CV  |  {n} structures  |  {args.epochs} epochs max")
    print('-'*50)

    kf      = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    results = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(n))):
        train_data = [data[i] for i in tr_idx]
        val_data   = [data[i] for i in val_idx]
        m = train_fold(fold, train_data, val_data, cfg, args, device)
        results.append(m)

    print(f"\n{'='*50}")
    print(f"  {args.folds}-FOLD CV RESULTS")
    print('-'*50)
    for k in ['rmse','mae','pearson_r','sam_deg']:
        vals = [r[k] for r in results]
        print(f"  {k:<14}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
    print('='*50)

    best_fold = int(np.argmin([r['rmse'] for r in results]))
    print(f"\n  Best fold : {best_fold+1}")
    print(f"  Checkpoint: checkpoints/fold{best_fold+1}_best.pt\n")

    summary = {
        'per_fold':  results,
        'mean':      {k:float(np.mean([r[k] for r in results])) for k in results[0]},
        'std':       {k:float(np.std( [r[k] for r in results])) for k in results[0]},
        'best_fold': best_fold+1,
    }
    json.dump(summary, open(os.path.join(args.out_dir,'kfold_summary.json'),'w'), indent=2)
    print(f"Results saved -> {args.out_dir}/kfold_summary.json")


if __name__ == '__main__':
    main()