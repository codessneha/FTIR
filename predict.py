"""
STEP 5 — predict.py
====================
Load trained model + a .log file -> predict real FTIR spectrum -> plot it.

Usage:
    # Predict all structures in dataset:
    python predict.py --checkpoint checkpoints/fold1_best.pt \
                      --dataset    data/dataset.json \
                      --output_dir results/

    # Predict one new .log file:
    python predict.py --checkpoint checkpoints/fold1_best.pt \
                      --log_file   data/logs/new_structure.log \
                      --output_dir results/
"""

import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

from model import FTIRNet


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt['cfg']
    model = FTIRNet(
        node_dim   = cfg['node_feat_dim'],
        edge_dim   = cfg['edge_feat_dim'],
        hidden_dim = 64,
        num_layers = 2,
        out_dim    = cfg['spectrum_bins'],
        dropout    = 0.0,
    ).to(device)
    model.load_state_dict(ckpt['state'])
    model.eval()
    print(f"Loaded model (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model, cfg


def entry_to_graph(entry):
    """Convert a dataset entry dict to a PyG graph."""
    import torch
    from torch_geometric.data import Data
    x          = torch.tensor(entry['node_features'], dtype=torch.float)
    if not entry['edge_index']:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(entry['edge_index'],    dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(entry['edge_features'], dtype=torch.float)
    y          = torch.tensor(entry['ftir_spectrum'], dtype=torch.float).unsqueeze(0)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                name=entry['name'])


@torch.no_grad()
def predict_entry(model, entry, device):
    """Predict FTIR spectrum for one dataset entry."""
    from torch_geometric.data import Batch
    graph = entry_to_graph(entry)
    batch = Batch.from_data_list([graph]).to(device)
    out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    return out.squeeze().cpu().numpy()


def parse_log_for_prediction(log_path):
    """
    Parse a NEW .log file that has no matching Excel.
    Uses molecular graph only (no FTIR target needed for prediction).
    """
    import re
    import numpy as np

    ATOMIC_NUM = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'S':16,'Cl':17,'Br':35,'I':53,'Zn':30,'Pt':78,'Co':27}
    ELECTRO    = {'H':2.20,'C':2.55,'N':3.04,'O':3.44,'F':3.98,
                  'Si':1.90,'S':2.58,'Cl':3.16,'Br':2.96,'I':2.66,
                  'Zn':1.65,'Pt':2.28,'Co':1.88}
    COV_RAD    = {'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,
                  'Si':1.11,'S':1.05,'Cl':1.02,'Br':1.20,'I':1.39,
                  'Zn':1.22,'Pt':1.36,'Co':1.26}

    text = Path(log_path).read_text(errors='ignore')

    # Coordinates
    atoms = None
    for pat in [r"Standard orientation:.*?[-]{10,}\n.*?[-]{10,}\n(.*?)\n\s*[-]{10,}",
                r"Input orientation:.*?[-]{10,}\n.*?[-]{10,}\n(.*?)\n\s*[-]{10,}"]:
        blocks = re.findall(pat, text, re.DOTALL)
        if blocks:
            atoms = []
            for line in blocks[-1].strip().split('\n'):
                p = line.split()
                if len(p) >= 6:
                    try:
                        anum = int(p[1])
                        sym  = next((k for k,v in ATOMIC_NUM.items() if v==anum), f'X{anum}')
                        atoms.append({'symbol':sym,'x':float(p[3]),'y':float(p[4]),'z':float(p[5])})
                    except ValueError:
                        continue
            if atoms:
                break

    if not atoms:
        print(f"Cannot parse coordinates from {log_path}")
        return None

    # Mulliken charges
    blocks = re.findall(r"Mulliken charges.*?:\n\s+1\n(.*?)\nSum of Mulliken", text, re.DOTALL)
    charges = [0.0]*len(atoms)
    if blocks:
        ch = []
        for line in blocks[-1].strip().split('\n'):
            p = line.split()
            if len(p)>=3:
                try: ch.append(float(p[2]))
                except: pass
        if len(ch)==len(atoms):
            charges = ch

    # Bonds
    coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
    bonds  = []
    for i in range(len(atoms)):
        for j in range(i+1,len(atoms)):
            ri=COV_RAD.get(atoms[i]['symbol'],0.77)
            rj=COV_RAD.get(atoms[j]['symbol'],0.77)
            d=float(np.linalg.norm(coords[i]-coords[j]))
            if d<1.2*(ri+rj):
                bonds.append((i,j,d))

    degrees=[0]*len(atoms)
    for i,j,_ in bonds:
        degrees[i]+=1; degrees[j]+=1

    node_feats=[]
    for k,a in enumerate(atoms):
        s=a['symbol']
        node_feats.append([
            ATOMIC_NUM.get(s,0)/10.0, ELECTRO.get(s,0.0)/4.0,
            COV_RAD.get(s,0.77)/1.5,
            float(s=='C'),float(s=='H'),float(s=='O'),float(s=='N'),
            float(s in ('Zn','Pt','Co','Si','S','F','Cl','Br')),
            charges[k], degrees[k]/6.0,
        ])

    edge_index,edge_feats=[],[]
    for i,j,d in bonds:
        ri=COV_RAD.get(atoms[i]['symbol'],0.77)
        rj=COV_RAD.get(atoms[j]['symbol'],0.77)
        ratio=d/(ri+rj)
        si,sj=atoms[i]['symbol'],atoms[j]['symbol']
        ef=[d/3.0,ratio,float(ratio<0.87),
            float(si=='C' and sj=='C'),
            float({si,sj}=={'C','O'}),float({si,sj}=={'C','H'})]
        edge_index+=[[i,j],[j,i]]
        edge_feats+=[ef,ef]

    return {
        'name':          Path(log_path).stem,
        'n_atoms':       len(atoms),
        'node_features': node_feats,
        'edge_index':    edge_index,
        'edge_features': edge_feats,
        'ftir_spectrum': [0.0]*500,
        'freq_min':      400,
        'freq_max':      4000,
        'n_bins':        500,
    }


def plot_spectrum(pred, target=None, name='', freq_min=1000, freq_max=3000, save_path=None):
    """
    Plot predicted FTIR spectrum.
    If target is provided (real Excel data), overlay both for comparison.
    """
    x = np.linspace(freq_min, freq_max, len(pred))

    fig, ax = plt.subplots(figsize=(13, 5))

    # Plot real spectrum if available
    if target is not None:
        ax.plot(x, target, color='steelblue', lw=2.0, alpha=0.85,
                label='Real FTIR (experimental)', zorder=2)

    # Plot prediction
    ax.plot(x, pred, color='crimson', lw=1.8, ls='--', alpha=0.9,
            label='GNN Prediction', zorder=3)

    # Find and annotate peaks
    peaks_idx, _ = find_peaks(pred, height=0.05, distance=5)
    freqs_at_peaks = x[peaks_idx]
    intens_at_peaks = pred[peaks_idx]
    top_peaks = sorted(zip(freqs_at_peaks, intens_at_peaks), key=lambda p:-p[1])[:10]
    for freq, intens in top_peaks:
        ax.annotate(f"{freq:.0f}", xy=(freq, intens),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=7, color='#c0392b')

    # Metrics if we have ground truth
    if target is not None:
        r = np.corrcoef(pred, target)[0,1]
        rmse = np.sqrt(np.mean((pred-target)**2))
        ax.text(0.02, 0.95, f"Pearson r = {r:.3f}   RMSE = {rmse:.4f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Absorbance (normalised)', fontsize=12)
    ax.set_title(f'FTIR Spectrum — {name}', fontsize=13)
    ax.set_xlim(freq_max, freq_min)   # FTIR convention: high to low
    ax.set_ylim(-0.03, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved -> {save_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True,
                    help='e.g. checkpoints/fold1_best.pt')
    ap.add_argument('--dataset',    default=None,
                    help='data/dataset.json — predict all + compare to real FTIR')
    ap.add_argument('--log_file',   default=None,
                    help='A single new .log file with no Excel (blind prediction)')
    ap.add_argument('--output_dir', default='results')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(args.checkpoint, device)

    freq_min = cfg.get('freq_min', 1000)
    freq_max = cfg.get('freq_max', 3000)

    # ── Predict all structures in dataset (with real FTIR comparison) ─────────
    if args.dataset:
        raw = json.load(open(args.dataset))
        print(f"\nPredicting {len(raw)} structures ...\n")

        import csv
        all_metrics = []

        for entry in raw:
            pred   = predict_entry(model, entry, device)
            target = np.array(entry['ftir_spectrum'], dtype=np.float32)

            r    = float(np.corrcoef(pred, target)[0,1])
            rmse = float(np.sqrt(np.mean((pred-target)**2)))
            all_metrics.append({'name':entry['name'],'pearson_r':r,'rmse':rmse})

            plot_spectrum(pred, target=target,
                          name=entry['name'],
                          freq_min=freq_min, freq_max=freq_max,
                          save_path=os.path.join(args.output_dir,
                                                  f"{entry['name']}_ftir.png"))
            print(f"  {entry['name']:30s}  r={r:.3f}  RMSE={rmse:.4f}")

        # Save metrics CSV
        csv_path = os.path.join(args.output_dir, 'metrics.csv')
        with open(csv_path,'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name','pearson_r','rmse'])
            w.writeheader()
            w.writerows(all_metrics)

        print(f"\nMean Pearson r : {np.mean([m['pearson_r'] for m in all_metrics]):.3f}")
        print(f"Mean RMSE      : {np.mean([m['rmse']      for m in all_metrics]):.4f}")
        print(f"\nPlots   -> {args.output_dir}/")
        print(f"Metrics -> {csv_path}")

    # ── Predict a single new .log file (no Excel needed) ─────────────────────
    elif args.log_file:
        entry = parse_log_for_prediction(args.log_file)
        if entry is None:
            return
        pred = predict_entry(model, entry, device)
        plot_spectrum(pred, target=None,
                      name=entry['name'],
                      freq_min=freq_min, freq_max=freq_max,
                      save_path=os.path.join(args.output_dir,
                                             f"{entry['name']}_ftir.png"))
        print(f"\nDone! Spectrum saved -> {args.output_dir}/{entry['name']}_ftir.png")

    else:
        print("Provide --dataset or --log_file")
        print("Example:")
        print("  python predict.py --checkpoint checkpoints/fold1_best.pt --dataset data/dataset.json --output_dir results/")


if __name__ == '__main__':
    main()