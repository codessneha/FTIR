"""
STEP 5 — predict.py  (improved)
==================================
Changes vs original:
  - Passes `u` (global features) to the model.
  - parse_log_for_prediction extracts HOMO/LUMO/dipole/ZPE for blind prediction.
  - Node/edge feature dims updated to 15/8.
  - Pd and other metals supported in atom tables.

Usage:
    # Predict all structures in dataset (with real FTIR comparison):
    python predict.py --checkpoint checkpoints/fold1_best.pt \
                      --dataset    data/dataset.json \
                      --output_dir results/

    # Predict one new .log file:
    python predict.py --checkpoint checkpoints/fold1_best.pt \
                      --log_file   data/logs/new_structure.log \
                      --output_dir results/
"""

import os, json, argparse, time
import numpy as np
import torch
import matplotlib
# Use 'Agg' backend for non-interactive plotting (no window will pop up)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

from model import FTIRNet

# ─────────────────────────────────────────────────────────────────────────────
# ── Shared Atomic and Chemical Properties ─────────────────────────────────────
# This dictionary maps atom symbols to their atomic numbers.
# Must be kept in sync with parse_data.py for consistent feature extraction.
# ─────────────────────────────────────────────────────────────────────────────
ATOMIC_NUM = {
    'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'S':16,'Cl':17,
    'Br':35,'I':53,'P':15,'B':5,
    'Pd':46,'Pt':78,'Ni':28,'Cu':29,'Zn':30,'Co':27,'Fe':26,
    'Mn':25,'Cr':24,'Ti':22,'Ru':44,'Rh':45,'Ag':47,'Au':79,
    'Mo':42,'Se':34,'V':23,'Hf':72,'Ir':77,'Os':76,'Re':75,'Sc':21,'Ta':73,'Zr':40
}
NUM_ATOMIC = {v: k for k, v in ATOMIC_NUM.items()}

ELECTRO = {
    'H':2.20,'C':2.55,'N':3.04,'O':3.44,'F':3.98,'P':2.19,'B':2.04,
    'Si':1.90,'S':2.58,'Cl':3.16,'Br':2.96,'I':2.66,
    'Pd':2.20,'Pt':2.28,'Ni':1.91,'Cu':1.90,'Zn':1.65,'Co':1.88,
    'Fe':1.83,'Mn':1.55,'Cr':1.66,'Ti':1.54,'Ru':2.20,'Rh':2.28,
    'Ag':1.93,'Au':2.54,
    'Mo':2.16,'Se':2.55,'V':1.63,'Hf':1.30,'Ir':2.20,'Os':2.20,'Re':1.90,'Sc':1.36,'Ta':1.50,'Zr':1.33
}
COV_RAD = {
    'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'P':1.07,'B':0.84,
    'Si':1.11,'S':1.05,'Cl':1.02,'Br':1.20,'I':1.39,
    'Pd':1.39,'Pt':1.36,'Ni':1.24,'Cu':1.32,'Zn':1.22,'Co':1.26,
    'Fe':1.32,'Mn':1.61,'Cr':1.39,'Ti':1.60,'Ru':1.46,'Rh':1.42,
    'Ag':1.45,'Au':1.36,
    'Mo':1.45,'Se':1.16,'V':1.34,'Hf':1.50,'Ir':1.37,'Os':1.28,'Re':1.59,'Sc':1.44,'Ta':1.38,'Zr':1.48
}
PERIOD = {
    'H':1,'B':2,'C':2,'N':2,'O':2,'F':2,
    'Si':3,'P':3,'S':3,'Cl':3,
    'Br':4,'Fe':4,'Co':4,'Ni':4,'Cu':4,'Zn':4,'Ti':4,'Cr':4,'Mn':4,
    'I':5,'Pd':5,'Rh':5,'Ru':5,'Ag':5,
    'Pt':6,'Au':6,
    'Mo':5,'Se':4,'V':4,'Hf':6,'Ir':6,'Os':6,'Re':6,'Sc':4,'Ta':6,'Zr':5
}
GROUP = {
    'H':1,'C':14,'N':15,'O':16,'F':17,'B':13,
    'Si':14,'P':15,'S':16,'Cl':17,'Br':17,'I':17,
    'Ti':4,'Cr':6,'Mn':7,'Fe':8,'Co':9,'Ni':10,'Cu':11,'Zn':12,
    'Ru':8,'Rh':9,'Pd':10,'Ag':11,'Pt':10,'Au':11,
    'Mo':6,'Se':16,'V':5,'Hf':4,'Ir':9,'Os':8,'Re':7,'Sc':3,'Ta':5,'Zr':4
}
METALS = {'Pd','Pt','Ni','Cu','Zn','Co','Fe','Mn','Cr','Ti','Ru','Rh','Ag','Au','Mo','V','Hf','Ir','Os','Re','Sc','Ta','Zr'}

# ─────────────────────────────────────────────────────────────────────────────
# ── Model Initialization Function ─────────────────────────────────────────────
# Loads a trained model from a checkpoint (.pt file).
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt['cfg']
    state = ckpt['state']
    
    inferred_hidden = state['input_proj.0.weight'].shape[0] if 'input_proj.0.weight' in state else cfg.get('hidden_dim', 128)
    inferred_layers = max([int(k.split('.')[1]) for k in state.keys() if k.startswith('convs.')]) + 1 if any(k.startswith('convs.') for k in state.keys()) else cfg.get('num_layers', 3)
    
    model = FTIRNet(
        node_dim   = cfg['node_feat_dim'],
        edge_dim   = cfg['edge_feat_dim'],
        global_dim = cfg.get('global_feat_dim', 0),
        hidden_dim = inferred_hidden,
        num_layers = inferred_layers,
        out_dim    = cfg['spectrum_bins'],
        dropout    = 0.0,
    ).to(device)
    model.load_state_dict(ckpt['state'])
    model.eval()
    print(f"Loaded model (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model, cfg

# ─────────────────────────────────────────────────────────────────────────────
# ── Data Transformation Function ──────────────────────────────────────────────
# Converts a dictionary-based data entry (from JSON) into a
# PyTorch Geometric 'Data' object suitable for GNN input.
# ─────────────────────────────────────────────────────────────────────────────
def entry_to_graph(entry):
    from torch_geometric.data import Data
    x          = torch.tensor(entry['node_features'],  dtype=torch.float)
    if not entry['edge_index']:
        n_ef = len(entry['edge_features'][0]) if entry.get('edge_features') else 8
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, n_ef), dtype=torch.float)
    else:
        edge_index = torch.tensor(entry['edge_index'],    dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(entry['edge_features'], dtype=torch.float)
    y = torch.tensor(entry['ftir_spectrum'], dtype=torch.float).unsqueeze(0)
    u = torch.tensor(entry.get('global_features', [0.0]*9), dtype=torch.float).unsqueeze(0)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u,
                name=entry['name'])

# ─────────────────────────────────────────────────────────────────────────────
# ── Prediction Function ───────────────────────────────────────────────────────
# Takes a single data entry, converts it to a graph, and runs inference.
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_entry(model, entry, device):
    from torch_geometric.data import Batch
    graph = entry_to_graph(entry)
    batch = Batch.from_data_list([graph]).to(device)
    u     = batch.u if hasattr(batch, 'u') else None
    out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, u=u)
    return out.squeeze().cpu().numpy()

# ─────────────────────────────────────────────────────────────────────────────
# ── Log File Parser (for Blind Prediction) ────────────────────────────────────
# Extracts required features from a Gaussian .log file directly.
# This allows making predictions for new structures not in the training set.
# ─────────────────────────────────────────────────────────────────────────────
def parse_log_for_prediction(log_path):
    """Parse a new .log file (no Excel required) for blind prediction."""
    import re
    import numpy as np

    text = Path(log_path).read_text(errors='ignore')
    atoms = None
    # 1. Parse Coordinates: extracts atom symbols and XYZ positions from Gaussian output.
    for pat in [
        r"Standard orientation:.*?[-]{10,}\n.*?[-]{10,}\n(.*?)\n\s*[-]{10,}",
        r"Input orientation:.*?[-]{10,}\n.*?[-]{10,}\n(.*?)\n\s*[-]{10,}",
    ]:
        blocks = re.findall(pat, text, re.DOTALL)
        if blocks:
            atoms = []
            for line in blocks[-1].strip().split('\n'):
                p = line.split()
                if len(p) >= 6:
                    try:
                        anum = int(p[1])
                        sym  = NUM_ATOMIC.get(anum, f'X{anum}')
                        atoms.append({'symbol':sym,'x':float(p[3]),'y':float(p[4]),'z':float(p[5])})
                    except ValueError:
                        continue
            if atoms:
                break

    if not atoms:
        print(f"Cannot parse coordinates from {log_path}")
        return None

    # 2. Parse APT/Mulliken Charges: used as a node feature to represent distribution.
    charges = [0.0] * len(atoms)
    for pat, key in [
        (r"APT charges:(.*?)Sum of APT", 2),
        (r"Mulliken charges.*?:\n\s+1\n(.*?)\nSum of Mulliken", 2),
    ]:
        blocks = re.findall(pat, text, re.DOTALL)
        if blocks:
            ch = []
            for line in blocks[-1].strip().split('\n'):
                p = line.split()
                if len(p) >= 3:
                    try: ch.append(float(p[key]))
                    except: pass
            if len(ch) == len(atoms):
                charges = ch
                break

    # 3. Parse Global Features: HOMO, LUMO, Dipole Moment, SCF Energy, and ZPE.
    # These provide molecular-level context to the graph.
    global_feats = []
    alpha_occ  = re.findall(r'Alpha  occ\. eigenvalues --\s+([\-\d\.\s]+)', text)
    alpha_virt = re.findall(r'Alpha virt\. eigenvalues --\s+([\-\d\.\s]+)', text)
    if alpha_occ and alpha_virt:
        homo = max(float(x) for line in alpha_occ  for x in line.split())
        lumo = min(float(x) for line in alpha_virt for x in line.split())
        # Convert Hartree to eV (approx x 27.211)
        global_feats += [homo*27.211, lumo*27.211, (lumo-homo)*27.211]
    else:
        global_feats += [0.0, 0.0, 0.0]

    dip = re.findall(r'X=\s*([-\d\.]+)\s+Y=\s*([-\d\.]+)\s+Z=\s*([-\d\.]+)\s+Tot=\s*([-\d\.]+)', text)
    if dip:
        global_feats += [float(x) for x in dip[-1]]
    else:
        global_feats += [0.0, 0.0, 0.0, 0.0]

    scf = re.findall(r'SCF Done.*?=\s*([-\d\.]+)', text)
    global_feats.append(float(scf[-1]) if scf else 0.0)
    zpe = re.findall(r'Zero-point correction=\s*([-\d\.]+)', text)
    global_feats.append(float(zpe[-1]) if zpe else 0.0)

    # 4. Infer Bonds: Based on covalent radii. Supports Metal-Ligand distance scaling.
    coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
    metal_idx = [i for i,a in enumerate(atoms) if a['symbol'] in METALS]
    if metal_idx:
        metal_pos = coords[metal_idx]
        min_dists = np.min(np.linalg.norm(coords[:,None,:]-metal_pos[None,:,:],axis=2),axis=1)
    else:
        min_dists = np.zeros(len(atoms))

    bonds = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            si, sj = atoms[i]['symbol'], atoms[j]['symbol']
            ri, rj = COV_RAD.get(si,0.77), COV_RAD.get(sj,0.77)
            d = float(np.linalg.norm(coords[i]-coords[j]))
            # Use larger scale for metals to capture coordination bonds
            scale = 1.35 if (si in METALS or sj in METALS) else 1.20
            if d < scale*(ri+rj):
                bonds.append((i,j,d))

    degrees = [0]*len(atoms)
    bond_map = {i:[] for i in range(len(atoms))}
    for i,j,_ in bonds:
        degrees[i]+=1; degrees[j]+=1
        bond_map[i].append(j); bond_map[j].append(i)

    # 5. Build Node Features: Atomic number, electronegativity, radii, etc.
    node_feats = []
    for k, a in enumerate(atoms):
        s = a['symbol']
        h_count  = sum(1 for nb in bond_map[k] if atoms[nb]['symbol']=='H')
        nb_syms  = [atoms[nb]['symbol'] for nb in bond_map[k]]
        aromatic = float(s=='C' and nb_syms.count('C')>=2 and degrees[k]==3)
        node_feats.append([
            ATOMIC_NUM.get(s,0)/10.0, ELECTRO.get(s,0.0)/4.0,
            COV_RAD.get(s,0.77)/1.5,
            PERIOD.get(s,4)/6.0, GROUP.get(s,10)/18.0,
            float(s=='C'), float(s=='H'), float(s=='O'), float(s=='N'),
            float(s in METALS),
            charges[k], degrees[k]/6.0,
            float(min_dists[k])/5.0, h_count/4.0, aromatic,
        ])

    # 6. Build Edge Features: Bond length, element pairs, etc.
    edge_index, edge_feats = [], []
    for i,j,d in bonds:
        si,sj = atoms[i]['symbol'],atoms[j]['symbol']
        ri,rj = COV_RAD.get(si,0.77),COV_RAD.get(sj,0.77)
        ratio = d/(ri+rj)
        ei,ej = ELECTRO.get(si,2.0),ELECTRO.get(sj,2.0)
        ef = [d/3.0, ratio, float(ratio<0.87),
              float(si=='C' and sj=='C'),
              float({si,sj}=={'C','O'}), float({si,sj}=={'C','H'}),
              float(si in METALS or sj in METALS), abs(ei-ej)/4.0]
        edge_index+=[[i,j],[j,i]]
        edge_feats+=[ef,ef]

    return {
        'name':            Path(log_path).stem,
        'n_atoms':         len(atoms),
        'node_features':   node_feats,
        'edge_index':      edge_index,
        'edge_features':   edge_feats,
        'global_features': global_feats,
        'ftir_spectrum':   [0.0]*500,
        'freq_min':        400,
        'freq_max':        3000,
        'n_bins':          500,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ── Visualization Function ────────────────────────────────────────────────────
# Plots the predicted spectrum (and the target if available) and
# annotates peaks for easier interpretation.
# ─────────────────────────────────────────────────────────────────────────────
def plot_spectrum(pred, target=None, name='', freq_min=400, freq_max=3000, save_path=None):
    x   = np.linspace(freq_min, freq_max, len(pred))
    fig, ax = plt.subplots(figsize=(13, 5))

    if target is not None:
        ax.plot(x, target, color='steelblue', lw=2.0, alpha=0.85,
                label='Computed FTIR (PM6)', zorder=2)

    ax.plot(x, pred, color='crimson', lw=1.8, ls='--', alpha=0.9,
            label='GNN Prediction', zorder=3)

    peaks_idx, _ = find_peaks(pred, height=0.05, distance=5)
    freqs_at_peaks  = x[peaks_idx]
    intens_at_peaks = pred[peaks_idx]
    top_peaks = sorted(zip(freqs_at_peaks, intens_at_peaks), key=lambda p:-p[1])[:10]
    for freq, intens in top_peaks:
        ax.annotate(f"{freq:.0f}", xy=(freq, intens),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=7, color='#c0392b')

    if target is not None:
        r    = np.corrcoef(pred, target)[0,1]
        rmse = np.sqrt(np.mean((pred-target)**2))
        ax.text(0.02, 0.95, f"Pearson r = {r:.3f}   RMSE = {rmse:.4f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Absorbance (normalised)', fontsize=12)
    ax.set_title(f'FTIR Spectrum — {name}', fontsize=13)
    ax.set_xlim(freq_max, freq_min)
    ax.set_ylim(-0.03, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved -> {save_path}")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# ── Main Entry Point ──────────────────────────────────────────────────────────
# Handles CLI arguments and orchestrates the prediction process.
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset',    default=None)
    ap.add_argument('--log_file',   default=None)
    ap.add_argument('--output_dir', default='results')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(args.checkpoint, device)

    freq_min = cfg.get('freq_min', 400)
    freq_max = cfg.get('freq_max', 3000)

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
            plot_spectrum(pred, target=target, name=entry['name'],
                          freq_min=freq_min, freq_max=freq_max,
                          save_path=os.path.join(args.output_dir,
                                                  f"{entry['name']}_ftir.png"))
            print(f"  {entry['name']:30s}  r={r:.3f}  RMSE={rmse:.4f}")

        csv_path = os.path.join(args.output_dir, 'metrics.csv')
        try:
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['name', 'pearson_r', 'rmse'])
                w.writeheader()
                w.writerows(all_metrics)
            print(f"\nMetrics CSV    : {csv_path}")
        except PermissionError:
            timestamp_csv = os.path.join(args.output_dir, f'metrics_{int(time.time())}.csv')
            print(f"\n[WARN] Could not write to {csv_path} (file likely open in Excel).")
            print(f"       Saving to alternate path instead: {timestamp_csv}")
            with open(timestamp_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['name', 'pearson_r', 'rmse'])
                w.writeheader()
                w.writerows(all_metrics)

        print(f"\nMean Pearson r : {np.mean([m['pearson_r'] for m in all_metrics]):.3f}")
        print(f"Mean RMSE      : {np.mean([m['rmse'] for m in all_metrics]):.4f}")

    elif args.log_file:
        entry = parse_log_for_prediction(args.log_file)
        if entry is None:
            return
        pred = predict_entry(model, entry, device)
        plot_spectrum(pred, name=entry['name'],
                      freq_min=freq_min, freq_max=freq_max,
                      save_path=os.path.join(args.output_dir,
                                             f"{entry['name']}_ftir.png"))
        print(f"\nDone! Spectrum -> {args.output_dir}/{entry['name']}_ftir.png")
    else:
        print("Provide --dataset or --log_file")

if __name__ == '__main__':
    main()