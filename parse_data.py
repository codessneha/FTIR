"""
STEP 1 — parse_data.py  (auto frequency range detection)
==========================================================
Automatically finds the overlapping frequency range across ALL Excel files,
then interpolates every spectrum onto that common fixed grid.

This solves the problem of different Excel files having different ranges.

Usage:
    python parse_data.py --log_dir data/logs --ftir_dir data/ftir --output data/dataset.json
"""

import re, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Atom tables ───────────────────────────────────────────────────────────────
ATOMIC_NUM = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'S':16,'Cl':17,'Br':35,'I':53,'Zn':30,'Pt':78,'Co':27}
ELECTRO    = {'H':2.20,'C':2.55,'N':3.04,'O':3.44,'F':3.98,
              'Si':1.90,'S':2.58,'Cl':3.16,'Br':2.96,'I':2.66,
              'Zn':1.65,'Pt':2.28,'Co':1.88}
COV_RAD    = {'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,
              'Si':1.11,'S':1.05,'Cl':1.02,'Br':1.20,'I':1.39,
              'Zn':1.22,'Pt':1.36,'Co':1.26}

N_BINS = 500   # fixed number of output points for all spectra

# ─────────────────────────────────────────────────────────────────────────────

def read_ftir_excel_raw(xlsx_path):
    """
    Read raw (frequency, absorbance) pairs from an Excel file.
    Returns a sorted DataFrame with columns ['freq', 'abs'].
    """
    df   = pd.read_excel(xlsx_path, header=None)
    data = df.iloc[1:, [1, 5]].copy()
    data.columns = ['freq', 'abs']
    data['freq'] = pd.to_numeric(data['freq'], errors='coerce')
    data['abs']  = pd.to_numeric(data['abs'],  errors='coerce')
    data = data.dropna().sort_values('freq').reset_index(drop=True)
    if len(data) < 10:
        raise ValueError(f"Too few data points in {xlsx_path.name}: {len(data)}")
    return data


def find_common_range(ftir_dir, log_stems):
    """
    Step through all paired Excel files and find:
      - global_min = the HIGHEST of all individual min frequencies
      - global_max = the LOWEST  of all individual max frequencies
    This gives the range that ALL files share.
    """
    print("\n  Scanning Excel files for frequency ranges:")
    print(f"  {'File':<30} {'Min':>8} {'Max':>8}  {'Points':>7}")
    print("  " + "-"*58)

    all_mins, all_maxs = [], []

    for stem in log_stems:
        xlsx = Path(ftir_dir) / (stem + '.xlsx')
        if not xlsx.exists():
            continue
        try:
            data = read_ftir_excel_raw(xlsx)
            fmin = float(data['freq'].min())
            fmax = float(data['freq'].max())
            all_mins.append(fmin)
            all_maxs.append(fmax)
            print(f"  {stem:<30} {fmin:>8.1f} {fmax:>8.1f}  {len(data):>7}")
        except Exception as e:
            print(f"  {stem:<30}  ERROR: {e}")

    if not all_mins:
        raise ValueError("No valid Excel files found!")

    # Overlap = highest min to lowest max
    common_min = max(all_mins)
    common_max = min(all_maxs)

    print("  " + "-"*58)
    print(f"  {'Each file min/max':<30} {min(all_mins):>8.1f} {max(all_maxs):>8.1f}")
    print(f"  {'COMMON OVERLAP':<30} {common_min:>8.1f} {common_max:>8.1f}")

    if common_min >= common_max:
        print("\n  WARNING: No overlapping range found across all files!")
        print("  Using the widest possible range instead (union).")
        print("  Regions with no data will be zero-padded.")
        common_min = min(all_mins)
        common_max = max(all_maxs)

    print(f"\n  -> Using range: {common_min:.1f} - {common_max:.1f} cm-1")
    print(f"     Step size   : {(common_max - common_min) / (N_BINS-1):.2f} cm-1")
    print(f"     Points      : {N_BINS}\n")

    return common_min, common_max


def interpolate_spectrum(data, freq_min, freq_max, n_bins=N_BINS):
    """
    Interpolate a raw spectrum onto the fixed common grid.
    Regions outside the original data range are zero-padded.
    Normalises result to [0, 1].
    """
    target_freqs = np.linspace(freq_min, freq_max, n_bins)
    absorbances  = np.interp(
        target_freqs,
        data['freq'].values,
        data['abs'].values,
        left=0.0,
        right=0.0,
    ).astype(np.float32)

    # Normalise to [0, 1]
    peak = absorbances.max()
    if peak > 0:
        absorbances /= peak

    return target_freqs, absorbances


# ── Molecular graph parsers ───────────────────────────────────────────────────

def parse_coordinates(text):
    for pattern in [
        r"Standard orientation:.*?[-]{10,}\n.*?[-]{10,}\n(.*?)\n\s*[-]{10,}",
        r"Input orientation:.*?[-]{10,}\n.*?[-]{10,}\n(.*?)\n\s*[-]{10,}",
    ]:
        blocks = re.findall(pattern, text, re.DOTALL)
        if blocks:
            atoms = []
            for line in blocks[-1].strip().split('\n'):
                p = line.split()
                if len(p) >= 6:
                    try:
                        anum = int(p[1])
                        sym  = next((k for k,v in ATOMIC_NUM.items() if v==anum), f'X{anum}')
                        atoms.append({'symbol':sym,
                                      'x':float(p[3]),
                                      'y':float(p[4]),
                                      'z':float(p[5])})
                    except ValueError:
                        continue
            if atoms:
                return atoms
    return None


def parse_mulliken(text, n_atoms):
    blocks = re.findall(
        r"Mulliken charges.*?:\n\s+1\n(.*?)\nSum of Mulliken",
        text, re.DOTALL
    )
    if not blocks:
        return [0.0] * n_atoms
    charges = []
    for line in blocks[-1].strip().split('\n'):
        p = line.split()
        if len(p) >= 3:
            try:
                charges.append(float(p[2]))
            except ValueError:
                continue
    return charges if len(charges) == n_atoms else [0.0] * n_atoms


def infer_bonds(atoms, scale=1.20):
    coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
    bonds  = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            ri = COV_RAD.get(atoms[i]['symbol'], 0.77)
            rj = COV_RAD.get(atoms[j]['symbol'], 0.77)
            d  = float(np.linalg.norm(coords[i]-coords[j]))
            if d < scale*(ri+rj):
                bonds.append((i,j,d))
    return bonds


def build_graph_features(atoms, bonds, charges):
    """Build node and edge feature matrices."""
    degrees = [0]*len(atoms)
    for i,j,_ in bonds:
        degrees[i]+=1; degrees[j]+=1

    node_feats = []
    for k,a in enumerate(atoms):
        s = a['symbol']
        node_feats.append([
            ATOMIC_NUM.get(s,0) / 10.0,
            ELECTRO.get(s,0.0)  / 4.0,
            COV_RAD.get(s,0.77) / 1.5,
            float(s=='C'),
            float(s=='H'),
            float(s=='O'),
            float(s=='N'),
            float(s in ('Zn','Pt','Co','Si','S','F','Cl','Br')),
            charges[k],
            degrees[k] / 6.0,
        ])

    edge_index, edge_feats = [], []
    for i,j,d in bonds:
        ri = COV_RAD.get(atoms[i]['symbol'],0.77)
        rj = COV_RAD.get(atoms[j]['symbol'],0.77)
        ratio = d/(ri+rj)
        si,sj = atoms[i]['symbol'],atoms[j]['symbol']
        ef = [
            d/3.0,
            ratio,
            float(ratio<0.87),
            float(si=='C' and sj=='C'),
            float({si,sj}=={'C','O'}),
            float({si,sj}=={'C','H'}),
        ]
        edge_index+=[[i,j],[j,i]]
        edge_feats+=[ef,ef]

    return node_feats, edge_index, edge_feats


# ── Main pair processor ───────────────────────────────────────────────────────

def process_pair(log_path, xlsx_path, freq_min, freq_max):
    """Process one (log, xlsx) pair using the pre-computed common range."""

    # ── Molecular graph from .log ─────────────────────────────────────────────
    text = log_path.read_text(errors='ignore')
    if 'Normal termination' not in text:
        print(f"    [WARN] {log_path.name}: no Normal termination")

    atoms = parse_coordinates(text)
    if atoms is None:
        print(f"    [SKIP] {log_path.name}: cannot parse coordinates")
        return None

    charges = parse_mulliken(text, len(atoms))
    bonds   = infer_bonds(atoms)
    node_feats, edge_index, edge_feats = build_graph_features(atoms, bonds, charges)

    # ── Real FTIR spectrum from .xlsx ─────────────────────────────────────────
    try:
        raw_data        = read_ftir_excel_raw(xlsx_path)
        freqs, spectrum = interpolate_spectrum(raw_data, freq_min, freq_max)
    except Exception as e:
        print(f"    [SKIP] {xlsx_path.name}: {e}")
        return None

    # Composition
    sym_counts = {}
    for a in atoms:
        sym_counts[a['symbol']] = sym_counts.get(a['symbol'],0)+1

    return {
        'name':          log_path.stem,
        'n_atoms':       len(atoms),
        'atoms':         [a['symbol'] for a in atoms],
        'node_features': node_feats,
        'edge_index':    edge_index,
        'edge_features': edge_feats,
        'ftir_spectrum': spectrum.tolist(),   # [N_BINS] real FTIR on common grid
        'ftir_freqs':    freqs.tolist(),      # [N_BINS] wavenumber axis
        'freq_min':      float(freq_min),
        'freq_max':      float(freq_max),
        'n_bins':        N_BINS,
        'composition':   sym_counts,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir',  default='data/logs')
    ap.add_argument('--ftir_dir', default='data/ftir')
    ap.add_argument('--output',   default='data/dataset.json')
    ap.add_argument('--log_ext',  default='.txt')
    ap.add_argument('--freq_min', type=float, default=None,
                    help='Override min frequency (auto-detected if not set)')
    ap.add_argument('--freq_max', type=float, default=None,
                    help='Override max frequency (auto-detected if not set)')
    args = ap.parse_args()

    log_dir  = Path(args.log_dir)
    ftir_dir = Path(args.ftir_dir)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Find all log files that have a matching Excel
    log_files = sorted(log_dir.glob(f'*{args.log_ext}'))
    print(f"\nFound {len(log_files)} log files")

    paired_stems = []
    for lf in log_files:
        xlsx = ftir_dir / (lf.stem + '.xlsx')
        if xlsx.exists():
            paired_stems.append(lf.stem)
        else:
            print(f"  [NO EXCEL] {lf.name} — skipping")

    print(f"Found {len(paired_stems)} matched pairs: {paired_stems}")

    if not paired_stems:
        print("\nERROR: No matched pairs found.")
        print("Make sure .log and .xlsx files have identical stem names.")
        print("Example: NH3.log must match NH3.xlsx")
        return

    # ── Auto-detect common frequency range ───────────────────────────────────
    if args.freq_min is not None and args.freq_max is not None:
        freq_min = args.freq_min
        freq_max = args.freq_max
        print(f"\nUsing manual range: {freq_min} - {freq_max} cm-1")
    else:
        freq_min, freq_max = find_common_range(ftir_dir, paired_stems)

    # ── Process each pair ─────────────────────────────────────────────────────
    dataset = []
    for stem in paired_stems:
        log_path  = log_dir  / (stem + args.log_ext)
        xlsx_path = ftir_dir / (stem + '.xlsx')
        print(f"  Processing: {stem}")
        entry = process_pair(log_path, xlsx_path, freq_min, freq_max)
        if entry:
            dataset.append(entry)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Successfully parsed : {len(dataset)} / {len(paired_stems)}")
    print(f"  Common freq range   : {freq_min:.1f} - {freq_max:.1f} cm-1")
    print(f"  Output bins         : {N_BINS}")
    print(f"  Step size           : {(freq_max-freq_min)/(N_BINS-1):.2f} cm-1")
    print(f"{'='*55}")

    if not dataset:
        print("\nERROR: Nothing was parsed successfully.")
        return

    json.dump(dataset, open(args.output,'w'), indent=2)
    print(f"\nSaved -> {args.output}")
    print(f"Node feature dim : {len(dataset[0]['node_features'][0])}")
    print(f"Edge feature dim : {len(dataset[0]['edge_features'][0])}")
    print(f"Target dim       : {len(dataset[0]['ftir_spectrum'])}\n")

    # ── Warn if overlap is very small ─────────────────────────────────────────
    overlap = freq_max - freq_min
    if overlap < 200:
        print(f"WARNING: Overlap range is only {overlap:.1f} cm-1 wide.")
        print("Your Excel files have very different frequency ranges.")
        print("Consider using --freq_min and --freq_max to set a manual range")
        print("that covers the most scientifically important region.")
        print("Example: python parse_data.py --freq_min 400 --freq_max 4000")


if __name__ == '__main__':
    main()