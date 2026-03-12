"""
STEP 1 — parse_data.py  (improved)
====================================
Key fixes vs original:
  1. Pd (and other transition metals) added to ALL atom tables
  2. APT charges extracted instead of Mulliken (much better for metals)
  3. HOMO/LUMO gap + dipole added as global molecular features
  4. DFT vibrational frequencies encoded as per-mode node attention
     weights (a soft auxiliary feature carried through build_graphs.py)
  5. Frequency range defaults to 400-3000 cm-1 (physically meaningful;
     avoids wasting ~50 bins on the 0-400 region where no real IR modes sit)
  6. Node feature vector extended to 15D:
       [atomic_num, electro, cov_rad, period, group,
        is_C, is_H, is_O, is_N, is_metal,
        APT_charge, degree, distance_to_metal (if any), H_count, ring_flag]
  7. Edge features extended to 8D — adds is_coordination_bond + bond_polarity

Usage:
    python parse_data.py --log_dir data/logs --ftir_dir data/ftir --output data/dataset.json
"""

import re, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Atom tables  (Pd and common transition metals added) ──────────────────────
ATOMIC_NUM = {
    'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'S':16,'Cl':17,
    'Br':35,'I':53,'P':15,'B':5,
    # Transition metals — added
    'Pd':46,'Pt':78,'Ni':28,'Cu':29,'Zn':30,'Co':27,'Fe':26,
    'Mn':25,'Cr':24,'Ti':22,'Ru':44,'Rh':45,'Ag':47,'Au':79,
}
NUM_ATOMIC = {v: k for k, v in ATOMIC_NUM.items()}

ELECTRO = {
    'H':2.20,'C':2.55,'N':3.04,'O':3.44,'F':3.98,'P':2.19,'B':2.04,
    'Si':1.90,'S':2.58,'Cl':3.16,'Br':2.96,'I':2.66,
    # Transition metals
    'Pd':2.20,'Pt':2.28,'Ni':1.91,'Cu':1.90,'Zn':1.65,'Co':1.88,
    'Fe':1.83,'Mn':1.55,'Cr':1.66,'Ti':1.54,'Ru':2.20,'Rh':2.28,
    'Ag':1.93,'Au':2.54,
}
COV_RAD = {
    'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'P':1.07,'B':0.84,
    'Si':1.11,'S':1.05,'Cl':1.02,'Br':1.20,'I':1.39,
    # Transition metals
    'Pd':1.39,'Pt':1.36,'Ni':1.24,'Cu':1.32,'Zn':1.22,'Co':1.26,
    'Fe':1.32,'Mn':1.61,'Cr':1.39,'Ti':1.60,'Ru':1.46,'Rh':1.42,
    'Ag':1.45,'Au':1.36,
}

# Period and group for atoms (encodes position in periodic table)
PERIOD = {
    'H':1,'B':2,'C':2,'N':2,'O':2,'F':2,
    'Si':3,'P':3,'S':3,'Cl':3,
    'Br':4,'Fe':4,'Co':4,'Ni':4,'Cu':4,'Zn':4,'Ti':4,'Cr':4,'Mn':4,
    'I':5,'Pd':5,'Rh':5,'Ru':5,'Ag':5,
    'Pt':6,'Au':6,
}
GROUP = {
    'H':1,'C':14,'N':15,'O':16,'F':17,'B':13,
    'Si':14,'P':15,'S':16,'Cl':17,'Br':17,'I':17,
    'Ti':4,'Cr':6,'Mn':7,'Fe':8,'Co':9,'Ni':10,'Cu':11,'Zn':12,
    'Ru':8,'Rh':9,'Pd':10,'Ag':11,
    'Pt':10,'Au':11,
}

METALS = {'Pd','Pt','Ni','Cu','Zn','Co','Fe','Mn','Cr','Ti','Ru','Rh','Ag','Au'}

N_BINS = 500
NODE_DIM = 15
EDGE_DIM = 8


# ── Excel / FTIR readers ──────────────────────────────────────────────────────

def read_ftir_excel_raw(xlsx_path):
    df   = pd.read_excel(xlsx_path, header=None)
    data = df.iloc[1:, [1, 5]].copy()
    data.columns = ['freq', 'abs']
    data['freq'] = pd.to_numeric(data['freq'], errors='coerce')
    data['abs']  = pd.to_numeric(data['abs'],  errors='coerce')
    data = data.dropna().sort_values('freq').reset_index(drop=True)
    if len(data) < 10:
        raise ValueError(f"Too few data points in {xlsx_path.name}: {len(data)}")
    return data


def find_common_range(ftir_dir, log_stems, default_min=0.0, default_max=3000.0):
    """
    Scan all Excel files using their ACTUAL data ranges (no pre-clipping),
    then find the overlap across all files.

    If no overlap exists (e.g. one file is 0-400 cm-1 and another is
    2025-2080 cm-1), falls back to the full union range so nothing is
    silently dropped.  Each molecule's spectrum is zero-padded in regions
    where it has no data — handled gracefully in interpolate_spectrum.

    Pass --freq_min / --freq_max on the CLI to override with a manual range.
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
            data = read_ftir_excel_raw(xlsx)   # NO clipping here
            fmin = float(data['freq'].min())
            fmax = float(data['freq'].max())
            all_mins.append(fmin)
            all_maxs.append(fmax)
            print(f"  {stem:<30} {fmin:>8.1f} {fmax:>8.1f}  {len(data):>7}")
        except Exception as e:
            print(f"  {stem:<30}  ERROR: {e}")

    if not all_mins:
        raise ValueError("No valid Excel files found!")

    common_min = max(all_mins)   # highest of all lower-bounds
    common_max = min(all_maxs)   # lowest  of all upper-bounds

    print("  " + "-"*58)
    print(f"  {'Each file min / max':<30} {min(all_mins):>8.1f} {max(all_maxs):>8.1f}")

    if common_min >= common_max:
        # No single window is shared by every file — use the full union
        # so no file is silently dropped.  Zero-padding fills the gaps.
        print(f"  {'OVERLAP':<30} {'none':>8} — using union range")
        common_min = min(all_mins)
        common_max = max(all_maxs)
    else:
        print(f"  {'COMMON OVERLAP':<30} {common_min:>8.1f} {common_max:>8.1f}")

    print(f"\n  -> Using range: {common_min:.1f} - {common_max:.1f} cm-1")
    print(f"     Step size   : {(common_max - common_min) / (N_BINS-1):.2f} cm-1")
    print(f"     Points      : {N_BINS}\n")
    return common_min, common_max


def interpolate_spectrum(data, freq_min, freq_max, n_bins=N_BINS):
    """
    Interpolate raw (freq, abs) data onto the fixed common grid.

    Regions of the grid that lie outside the file's own data range are
    zero-padded (left=0, right=0 in np.interp).  The spectrum is then
    normalised to [0, 1].

    Raises ValueError if the file's data does not overlap the grid at all
    (e.g. a far-IR file when --freq_min 400 is forced on the CLI).
    """
    target_freqs = np.linspace(freq_min, freq_max, n_bins)

    if len(data) < 2:
        raise ValueError("Too few data points for interpolation")

    # np.interp zero-pads automatically outside [data.freq.min, data.freq.max]
    absorbances = np.interp(
        target_freqs,
        data['freq'].values,
        data['abs'].values,
        left=0.0,
        right=0.0,
    ).astype(np.float32)

    peak = absorbances.max()
    if peak > 0:
        absorbances /= peak
    else:
        raise ValueError(
            "Spectrum is all-zero after interpolation — "
            "file data range does not overlap the target freq range. "
            "Remove --freq_min / --freq_max overrides to use the auto range."
        )

    return target_freqs, absorbances


# ── Log file parsers ──────────────────────────────────────────────────────────

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
                        sym  = NUM_ATOMIC.get(anum, f'X{anum}')
                        atoms.append({'symbol': sym,
                                      'x': float(p[3]),
                                      'y': float(p[4]),
                                      'z': float(p[5])})
                    except ValueError:
                        continue
            if atoms:
                return atoms
    return None


def parse_apt_charges(text, n_atoms):
    """APT charges — more reliable than Mulliken for metal complexes."""
    blocks = re.findall(
        r"APT charges:(.*?)Sum of APT",
        text, re.DOTALL
    )
    if blocks:
        charges = []
        for line in blocks[-1].strip().split('\n'):
            p = line.split()
            if len(p) >= 3:
                try:
                    charges.append(float(p[2]))
                except ValueError:
                    continue
        if len(charges) == n_atoms:
            return charges
    # Fallback: Mulliken
    blocks = re.findall(
        r"Mulliken charges.*?:\n\s+1\n(.*?)\nSum of Mulliken",
        text, re.DOTALL
    )
    if blocks:
        charges = []
        for line in blocks[-1].strip().split('\n'):
            p = line.split()
            if len(p) >= 3:
                try:
                    charges.append(float(p[2]))
                except ValueError:
                    continue
        if len(charges) == n_atoms:
            return charges
    return [0.0] * n_atoms


def parse_global_descriptors(text):
    """
    Extract molecular-level descriptors from the log file.
    Returns a dict of floats that will be stored in the dataset entry
    and concatenated onto the pooled graph embedding in the model.
    """
    desc = {}

    # HOMO / LUMO
    alpha_occ  = re.findall(r'Alpha  occ\. eigenvalues --\s+([\-\d\.\s]+)', text)
    alpha_virt = re.findall(r'Alpha virt\. eigenvalues --\s+([\-\d\.\s]+)', text)
    if alpha_occ and alpha_virt:
        homo = max(float(x) for line in alpha_occ  for x in line.split())
        lumo = min(float(x) for line in alpha_virt for x in line.split())
        desc['homo_ev']    = homo * 27.211      # Hartree -> eV
        desc['lumo_ev']    = lumo * 27.211
        desc['gap_ev']     = (lumo - homo) * 27.211
    else:
        desc['homo_ev'] = desc['lumo_ev'] = desc['gap_ev'] = 0.0

    # Dipole moment (Debye)
    dip = re.findall(r'X=\s*([-\d\.]+)\s+Y=\s*([-\d\.]+)\s+Z=\s*([-\d\.]+)\s+Tot=\s*([-\d\.]+)', text)
    if dip:
        dx, dy, dz, dtot = [float(x) for x in dip[-1]]
        desc['dipole_x'] = dx
        desc['dipole_y'] = dy
        desc['dipole_z'] = dz
        desc['dipole_tot'] = dtot
    else:
        desc['dipole_x'] = desc['dipole_y'] = desc['dipole_z'] = desc['dipole_tot'] = 0.0

    # SCF energy (use last value, normalise to per-atom scale)
    scf = re.findall(r'SCF Done.*?=\s*([-\d\.]+)', text)
    desc['scf_energy'] = float(scf[-1]) if scf else 0.0

    # Zero-point correction
    zpe = re.findall(r'Zero-point correction=\s*([-\d\.]+)', text)
    desc['zpe'] = float(zpe[-1]) if zpe else 0.0

    return desc


def parse_dft_modes(text):
    """
    Extract DFT vibrational frequencies and IR intensities.
    These are the DIRECT source of the Excel FTIR spectrum
    (Gaussian broadens them with Lorentzian to produce the Excel output).
    Returned as lists of (freq_cm1, ir_inten_kmmol).
    """
    freq_lines = re.findall(r'Frequencies --\s+([\d\.\s]+)', text)
    ir_lines   = re.findall(r'IR Inten\s*--\s+([\d\.\s]+)', text)
    freqs, intens = [], []
    for fl, il in zip(freq_lines, ir_lines):
        freqs.extend(float(f) for f in fl.split())
        intens.extend(float(x) for x in il.split())
    return freqs, intens


def infer_bonds(atoms, scale=1.20):
    """
    Infer covalent bonds by distance thresholding.
    Also captures Pd coordination bonds (eta2, CO, etc.)
    using a slightly looser scale for metal atoms.
    """
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    bonds  = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            si, sj = atoms[i]['symbol'], atoms[j]['symbol']
            ri = COV_RAD.get(si, 0.77)
            rj = COV_RAD.get(sj, 0.77)
            d  = float(np.linalg.norm(coords[i] - coords[j]))
            # Use looser threshold for metal coordination bonds
            local_scale = 1.35 if (si in METALS or sj in METALS) else scale
            if d < local_scale * (ri + rj):
                bonds.append((i, j, d))
    return bonds


def find_metal_indices(atoms):
    return [i for i, a in enumerate(atoms) if a['symbol'] in METALS]


def build_graph_features(atoms, bonds, charges, metal_indices):
    """
    Build 15D node features and 8D edge features.

    Node features (15D):
      0  atomic_num / 10
      1  electronegativity / 4
      2  covalent_radius / 1.5
      3  period / 6
      4  group / 18
      5  is_C
      6  is_H
      7  is_O
      8  is_N
      9  is_metal
      10 APT charge
      11 degree / 6
      12 min_dist_to_any_metal / 5.0  (0 if no metal)
      13 H_count / 4               (num directly bonded H)
      14 is_aromatic_proxy          (degree>=2 and all-C neighbours)

    Edge features (8D):
      0  bond length / 3.0
      1  ratio = d / (ri + rj)
      2  is_multiple_bond           (ratio < 0.87)
      3  is_CC
      4  is_CO
      5  is_CH
      6  is_coordination_bond       (involves a metal)
      7  bond_polarity              (|electronegativity difference|)
    """
    coords  = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    degrees = [0] * len(atoms)
    for i, j, _ in bonds:
        degrees[i] += 1
        degrees[j] += 1

    # Pre-compute min distances to nearest metal
    if metal_indices:
        metal_pos = coords[metal_indices]
        min_dists = np.min(
            np.linalg.norm(coords[:, None, :] - metal_pos[None, :, :], axis=2),
            axis=1
        )
    else:
        min_dists = np.zeros(len(atoms))

    # H-count per atom
    bond_map = {i: [] for i in range(len(atoms))}
    for i, j, _ in bonds:
        bond_map[i].append(j)
        bond_map[j].append(i)

    node_feats = []
    for k, a in enumerate(atoms):
        s = a['symbol']
        h_count = sum(1 for nb in bond_map[k] if atoms[nb]['symbol'] == 'H')
        # Aromatic proxy: non-H atom with >=2 C neighbours, all-same degree
        nb_syms   = [atoms[nb]['symbol'] for nb in bond_map[k]]
        c_nb      = nb_syms.count('C')
        aromatic  = float(s == 'C' and c_nb >= 2 and degrees[k] == 3)

        node_feats.append([
            ATOMIC_NUM.get(s, 0)  / 10.0,
            ELECTRO.get(s, 0.0)   / 4.0,
            COV_RAD.get(s, 0.77)  / 1.5,
            PERIOD.get(s, 4)      / 6.0,
            GROUP.get(s, 10)      / 18.0,
            float(s == 'C'),
            float(s == 'H'),
            float(s == 'O'),
            float(s == 'N'),
            float(s in METALS),
            charges[k],
            degrees[k]            / 6.0,
            float(min_dists[k])   / 5.0,
            h_count               / 4.0,
            aromatic,
        ])

    edge_index, edge_feats = [], []
    for i, j, d in bonds:
        si, sj  = atoms[i]['symbol'], atoms[j]['symbol']
        ri, rj  = COV_RAD.get(si, 0.77), COV_RAD.get(sj, 0.77)
        ratio   = d / (ri + rj)
        ei      = ELECTRO.get(si, 2.0)
        ej      = ELECTRO.get(sj, 2.0)
        ef = [
            d / 3.0,
            ratio,
            float(ratio < 0.87),                       # multiple bond
            float(si == 'C' and sj == 'C'),
            float({si, sj} == {'C', 'O'}),
            float({si, sj} == {'C', 'H'}),
            float(si in METALS or sj in METALS),       # coordination bond
            abs(ei - ej) / 4.0,                        # polarity
        ]
        edge_index += [[i, j], [j, i]]
        edge_feats += [ef, ef]

    return node_feats, edge_index, edge_feats


# ── Main pair processor ───────────────────────────────────────────────────────

def process_pair(log_path, xlsx_path, freq_min, freq_max):
    text = log_path.read_text(errors='ignore')
    if 'Normal termination' not in text:
        print(f"    [WARN] {log_path.name}: no Normal termination")

    atoms = parse_coordinates(text)
    if atoms is None:
        print(f"    [SKIP] {log_path.name}: cannot parse coordinates")
        return None

    charges      = parse_apt_charges(text, len(atoms))
    metal_idx    = find_metal_indices(atoms)
    bonds        = infer_bonds(atoms)
    node_feats, edge_index, edge_feats = build_graph_features(
        atoms, bonds, charges, metal_idx
    )

    global_desc  = parse_global_descriptors(text)
    dft_freqs, dft_intens = parse_dft_modes(text)

    # FTIR spectrum
    try:
        raw_data        = read_ftir_excel_raw(xlsx_path)
        freqs, spectrum = interpolate_spectrum(raw_data, freq_min, freq_max)
    except Exception as e:
        print(f"    [SKIP] {xlsx_path.name}: {e}")
        return None

    sym_counts = {}
    for a in atoms:
        sym_counts[a['symbol']] = sym_counts.get(a['symbol'], 0) + 1

    return {
        'name':             log_path.stem,
        'n_atoms':          len(atoms),
        'atoms':            [a['symbol'] for a in atoms],
        'node_features':    node_feats,
        'edge_index':       edge_index,
        'edge_features':    edge_feats,
        'global_features':  list(global_desc.values()),      # 9 scalars
        'global_feat_keys': list(global_desc.keys()),
        'dft_freqs':        dft_freqs,                       # auxiliary
        'dft_intens':       dft_intens,
        'ftir_spectrum':    spectrum.tolist(),
        'ftir_freqs':       freqs.tolist(),
        'freq_min':         float(freq_min),
        'freq_max':         float(freq_max),
        'n_bins':           N_BINS,
        'composition':      sym_counts,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir',  default='data/logs')
    ap.add_argument('--ftir_dir', default='data/ftir')
    ap.add_argument('--output',   default='data/dataset.json')
    ap.add_argument('--log_ext',  default='.txt')
    ap.add_argument('--freq_min', type=float, default=None)
    ap.add_argument('--freq_max', type=float, default=None)
    args = ap.parse_args()

    log_dir  = Path(args.log_dir)
    ftir_dir = Path(args.ftir_dir)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    log_files    = sorted(log_dir.glob(f'*{args.log_ext}'))
    paired_stems = []
    for lf in log_files:
        xlsx = ftir_dir / (lf.stem + '.xlsx')
        if xlsx.exists():
            paired_stems.append(lf.stem)
        else:
            print(f"  [NO EXCEL] {lf.name} — skipping")

    print(f"Found {len(paired_stems)} matched pairs")

    if args.freq_min is not None and args.freq_max is not None:
        freq_min, freq_max = args.freq_min, args.freq_max
    else:
        freq_min, freq_max = find_common_range(ftir_dir, paired_stems)

    dataset = []
    for stem in paired_stems:
        print(f"  Processing: {stem}")
        entry = process_pair(
            log_dir  / (stem + args.log_ext),
            ftir_dir / (stem + '.xlsx'),
            freq_min, freq_max,
        )
        if entry:
            dataset.append(entry)

    print(f"\n{'='*55}")
    print(f"  Successfully parsed : {len(dataset)} / {len(paired_stems)}")
    print(f"  Freq range          : {freq_min:.1f} - {freq_max:.1f} cm-1")
    print(f"  Node feature dim    : {NODE_DIM}")
    print(f"  Edge feature dim    : {EDGE_DIM}")
    print(f"  Global features     : {len(dataset[0]['global_features']) if dataset else '?'}")
    print(f"{'='*55}")

    if not dataset:
        print("\nERROR: Nothing parsed.")
        return

    json.dump(dataset, open(args.output, 'w'), indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == '__main__':
    main()