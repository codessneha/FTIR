# GNN-FTIR: Predict FTIR Spectra from Gaussian Log Files
## Complete guide — from scratch

---

## What this does

```
Your Gaussian .log file  →  Molecular graph  →  GNN  →  Full FTIR spectrum plot
```

You give it `.log` files (which contain xyz coordinates + IR peaks).
The GNN learns the relationship between molecular structure and IR spectrum.
At prediction time, you give it a new `.log` file and it outputs the full FTIR spectrum curve.

---

## The key concept: how DFT peaks become an FTIR graph

Your `.log` file has discrete peaks like this:
```
 Frequencies --   1620.4   3412.1   890.3
 IR Inten    --     45.3    120.1     8.7
```

A real FTIR spectrum is a **continuous curve** — each peak is broadened
into a bell (Gaussian) shape, then all bells are added together:

```
                 ___
                /   \          _____
_______________/     \________/     \________  ← FTIR curve
         890        1620            3412      wavenumber (cm⁻¹)
```

This script does that broadening automatically. The GNN learns to predict
that curve directly from the molecular graph (atoms + bonds + charges).

---

## Setup

```bash
# 1. Install dependencies
pip install torch torchvision
pip install torch-geometric torch-scatter torch-sparse
pip install numpy scipy matplotlib scikit-learn

# 2. Organise your files
mkdir logs
# copy all your .log files into logs/
```

Your folder should look like:
```
ftir_gnn/
├── logs/
│   ├── graphene_01.log
│   ├── graphene_02.log
│   └── ...  (all your Gaussian .log files here)
├── step1_parse_logs.py
├── step2_build_dataset.py
├── step3_model.py
├── step4_train.py
└── step5_predict.py
```

---

## Run it — 5 commands

### Command 1 — Parse your .log files
```bash
python step1_parse_logs.py --log_dir ./logs --output data/raw_dataset.json
```
Reads every `.log` file and extracts:
- Atom positions (xyz)
- Bonds (inferred from distances)
- Mulliken charges
- IR frequencies + intensities

Output: `data/raw_dataset.json`

---

### Command 2 — Build the FTIR spectrum + graph objects
```bash
python step2_build_dataset.py --input data/raw_dataset.json --output_dir data/graphs
```
- Converts each structure into a molecular graph (PyTorch Geometric format)
- Converts discrete IR peaks → continuous 360-bin FTIR spectrum (the training target)

Output: `data/graphs/graph_0000.pt`, `graph_0001.pt`, ... + `config.json`

---

### Command 3 — Train the GNN
```bash
python step4_train.py --graph_dir data/graphs --out_dir checkpoints --epochs 500
```
- Uses 5-fold cross-validation (because you have < 50 structures)
- Augments training data 4× automatically
- Saves best model for each fold

Output: `checkpoints/fold1_best.pt` ... `fold5_best.pt` + `kfold_summary.json`

---

### Command 4 — Check which fold performed best
```bash
cat checkpoints/kfold_summary.json
```
Look for `"best_fold"`. Use that fold's checkpoint for prediction.

---

### Command 5 — Predict FTIR for a new structure
```bash
python step5_predict.py \
    --checkpoint checkpoints/fold1_best.pt \
    --log_file   logs/your_new_structure.log \
    --output_dir results/
```
Outputs:
- `results/your_new_structure_ftir.png` — the FTIR spectrum plot
- Peak table printed to terminal

Or predict all your structures at once and compare to DFT:
```bash
python step5_predict.py \
    --checkpoint  checkpoints/fold1_best.pt \
    --dataset_json data/raw_dataset.json \
    --output_dir   results/
```

---

## What the output plot looks like

```
Normalised
Intensity
   1.0 |        ___
       |       /   \          O-H stretch
       |      /     \              ___
   0.5 |_____/       \____________/   \____
       |
   0.0 |___________________________________
       4000   3000   2000   1000   400
                  Wavenumber (cm⁻¹)

    ── DFT/Gaussian (ground truth)
    -- GNN Prediction
```

Key FTIR regions for graphene are colour-shaded:
| Region | Wavenumber | Meaning |
|--------|-----------|---------|
| O–H stretch | 3200–3700 | Hydroxyl groups (graphene oxide) |
| C–H stretch | 2850–3100 | Edge hydrogen termination |
| C=O stretch | 1700–1850 | Carboxyl groups (GO) |
| C=C (G band) | 1550–1700 | Graphene backbone |
| C–O–C epoxy | 1050–1280 | Epoxide groups (GO) |

---

## Node features (what the GNN sees per atom)

| # | Feature | Why |
|---|---------|-----|
| 0 | Atomic number (normalised) | Atom identity |
| 1 | Electronegativity | Bond polarity → IR activity |
| 2 | Covalent radius | Bond length context |
| 3 | Is carbon | Graphene backbone |
| 4 | Is hydrogen | Edge termination |
| 5 | Is oxygen | GO functional groups |
| 6 | Is heteroatom (N/S/F/Cl) | Dopants |
| 7 | Mulliken charge | Electronic environment |
| 8 | Bond degree (normalised) | Local connectivity |

---

## Good results vs bad results

| Pearson r | What it means |
|-----------|--------------|
| > 0.90 | Excellent — peaks in right positions and heights |
| 0.75–0.90 | Good — main peaks correct, minor ones off |
| 0.50–0.75 | Fair — needs more structures or epochs |
| < 0.50 | Poor — add more diverse structures |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No IR data" | Your .log must use the `Freq` keyword, not just `Opt` |
| "Cannot parse coordinates" | Check the job ended with "Normal termination" |
| Imaginary frequencies | Structure didn't converge — redo with `opt=tight` |
| Poor Pearson r | Add more structures with diverse O/C ratios |
| CUDA out of memory | Use `--batch_size 4` |