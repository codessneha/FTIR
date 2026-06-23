# GNN-FTIR: Predict FTIR Spectra from Gaussian Log Files

**Project Summary**: This repository provides a Graph Neural Network (GNN) pipeline to predict continuous FTIR spectra directly from molecular structures. By leveraging Gaussian `.log` files as input, the GNN learns the complex relationship between 3D geometry, electronic properties, and vibrational infrared activity.

---

## 🚀 The Entire Process — At a Glance

The pipeline consists of four main stages:

1.  **Data Extraction** (`parse_data.py`): Parses Gaussian `.log` files and experimental/computed Excel spectra. It extracts 15D node features (including APT charges), 8D edge features, and 9 global molecular descriptors (HOMO/LUMO, Dipole, ZPE).
2.  **Dataset Preparation** (`build_graphs.py`): Converts the extracted data into PyTorch Geometric graph objects and interpolates discrete IR intensities into a continuous 500-bin spectrum.
3.  **GNN Training** (`train.py`): Implements K-Fold cross-validation with robust data augmentation (noise injection) to ensure high generalization even on small molecular datasets.
4.  **Spectral Prediction** (`predict.py`): A standalone inference tool that takes a model checkpoint and a new structure (or dataset) to generate spectral plots and calculate accuracy metrics (Pearson R, SAM, RMSE).

---

## 🛠️ Key Technical Improvements

-   **Wide Element Support**: Native support for **transition metals** (Pd, Pt, Ni, Cu, Zn, Co, Fe, Mn, Cr, Ti, Ru, Rh, Ag, Au).
-   **Enhanced Graph Features**: 15D node features capturing local chemistry (electronegativity, radii, aromaticity) and 8D edge features for bond characterisation.
-   **Global Descriptors**: Conditions the GNN on molecular properties like HOMO-LUMO gap and dipole moment for state-of-the-art accuracy.
-   **Advanced Loss Function**: Uses a combination of MSE, Cosine Similarity, and Gradient matching (`FTIRLoss`) to capture both peak position and shape.
-   **Automated Evaluation**: Integrated **Spectral Angle Mapper (SAM)** and Pearson Correlation for rigorous spectral validation.

---

## 📦 Project Structure

```bash
ftir_gnn/
├── data/
│   ├── logs/             # Input Gaussian .log files
│   ├── ftir/             # Input Reference Spectra (.xlsx)
│   └── graphs/           # Generated PyTorch graph files
├── parse_data.py         # Step 1: Raw data parser
├── build_graphs.py       # Step 2: Graph dataset builder
├── model.py              # Step 3: GAT-based GNN Architecture
├── train.py              # Step 4: K-Fold Training Pipeline
└── predict.py            # Step 5: Inference & Visualisation
```

---

## 📖 Usage Guide

### 1. Data Parsing
```bash
python parse_data.py --log_dir data/logs --ftir_dir data/ftir --output data/dataset.json
```
Extracts coordinates, charges, and spectral targets. Supports transition metals automatically.

### 2. Graph Building
```bash
python build_graphs.py --input data/dataset.json --output_dir data/graphs
```
Generates the graph representation used by PyTorch Geometric.

### 3. Training
```bash
python train.py --graph_dir data/graphs --out_dir checkpoints --epochs 500
```
Trains the model using 5-fold cross-validation and performs automatic data augmentation.

### 4. Prediction
```bash
python predict.py --checkpoint checkpoints/fold1_best.pt --log_file your_molecule.log --output_dir results/
```
Generates a detailed FTIR plot with annotated peaks and accuracy metrics.

---

## 📊 Evaluation Metrics

| Metric | Goal | Meaning |
|---|---|---|
| **Pearson r** | > 0.90 | Peak positions and relative heights are well-matched. |
| **SAM (deg)** | < 10° | Excellent spectral shape similarity (Spectral Angle Mapper). |
| **RMSE** | < 0.05 | Low overall intensity error across all 500 bins. |

---

## ⚠️ Troubleshooting

-   **"No IR data"**: Ensure your Gaussian job used the `Freq` keyword.
-   **"Transition Metal Mismatch"**: If using an unsupported metal, update the `ATOMIC_NUM` table in `parse_data.py` and `predict.py`.
-   **"CUDA OOM"**: Reduce `--batch_size` in `train.py`.
-   **"Imaginary frequencies"**: Structure didn't converge — redo with `opt=tight`.