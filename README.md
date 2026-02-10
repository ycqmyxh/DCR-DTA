# DCR-DTA
# [Project Name] DCR-DTA

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Link-green.svg)](https://doi.org/your-paper-doi-here)

> Official PyTorch implementation of **"DCR-DTA: An Efficient Hybrid Graph-Sequence Network with Dynamic Contextual Regularization and Induced-Fit Interaction for Drug-Target Affinity Prediction"**.

## 📖 Abstract
Accurate drug–target affinity (DTA) prediction is crucial for virtual screening. This paper proposes **DCR-DTA**, a robust deep learning framework that:
- Explicitly models **Bidirectional Induced-Fit Interactions**.
- Enforces **Dynamic Contextual Regularization (DCR)** to mitigate feature anisotropy.
- Achieves state-of-the-art performance on **Davis** and **KIBA** datasets.


*(Figure 1: The schematic overview of the DCR-DTA framework.)*

## 🚀 Key Features
- **FlashAttention & Torch Compile**: Optimized for maximum training speed.
- **Mixed Precision (AMP)**: Automatic BFloat16/Float16 support.
- **Bi-directional Interaction**: Captures dynamic ligand-receptor binding.
- **Rigorous Metrics**: Computes MSE, CI, $r_m^2$, Pearson, and Spearman.

## 🛠️ Environment
We recommend using **PyTorch 2.0+** to enable all acceleration features.

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/DCR-DTA.git](https://github.com/your-username/DCR-DTA.git)
cd DCR-DTA

# 2. Install dependencies
# pip install numpy pandas scipy scikit-learn networkx tqdm rdkit torch
DCR-DTA/
├── Data/
│   └── KIBA/
│       ├── drug_smiles.json        # Input SMILES
│       ├── protein_seq.json        # Input Sequences
│       ├── Y.txt                   # Affinity Labels
│       └── plm_features_SOTA/      # Pre-trained Features
│           └── plm_embeddings_sota_final.pt
├── model_utils.py
├── main.py
└── ...
> **⚠️ Note regarding Pre-trained Features:**
> The pre-computed PLM feature file (`plm_embeddings_sota_final.pt`) is **excluded** from this repository due to GitHub's file size limitations (exceeding 100MB).
>
> We provide a generation script to reproduce these features locally. Please run the following command before training:
> ```bash
> python gen_plm_features.py
> ```
