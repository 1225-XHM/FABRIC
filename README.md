# FABRIC

FABRIC is a deep learning framework for drug–target interaction (DTI) prediction. It separates drug and protein representations into low- and high-frequency bands and pretrains the model by reconstructing masked bands. For prediction, it uses bilinear attention to capture interactions within and across frequency bands, then adaptively fuses the resulting features.

## Environment

Use Python 3.10 with PyTorch installed for your CPU or GPU. The training pipeline was smoke-tested on CPU with PyTorch 2.3.1, RDKit 2025.9.2, NumPy 1.26.4, pandas 2.3.3 and scikit-learn 1.7.2. To install the remaining dependencies in a Python environment containing PyTorch, run

```bash
python -m pip install numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2 rdkit==2025.9.2 transformers==4.51.3
```

The embedding step downloads `facebook/esm2_t30_150M_UR50D` on first use or reads it from the local Hugging Face cache.

## Train one fold

Run these commands from the directory **containing** the cloned `FABRIC/` folder. The example trains Human, random split, fold 0. Each command runs in Bash or PowerShell.

1. Restore the published splits. The archive includes a manifest reconstructed from the published catalog and fold indices. The restoration script verifies each output CSV against its SHA-256 digest.

   ```bash
   python -m zipfile -e FABRIC/datasets.zip fabric-data-package
   python fabric-data-package/restore_datasets.py --output fabric-data
   ```

2. Cache the frozen ESM-2 embeddings for the sequences in the chosen fold.

   ```bash
   python -m FABRIC.esm_cache --csv fabric-data/data/human/random/0/train.csv fabric-data/data/human/random/0/validation.csv fabric-data/data/human/random/0/test.csv --out fabric-cache/human
   ```

3. Start MFBR pretraining followed by supervised training.

   ```bash
   python -m FABRIC.train --split-dir fabric-data/data/human/random/0 --esm-cache fabric-cache/human --out fabric-runs/human-random-0
   ```

   `python FABRIC/train.py` accepts the same arguments. Use a separate output folder per run. The default device is CUDA when available, otherwise CPU. Pass `--seed` to select a run seed. For other datasets and folds, change the three CSV paths, split directory and output directory consistently.

The distributed splits contain 5,991 Human pairs, 27,437 BioSNAP pairs, 23,621 BindingDB pairs, and 33,582 DrugBank pairs in each complete random fold. These are the counts of the released rows. Dataset directory names are `human`, `biosnap`, `bindingdb`, and `DrugBank`. Each has folds 0–4 for `random`, `e2` (cold drug), `e3` (cold target), and `e4` (cold pair).

## Method and defaults

MFBR uses equation (13) in both modalities. Given the unmasked input `X` and independent band masks `m_k`, the encoder receives `X - sum_k(m_k * z_k)`. Here `X` contains atom features or normalized ESM-2 embeddings before applying the spectral window. The reconstruction targets are detached. Each modality has a decoder shared across its bands. The weighted Smooth L1 error is computed on masked bands over valid atoms or residues. Both reconstruction losses and the band separation and occupancy penalties enter the pretraining objective.

| Setting | Default |
| --- | --- |
| Samples per optimizer update | 64 |
| Supervised epochs | Up to 100 |
| Hidden dimension | 256 |
| Initial learning rate | 0.001 for supervised training, MFBR, and spectral parameters |
| Frequency bands | 2 |
| Optimizer | AdamW |
| Weight decay | 0.01 on the decay parameter group |
| Early stopping | 20 epochs on validation AUPRC |
| MFBR epochs | 20 |
| Mask probabilities (low, high) | 0.15, 0.50 |
| Reconstruction weights (low, high) | 1.0, 2.0 |

The loaders divide a logical batch into microbatches when padded atom–residue dimensions exceed `--token-budget`. Gradients are weighted by the number of samples and accumulated, with one optimizer update per 64 samples or the final partial batch of an epoch. Lower the budget if needed. Each individual sample must still fit on the device.

The training split supplies the default unlabeled pretraining corpora. The validation split determines the best checkpoint by average precision and the classification threshold by maximum F1. The test split uses this checkpoint and threshold. The output directory contains `mfbr_pretrained.pt`, `best.pt`, training histories, validation and test predictions, and `results.json` with the run settings and metrics. AUPRC is calculated as average precision.

## Quick demo

```bash
python -m FABRIC.demo --device cpu
```

The demo uses four synthetic examples and runs one optimizer step in each stage. Both the demo and the formal training command default to the residual reconstruction above. The demo's optional `--recomposition retained` selects the earlier sum-of-retained-bands variant for comparison.
