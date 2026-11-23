# FABRIC

FABRIC is a deep learning framework for drug–target interaction (DTI) prediction. It separates drug and protein representations into low- and high-frequency bands and pretrains the model by reconstructing masked bands. For prediction, it uses bilinear attention to capture interactions within and across frequency bands, then adaptively fuses the resulting features.

---

## Environment Setup

### Requirements

- Python 3.9
- PyTorch 2.1.1
- CUDA 12.1
- RDKit 2025.9.2

### Main Dependencies

```text
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.6.1
joblib==1.3.2
transformers==4.51.3
```

`transformers` is used to extract protein embeddings with ESM-2.

---

## Run

Clone the repository, then run the GPU demo from the directory that contains `FABRIC/`.

```bash
python -m FABRIC.demo --device cuda
```

For the Human dataset, restore the published splits and cache the ESM-2 embeddings for fold 0.

```bash
python -m zipfile -e FABRIC/datasets.zip fabric-data-package
python fabric-data-package/restore_datasets.py --output fabric-data
python -m FABRIC.esm_cache --csv fabric-data/data/human/random/0/train.csv fabric-data/data/human/random/0/validation.csv fabric-data/data/human/random/0/test.csv --out fabric-cache/human
```

Run MFBR pretraining and supervised training on the random split.

```bash
python -m FABRIC.train --split-dir fabric-data/data/human/random/0 --esm-cache fabric-cache/human --out fabric-runs/human-random-0
```

For cold-drug, cold-target, or cold-pair evaluation, use `e2`, `e3`, or `e4` in the split paths and choose a separate output directory. The validation AUPRC determines the checkpoint used for test evaluation.
