import argparse
import csv
import re
from itertools import islice
from pathlib import Path

import torch
from torch.nn import functional as F

from . import config as C
from .data import AA, ProteinCache, collate, mol_graph, standardize_smiles
from .model import MFBR, FreqAwareDTI, ModelConfig


def synthetic_batch():
    smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN"]
    sequences = ["ACDEFG", "AVKTLRSM", "MPSTWY", "GLANVSEPR"]
    generator = torch.Generator().manual_seed(4)
    return collate([
        (mol_graph(s), torch.randn(len(q), C.ESM_DIM, generator=generator),
         float(i % 2), s, q, i)
        for i, (s, q) in enumerate(zip(smiles, sequences))
    ])


def read_batch(csv_path, cache, max_rows):
    items = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"smiles", "sequence", "label"}
        if not required.issubset(rows.fieldnames or ()):
            raise ValueError("CSV columns must include smiles, sequence, and label")
        for i, row in enumerate(islice(rows, max_rows)):
            smiles = standardize_smiles(row["smiles"])
            seq = (row["sequence"] or "").strip().upper()
            if smiles is None or re.fullmatch(AA, seq) is None:
                raise ValueError(f"Invalid SMILES or protein sequence on CSV line {i + 2}")
            label = float(row["label"])
            if label not in (0.0, 1.0):
                raise ValueError(f"Expected a binary label on CSV line {i + 2}")
            items.append((mol_graph(smiles), cache.get(seq), label, smiles, seq, i))
    if not items:
        raise ValueError("CSV contains no examples")
    return collate(items)


def training_step(loss, module, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()


def regularization(model):
    terms = model.regularizers()
    return 1e-3 * terms["separation"] + 1e-2 * terms["occupancy"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one MFBR step, one DTI step, and inference")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--esm-cache", type=Path)
    parser.add_argument("--max-rows", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recomposition", choices=["retained", "residual"], default=C.MFBR_RECOMPOSITION)
    args = parser.parse_args(argv)
    if (args.csv is None) != (args.esm_cache is None):
        parser.error("--csv and --esm-cache must be provided together")
    if args.max_rows < 1:
        parser.error("--max-rows must be positive")

    torch.manual_seed(7)
    if args.csv is None:
        batch = synthetic_batch()
        input_name = "synthetic ESM-shaped embeddings"
    else:
        cache = ProteinCache(args.esm_cache)
        batch = read_batch(args.csv, cache, args.max_rows)
        input_name = "CSV with cached ESM-2 embeddings"
    batch = batch.to(args.device)
    model = FreqAwareDTI(ModelConfig(esm_dim=batch.protein.x.shape[-1])).to(args.device)
    pretrain = MFBR(model, recomposition=args.recomposition).to(args.device)
    optimizer = torch.optim.AdamW(pretrain.parameters(), lr=C.LR_PRETRAIN)
    loss, info = pretrain(batch)
    training_step(loss + regularization(model), pretrain, optimizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR)
    loss = F.binary_cross_entropy_with_logits(model(batch), batch.label)
    training_step(loss + regularization(model), model, optimizer)

    model.eval()
    with torch.inference_mode():
        logits, details = model(batch, return_details=True)
        print(f"Input: {input_name} ({batch.drug.size} pairs)")
        print(f"MFBR ({args.recomposition}): {info}")
        print(f"DTI classification loss: {loss.item():.6f}")
        print(f"LBD shapes: drug {tuple(details['zd'].shape)}, protein {tuple(details['zp'].shape)}")
        print(f"FABA paths: {', '.join(details['paths'])}")
        print(f"Fusion weights (first pair): {details['fusion_weights'][0].tolist()}")
        print(f"Demo probabilities: {logits.sigmoid().tolist()}")


if __name__ == "__main__":
    main()
