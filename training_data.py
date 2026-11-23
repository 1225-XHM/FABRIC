from functools import partial
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data import AA, collate, mol_graph, standardize_smiles


class InteractionDataset(Dataset):
    def __init__(self, frame, cache):
        self.rows = list(frame[["smiles", "sequence", "label"]].itertuples(index=False, name=None))
        self.cache = cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        smiles, seq, label = self.rows[index]
        return mol_graph(smiles), self.cache.get(seq), label, smiles, seq, index


class PretrainDataset(Dataset):
    """Independently shuffle the two training corpora. MFBR ignores pair identity."""
    def __init__(self, drugs, proteins, cache, seed):
        self.drugs, self.proteins = drugs, proteins
        self.cache, self.seed = cache, seed
        self.reshuffle(0)

    def reshuffle(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        size = len(self)
        self.di = np.resize(rng.permutation(len(self.drugs)), size)
        self.pi = np.resize(rng.permutation(len(self.proteins)), size)

    def __len__(self):
        return max(len(self.drugs), len(self.proteins))

    def __getitem__(self, index):
        smiles = self.drugs[self.di[index]]
        seq = self.proteins[self.pi[index]]
        return mol_graph(smiles), self.cache.get(seq), None, smiles, seq, index


def build_corpus(train, heldout, extra_drugs, extra_proteins):
    drugs, proteins = set(train.smiles), set(train.sequence)
    if extra_drugs:
        values = pd.read_csv(extra_drugs, keep_default_na=False)["smiles"]
        external = {standardize_smiles(value) for value in values}
        if None in external:
            raise ValueError("Invalid SMILES in the external pretraining corpus")
        drugs.update(external.difference(heldout.smiles))
    if extra_proteins:
        values = pd.read_csv(extra_proteins, keep_default_na=False)["sequence"]
        external = {str(value).strip().upper() for value in values}
        if any(re.fullmatch(AA, seq) is None for seq in external):
            raise ValueError("Invalid sequence in the external pretraining corpus")
        proteins.update(external.difference(heldout.sequence))
    return sorted(drugs), sorted(proteins), {"drugs": len(drugs), "proteins": len(proteins)}


def collate_update(items, token_budget):
    """Collate memory-sized microbatches belonging to one optimizer update."""
    groups, current = [], []
    max_atoms = max_residues = 0
    for item in items:
        atoms, residues = item[0].x.shape[0], item[1].shape[0]
        new_atoms, new_residues = max(atoms, max_atoms), max(residues, max_residues)
        cost = (len(current) + 1) * new_atoms * new_residues
        if current and token_budget and cost > token_budget:
            groups.append(collate(current))
            current = []
            new_atoms, new_residues = atoms, residues
        current.append(item)
        max_atoms, max_residues = new_atoms, new_residues
    groups.append(collate(current))
    return groups


def make_loader(dataset, batch_size, token_budget, shuffle, seed, workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=partial(collate_update, token_budget=token_budget))


def audit(train, split, hits=None):
    records = pd.DataFrame({
        "unseen_drug": ~split.smiles.isin(train.smiles),
        "unseen_protein": ~split.sequence.isin(train.sequence),
    }).reset_index(drop=True)
    summary = {key: int(records[key].sum()) for key in records}
    if hits is not None:
        if "sequence" not in hits or hits.sequence.duplicated().any():
            raise ValueError("Protein hits CSV requires one row per sequence")
        extra = hits.set_index("sequence").reindex(split.sequence).reset_index(drop=True)
        records = records.join(extra.add_prefix("hit_"))
    return summary, records
