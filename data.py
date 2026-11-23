from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "other"]
HYBRID = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
          Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
          Chem.rdchem.HybridizationType.SP3D2]
ATOM_DIM = len(ELEMENTS) + 6 + 5 + len(HYBRID) + 5


def _onehot(value, choices):
    return [float(value == c) for c in choices]


def atom_features(atom):
    sym = atom.GetSymbol() if atom.GetSymbol() in ELEMENTS else "other"
    f = _onehot(sym, ELEMENTS)
    f += _onehot(min(atom.GetDegree(), 5), range(6))
    f += _onehot(min(atom.GetTotalNumHs(), 4), range(5))
    f += _onehot(atom.GetHybridization(), HYBRID)
    f += [float(atom.GetIsAromatic()), float(atom.IsInRing()),
          float(atom.GetFormalCharge()) / 2.0,
          float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
          float(atom.GetNumRadicalElectrons() > 0)]
    return f


_chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


def standardize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = _chooser.choose(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


@dataclass
class MolGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    evals: torch.Tensor
    evecs: torch.Tensor
    lap: torch.Tensor


@lru_cache(maxsize=200000)
def mol_graph(smiles: str) -> MolGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    n = mol.GetNumAtoms()
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    adj = torch.zeros((n, n), dtype=torch.float64)
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = 1.0
        src += [i, j]
        dst += [j, i]
    deg = adj.sum(1)
    dinv = torch.where(deg > 0, deg.clamp_min(1e-12).rsqrt(), torch.zeros_like(deg))
    lap = torch.diag((deg > 0).double()) - dinv[:, None] * adj * dinv[None, :]
    evals, evecs = torch.linalg.eigh(lap)
    return MolGraph(x, torch.tensor([src, dst], dtype=torch.long).reshape(2, -1),
                    evals.clamp(0.0, 2.0).float(), evecs.float(), lap.float())


@dataclass
class DrugBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    pos: torch.Tensor
    mask: torch.Tensor
    evals: torch.Tensor
    evecs: torch.Tensor
    lap: torch.Tensor

    def to(self, device):
        return DrugBatch(*[getattr(self, f).to(device) for f in self.__dataclass_fields__])

    @property
    def size(self):
        return self.mask.shape[0]

    def dense(self, h: torch.Tensor) -> torch.Tensor:
        out = h.new_zeros(self.mask.shape[0], self.mask.shape[1], h.shape[-1])
        return out.index_put((self.batch, self.pos), h)

    def sparse(self, hd: torch.Tensor) -> torch.Tensor:
        return hd[self.batch, self.pos]


@dataclass
class ProteinBatch:
    x: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor

    def to(self, device):
        return ProteinBatch(self.x.to(device), self.mask.to(device), self.lengths.to(device))

    @property
    def size(self):
        return self.x.shape[0]


@dataclass
class PairBatch:
    drug: DrugBatch
    protein: ProteinBatch
    label: Optional[torch.Tensor]
    drug_ids: List[str]
    protein_ids: List[str]
    index: Optional[List[int]] = None

    def to(self, device):
        lab = None if self.label is None else self.label.to(device)
        return PairBatch(self.drug.to(device), self.protein.to(device), lab,
                         self.drug_ids, self.protein_ids, self.index)


def collate_drugs(graphs: List[MolGraph]) -> DrugBatch:
    sizes = [g.x.shape[0] for g in graphs]
    b, nmax = len(graphs), max(sizes)
    xs, edges, batch, pos = [], [], [], []
    evals = torch.zeros(b, nmax)
    evecs = torch.zeros(b, nmax, nmax)
    lap = torch.zeros(b, nmax, nmax)
    mask = torch.zeros(b, nmax, dtype=torch.bool)
    offset = 0
    for i, (g, n) in enumerate(zip(graphs, sizes)):
        xs.append(g.x)
        edges.append(g.edge_index + offset)
        batch.append(torch.full((n,), i, dtype=torch.long))
        pos.append(torch.arange(n))
        evals[i, :n] = g.evals
        evecs[i, :n, :n] = g.evecs
        lap[i, :n, :n] = g.lap
        mask[i, :n] = True
        offset += n
    return DrugBatch(torch.cat(xs), torch.cat(edges, 1), torch.cat(batch), torch.cat(pos),
                     mask, evals, evecs, lap)


def collate_proteins(embs: List[torch.Tensor]) -> ProteinBatch:
    lengths = torch.tensor([e.shape[0] for e in embs])
    b, lmax, d = len(embs), int(lengths.max()), embs[0].shape[1]
    x = torch.zeros(b, lmax, d)
    mask = torch.zeros(b, lmax, dtype=torch.bool)
    for i, e in enumerate(embs):
        x[i, :e.shape[0]] = e
        mask[i, :e.shape[0]] = True
    return ProteinBatch(x, mask, lengths)


AA = r"[ACDEFGHIKLMNPQRSTVWYXUOBZ]+"


def seq_key(seq):
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


class ProteinCache:
    def __init__(self, folder, max_bytes=6 * 2 ** 30):
        self.folder = Path(folder)
        mf = self.folder / "manifest.json"
        if not mf.exists():
            raise FileNotFoundError(f"{mf} missing: run `python -m freqdti.esm_cache` first")
        self.manifest = json.loads(mf.read_text(encoding="utf-8"))
        if self.manifest.get("source") != "esm2":
            raise ValueError("Cache is not real ESM-2; run `python -m freqdti.esm_cache` first")
        self.dim = int(self.manifest["embedding_dim"])
        self.store = np.float32 if self.manifest.get("dtype") == "float32" else np.float16
        self.mem, self.max_bytes, self.bytes = OrderedDict(), int(max_bytes), 0

    def has(self, seq):
        return (self.folder / f"{seq_key(seq)}.npy").exists()

    def get(self, seq):
        key = seq_key(seq)
        if key not in self.mem:
            arr = np.load(self.folder / f"{key}.npy", allow_pickle=False)
            if arr.shape != (len(seq), self.dim):
                raise ValueError(f"Embedding shape {arr.shape} != ({len(seq)}, {self.dim})")
            t = torch.from_numpy(arr.astype(self.store, copy=False))
            self.mem[key] = t
            self.bytes += t.numel() * t.element_size()
            while self.bytes > self.max_bytes and len(self.mem) > 1:
                _, old = self.mem.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()
        self.mem.move_to_end(key)
        return self.mem[key].float()


def collate(items):
    g, p, lab, s, q, idx = zip(*items)
    label = None if lab[0] is None else torch.tensor(lab, dtype=torch.float32)
    return PairBatch(collate_drugs(list(g)), collate_proteins(list(p)), label, list(s), list(q),
                     list(idx))
