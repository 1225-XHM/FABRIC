from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
DATASETS = ('bindingdb', 'biosnap', 'DrugBank', 'human')
PROTOCOLS = ('random', 'e2', 'e3', 'e4')
ROLES = ('train', 'validation', 'test')
MANIFEST = 'split_manifest.json'

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for buf in iter(lambda: fh.read(2 ** 20), b''):
            digest.update(buf)
    return digest.hexdigest()

def read_manifest(project: Path) -> dict:
    path = Path(project).resolve() / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f'Fixed-fold integrity manifest is missing: {path}')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if obj.get('schema_version') != 1:
        raise ValueError(f'Unsupported split manifest version: {path}')
    return obj

def fold_location(project: Path, dataset: str, protocol: str, fold: int):
    if dataset not in DATASETS or protocol not in PROTOCOLS:
        raise ValueError(f'Unsupported dataset/protocol: {dataset}/{protocol}')
    n = 25 if protocol == 'e4' else 5
    if not isinstance(fold, int) or not 0 <= fold < n:
        raise ValueError(f'Invalid fold {fold} for {protocol}; expected 0..{n - 1}')
    project = Path(project).resolve()
    return (project / 'data' / dataset / protocol / str(fold), project / MANIFEST)

def get_fold_metadata(project: Path, dataset: str, protocol: str, fold: int) -> dict:
    fold_location(project, dataset, protocol, fold)
    manifest = read_manifest(project)
    try:
        m = manifest['folds'][dataset][protocol][str(fold)]
    except KeyError as e:
        raise ValueError(f'No frozen split record for {dataset}/{protocol}/{fold}') from e
    if m.get('dataset') != dataset or m.get('protocol') != protocol or m.get('fold') != fold or (not m.get('frozen_fold')):
        raise ValueError('Frozen fold metadata identity mismatch')
    if m.get('n_cleaned_rows') != manifest['datasets'][dataset]['cleaned_rows']:
        raise ValueError('Dataset total does not match frozen fold metadata')
    return m

def load_fixed_split(folder: Path, *, verify_source: bool=True):
    folder = Path(folder).resolve()
    fold_text, protocol, dataset = (folder.name, folder.parent.name, folder.parent.parent.name)
    if not fold_text.isdecimal():
        raise ValueError(f'Fold directory must be numeric: {folder}')
    project = folder.parents[3]
    expected, _ = fold_location(project, dataset, protocol, int(fold_text))
    if expected != folder:
        raise ValueError(f'Unrecognized fold path {folder}')
    meta = get_fold_metadata(project, dataset, protocol, int(fold_text))
    names = {p.name for p in folder.iterdir() if p.is_file()}
    if names != {f'{name}.csv' for name in ROLES}:
        raise ValueError('Each fold directory must contain exactly three CSVs: train, validation, test')
    frames = {}
    for name in ROLES:
        path = folder / f'{name}.csv'
        if sha256(path) != meta['csv_sha256'][name]:
            raise ValueError(f'Fixed CSV checksum mismatch: {path}')
        df = pd.read_csv(path, keep_default_na=False)
        required = {'source_row_id', 'smiles', 'sequence', 'label', 'drug_group', 'protein_group'}
        if not required.issubset(df.columns):
            raise ValueError(f'Missing CSV columns: {required - set(df.columns)} in {path}')
        ex = meta['csv_sizes'][name]
        if len(df) != ex['rows'] or not df.label.isin((0, 1)).all() or (not pd.api.types.is_integer_dtype(df.label)) or (df.label.nunique() != 2) or (int(df.label.sum()) != ex['positive']) or (not pd.api.types.is_integer_dtype(df.source_row_id)) or df.source_row_id.duplicated().any() or df.duplicated(['smiles', 'sequence']).any() or df[['smiles', 'sequence', 'drug_group', 'protein_group']].isna().any().any() or df[['smiles', 'sequence', 'drug_group', 'protein_group']].eq('').any().any() or df.source_row_id.lt(0).any() or df.source_row_id.ge(meta['n_cleaned_rows']).any():
            raise ValueError(f'Invalid label, record count or duplicated rows: {path}')
        frames[name] = df
    vals = [frames[r] for r in ROLES]
    if sum(map(len, vals)) + int(meta['bridge_pairs_discarded']) != meta['n_cleaned_rows']:
        raise ValueError('Train/validation/test counts plus excluded bridge rows are inconsistent')
    for i in range(3):
        for j in range(i + 1, 3):
            x, y = (vals[i], vals[j])
            if set(x.source_row_id) & set(y.source_row_id) or set(zip(x.smiles, x.sequence)) & set(zip(y.smiles, y.sequence)):
                raise ValueError('Data leakage: DTI pairs shared across train/validation/test')
            if protocol in ('e2', 'e4') and set(x.drug_group) & set(y.drug_group):
                raise ValueError('Data leakage: drug scaffold groups shared across splits')
            if protocol in ('e3', 'e4') and set(x.protein_group) & set(y.protein_group):
                raise ValueError('Data leakage: exact-sequence protein groups shared across splits')
    return (frames['train'], frames['validation'], frames['test'], meta)
