import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

# Allow both 'python train.py' and 'python -m FABRIC.train'.
if __name__ == "__main__" and not __package__:
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    __package__ = package_dir.name

from . import config as C
from .data import ProteinCache
from .data_process import load_fixed_split
from .metric import best_f1_threshold, evaluate
from .model import FreqAwareDTI, MFBR, ModelConfig
from .training_data import InteractionDataset, PretrainDataset, audit, build_corpus, make_loader
from .utils import param_groups, predict, reg_loss, set_seed


def update_batch(group, model, optimizer, args, device, mfbr=None):
    """One AdamW update for the sample mean over one logical batch."""
    module = model if mfbr is None else mfbr
    size = sum(batch.drug.size for batch in group)
    optimizer.zero_grad(set_to_none=True)
    total, stats = 0.0, {}
    for batch in group:
        batch = batch.to(device)
        weight = batch.drug.size / size
        if mfbr is None:
            loss = F.binary_cross_entropy_with_logits(model(batch), batch.label)
        else:
            loss, info = mfbr(batch)
            for key, value in info.items():
                stats[key] = stats.get(key, 0.0) + weight * value
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite batch loss")
        (weight * loss).backward()
        total += weight * loss.detach().item()

    penalty = reg_loss(model, args)
    if not torch.isfinite(penalty):
        raise FloatingPointError("Non-finite band regularizer")
    penalty.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total + penalty.detach().item(), size, stats


def pretrain(model, train, heldout, cache, args, device, log):
    drugs, proteins, report = build_corpus(train, heldout, args.pretrain_drugs, args.pretrain_proteins)
    log["pretrain_corpus"] = report
    missing = [seq for seq in proteins if not cache.has(seq)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} pretraining proteins lack ESM-2 embeddings")
    dataset = PretrainDataset(drugs, proteins, cache, args.seed)
    loader = make_loader(dataset, args.batch_size, args.token_budget, True, args.seed, args.workers)
    mfbr = MFBR(model, args.q_low, args.q_high, args.w_low, args.w_high,
                recomposition=C.MFBR_RECOMPOSITION).to(device)
    optimizer = torch.optim.AdamW(param_groups(mfbr, args.weight_decay, args.lr_pretrain, args.spectral_lr))
    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        dataset.reshuffle(epoch)
        mfbr.train()
        total, seen, sums = 0.0, 0, {}
        for group in loader:
            loss, size, info = update_batch(group, model, optimizer, args, device, mfbr)
            total += loss * size
            seen += size
            for key, value in info.items():
                sums[key] = sums.get(key, 0.0) + size * value
        history.append({"epoch": epoch, "loss": total / seen,
                        **{key: value / seen for key, value in sums.items()}})
        print(f"[MFBR] epoch {epoch} loss {total / seen:.4f}", flush=True)
    out = Path(args.out)
    pd.DataFrame(history).to_csv(out / "mfbr_history.csv", index=False)
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(),
                "mfbr_recomposition": mfbr.recomposition}, out / "mfbr_pretrained.pt")


def run(args):
    if min(args.batch_size, args.epochs, args.patience) < 1 or args.pretrain_epochs < 0 or args.token_budget < 0:
        raise ValueError("Invalid batch size, epoch count, patience or token budget")
    set_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = {'args': vars(args), 'torch': torch.__version__, 'python': platform.python_version(), 'data_cleaning': {}}
    if args.split_dir:
        train, val, test, split_meta = load_fixed_split(args.split_dir)
        protocol = split_meta['protocol']
        if protocol in ('e2', 'e4'):
            args.drug_group = 'scaffold'
        df = pd.concat([train, val, test], ignore_index=True)
        log['data_cleaning'] = {'status': 'frozen CSVs verified against split_manifest.json (SHA256, counts, disjointness)'}
        log['split_source'] = str(Path(args.split_dir).resolve())
        log['split_metadata'] = split_meta
        log['bridge_pairs_discarded'] = int(split_meta.get('bridge_pairs_discarded', 0))
    else:
        raise ValueError('--split-dir with three immutable CSVs is required (see the dataset restoration command in README.md)')
    if args.allow_transductive:
        raise ValueError('Transductive use of held-out entities is prohibited by this frozen CV protocol')
    if split_meta['protocol'] in ('e2', 'e3', 'e4') and (args.pretrain_drugs or args.pretrain_proteins):
        raise ValueError('External pretraining corpora are disabled for cold-start protocols until scaffold/homology leakage is independently audited')
    for name, part in (('train', train), ('val', val), ('test', test)):
        part.to_csv(out / f'{name}.csv', index=False)
    log['sizes'] = {'train': len(train), 'val': len(val), 'test': len(test)}
    cache = ProteinCache(args.esm_cache)
    missing = [q for q in df.sequence.unique() if not cache.has(q)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} sequences lack ESM-2 embeddings; rerun esm_cache')
    log['esm_cache'] = cache.manifest
    if not args.allow_other_esm and (cache.manifest.get('model') != C.ESM_MODEL or cache.dim != C.ESM_DIM):
        raise ValueError(f"ESM cache must be {C.ESM_MODEL} (dim {C.ESM_DIM}); found {cache.manifest.get('model')} (dim {cache.dim}). Rebuild with `python -m FABRIC.esm_cache`.")
    cfg = ModelConfig(esm_dim=cache.dim, hidden=args.hidden, gin_layers=args.gin_layers, adapter_bottleneck=args.adapter_bottleneck, dropout=args.dropout)
    model = FreqAwareDTI(cfg).to(device)
    log['n_parameters'] = sum((p.numel() for p in model.parameters()))
    print('sizes', log['sizes'], 'params', log['n_parameters'], 'device', device, flush=True)
    log['mfbr_recomposition'] = C.MFBR_RECOMPOSITION
    t0 = time.time()
    if args.pretrain_epochs > 0:
        pretrain(model, train, pd.concat([val, test]), cache, args, device, log)
    log['pretrain_seconds'] = time.time() - t0
    loaders = [make_loader(InteractionDataset(frame, cache), args.batch_size, args.token_budget,
                           i == 0, args.seed, args.workers)
               for i, frame in enumerate((train, val, test))]
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay, args.lr, args.spectral_lr))
    best, wait, hist, t1 = (-1.0, 0, [], time.time())
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = (0.0, 0)
        for group in loaders[0]:
            batch_loss, size, _ = update_batch(group, model, opt, args, device)
            tot += batch_loss * size
            n += size
        y, p = predict(model, loaders[1], device)
        vm = evaluate(y, p, 0.5)
        hist.append({'epoch': ep, 'train_loss': tot / n, 'val_AUROC': vm['AUROC'], 'val_AUPRC': vm['AUPRC']})
        print(f"[DTI] epoch {ep} loss {tot / n:.4f} val AUROC {vm['AUROC']:.4f} AUPRC {vm['AUPRC']:.4f}", flush=True)
        if vm['AUPRC'] > best + 0.0001:
            best, wait = (vm['AUPRC'], 0)
            torch.save({'model': model.state_dict(), 'config': cfg.to_dict(), 'epoch': ep}, out / 'best.pt')
        else:
            wait += 1
            if wait >= args.patience:
                break
    log['finetune_seconds'] = time.time() - t1
    pd.DataFrame(hist).to_csv(out / 'history.csv', index=False)
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True)['model'])
    yv, pv = predict(model, loaders[1], device)
    thr = best_f1_threshold(yv, pv)
    results, log['novelty'] = ({}, {})
    hits = pd.read_csv(args.protein_hits) if args.protein_hits else None
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    for name, d, loader in (('val', val, loaders[1]), ('test', test, loaders[2])):
        t2 = time.time()
        y, p, w = predict(model, loader, device, fusion=True)
        if cuda:
            torch.cuda.synchronize(device)
        log[f'{name}_inference_ms_per_pair'] = 1000 * (time.time() - t2) / len(d)
        results[name] = evaluate(y, p, thr)
        rep, nov = audit(train, d, hits)
        log['novelty'][name] = rep
        pd.concat([d.reset_index(drop=True).assign(prob=p), w, nov.reset_index(drop=True)], axis=1).to_csv(out / f'{name}_predictions.csv', index=False)
    if cuda:
        log['peak_inference_mem_mb'] = torch.cuda.max_memory_allocated(device) / 2 ** 20
    mu_d, sd_d, _ = model.lbd.drug_gates.mu_sigma()
    mu_p, sd_p, _ = model.lbd.protein_gates.mu_sigma()
    log['learned_gates'] = {'drug_mu': mu_d.tolist(), 'drug_sigma': sd_d.tolist(), 'protein_mu': mu_p.tolist(), 'protein_sigma': sd_p.tolist()}
    if model.lbd.window is not None:
        log['learned_window_coef'] = model.lbd.window.coef.tolist()
    log['results'] = results
    (out / 'results.json').write_text(json.dumps(log, indent=2, default=str), encoding='utf-8')
    print('TEST', json.dumps(results['test']), flush=True)
    return results

def parser():
    a = argparse.ArgumentParser(description='FreqAwareDTI: MFBR pre-training + fine-tuning')
    add = a.add_argument
    add('--split-dir', required=True, help='data/<dataset>/<protocol>/<fold>, reads and verifies immutable train/validation/test CSVs')
    add('--esm-cache', required=True)
    add('--out', required=True)
    add('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    add('--seed', type=int, default=42)
    add('--workers', type=int, default=0)
    add('--drug-group', default='smiles', choices=['smiles', 'scaffold', 'column'])
    add('--hidden', type=int, default=C.HIDDEN)
    add('--gin-layers', type=int, default=C.GIN_LAYERS)
    add('--adapter-bottleneck', type=int, default=C.ADAPTER_BOTTLENECK)
    add('--allow-other-esm', action='store_true', help='skip the ESM-2 150M cache check (other backbones)')
    add('--dropout', type=float, default=C.DROPOUT)
    add('--batch-size', type=int, default=C.BATCH_SIZE, help='logical batch size per optimizer update')
    add('--token-budget', type=int, default=C.TOKEN_BUDGET, help='padded atom-residue pairs per microbatch (0 disables splitting); gradients accumulate over the logical batch')
    add('--protein-hits', default=None, help='optional CSV of per-sequence annotations, one row per sequence')
    add('--epochs', type=int, default=C.EPOCHS)
    add('--patience', type=int, default=C.PATIENCE)
    add('--lr', type=float, default=C.LR)
    add('--weight-decay', type=float, default=C.WEIGHT_DECAY)
    add('--spectral-lr', type=float, default=C.SPECTRAL_LR, help='lr of LBD gate / window parameters')
    add('--sep-weight', type=float, default=0.001)
    add('--occ-weight', type=float, default=0.01)
    add('--pretrain-epochs', type=int, default=C.PRETRAIN_EPOCHS)
    add('--lr-pretrain', type=float, default=C.LR_PRETRAIN)
    add('--q-low', type=float, default=C.Q_LOW)
    add('--q-high', type=float, default=C.Q_HIGH)
    add('--w-low', type=float, default=C.W_LOW)
    add('--w-high', type=float, default=C.W_HIGH)
    add('--pretrain-drugs', default=None, help='CSV with a smiles column (ChEMBL/ZINC)')
    add('--pretrain-proteins', default=None, help='CSV with a sequence column (UniProt)')
    add('--allow-transductive', action='store_true')
    return a

def main(argv=None):
    args = parser().parse_args(argv)
    if args.device.startswith('cuda') and (not torch.cuda.is_available()):
        raise SystemExit('CUDA requested but unavailable')
    return run(args)
if __name__ == '__main__':
    main()
