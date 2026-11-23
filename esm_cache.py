import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from .data import AA, seq_key


@torch.inference_mode()
def embed(seq, tok, model, device, window):
    def run(s):
        t = tok(s, return_tensors="pt").to(device)
        return model(**t).last_hidden_state[0, 1:-1].float()
    if len(seq) <= window:
        return run(seq)
    acc = torch.zeros(len(seq), model.config.hidden_size, device=device)
    cnt = torch.zeros(len(seq), 1, device=device)
    starts = list(range(0, len(seq) - window, window // 2)) + [len(seq) - window]
    for s in starts:
        acc[s:s + window] += run(seq[s:s + window])
        cnt[s:s + window] += 1
    return acc / cnt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="CSVs with a `sequence` column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=config.ESM_MODEL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--window", type=int, default=1022)
    ap.add_argument("--long-seq", default="chunk", choices=["chunk", "refuse"])
    ap.add_argument("--dtype", default=config.ESM_CACHE_DTYPE, choices=["float16", "float32"])
    a = ap.parse_args(argv)
    if a.window < 2:
        ap.error("--window must be at least 2")
    from transformers import AutoModel, AutoTokenizer
    seqs = set()
    for f in a.csv:
        s = pd.read_csv(f).sequence.dropna().astype(str).str.strip().str.upper()
        seqs |= set(s[s.str.fullmatch(AA)])
    seqs = sorted(seqs)
    n_long = sum(len(s) > a.window for s in seqs)
    if n_long and a.long_seq == "refuse":
        raise SystemExit(f"{n_long} sequences exceed {a.window} residues")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, add_pooling_layer=False)
    model = model.to(a.device).eval().requires_grad_(False)
    dim = model.config.hidden_size
    layers = model.config.num_hidden_layers
    if a.model == config.ESM_MODEL and (dim, layers) != (config.ESM_DIM, config.ESM_LAYERS):
        raise RuntimeError(f"{a.model}: expected {config.ESM_LAYERS} layers / {config.ESM_DIM} dim, got {layers} / {dim}")
    for i, s in enumerate(seqs):
        path = out / f"{seq_key(s)}.npy"
        if path.exists() and np.load(path, mmap_mode="r").shape == (len(s), dim):
            continue
        h = embed(s, tok, model, a.device, a.window)
        if h.shape[0] != len(s):
            raise RuntimeError(f"token/residue mismatch {h.shape[0]} vs {len(s)}")
        tmp = str(path) + ".tmp.npy"
        np.save(tmp, h.cpu().numpy().astype(a.dtype))
        os.replace(tmp, path)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(seqs)}", flush=True)
    manifest = {"source": "esm2", "model": a.model, "embedding_dim": dim, "num_layers": layers, "frozen": True,
                "num_sequences": len(seqs), "long_sequences_chunked": n_long,
                "window": a.window, "dtype": a.dtype}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("done", manifest)


if __name__ == "__main__":
    main()
