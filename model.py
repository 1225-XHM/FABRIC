from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import List, Tuple, Dict

import torch
import numpy as np
import math
from torch import nn
import torch.nn.functional as F

from . import config as C
from .data import ATOM_DIM, DrugBatch, ProteinBatch, PairBatch


def _gin_aggregate(h, edge_index):
    agg = torch.zeros_like(h)
    if edge_index.numel():
        agg = agg.index_add(0, edge_index[1], h[edge_index[0]])
    return agg


class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=3, dropout=0.1):
        super().__init__()
        if not 1 <= layers:
            raise ValueError("GIN needs >= 1 layer")
        self.inp = nn.Linear(in_dim, hidden)
        self.eps = nn.Parameter(torch.zeros(layers))
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                                 nn.Linear(hidden, hidden))
                                   for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.jk = nn.Sequential(nn.Linear(hidden * (layers + 1), hidden), nn.LayerNorm(hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, drug: DrugBatch):
        h = F.relu(self.inp(x))
        hs = [h]
        for eps, mlp, ln in zip(self.eps, self.mlps, self.norms):
            upd = mlp((1.0 + eps) * h + _gin_aggregate(h, drug.edge_index))
            h = ln(h + self.dropout(upd))
            hs.append(h)
        return drug.dense(self.jk(torch.cat(hs, -1)))


class GPSEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=3, dropout=0.1, heads=4):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.eps = nn.Parameter(torch.zeros(layers))
        self.local = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                                  nn.Linear(hidden, hidden))
                                    for _ in range(layers)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                                         batch_first=True)
                                   for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                                nn.Linear(2 * hidden, hidden))
                                  for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.jk = nn.Sequential(nn.Linear(hidden * (layers + 1), hidden), nn.LayerNorm(hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, drug: DrugBatch):
        h = F.relu(self.inp(x))
        hs = [h]
        for i in range(len(self.local)):
            loc = self.local[i]((1.0 + self.eps[i]) * h + _gin_aggregate(h, drug.edge_index))
            hd = drug.dense(h)
            glob = self.attn[i](hd, hd, hd, key_padding_mask=~drug.mask, need_weights=False)[0]
            h = self.n1[i](h + self.dropout(loc) + self.dropout(drug.sparse(glob)))
            h = self.n2[i](h + self.dropout(self.ffn[i](h)))
            hs.append(h)
        return drug.dense(self.jk(torch.cat(hs, -1)))


class ProteinAdapter(nn.Module):
    def __init__(self, esm_dim, hidden=256, bottleneck=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, bottleneck),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(bottleneck, hidden), nn.LayerNorm(hidden))

    def forward(self, x, mask):
        return self.net(x) * mask[..., None].to(x.dtype)


def _inv_softplus(y):
    return math.log(math.expm1(y))


class BandGates(nn.Module):
    def __init__(self, k=2, sigma0=0.2):
        super().__init__()
        if k < 2:
            raise ValueError("K >= 2 bands required")
        step = (1.0 / sigma0 ** 2) / k
        self.alpha = nn.Parameter(torch.tensor([_inv_softplus(step / 2)] +
                                               [_inv_softplus(step)] * (k - 1)))
        self.gamma = nn.Parameter(torch.tensor([_inv_softplus(step)] * (k - 1) +
                                               [_inv_softplus(step / 2)]))
        self.bias = nn.Parameter(torch.zeros(k))
        self.k = k

    def mu_sigma(self):
        a = torch.cumsum(F.softplus(self.alpha), 0)
        c = torch.flip(torch.cumsum(torch.flip(F.softplus(self.gamma), [0]), 0), [0])
        tau = a + c
        return a / tau, tau.rsqrt(), tau

    def forward(self, nu):
        mu, _, tau = self.mu_sigma()
        shape = (-1,) + (1,) * nu.dim()
        logits = -0.5 * tau.view(shape) * (nu[None] - mu.view(shape)).square()
        return torch.softmax(logits + self.bias.view(shape), dim=0)

    def regularizers(self, rho_min, points=257):
        nu = torch.linspace(0, 1, points, device=self.alpha.device)
        g = self(nu)
        coverage = (g.sum(0) - 1).square().mean()
        sep = sum(torch.trapezoid(g[i] * g[j], nu)
                  for i in range(self.k) for j in range(i + 1, self.k))
        occupancy = torch.trapezoid(g, nu, dim=-1)
        occ = F.relu(rho_min - occupancy).square().sum()
        return coverage, sep, occ


class LearnableWindow(nn.Module):
    def __init__(self, terms=3, learnable=True):
        super().__init__()
        coef = torch.zeros(terms)
        coef[0], coef[1] = 0.5, -0.5
        if learnable:
            self.coef = nn.Parameter(coef)
        else:
            self.register_buffer("coef", coef)
        self._basis = {}

    def forward(self, n, device, dtype):
        if n == 1:
            return torch.ones(1, device=device, dtype=dtype)
        key = (n, str(device), dtype)
        basis = self._basis.get(key)
        if basis is None:
            t = torch.arange(n, device=device, dtype=dtype) / (n - 1)
            m = torch.arange(self.coef.numel(), device=device, dtype=dtype)
            basis = torch.cos(2 * math.pi * m[:, None] * t[None])
            if len(self._basis) < 8192:
                self._basis[key] = basis
        return self.coef.to(dtype) @ basis


def chebyshev_lowpass_init(order, cutoff=0.5, width=0.1):
    f = lambda x: 1.0 / (1.0 + np.exp((((x + 1) / 2) - cutoff) / width))
    return torch.tensor(np.polynomial.chebyshev.chebinterpolate(f, order), dtype=torch.float32)


class LBD(nn.Module):
    def __init__(self, k=2, sigma0=0.2, gate_mode="learnable", window="learnable",
                 drug_filter="gft", cheb_order=8):
        super().__init__()
        if gate_mode not in ("learnable", "frozen"):
            raise ValueError(gate_mode)
        if window not in ("learnable", "hann", "none", "mirror"):
            raise ValueError(window)
        if drug_filter not in ("gft", "chebyshev"):
            raise ValueError(drug_filter)
        if drug_filter == "chebyshev" and k != 2:
            raise ValueError("Chebyshev low/high split is defined for K=2 only")
        self.k, self.drug_filter = k, drug_filter
        self.drug_gates = BandGates(k, sigma0)
        self.protein_gates = BandGates(k, sigma0)
        if gate_mode == "frozen":
            for p in list(self.drug_gates.parameters()) + list(self.protein_gates.parameters()):
                p.requires_grad_(False)
        self.mirror = window == "mirror"
        self.window = (None if window in ("none", "mirror")
                       else LearnableWindow(learnable=window == "learnable"))
        if drug_filter == "chebyshev":
            self.theta = nn.Parameter(chebyshev_lowpass_init(cheb_order))

    def drug_bands(self, h, drug: DrugBatch):
        if self.drug_filter == "chebyshev":
            return self._chebyshev(h, drug)
        coeff = torch.einsum("bnm,bnd->bmd", drug.evecs, h)
        g = self.drug_gates(drug.evals / 2.0) * drug.mask[None].to(h.dtype)
        return torch.einsum("bnm,kbmd->kbnd", drug.evecs, g[..., None] * coeff[None])

    def _chebyshev(self, h, drug):
        m = drug.mask.to(h.dtype)
        lt = drug.lap - torch.diag_embed(m)
        t_prev, t_cur = h, lt @ h
        low = self.theta[0] * t_prev + self.theta[1] * t_cur
        for t in range(2, self.theta.numel()):
            t_prev, t_cur = t_cur, 2 * (lt @ t_cur) - t_prev
            low = low + self.theta[t] * t_cur
        return torch.stack([low, h - low])

    def protein_window(self, n, device, dtype):
        if self.window is None:
            return torch.ones(n, device=device, dtype=dtype)
        return self.window(n, device, dtype)

    def protein_bands(self, h, prot: ProteinBatch):
        lmax, lengths = h.shape[1], prot.lengths.tolist()
        nus = [torch.fft.rfftfreq(2 * n if self.mirror else n, device=h.device, dtype=h.dtype) * 2.0
               for n in lengths]
        gates = self.protein_gates(torch.cat(nus)).split([v.numel() for v in nus], dim=1)
        wins = [self.protein_window(n, h.device, h.dtype) for n in lengths]
        outs = []
        for i, n in enumerate(lengths):
            x = h[i, :n]
            if self.mirror:
                x = torch.cat([x, x.flip(0)], 0)
                m = 2 * n
            else:
                x, m = x * wins[i][:, None], n
            spec = torch.fft.rfft(x, dim=0)
            z = torch.fft.irfft(gates[i][..., None] * spec[None], n=m, dim=1)[:, :n]
            outs.append(F.pad(z, (0, 0, 0, lmax - n)))
        return torch.stack(outs, 1)

    def regularizers(self):
        rho = 0.25 / self.k
        gates = [self.protein_gates] + ([self.drug_gates] if self.drug_filter == "gft" else [])
        terms = [g.regularizers(rho) for g in gates]
        return {"coverage": sum(t[0] for t in terms), "separation": sum(t[1] for t in terms),
                "occupancy": sum(t[2] for t in terms)}


def mean_max_pool(z, mask):
    m = mask[..., None].to(z.dtype)
    mean = (z * m).sum(1) / m.sum(1).clamp_min(1.0)
    mx = z.masked_fill(~mask[..., None], float("-inf")).amax(1)
    return torch.cat([mean, mx], -1)


class BandCalibration(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w_p = nn.Linear(4 * d, d)
        self.w_d = nn.Linear(4 * d, d)

    def forward(self, zd, dmask, zp, pmask):
        cd, cp = mean_max_pool(zd, dmask), mean_max_pool(zp, pmask)
        gp = torch.sigmoid(self.w_p(torch.cat([cp, cd], -1)))
        gd = torch.sigmoid(self.w_d(torch.cat([cd, cp], -1)))
        return zd * gd[:, None], zp * gp[:, None], gd, gp


class BAN(nn.Module):
    def __init__(self, d, out_dim=256, heads=2, k=3, dropout=0.2):
        super().__init__()
        r = out_dim * k
        self.u = nn.Linear(d, r)
        self.v = nn.Linear(d, r)
        p = torch.ones(heads, r) / math.sqrt(r)
        self.p = nn.Parameter(p + 0.01 * torch.randn_like(p) / math.sqrt(r))
        self.p_bias = nn.Parameter(torch.zeros(heads))
        self.pool = nn.AvgPool1d(k, stride=k)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.k = k

    def forward(self, zd, dmask, zp, pmask):
        ud = self.drop(F.relu(self.u(zd)))
        vp = self.drop(F.relu(self.v(zp)))
        logits = (ud[:, None] * self.p[None, :, None]) @ vp[:, None].transpose(-1, -2)
        logits = logits + self.p_bias[None, :, None, None]
        valid = dmask[:, None, :, None] & pmask[:, None, None, :]
        logits = logits.masked_fill(~valid, float("-inf"))
        b, h, n, l = logits.shape
        att = torch.softmax(logits.reshape(b, h, n * l), -1).reshape(b, h, n, l)
        joint = ((att.sum(1).transpose(1, 2) @ ud) * vp).sum(1)
        f = self.pool(joint[:, None]).squeeze(1) * self.k
        return self.norm(f), att


def path_spec(k: int, cross: str) -> List[Tuple[str, int, int]]:
    label = (lambda i: "LH"[i]) if k == 2 else (lambda i: f"B{i}")
    intra = [(label(i) + label(i), i, i) for i in range(k)]
    hl = [(label(j) + label(i), j, i) for j in range(k) for i in range(k) if j > i]
    lh = [(label(i) + label(j), i, j) for j in range(k) for i in range(k) if j > i]
    extra = {"hl": hl, "lh": lh, "both": hl + lh, "none": []}[cross]
    return intra + extra


class FABA(nn.Module):
    def __init__(self, d, k=2, cross="hl", out_dim=256, heads=2, dropout=0.2,
                 calibration=True, fusion="adaptive"):
        super().__init__()
        if fusion not in ("adaptive", "scalar", "mean", "content"):
            raise ValueError(fusion)
        self.paths = path_spec(k, cross)
        self.names = [p[0] for p in self.paths]
        self.calib = nn.ModuleList([BandCalibration(d) for _ in range(k)]) if calibration else None
        self.bans = nn.ModuleDict({n: BAN(d, out_dim, heads, dropout=dropout) for n in self.names})
        n_path = len(self.paths)
        self.fusion = fusion
        if fusion == "adaptive":
            self.fuse = nn.Sequential(nn.Linear(2 * n_path, 64), nn.GELU(), nn.Linear(64, n_path))
        elif fusion == "scalar":
            self.fuse_logits = nn.Parameter(torch.zeros(n_path))
        elif fusion == "content":
            self.fuse = nn.Sequential(nn.Linear(2 * n_path, 64), nn.GELU(), nn.Linear(64, n_path))
            self.score = nn.Sequential(nn.Linear(out_dim, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, zd, dmask, zp, pmask) -> Tuple[torch.Tensor, Dict]:
        k = zd.shape[0]
        gates = {}
        if self.calib is not None:
            cal = [self.calib[i](zd[i], dmask, zp[i], pmask) for i in range(k)]
            zd = [c[0] for c in cal]
            zp = [c[1] for c in cal]
            gates = {i: (c[2], c[3]) for i, c in enumerate(cal)}
        feats, energy, att = [], [], {}
        for name, j, i in self.paths:
            f, a = self.bans[name](zd[j], dmask, zp[i], pmask)
            feats.append(f)
            att[name] = a
            energy.append(torch.stack([mean_max_pool(zd[j], dmask).norm(dim=-1),
                                       mean_max_pool(zp[i], pmask).norm(dim=-1)], -1))
        stack = torch.stack(feats, 1)
        if self.fusion in ("adaptive", "content"):
            e = torch.log(torch.cat(energy, -1).clamp_min(1e-6))
            logits = self.fuse(e)
            if self.fusion == "content":
                logits = logits + self.score(stack).squeeze(-1)
            w = torch.softmax(logits, -1)
        elif self.fusion == "scalar":
            w = torch.softmax(self.fuse_logits, 0).expand(stack.shape[0], -1)
        else:
            w = stack.new_full(stack.shape[:2], 1.0 / stack.shape[1])
        fused = (stack * w[..., None]).sum(1)
        return fused, {"paths": self.names, "fusion_weights": w, "attention": att,
                       "calibration_gates": gates}


@dataclass
class ModelConfig:
    esm_dim: int = C.ESM_DIM
    hidden: int = C.HIDDEN
    adapter_bottleneck: int = C.ADAPTER_BOTTLENECK
    gin_layers: int = C.GIN_LAYERS
    drug_encoder: str = "gin"
    band_k: int = C.BAND_K
    sigma0: float = 0.2
    band_mode: str = "spectral"
    gate_mode: str = "learnable"
    window: str = "learnable"
    drug_filter: str = "gft"
    cross: str = "hl"
    calibration: bool = True
    fusion: str = "adaptive"
    ban_out: int = 256
    ban_heads: int = 2
    mlp_hidden: int = 512
    dropout: float = 0.2

    def to_dict(self):
        return asdict(self)


class FreqAwareDTI(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.band_mode not in ("spectral", "identity"):
            raise ValueError(cfg.band_mode)
        self.cfg = cfg
        enc = {"gin": GINEncoder, "gps": GPSEncoder}[cfg.drug_encoder]
        self.drug_encoder = enc(ATOM_DIM, cfg.hidden, cfg.gin_layers, dropout=0.1)
        self.protein_adapter = ProteinAdapter(cfg.esm_dim, cfg.hidden, cfg.adapter_bottleneck, 0.1)
        if cfg.band_k < 2 and cfg.band_mode != "identity":
            raise ValueError("band_k=1 (single-BAN baseline) requires band_mode='identity'")
        self.lbd = LBD(max(cfg.band_k, 2), cfg.sigma0, cfg.gate_mode, cfg.window,
                       cfg.drug_filter if cfg.band_k >= 2 else "gft")
        self.faba = FABA(cfg.hidden, cfg.band_k, cfg.cross, cfg.ban_out, cfg.ban_heads,
                         cfg.dropout, cfg.calibration, cfg.fusion)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.ban_out, cfg.mlp_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden // 4), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden // 4, 1))

    def encode_drug(self, x, drug):
        return self.drug_encoder(x, drug)

    def encode_protein(self, x, prot):
        return self.protein_adapter(x, prot.mask)

    def bands(self, hd, hp, batch: PairBatch):
        if self.cfg.band_mode == "identity":
            k = self.cfg.band_k
            return hd.expand(k, *hd.shape), hp.expand(k, *hp.shape)
        return self.lbd.drug_bands(hd, batch.drug), self.lbd.protein_bands(hp, batch.protein)

    def forward(self, batch: PairBatch, return_details=False):
        hd = self.encode_drug(batch.drug.x, batch.drug)
        hp = self.encode_protein(batch.protein.x, batch.protein)
        zd, zp = self.bands(hd, hp, batch)
        f, aux = self.faba(zd, batch.drug.mask, zp, batch.protein.mask)
        logit = self.classifier(f).squeeze(-1)
        if return_details:
            aux.update({"zd": zd, "zp": zp})
            return logit, aux
        return logit

    def regularizers(self):
        if self.cfg.band_mode == "identity":
            z = self.classifier[0].weight.sum() * 0.0
            return {"coverage": z, "separation": z, "occupancy": z}
        return self.lbd.regularizers()


def _interp(lo, hi, k):
    return [lo + (hi - lo) * i / max(k - 1, 1) for i in range(k)]


class MFBR(nn.Module):
    def __init__(self, model, q_low=0.15, q_high=0.5, w_low=1.0, w_high=2.0, unit="band",
                 token_rate=0.15, contrastive_weight=0.0, temperature=0.1, proj_dim=128,
                 recomposition=C.MFBR_RECOMPOSITION):
        super().__init__()
        if unit not in ("band", "token"):
            raise ValueError(unit)
        if model.cfg.band_mode != "spectral" and unit == "band":
            raise ValueError("Band masking requires band_mode='spectral'")
        if not (0 <= q_low <= 1 and 0 <= q_high <= 1):
            raise ValueError("mask probabilities must be in [0, 1]")
        if recomposition not in ("retained", "residual"):
            raise ValueError("recomposition must be 'retained' or 'residual'")
        self.model, self.unit, k = model, unit, model.cfg.band_k
        self.recomposition = recomposition
        h = model.cfg.hidden
        self.register_buffer("q", torch.tensor(_interp(q_low, q_high, k)))
        self.register_buffer("w", torch.tensor(_interp(w_low, w_high, k)))
        self.token_rate = token_rate
        self.drug_decoder = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, ATOM_DIM))
        self.protein_decoder = nn.Sequential(nn.Linear(h, h), nn.GELU(),
                                             nn.Linear(h, model.cfg.esm_dim))
        self.contrastive_weight, self.temperature = contrastive_weight, temperature
        self.proj_d = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, proj_dim))
        self.proj_p = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, proj_dim))

    def draw_mask(self, b, device):
        return (torch.rand(b, self.q.numel(), device=device) < self.q).to(torch.float32)

    @staticmethod
    def _per_band_loss(rec, target, mask):
        e = F.smooth_l1_loss(rec, target, reduction="none").mean(-1)
        m = mask[None].to(e.dtype)
        return (e * m).sum(-1) / m.sum(-1).clamp_min(1.0)

    def _modality(self, x, mask, bands_fn, encode_fn, decoder):
        if self.unit == "token":
            rows = (torch.rand(mask.shape, device=x.device) < self.token_rate) & mask
            enc = encode_fn(x * (~rows)[..., None].to(x.dtype))
            e = F.smooth_l1_loss(decoder(enc), x, reduction="none").mean(-1)
            return (e * rows).sum() / rows.sum().clamp_min(1)
        with torch.no_grad():
            target = bands_fn(x)
        m = self.draw_mask(x.shape[0], x.device)
        masked = m.T[..., None, None].to(target.dtype)
        if self.recomposition == "retained":
            corrupt = ((1 - masked) * target).sum(0)
        else:
            # Eq. (13) subtracts masked targets from the unwindowed input.
            corrupt = x - (masked * target).sum(0)
        rec = decoder(bands_fn(encode_fn(corrupt)))
        per = self._per_band_loss(rec, target, mask)
        return (self.w[:, None] * per * m.T).sum(0).mean()

    def reconstruction(self, batch: PairBatch):
        mdl, drug, prot = self.model, batch.drug, batch.protein
        xd = drug.dense(drug.x)
        ld = self._modality(xd, drug.mask, lambda h: mdl.lbd.drug_bands(h, drug),
                            lambda h: mdl.encode_drug(drug.sparse(h), drug), self.drug_decoder)
        xp = F.layer_norm(prot.x, prot.x.shape[-1:]) * prot.mask[..., None]
        lp = self._modality(xp, prot.mask, lambda h: mdl.lbd.protein_bands(h, prot),
                            lambda h: mdl.encode_protein(h, prot), self.protein_decoder)
        return ld, lp

    def contrastive(self, pos: PairBatch):
        mdl = self.model
        hd = mdl.encode_drug(pos.drug.x, pos.drug)
        hp = mdl.encode_protein(pos.protein.x, pos.protein)
        zd, zp = mdl.bands(hd, hp, pos)
        dm, pm = pos.drug.mask[None, ..., None].float(), pos.protein.mask[None, ..., None].float()
        vd = F.normalize(self.proj_d((zd * dm).sum(2) / dm.sum(2)), dim=-1)
        vp = F.normalize(self.proj_p((zp * pm).sum(2) / pm.sum(2)), dim=-1)
        b = vd.shape[1]
        same_d = torch.tensor([[a == c for c in pos.drug_ids] for a in pos.drug_ids], device=vd.device)
        same_p = torch.tensor([[a == c for c in pos.protein_ids] for a in pos.protein_ids], device=vd.device)
        eye = torch.eye(b, dtype=torch.bool, device=vd.device)
        false_neg = (same_d | same_p) & ~eye
        labels = torch.arange(b, device=vd.device)
        losses = []
        for k in range(vd.shape[0]):
            s = (vd[k] @ vp[k].T / self.temperature).masked_fill(false_neg, float("-inf"))
            losses.append(0.5 * (F.cross_entropy(s, labels) + F.cross_entropy(s.T, labels)))
        return torch.stack(losses).mean()

    def forward(self, batch: PairBatch, pos_batch: PairBatch = None):
        ld, lp = self.reconstruction(batch)
        loss = ld + lp
        info = {"recon_drug": float(ld.detach()), "recon_protein": float(lp.detach())}
        if self.contrastive_weight > 0 and pos_batch is not None and pos_batch.drug.size >= 2:
            c = self.contrastive(pos_batch)
            loss = loss + self.contrastive_weight * c
            info["contrastive"] = float(c.detach())
        return loss, info
