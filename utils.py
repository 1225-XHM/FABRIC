import random
import numpy as np
import pandas as pd
import torch
NO_DECAY = {'alpha', 'gamma', 'bias', 'coef', 'theta', 'p', 'p_bias', 'eps', 'fuse_logits'}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def param_groups(module, wd, lr, spectral_lr):
    decay, no_decay, spectral = ([], [], [])
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if '.lbd.' in f'.{n}':
            spectral.append(p)
        elif p.dim() < 2 or n.rsplit('.', 1)[-1] in NO_DECAY:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': wd, 'lr': lr}, {'params': no_decay, 'weight_decay': 0.0, 'lr': lr}, {'params': spectral, 'weight_decay': 0.0, 'lr': spectral_lr}]

def reg_loss(model, args):
    r = model.regularizers()
    return args.sep_weight * r['separation'] + args.occ_weight * r['occupancy']

def step(loss, params, opt):
    if not torch.isfinite(loss):
        raise FloatingPointError('non-finite loss')
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

@torch.no_grad()
def predict(model, loader, device, fusion=False):
    model.eval()
    n = len(loader.dataset)
    ys, ps, ws, names = (np.zeros(n, int), np.zeros(n), None, None)
    for group in loader:
        batches = group if isinstance(group, list) else [group]
        for b in batches:
            b = b.to(device)
            logit, aux = model(b, return_details=True)
            ps[b.index] = torch.sigmoid(logit).cpu().numpy()
            ys[b.index] = b.label.int().cpu().numpy()
            if fusion:
                names = aux['paths']
                ws = np.zeros((n, len(names))) if ws is None else ws
                ws[b.index] = aux['fusion_weights'].cpu().numpy()
    if fusion:
        return (ys, ps, pd.DataFrame(ws, columns=[f'w_{p}' for p in names]))
    return (ys, ps)
