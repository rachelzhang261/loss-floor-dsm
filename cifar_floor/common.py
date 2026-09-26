"""
common.py -- schedule, data, model, loss for the CIFAR-10 floor experiment.

CONVENTIONS (must match floor_core.py exactly)
----------------------------------------------
    alpha(l)^2 = sigmoid(l),  sigma(l)^2 = sigmoid(-l),  SNR(l) = e^l
    x_l = alpha(l) y + sigma(l) eps

Continuous-time diffusion loss (VDM, eps-prediction, ELBO weighting), on a
log-SNR interval [lo, hi]:

    L[lo,hi] = 1/2 * (hi-lo) * E_{l~U[lo,hi]} E_{y,eps} || eps - eps_theta ||^2

and the irreducible part of it is

    FLOOR[lo,hi] = 1/2 * (hi-lo) * E_l [ d - sigma(l) tr(d eps_theta/dx) ].

Everything is reported in nats per image (summed over the 3072 dimensions),
NOT per dimension, so that FLOOR is directly comparable with Delta I.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
def alpha_sigma(lam):
    """lam: tensor of log-SNR. returns (alpha, sigma) with alpha^2=sigmoid(lam)"""
    a2 = torch.sigmoid(lam)
    return a2.sqrt(), (1.0 - a2).sqrt()


# Reference configurations.  These are the log-SNR ranges over which training
# and evaluation are performed.  sigma_min = exp(-hi/2) approximately, so
# hi = 9.2 corresponds to sigma_min ~ 1e-2 and hi = 13.8 to sigma_min ~ 1e-3;
# both are values in common use.
CONFIGS = {
    "narrow": (-8.0, 6.0),
    "mid": (-8.0, 9.2),
    "wide": (-8.0, 13.8),
}
COMMON = "narrow"          # the yardstick every model is also evaluated on


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def cifar_tensors(root, train=True):
    """CIFAR-10 as a float tensor in [-1,1], shape (N,3,32,32). No augmentation:
    the floor is a property of the data distribution, so the distribution the
    models see must be fixed and identical across runs."""
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=root, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    return x.contiguous()


class TensorLoader:
    """Minimal infinite shuffled loader; avoids DataLoader worker overhead for
    an in-memory dataset."""

    def __init__(self, x, batch_size, seed=0, device="cuda"):
        self.x, self.bs, self.device = x, batch_size, device
        self.g = torch.Generator().manual_seed(seed)
        self._perm, self._i = None, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._perm is None or self._i + self.bs > len(self.x):
            self._perm = torch.randperm(len(self.x), generator=self.g)
            self._i = 0
        idx = self._perm[self._i:self._i + self.bs]
        self._i += self.bs
        return self.x[idx].to(self.device, non_blocking=True)


# ---------------------------------------------------------------------------
# model: small DDPM-style UNet
# ---------------------------------------------------------------------------
def timestep_embedding(lam, dim):
    """sinusoidal embedding of the log-SNR (not of t: the model is a function
    of lambda, which is the natural coordinate -- see the paper's Sec. 4)"""
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=lam.device)
                      / max(half - 1, 1))
    args = lam[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cemb, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(32, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(cemb, cout)
        self.n2 = nn.GroupNorm(32, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(32, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).unbind(1)
        a = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)
        h = (v @ a.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.out(h)


class UNet(nn.Module):
    """~12M params at ch=64. Deliberately small: the claim is about the loss
    decomposition, not about sample quality."""

    def __init__(self, ch=64, ch_mult=(1, 2, 2), nres=2, attn_res=(16,),
                 dropout=0.1):
        super().__init__()
        cemb = ch * 4
        self.lam_dim = ch
        self.emb = nn.Sequential(nn.Linear(ch, cemb), nn.SiLU(),
                                 nn.Linear(cemb, cemb))
        self.cin = nn.Conv2d(3, ch, 3, padding=1)

        self.nres = nres
        self.nlevel = len(ch_mult)

        # ---- down path, one ModuleList per resolution level ----------------
        self.down = nn.ModuleList()        # down[i] : nres blocks
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()  # downsample[i] : conv or Identity
        skip_ch, cur, res = [ch], ch, 32
        for i, m in enumerate(ch_mult):
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(nres):
                blocks.append(ResBlock(cur, ch * m, cemb, dropout))
                cur = ch * m
                attns.append(Attn(cur) if res in attn_res else nn.Identity())
                skip_ch.append(cur)
            self.down.append(blocks)
            self.down_attn.append(attns)
            if i < self.nlevel - 1:
                self.downsample.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1))
                skip_ch.append(cur)
                res //= 2
            else:
                self.downsample.append(nn.Identity())

        self.mid1 = ResBlock(cur, cur, cemb, dropout)
        self.midattn = Attn(cur)
        self.mid2 = ResBlock(cur, cur, cemb, dropout)

        # ---- up path: nres+1 blocks per level, consuming skips in reverse --
        self.up = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in reversed(range(self.nlevel)):
            m = ch_mult[i]
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(nres + 1):
                blocks.append(ResBlock(cur + skip_ch.pop(), ch * m, cemb, dropout))
                cur = ch * m
                attns.append(Attn(cur) if res in attn_res else nn.Identity())
            self.up.append(blocks)
            self.up_attn.append(attns)
            if i > 0:
                self.upsample.append(nn.Upsample(scale_factor=2, mode="nearest"))
                res *= 2
            else:
                self.upsample.append(nn.Identity())
        assert not skip_ch, f"skip bookkeeping mismatch, {len(skip_ch)} left over"

        self.nout = nn.GroupNorm(32, cur)
        self.cout = nn.Conv2d(cur, 3, 3, padding=1)
        nn.init.zeros_(self.cout.weight)
        nn.init.zeros_(self.cout.bias)

    def forward(self, x, lam):
        emb = self.emb(timestep_embedding(lam, self.lam_dim))
        h = self.cin(x)
        hs = [h]
        for i in range(self.nlevel):
            for blk, att in zip(self.down[i], self.down_attn[i]):
                h = att(blk(h, emb))
                hs.append(h)
            if i < self.nlevel - 1:
                h = self.downsample[i](h)
                hs.append(h)
        h = self.mid2(self.midattn(self.mid1(h, emb)), emb)
        for j in range(self.nlevel):
            for blk, att in zip(self.up[j], self.up_attn[j]):
                h = att(blk(torch.cat([h, hs.pop()], dim=1), emb))
            h = self.upsample[j](h)
        assert not hs, "skip stack not exhausted"
        return self.cout(F.silu(self.nout(h)))


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def sample_lambda(n, lo, hi, device, generator=None, antithetic=True):
    """stratified / antithetic uniform sampling of lambda on [lo,hi]"""
    if antithetic:
        u = torch.rand(1, device=device, generator=generator)
        u = (u + torch.arange(n, device=device) / n) % 1.0
    else:
        u = torch.rand(n, device=device, generator=generator)
    return lo + (hi - lo) * u


def dsm_loss_terms(model, y, lo, hi, generator=None):
    """
    Returns the per-image ELBO-weighted diffusion loss in nats:

        L = 1/2 (hi-lo) * mean_over_batch || eps - eps_theta ||^2

    (the ||.||^2 is summed over the 3072 dimensions)
    """
    n = y.shape[0]
    lam = sample_lambda(n, lo, hi, y.device, generator)
    a, s = alpha_sigma(lam)
    a = a.view(-1, 1, 1, 1)
    s = s.view(-1, 1, 1, 1)
    eps = torch.randn(y.shape, device=y.device, generator=generator)
    x = a * y + s * eps
    pred = model(x, lam)
    per_image = (eps - pred).pow(2).flatten(1).sum(1)
    return 0.5 * (hi - lo) * per_image


def count_params(m):
    return sum(p.numel() for p in m.parameters())
