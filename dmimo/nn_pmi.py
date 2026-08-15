"""nn_pmi.py — Type-I-wideband-PMI feature extraction and precoder wrapper.

* :func:`wideband_pmi` computes the Type I **wideband** PMI from the channel
  (must be called with the *error-free* channel, as the TX precoder uses clean
  CSI; the transmission then runs through the error-corrupted channel).
* :func:`expand_subband_to_subcarriers` maps the model's subband output
  ``[B, K, 2P, r, S]`` back to per-subcarrier ``[B, K, 2P, r, N]``.
* :class:`NNMixerPMI` is the :class:`~dmimo.precoding.Precoder`-protocol wrapper
  used for inference in the link-level example (``precoder(h)``).
* :func:`save_model` / :func:`load_model` handle checkpoint IO with scenario and
  architecture metadata.

Training lives in ``examples/modelTrain.py`` (flat structure), not here.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from .modelDesign import MLPMixerSubbandPMI
from .precoding import type1_wideband_selection

__all__ = ["NNMixerPMI", "wideband_pmi", "expand_subband_to_subcarriers",
           "save_model", "load_model"]


def wideband_pmi(h: torch.Tensor, rank: int = 2, oversmpl: int = 4):
    """Type I wideband PMI per TRP as a complex precoder.

    ``h : [B, K, D, 2P, N]`` (CLEAN channel) -> ``(W_wb, sel)`` with
    ``W_wb : [B, K, 2P, r]`` = ``1/sqrt(2)[v ; phi_wb*v]`` (wideband QPSK
    co-phasing). ``sel`` carries the wideband beams / projections (see
    :func:`dmimo.precoding.type1_wideband_selection`).
    """
    sel = type1_wideband_selection(h, rank, oversmpl)
    Vr, a, c, P, r = (sel["Vr"], sel["a"], sel["c"], sel["P"], sel["r"])
    qpsk = torch.tensor([1, 1j, -1, -1j], dtype=torch.complex64, device=h.device)
    comb = a.unsqueeze(-1) + c.unsqueeze(-1) * qpsk            # [B,K,r,D,N,4]
    power = comb.abs().square().sum(dim=3)                     # [B,K,r,N,4] (over D)
    power = power.sum(dim=3)                                   # [B,K,r,4]   (over N)
    phi_wb = qpsk[power.argmax(dim=-1)]                        # [B,K,r]
    top = Vr.permute(0, 1, 3, 2)                               # [B,K,P,r]
    bot = (phi_wb.unsqueeze(-1) * Vr).permute(0, 1, 3, 2)      # [B,K,P,r]
    w_wb = torch.cat([top, bot], dim=2) / math.sqrt(2)         # [B,K,2P,r]
    return w_wb, sel


def expand_subband_to_subcarriers(w_sub: torch.Tensor, n_subcarriers: int,
                                  subband_size: int) -> torch.Tensor:
    """Map a subband-level precoder ``[B, K, 2P, r, S]`` to per-subcarrier
    ``[B, K, 2P, r, N]`` (subcarrier ``k`` uses subband ``k // subband_size``)."""
    N = int(n_subcarriers)
    sub_id = (torch.arange(N, device=w_sub.device) // int(subband_size)).long()
    return w_sub[..., sub_id]

class NNMixerPMI(nn.Module):
    """Precoder-protocol wrapper for the MLP-Mixer subband-PMI network.

    Implements ``__call__(h) -> w : [B, K, 2P, r, N]`` for use as a
    ``Precoder`` in the link-level evaluation. Only the Type I wideband PMI of
    ``h`` (computed here) reaches the network; the network expands it to
    subbands, mixes over Tx/Layer/Subband (see
    :class:`dmimo.modelDesign.MLPMixerSubbandPMI`) and adds the learned
    residual to the expanded base.
    """

    kind = "mixer"

    def __init__(self, rank: int = 2, oversmpl: int = 4, subband_size: int = 12,
                 n_subcarriers: int = 240, num_ant: int = 32, num_trps: int = 3,
                 blocks: int = 2, hidden: int = 64, dropout: float = 0.0,
                 ckpt: str = None, device: str = None):
        super().__init__()
        self.rank = int(rank)
        self.oversmpl = int(oversmpl)
        self.subband_size = int(subband_size)
        self.n_subcarriers = int(n_subcarriers)
        self.num_ant = int(num_ant)
        self.num_trps = int(num_trps)
        self.num_subbands = int(math.ceil(self.n_subcarriers / self.subband_size))
        self.net = MLPMixerSubbandPMI(num_ant=self.num_ant, rank=self.rank,
                                      num_subbands=self.num_subbands,
                                      num_trps=self.num_trps, blocks=blocks,
                                      hidden=hidden, dropout=dropout)
        self.meta = {}
        if device is not None:
            self.net.to(device)
        if ckpt:
            ckpt_data = torch.load(ckpt, map_location="cpu")
            self.net.load_state_dict(ckpt_data["state"])
            self.meta = ckpt_data.get("meta", {})
            if device is not None:
                self.net.to(device)

    def config(self) -> dict:
        """Constructor kwargs (checkpoint reconstruction)."""
        return dict(rank=self.rank, oversmpl=self.oversmpl,
                    subband_size=self.subband_size, n_subcarriers=self.n_subcarriers,
                    num_ant=self.num_ant, num_trps=self.num_trps,
                    blocks=self.net.blocks, hidden=self.net.hidden,
                    dropout=self.net.dropout)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """``h : [B, K, D, 2P, N]`` (clean) -> precoders ``w : [B, K, 2P, r, N]``."""
        B, K, D, Nt, N = h.shape
        W_wb, sel = wideband_pmi(h, self.rank, self.oversmpl)   # [B,K,2P,r]
        kp = self.num_trps
        if K < kp:   # single-TRP evaluation: zero-pad, mix, then slice back
            pad = torch.zeros(B, kp - K, *W_wb.shape[2:],
                              dtype=W_wb.dtype, device=W_wb.device)
            W_wb = torch.cat([W_wb, pad], dim=1)
        ws = self.net(W_wb.to(h.device))                        # [B,kp,2P,r,S]
        ws = ws[:, :K]
        return expand_subband_to_subcarriers(ws, N, self.subband_size)  # [B,K,2P,r,N]


def save_model(model, path: str, meta: dict = None) -> None:
    """Save weights + architecture + metadata to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"kind": getattr(model, "kind", "mixer"),
                "config": model.config(), "state": model.net.state_dict(),
                "meta": meta or {}}, path)


def load_model(path: str, device: str = None):
    """Restore an :class:`NNMixerPMI` from ``path``; returns ``(model, meta)``."""
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("kind", "mixer") != "mixer":
        raise ValueError(f"unsupported model kind {ckpt.get('kind')!r} in {path}")
    m = NNMixerPMI(**ckpt["config"])
    m.net.load_state_dict(ckpt["state"])
    m.meta = ckpt.get("meta", {})
    if device is not None:
        m.net.to(device)
    return m, m.meta

