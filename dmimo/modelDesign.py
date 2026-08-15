"""modelDesign.py — neural subband-PMI model architecture.

Pure model file: no training loop, no checkpoint I/O. The network maps a Type I
*wideband* PMI ``W_wb : [B, K, 2P, r]`` to a *subband-level* precoder
``[B, K, 2P, r, S]``:

1. expand the wideband PMI to a subband-level base (broadcast over ``S``);
2. ``blocks`` MLP-Mixer stages mixing along the Tx (K), Layer (r) and Subband (S)
   axes plus a channel-mixing MLP over the ``[Re; Im]`` antenna features;
3. the mixer output is a **residual** added to the base;
4. the sum is normalised per (TRP, layer, subband) over the ``2P`` antennas.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _AxisMLP(nn.Module):
    """Shared per-axis MLP for MLP-Mixer (applied along the last dimension)."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPMixerSubbandPMI(nn.Module):
    """MLP-Mixer subband-PMI network (Type I wideband PMI -> subband precoder).

    Pipeline
    --------
    1. Expand the Type I *wideband* precoder ``W_wb : [B, K, 2P, r]`` to a
       subband-level base by broadcasting over the ``S`` subbands;
    2. ``blocks`` MLP-Mixer stages, each mixing along the **Tx (K)**, **Layer (r)**
       and **Subband (S)** axes (per-axis token-mixing MLPs) plus a channel-mixing
       MLP over the ``[Re; Im]`` ``2*2P`` antenna features (with LayerNorm);
    3. the mixer output is treated as a **residual** and added to the base;
    4. the sum is normalised per (TRP, layer, subband) over the ``2P`` antennas.

    The network consumes ONLY the Type I wideband PMI (no per-subcarrier channel
    information), so it learns the average subband structure of the optimal
    precoder conditioned on the wideband PMI.

    :input W_wb: ``[B, K, 2P, r]`` complex wideband PMI.
    :output:     ``[B, K, 2P, r, S]`` complex subband precoder (unit per-(TRP,layer,subband)).
    """

    def __init__(self, num_ant: int, rank: int, num_subbands: int, num_trps: int = 3,
                 blocks: int = 2, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_ant = int(num_ant)
        self.rank = int(rank)
        self.num_subbands = int(num_subbands)
        self.num_trps = int(num_trps)
        self.blocks = int(blocks)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        chan = 2 * self.num_ant                      # [Re; Im] antenna features
        self.mix_sub = nn.ModuleList([_AxisMLP(self.num_subbands, hidden)
                                      for _ in range(self.blocks)])
        self.mix_layer = nn.ModuleList([_AxisMLP(self.rank, hidden)
                                        for _ in range(self.blocks)])
        self.mix_trp = nn.ModuleList([_AxisMLP(self.num_trps, hidden)
                                      for _ in range(self.blocks)])
        self.mix_chan = nn.ModuleList([_AxisMLP(chan, hidden)
                                       for _ in range(self.blocks)])
        self.ln_chan = nn.ModuleList([nn.LayerNorm(chan)
                                      for _ in range(self.blocks)])

    def forward(self, W_wb: torch.Tensor) -> torch.Tensor:
        """``W_wb : [B, K, 2P, r]`` complex -> subband precoder ``[B, K, 2P, r, S]``."""
        B, K, P2, r = W_wb.shape
        S = self.num_subbands
        base = W_wb.unsqueeze(-1).expand(B, K, P2, r, S).contiguous()
        x = torch.cat([base.real, base.imag], dim=2)          # [B,K,2*2P,r,S]
        for i in range(self.blocks):
            # subband token-mixing (S is the last axis)
            x = x + self.mix_sub[i](x)
            # layer token-mixing (over r)
            xl = x.permute(0, 1, 2, 4, 3)                     # [B,K,C,S,r]
            x = x + self.mix_layer[i](xl).permute(0, 1, 2, 4, 3)
            # TRP token-mixing (over K)
            xk = x.permute(0, 2, 3, 4, 1)                     # [B,C,r,S,K]
            x = x + self.mix_trp[i](xk).permute(0, 4, 1, 2, 3)
            # channel mixing (over C, with LayerNorm)
            xn = self.ln_chan[i](x.permute(0, 1, 3, 4, 2))    # [B,K,r,S,C]
            x = x + self.mix_chan[i](xn).permute(0, 1, 4, 2, 3)
        res = x[:, :, :P2] + 1j * x[:, :, P2:]                # [B,K,2P,r,S]
        w = base + res
        return w / (w.abs().square().sum(dim=2, keepdim=True).sqrt() + 1e-12)

    def arch(self) -> dict:
        """Constructor kwargs (checkpoint metadata / reconstruction)."""
        return dict(num_ant=self.num_ant, rank=self.rank, num_subbands=self.num_subbands,
                    num_trps=self.num_trps, blocks=self.blocks, hidden=self.hidden,
                    dropout=self.dropout)
