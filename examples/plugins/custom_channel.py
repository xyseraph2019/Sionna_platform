"""
Example plugin: register a *custom propagation channel* at the surface level.

This registers a frequency-flat Rayleigh fading channel. After importing this
module, any ``SimConfig`` can use it by setting ``channel.channel_type =
"flat_rayleigh"`` — the core simulator never changes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from sionna5g import registry


class FlatRayleigh(nn.Module):
    """Single-tap frequency-flat Rayleigh fading channel (channel-model API).

    Conforms to the Sionna ``ChannelModel`` interface used by ``OFDMChannel``:
    calling with ``(batch_size, num_time_steps, sampling_frequency)`` returns
    path coefficients ``a`` and delays ``tau``.
    """

    def __init__(self, num_tx_ant=1, num_rx_ant=1, num_tx=1, num_rx=1, device="cpu"):
        super().__init__()
        self.num_tx_ant = int(num_tx_ant)
        self.num_rx_ant = int(num_rx_ant)
        self.num_tx = int(num_tx)
        self.num_rx = int(num_rx)
        self.device = device

    def forward(self, batch_size, num_time_steps, sampling_frequency):
        # Block-fading: one complex gain per (batch, tx-antenna, rx-antenna, path)
        # pair, constant across the whole slot (num_time_steps). This is the
        # physically meaningful "flat Rayleigh" channel and is trackable by the
        # LS channel estimator's time interpolation.
        base = [int(batch_size), self.num_rx, self.num_rx_ant,
                self.num_tx, self.num_tx_ant, 1]
        real = torch.randn(base, dtype=torch.float32, device=self.device)
        imag = torch.randn(base, dtype=torch.float32, device=self.device)
        a = ((real + 1j * imag) / math.sqrt(2.0)).unsqueeze(-1)
        a = a.expand(*base, int(num_time_steps))  # constant across the slot

        tau = torch.zeros(int(batch_size), self.num_rx, self.num_tx, 1,
                          dtype=torch.float32, device=self.device)
        return a, tau


def flat_rayleigh(cfg, resource_grid, device, num_tx_ant, num_rx_ant):
    """Registry builder -> (ofdm_channel, system_level_model_or_None)."""
    from sionna.phy.channel import OFDMChannel

    model = FlatRayleigh(num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant, device=device)
    ofdm = OFDMChannel(channel_model=model, resource_grid=resource_grid,
                       normalize_channel=True, return_channel=True, device=device)
    return ofdm, None


registry.register("channel", "flat_rayleigh", flat_rayleigh)

if __name__ == "__main__":
    print("Registered custom channels:", registry.names("channel"))