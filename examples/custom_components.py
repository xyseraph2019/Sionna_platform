"""
Custom components (flat version).

This module replaces the old ``examples/plugins/`` nesting. Importing it
registers the example channel/estimator/detector names so they can be selected
from YAML configs.

Registered names
----------------
- channel:   ``flat_rayleigh``
- estimator: ``ls_avg``
- detector:  ``mf``
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import math

import numpy as np
import torch
import torch.nn as nn

from sionna5g import registry
from sionna.phy.mimo import StreamManagement
from sionna.phy.nr import PUSCHLSChannelEstimator
from sionna.phy.ofdm.detection import LinearDetector


class FlatRayleigh(nn.Module):
    """Single-tap frequency-flat Rayleigh fading channel (Sionna model API)."""

    def __init__(self, num_tx_ant=1, num_rx_ant=1, num_tx=1, num_rx=1, device="cpu"):
        super().__init__()
        self.num_tx_ant = int(num_tx_ant)
        self.num_rx_ant = int(num_rx_ant)
        self.num_tx = int(num_tx)
        self.num_rx = int(num_rx)
        self.device = device

    def forward(self, batch_size, num_time_steps, sampling_frequency):
        base = [int(batch_size), self.num_rx, self.num_rx_ant,
                self.num_tx, self.num_tx_ant, 1]
        real = torch.randn(base, dtype=torch.float32, device=self.device)
        imag = torch.randn(base, dtype=torch.float32, device=self.device)
        a = ((real + 1j * imag) / math.sqrt(2.0)).unsqueeze(-1)
        a = a.expand(*base, int(num_time_steps))
        tau = torch.zeros(int(batch_size), self.num_rx, self.num_tx, 1,
                          dtype=torch.float32, device=self.device)
        return a, tau


def flat_rayleigh(cfg, resource_grid, device, num_tx_ant, num_rx_ant):
    """Registry builder for the custom flat-Rayleigh channel."""
    from sionna.phy.channel import OFDMChannel

    model = FlatRayleigh(num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant, device=device)
    ofdm = OFDMChannel(channel_model=model, resource_grid=resource_grid,
                       normalize_channel=True, return_channel=True, device=device)
    return ofdm, None


def ls_time_averaged(transmitter, device="cpu"):
    """LS channel estimator with linear + time-averaged interpolation."""
    return PUSCHLSChannelEstimator(
        transmitter.resource_grid,
        transmitter._dmrs_length,
        transmitter._dmrs_additional_position,
        transmitter._num_cdm_groups_without_data,
        interpolation_type="lin_time_avg",
        device=device,
    )


def mf_detector(transmitter, device="cpu"):
    """Matched-filter (maximal-ratio combining) MIMO detector."""
    sm = StreamManagement(np.ones([1, transmitter._num_tx], bool), transmitter._num_layers)
    return LinearDetector(
        "mf", "bit", "maxlog", transmitter.resource_grid, sm, "qam",
        transmitter._num_bits_per_symbol, device=device,
    )


# Register the custom components (idempotent).
registry.register("channel", "flat_rayleigh", flat_rayleigh)
registry.register("estimator", "ls_avg", ls_time_averaged)
registry.register("detector", "mf", mf_detector)
