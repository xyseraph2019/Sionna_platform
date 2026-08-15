"""
5G NR PUSCH transmitter wrapper around Sionna's ``PUSCHTransmitter``.

Builds the Sionna ``PUSCHConfig`` hierarchy from the platform configuration
dataclasses and exposes a thin ``torch.nn.Module`` wrapper that returns the
frequency-domain resource grid and the transmitted information bits.
"""
from __future__ import annotations

from typing import Union, Tuple

import torch
import torch.nn as nn

from .config import CarrierConfig, PUSCHConfig, TBConfig

import sionna
from sionna.phy.nr import (
    CarrierConfig as SionnaCarrierConfig,
    PUSCHConfig as SionnaPUSCHConfig,
    TBConfig as SionnaTBConfig,
    PUSCHTransmitter as SionnaPUSCHTransmitter,
)


def build_pusch_config(
    carrier: CarrierConfig,
    pusch: PUSCHConfig,
    tb: TBConfig,
    device: str = "cpu",
) -> SionnaPUSCHConfig:
    """Translate the platform dataclasses into a Sionna ``PUSCHConfig``."""
    s_carrier = SionnaCarrierConfig()
    s_carrier.subcarrier_spacing = carrier.subcarrier_spacing
    s_carrier.n_size_grid = carrier.n_size_grid
    s_carrier.n_cell_id = carrier.n_cell_id
    s_carrier.slot_number = carrier.slot_number
    s_carrier.frame_number = carrier.frame_number

    s_pusch = SionnaPUSCHConfig(s_carrier)
    s_pusch.num_layers = pusch.num_layers
    s_pusch.num_antenna_ports = pusch.num_antenna_ports
    s_pusch.mapping_type = pusch.mapping_type
    s_pusch.symbol_allocation = [pusch.symbol_start, pusch.symbol_length]
    s_pusch.precoding = pusch.precoding
    s_pusch.transform_precoding = pusch.transform_precoding

    s_pusch.tb.mcs_index = tb.mcs_index
    s_pusch.tb.mcs_table = tb.mcs_table
    s_pusch.tb.channel_type = tb.channel_type
    if tb.n_id is not None:
        s_pusch.tb.n_id = tb.n_id

    s_pusch.device = device
    return s_pusch


class PUSCHTransmitterWrapper(nn.Module):
    """Wrapper around the Sionna PUSCH transmitter.

    The forward pass returns ``(x, b)`` where ``x`` is the frequency-domain
    resource grid of shape ``[batch, num_tx, num_ant, n_symbols, fft_size]``
    and ``b`` the information bits of shape ``[batch, num_tx, tb_size]``.
    """

    def __init__(
        self,
        carrier: CarrierConfig,
        pusch: PUSCHConfig,
        tb: TBConfig,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.config = build_pusch_config(carrier, pusch, tb, device=device)
        # Single-UE uplink physical link.
        self._tx = SionnaPUSCHTransmitter(
            self.config,
            return_bits=True,
            output_domain="freq",
            device=device,
        )

    @property
    def transmitter(self) -> SionnaPUSCHTransmitter:
        return self._tx

    @property
    def resource_grid(self):
        return self._tx.resource_grid

    @property
    def info_bits_per_tb(self) -> int:
        """Number of information bits carried by one transport block (one slot)."""
        return int(self.config.tb_size)

    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._tx(batch_size)
