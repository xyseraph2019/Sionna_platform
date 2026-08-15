"""
Protocol contracts for swappable platform components.

These are *loose* interface agreements (documented, not type-enforced) so that
custom components dropped in via the registry need only conform to the method
shapes below. Built-ins follow these too. Three of these map 1:1 onto Sionna's
own injectable receiver seams (channel estimator / MIMO detector / TB decoder).
"""
from __future__ import annotations

from typing import Protocol, Tuple, Optional

import torch


class Transmitter(Protocol):
    """Generates the frequency-domain transmit resource grid and info bits.

    ``forward(batch_size) -> (x, b)`` with ``x`` of shape
    ``[batch, num_tx, num_ant, n_symbols, fft_size]`` and ``b`` the
    ``[batch, num_tx, tb_size]`` information bits.
    """

    resource_grid: object
    info_bits_per_tb: int

    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class ChannelModel(Protocol):
    """Applies a propagation channel to a frequency-domain grid.

    ``forward(x, no) -> (y, h)``; ``h`` is the perfect CSI when available
    (used by estimators flagged ``"perfect"``), else ``None``.
    """

    def forward(self, x: torch.Tensor, no: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...


class ChannelEstimator(Protocol):
    """``forward(y, no) -> (h_hat, err_var)`` (Sionna channel-estimator API).

    Returning ``None`` selects Sionna's default LS estimator; the string
    ``"perfect"`` selects the perfect-CSI receiver path that consumes ``h``.
    """


class MIMODetector(Protocol):
    """``forward(y, h_hat, err_var, no) -> llr`` (Sionna OFDMDetector API)."""


class TBDecoder(Protocol):
    """Transport-block decoder (Sionna ``sionna.phy.nr.TBDecoder`` API)."""


class Receiver(Protocol):
    """``forward(y, no, h=None) -> (b_hat, crc_status)``.

    ``crc_status`` is ``True`` for a failed transport block (normalised).
    """

    def forward(
        self, y: torch.Tensor, no: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...