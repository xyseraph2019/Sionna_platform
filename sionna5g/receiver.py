"""
5G NR PUSCH receiver wrapper around Sionna's ``PUSCHReceiver``.

The receiver combines, in a single block:
  * OFDM demodulation (frequency-domain input is used here),
  * channel estimation (configurable: ``ls`` | ``perfect`` | custom),
  * MIMO detection (configurable: ``lmmse`` | ``zf`` | custom),
  * transport-block LDPC decoding with CRC check (``tb_decoder``).

The algorithmic pieces are resolved through the :mod:`sionna5g.registry`, so an
advanced user can register a custom estimator / detector / decoder and select it
with a YAML ``receiver:`` section instead of editing code here.

It returns the estimated information bits and a transport-block CRC status
(``False`` = CRC OK / no error), i.e. ``True`` marks a failed block.
"""
from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn

from .transmitter import PUSCHTransmitterWrapper
from .config import ReceiverConfig
from . import registry

from sionna.phy.nr import PUSCHReceiver as SionnaPUSCHReceiver


class PUSCHReceiverWrapper(nn.Module):
    """Thin, config-driven wrapper so the receiver is interchangeable."""

    def __init__(
        self,
        transmitter: PUSCHTransmitterWrapper,
        return_crc_status: bool = True,
        device: str = "cpu",
        receiver_cfg: Optional[ReceiverConfig] = None,
    ) -> None:
        super().__init__()
        receiver_cfg = receiver_cfg or ReceiverConfig()
        self.return_crc_status = return_crc_status
        s_tx = transmitter.transmitter

        # Resolve built-in algorithms directly; custom names go through registry.
        estimator = self._build_estimator(receiver_cfg.channel_estimator, s_tx, device)
        detector = self._build_detector(receiver_cfg.mimo_detector, s_tx, device)
        decoder = self._build_decoder(receiver_cfg.tb_decoder, s_tx, device)

        self._perfect_csi = (estimator == "perfect")
        self._rx = SionnaPUSCHReceiver(
            transmitter.transmitter,
            return_tb_crc_status=return_crc_status,
            input_domain="freq",
            channel_estimator=estimator,
            mimo_detector=detector,
            tb_decoder=decoder,
            device=device,
        )

    @staticmethod
    def _build_estimator(name: str, s_tx, device: str):
        """Map a config name to a Sionna channel-estimator object/flag."""
        if name == "ls":
            return None  # Sionna default PUSCH LS estimator
        if name == "perfect":
            return "perfect"
        if not registry.has("estimator", name):
            raise ValueError(
                f"Unknown channel_estimator '{name}'. Registered: {registry.names('estimator')}"
            )
        return registry.build("estimator", name, s_tx, device)

    @staticmethod
    def _build_detector(name: str, s_tx, device: str):
        """Map a config name to a Sionna MIMO detector object."""
        if name == "lmmse":
            return None  # Sionna default linear detector (LMMSE)
        if name == "zf":
            import numpy as np

            from sionna.phy.mimo import StreamManagement
            from sionna.phy.ofdm.detection import LinearDetector

            sm = StreamManagement(np.ones([1, s_tx._num_tx], bool), s_tx._num_layers)
            return LinearDetector(
                "zf", "bit", "maxlog", s_tx.resource_grid, sm, "qam",
                s_tx._num_bits_per_symbol, precision=None, device=device,
            )
        if not registry.has("detector", name):
            raise ValueError(
                f"Unknown mimo_detector '{name}'. Registered: {registry.names('detector')}"
            )
        return registry.build("detector", name, s_tx, device)

    @staticmethod
    def _build_decoder(name: str, s_tx, device: str):
        """Map a config name to a Sionna TB decoder object."""
        if name == "default":
            return None
        if not registry.has("decoder", name):
            raise ValueError(
                f"Unknown tb_decoder '{name}'. Registered: {registry.names('decoder')}"
            )
        return registry.build("decoder", name, s_tx, device)


    @property
    def receiver(self) -> SionnaPUSCHReceiver:
        return self._rx

    @property
    def perfect_csi(self) -> bool:
        return self._perfect_csi

    def forward(
        self, y: torch.Tensor, no: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a received resource grid.

        Parameters
        ----------
        y : [batch, num_rx, num_ant, n_symbols, fft] complex
            Received signal (noise already embedded in ``y``).
        no : float tensor
            AWGN variance per complex dimension.
        h : optional perfect CSI (required when ``channel_estimator="perfect"``).

        Returns
        -------
        b_hat : decoded information bits.
        crc : bool tensor, ``True`` marks a failed CRC.
        """
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, dtype=torch.float32, device=y.device)
        no = no.to(dtype=torch.float32, device=y.device)

        if self._perfect_csi and h is None:
            raise ValueError("channel_estimator='perfect' requires h (use a fading channel)")

        if self.return_crc_status:
            out = self._rx(y, no, h)
            b_hat, crc = out
            # Normalise: ``crc = True`` always means a FAILED transport block.
            crc = (~crc.bool())
            return b_hat, crc
        return self._rx(y, no, h), None
