"""
High-level link-level simulator.

The :class:`LinkSimulator` chains the three physical-layer stages
(TX -> channel -> RX) into a single ``torch.nn.Module`` and provides methods
to run a batch and to sweep a full SNR curve returning
:class:`~sionna5g.metrics.LinkMetrics` for each point.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import SimConfig, resolve_device
from .transmitter import PUSCHTransmitterWrapper
from .channel import ChannelModelWrapper
from .receiver import PUSCHReceiverWrapper
from .metrics import LinkMetrics, snr_to_no, evaluate_link


class LinkSimulator(nn.Module):
    """End-to-end 5G NR PUSCH link (single-user uplink).

    Parameters
    ----------
    cfg : SimConfig
        Scenario configuration.
    """

    def __init__(self, cfg: SimConfig, components: Optional[dict] = None) -> None:
        super().__init__()
        self.cfg = cfg
        # Resolve "auto" -> cuda:0 (if available) or cpu, and store the
        # effective device back so downstream blocks/tensors use a real value.
        self.device = resolve_device(cfg.device)
        cfg.device = self.device
        components = components or {}

        # Advanced usage: inject a custom transmitter / channel / receiver.
        # Any piece that is *not* injected is built from ``cfg`` as usual.
        self.tx = components.get("tx")
        if self.tx is None:
            self.tx = PUSCHTransmitterWrapper(
                cfg.carrier, cfg.pusch, cfg.tb, device=self.device
            )

        num_tx_ant = cfg.pusch.num_antenna_ports
        num_rx_ant = cfg.channel.num_rx_ant or num_tx_ant
        self.channel = components.get("channel")
        if self.channel is None:
            self.channel = ChannelModelWrapper(
                cfg.channel,
                self.tx.resource_grid,
                device=self.device,
                num_tx_ant=num_tx_ant,
                num_rx_ant=num_rx_ant,
            )

        self.rx = components.get("rx")
        if self.rx is None:
            self.rx = PUSCHReceiverWrapper(
                self.tx,
                return_crc_status=cfg.return_crc_status,
                device=self.device,
                receiver_cfg=cfg.receiver,
            )

    # ------------------------------------------------------------------
    @property
    def info_bits_per_tb(self) -> int:
        return self.tx.info_bits_per_tb

    @property
    def num_layers(self) -> int:
        return self.cfg.pusch.num_layers

    def forward(
        self, batch_size: int, snr_db: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one batched link transmission.

        Parameters
        ----------
        batch_size : int
            Number of parallel transport blocks (Monte-Carlo trials).
        snr_db : float, optional
            Point SNR used to derive the noise variance. If ``None``, a low
            default of 0 dB is assumed.

        Returns
        -------
        b      : transmitted bits [batch, num_tx, tb_size].
        b_hat  : decoded bits [batch, num_tx, tb_size].
        crc    : CRC status [batch, num_tx] (``True`` = failure).
        """
        if snr_db is None:
            snr_db = self.cfg.snr_grid[0] if self.cfg.snr_grid else 0.0
        no = torch.tensor(
            snr_to_no(snr_db, num_layers=self.num_layers),
            dtype=torch.float32,
            device=self.device,
        )
        x, b = self.tx(batch_size)
        y, h = self.channel(x, no)
        b_hat, crc = self.rx(y, no, h)
        if crc is None:
            # No CRC status: fall back to hard BER comparison to count errors.
            crc = ~torch.all(b == b_hat, dim=-1)
        return b, b_hat, crc

    # ------------------------------------------------------------------
    def run_snr(self, snr_db: float, num_trials: int) -> LinkMetrics:
        """Evaluate all metrics at a single SNR point."""
        cfg = self.cfg
        num_trials = int(num_trials)
        batch_size = min(cfg.batch_size, num_trials)

        b_parts, bhat_parts, crc_parts = [], [], []
        remaining = num_trials
        with torch.no_grad():
            while remaining > 0:
                bs = min(batch_size, remaining)
                b, b_hat, crc = self.forward(bs, snr_db)
                b_parts.append(b)
                bhat_parts.append(b_hat)
                crc_parts.append(crc)
                remaining -= bs

        b_all = torch.cat(b_parts, dim=0)
        bhat_all = torch.cat(bhat_parts, dim=0)
        crc_all = torch.cat(crc_parts, dim=0)
        return evaluate_link(b_all, bhat_all, crc_all, snr_db, cfg)

    def run_curve(
        self, snr_db: Optional[List[float]] = None, num_trials: Optional[int] = None
    ) -> List[LinkMetrics]:
        """Sweep an SNR range and return per-point metrics."""
        snr_range = snr_db if snr_db is not None else self.cfg.snr_grid
        trials = num_trials if num_trials is not None else self.cfg.num_trials
        results = []
        for s in snr_range:
            results.append(self.run_snr(float(s), trials))
        return results
