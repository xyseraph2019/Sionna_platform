"""
Error-rate and throughput metrics for link-level simulations.

These are computed from Monte-Carlo runs of many transport blocks:
  * BLER  - transport-block error (CRC) rate,
  * BER   - bit error rate (across all decoded bits),
  * throughput - number of delivered information bits per second
    (higher-layer / achievable throughput, assuming CRC-passing TBs are
    delivered error free).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .config import SimConfig


def snr_to_no(snr_db: float, num_layers: int = 1) -> float:
    """SNR (dB) -> AWGN variance per complex dimension.

    Sionna interprets ``no`` as the per-complex-dimension noise variance, so
    the total noise power per resource element is ``2 * no``. With a unit-power
    transmit resource grid this gives Es/No: ``SNR = 1 / (2 * no)``.
    """
    # Note: ``num_layers`` is accepted for API symmetry but the Es/No relation
    # already accounts for the received signal power; we do not rescale by it.
    return 1.0 / (2.0 * 10.0 ** (snr_db / 10.0))


@dataclass
class LinkMetrics:
    """Metric snapshot for a single SNR point."""

    snr_db: float
    bler: float = 0.0               # transport-block error rate
    ber: float = 0.0                # bit error rate
    throughput_bps: float = 0.0     # delivered info bits per second
    num_blocks: int = 0
    num_info_bits_per_tb: int = 0
    info_bits_failed: int = 0

    def as_dict(self) -> dict:
        return {
            "snr_db": self.snr_db,
            "bler": self.bler,
            "ber": self.ber,
            "throughput_bps": self.throughput_bps,
            "num_blocks": self.num_blocks,
            "num_info_bits_per_tb": self.num_info_bits_per_tb,
            "num_tb_errors": self.info_bits_failed,
        }

    def __str__(self) -> str:
        return (
            f"SNR={self.snr_db:6.2f} dB  BLER={self.bler:.4e}  "
            f"BER={self.ber:.4e}  Throughput={self.throughput_bps/1e6:9.3f} Mbps"
        )


def evaluate_link(
    b_all: torch.Tensor,
    b_hat_all: torch.Tensor,
    crc_all: torch.Tensor,
    snr_db: float,
    cfg: SimConfig,
) -> LinkMetrics:
    """Compute metrics from accumulated transmit/decode tensors.

    Parameters
    ----------
    b_all      : [N, num_tx, tb_size] transmitted bits (all trials concatenated).
    b_hat_all  : [N, num_tx, tb_size] decoded bits.
    crc_all    : [N, num_tx] bool, ``True`` = CRC failed.
    snr_db     : SNR point.
    cfg        : simulation config (used for slot duration and layers).
    """
    # Number of transport blocks in the accumulated batch.
    num_blocks = crc_all.numel()
    tb_size = int(b_all.numel() // max(num_blocks, 1))

    # Reshape to [num_blocks, tb_size]; per-block CRC.
    b2 = b_all.reshape(num_blocks, -1)
    bhat2 = b_hat_all.reshape(num_blocks, -1)
    crc = crc_all.reshape(-1)

    num_tb_errors = int(crc.sum().item())
    bler = num_tb_errors / num_blocks

    # Raw bit-error rate over ALL transmitted bits, i.e. the physical-layer BER
    # seen by the decoder across both CRC-passing and CRC-failing transport
    # blocks. (Measuring BER only over passing blocks is degenerate: a passing
    # block is by definition error-free under the CRC, so that BER is always 0.)
    bit_err = (b2 != bhat2).float()
    ber = bit_err.mean().item()

    slot_duration = cfg.carrier.slot_duration
    delivered_bits = (num_blocks - num_tb_errors) * tb_size
    throughput_bps = delivered_bits / slot_duration / max(num_blocks, 1)

    return LinkMetrics(
        snr_db=snr_db,
        bler=bler,
        ber=ber,
        throughput_bps=throughput_bps,
        num_blocks=num_blocks,
        num_info_bits_per_tb=tb_size,
        info_bits_failed=num_tb_errors,
    )


def metrics_to_records(metrics: List[LinkMetrics]) -> List[dict]:
    """Flatten a list of :class:`LinkMetrics` into JSON-serialisable dict rows."""
    return [m.as_dict() for m in metrics]


def save_metrics_csv(metrics: List[LinkMetrics], path: str) -> None:
    """Write an SNR-sweep metric table to a CSV file."""
    import csv
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    records = metrics_to_records(metrics)
    if not records:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_metrics_json(metrics: List[LinkMetrics], path: str) -> None:
    """Write an SNR-sweep metric table to a JSON file."""
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics_to_records(metrics), fh, indent=2)


def bler_10db(metrics: List[LinkMetrics]) -> Optional[float]:
    """Interpolate the SNR (in dB) at which the BLER crosses 0.1 (10%).

    Uses linear interpolation on the SNR axis between the two bracketing BLER
    samples. Returns ``None`` when the curve stays entirely above or below 0.1
    (no crossing in the swept range).
    """
    import math

    points = sorted((m.snr_db, m.bler) for m in metrics)
    # Find where BLER goes from above to at/below 0.1.
    prev_snr, prev_bler = points[0]
    for snr, bler in points[1:]:
        if prev_bler > 0.1 >= bler or prev_bler >= 0.1 > bler:
            # linear interpolation between (prev_snr, prev_bler) and (snr, bler)
            if prev_bler == bler:
                return float(snr)
            t = (prev_bler - 0.1) / (prev_bler - bler)
            return float(prev_snr + t * (snr - prev_snr))
        prev_snr, prev_bler = snr, bler
    return None

