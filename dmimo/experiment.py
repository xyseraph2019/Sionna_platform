"""
Dataset generation and evaluation harness for the DMIMO downlink model.

Useful for the later NN-precoding experiments:

  * :func:`generate_dataset` draws many (channel, error, baseline-metrics)
    samples and saves them to ``.pt`` for offline training.
  * :func:`evaluate_precoder` runs an arbitrary :class:`~dmimo.precoding.Precoder`
    (baseline or a learned neural precoder) and returns the DMIMO metrics, so a
    learned precoder can be compared against the independent-MRT baseline.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from .link import DMIMODownlink, DMIMMetrics
from .precoding import Precoder, IndependentMRT


def build_link(num_trps=3, num_tx_ant=4, num_ue_ant=1, n_subcarriers=64,
               subcarrier_spacing=30e3, tau_seconds=None, cal_amp_error=None,
               cal_pha_error=None, granularity="SC",
               channel_kind="simple", pathloss=True,
               trp_distances=(100.0, 200.0, 350.0), shadow_fading=True,
               carrier_frequency=3.5e9, delay_spread=100e-9,
               cdl_model="A", tdl_model="C", bs_height=25.0, beta=None):
    return DMIMODownlink(num_trps=num_trps, num_tx_ant=num_tx_ant, num_ue_ant=num_ue_ant,
                         n_subcarriers=n_subcarriers,
                         subcarrier_spacing=subcarrier_spacing,
                         tau_seconds=tau_seconds,
                         cal_amp_error=cal_amp_error, cal_pha_error=cal_pha_error,
                         granularity=granularity,
                         channel_kind=channel_kind, pathloss=pathloss,
                         trp_distances=trp_distances, shadow_fading=shadow_fading,
                         carrier_frequency=carrier_frequency, delay_spread=delay_spread,
                         cdl_model=cdl_model, tdl_model=tdl_model, bs_height=bs_height,
                         beta=beta)


def evaluate_precoder(link: DMIMODownlink, batch_size: int, snr_db: float,
                      precoder: Optional[Precoder] = None, device=None) -> DMIMMetrics:
    """Run ``precoder`` (default independent MRT) over one batch."""
    return link(batch_size, snr_db, precoder=precoder, device=device)


def generate_dataset(link: DMIMODownlink, batch_size: int, snr_db: float,
                     num_batches: int = 1, device=None) -> dict:
    """Draw per-sample tensors for (channel, baseline precoder, error, metrics).

    The output tensors are ready for training a neural precoder: the network
    observes ``channel`` (and optionally ``error``) and is asked to return
    precoders that maximise ``rate``.
    """
    out = None
    for _ in range(num_batches):
        t = link.forward_tensors(batch_size, snr_db, device=device)
        if out is None:
            out = {k: [v] if torch.is_tensor(v) else v for k, v in t.items()}
        else:
            for k, v in t.items():
                if torch.is_tensor(v):
                    out[k].append(v)
    for k, v in out.items():
        if isinstance(v, list):
            out[k] = torch.cat(v, dim=0)
    return out


def save_dataset(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(data, path)


def load_dataset(path: str) -> dict:
    return torch.load(path, map_location="cpu")
