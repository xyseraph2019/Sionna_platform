"""
DMIMO downlink link model: combine multiple TRPs at the UE and compute metrics.

For a single UE with one receive antenna, per subcarrier ``k``:

    z[k] = sum_t  g_t[k] * p_t[k],    p_t[k] = w_t[k]^H h_t[k]

where ``h_t`` is the downlink channel of TRP t, ``w_t`` its (independent)
precoder and ``g_t[k]`` the TRP error scalar (timing + calibration). The
instantaneous received power is ``gain[k] = |z[k]|^2`` and the achievable rate
(no intra-cell interference, single user) is

    R = mean_k  log2( 1 + gain[k] / no )

Metrics also include the **coherent-combining baseline** (g = 1, no error) and
the **incoherent power** ``sum_t |p_t|^2``, so the gain eroded by the errors can
be quantified in dB.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import os

import torch
import torch.nn as nn

from .errors import TRPErrorModel
from .channels import build_downlink_channel
from .precoding import Precoder, IndependentMRT


@dataclass
class DMIMMetrics:
    """Per-batch scalar metrics for the DMIMO downlink link."""

    rate_bpshz: float            # with errors
    rate_coherent_bpshz: float   # no errors (g=1)
    gain_lin: float              # mean |z|^2 with errors
    gain_coherent_lin: float     # mean |z0|^2 (no errors)
    gain_incoherent_lin: float   # mean sum_t |p_t|^2
    gain_loss_db: float          # 10*log10(gain_coherent / gain)

    def as_dict(self) -> dict:
        return asdict(self)


class DMIMODownlink(nn.Module):
    """End-to-end differentiable multi-TRP downlink with per-TRP errors.

    Parameters
    ----------
    num_trps, num_tx_ant, n_subcarriers : int
        Geometry of the scenario.
    subcarrier_spacing : float
        Sub-carrier spacing in Hz (drives the timing phase ramp).
    tau_seconds : sequence[float]
        Per-TRP timing offset relative to TRP 1 (TRP 1 must be 0).
    cal_amplitude, cal_phase : sequence[float]
        Per-TRP calibration amplitude / phase (default = no error).
    beta : sequence[float] | None
        Per-TRP large-scale gains.
    """

    def __init__(self, num_trps: int = 3, num_tx_ant: int = 4, num_ue_ant: int = 1,
                 n_subcarriers: int = 64, subcarrier_spacing: float = 30e3,
                 tau_seconds=None, cal_amp_error=None, cal_pha_error=None,
                 granularity="SC",
                 channel_kind="simple", pathloss=True,
                 trp_distances=(100.0, 200.0, 350.0), shadow_fading=True,
                 carrier_frequency=3.5e9, delay_spread=100e-9,
                 cdl_model="A", tdl_model="C", bs_height=25.0, beta=None):
        super().__init__()
        self.n_subcarriers = n_subcarriers
        # Auto-size the per-TRP timing offsets to the number of TRPs so the
        # model is self-consistent whatever K is (e.g. single-TRP baselines).
        k = int(num_trps)
        if tau_seconds is None:
            tau_seconds = [0.0] * k
        self.channel = build_downlink_channel(
            channel_kind, num_trps=k, num_tx_ant=num_tx_ant, num_ue_ant=num_ue_ant,
            n_subcarriers=n_subcarriers, subcarrier_spacing=subcarrier_spacing,
            carrier_frequency=carrier_frequency, pathloss=pathloss,
            trp_distances=trp_distances, shadow_fading=shadow_fading,
            delay_spread=delay_spread, cdl_model=cdl_model, tdl_model=tdl_model,
            bs_height=bs_height, beta=beta)
        self.error = TRPErrorModel(subcarrier_spacing, n_subcarriers,
                                   tau_seconds, num_tx_ant=num_tx_ant,
                                   cal_amp_error=cal_amp_error,
                                   cal_pha_error=cal_pha_error,
                                   granularity=granularity)

    @staticmethod
    def _combine(h, w):
        """Sum per-TRP precoded channels ``sum_t H_t W_t : [B,D,r,N]``."""
        Hw = torch.einsum("bkdan,bkaln->bkdln", h, w)   # [B,K,D,r,N]
        return Hw.sum(dim=1)                            # [B,D,r,N]

    @staticmethod
    def _rate(H, no):
        """Multi-stream rate ``log2 det(I + H^H H / no)`` averaged over subcarriers.

        ``H : [B,D,r,N]`` -> ``[B]``.
        """
        G = torch.einsum("bdln,bdmn->blmn", H.conj(), H)   # [B,r,r,N]
        G = G.permute(0, 3, 1, 2)                           # [B,N,r,r]
        r = G.shape[-1]
        eye = torch.eye(r, dtype=G.real.dtype, device=G.device)
        det = torch.linalg.det(eye + G / no).real            # Hermitian PSD -> real det
        return torch.log2(det.clamp(min=1e-12)).mean(dim=1)  # [B]

    def forward(self, batch_size: int, snr_db: float,
                precoder: Precoder = None, device=None) -> DMIMMetrics:
        """Run one batch and return scalar metrics.

        Precoding is computed from the *clean* channel by default (limited
        feedback: Type I / NN-PMI), or from the *error-corrupted* channel when
        the precoder opts in via ``precodes_from_errors = True`` (CJT observes
        and compensates timing/calibration errors). The error-corrupted channel
        is always used for the actual combining/transmission. The coherent upper
        bound uses the clean channel for both.
        """
        precoder = precoder or IndependentMRT()
        h = self.channel.sample(batch_size, device)   # [B,K,D,Nt,N] clean
        h_err = self.error.apply(h)                   # with timing + calibration
        w = precoder(h_err) if getattr(precoder, "precodes_from_errors", False) else precoder(h)
        H_eff = self._combine(h_err, w)               # [B,D,r,N] with errors (transmission)
        H0 = self._combine(h, w)                      # [B,D,r,N] coherent (clean)

        no = 10.0 ** (-snr_db / 10.0)
        rate = self._rate(H_eff, no)                  # [B]
        rate0 = self._rate(H0, no)

        gain = H_eff.abs().square().sum(dim=(1, 2)).mean(dim=1)  # [B]
        gain0 = H0.abs().square().sum(dim=(1, 2)).mean(dim=1)
        # incoherent power = sum_t ||H_err_t W_t||_F^2
        Hw = torch.einsum("bkdan,bkaln->bkdln", h_err, w)
        gain_incoh = Hw.abs().square().sum(dim=(2, 3)).sum(dim=1).mean(dim=1)  # [B]

        return DMIMMetrics(
            rate_bpshz=float(rate.mean().item()),
            rate_coherent_bpshz=float(rate0.mean().item()),
            gain_lin=float(gain.mean().item()),
            gain_coherent_lin=float(gain0.mean().item()),
            gain_incoherent_lin=float(gain_incoh.mean().item()),
            gain_loss_db=float(10.0 * torch.log10(gain0.mean() / (gain.mean() + 1e-12)).item()),
        )

    def forward_tensors(self, batch_size: int, snr_db: float,
                        precoder: Precoder = None, device=None):
        """Like :meth:`forward` but returns raw per-batch tensors (for datasets)."""
        precoder = precoder or IndependentMRT()
        h = self.channel.sample(batch_size, device)
        h_err = self.error.apply(h)
        w = precoder(h_err) if getattr(precoder, "precodes_from_errors", False) else precoder(h)
        H_eff = self._combine(h_err, w)                # with errors (transmission)
        H0 = self._combine(h, w)                       # coherent (clean)
        no = 10.0 ** (-snr_db / 10.0)
        return {
            "channel": h, "channel_err": h_err, "precoder": w,
            "Heff": H_eff, "coherent": H0,
            "rate": self._rate(H_eff, no),
            "rate_coherent": self._rate(H0, no),
            "gain": H_eff.abs().square().sum(dim=(1, 2)).mean(dim=1),
            "gain_coherent": H0.abs().square().sum(dim=(1, 2)).mean(dim=1),
            "tau_seconds": self.error.timing.tau,
        }


def build_link(num_trps=3, num_tx_ant=4, num_ue_ant=1, n_subcarriers=64,
               subcarrier_spacing=30e3, tau_seconds=None, cal_amp_error=None,
               cal_pha_error=None, granularity="SC",
               channel_kind="simple", pathloss=True,
               trp_distances=(100.0, 200.0, 350.0), shadow_fading=True,
               carrier_frequency=3.5e9, delay_spread=100e-9,
               cdl_model="A", tdl_model="C", bs_height=25.0, beta=None):
    """Build a :class:`DMIMODownlink` from plain keyword arguments."""
    return DMIMODownlink(
        num_trps=num_trps, num_tx_ant=num_tx_ant, num_ue_ant=num_ue_ant,
        n_subcarriers=n_subcarriers, subcarrier_spacing=subcarrier_spacing,
        tau_seconds=tau_seconds, cal_amp_error=cal_amp_error,
        cal_pha_error=cal_pha_error, granularity=granularity,
        channel_kind=channel_kind, pathloss=pathloss,
        trp_distances=trp_distances, shadow_fading=shadow_fading,
        carrier_frequency=carrier_frequency, delay_spread=delay_spread,
        cdl_model=cdl_model, tdl_model=tdl_model, bs_height=bs_height,
        beta=beta,
    )


def evaluate_precoder(link, batch_size, snr_db, precoder=None, device=None):
    """Run ``precoder`` (default independent MRT) over one batch."""
    return link(batch_size, snr_db, precoder=precoder, device=device)


def generate_dataset(link, batch_size, snr_db, num_batches=1, device=None):
    """Draw per-sample tensors for (channel, baseline precoder, error, metrics)."""
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


def save_dataset(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(data, path)


def load_dataset(path):
    return torch.load(path, map_location="cpu")
