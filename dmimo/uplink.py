"""
Uplink DMIMO (UDMIMO): single UE, multiple TRPs receiving jointly.

The downlink DMIMO story (``dmimo.link``) is "per-TRP *transmit* precoding is
eroded by inter-TRP timing / calibration errors". The uplink story is its
mirror at the *receive* side: each TRP's receive chain carries its own timing /
calibration error, and a central combiner either loses coherent-combining gain
to those errors or learns / estimates to compensate.

This module provides the **system-level** (rate-domain) model:

* :class:`UDMIMOLink` - channel (reuses the per-TRP downlink channel generator
  via reciprocity: ``h_ul = h_dl^T``), receive-side error application, transmit
  power normalisation, and two combining layers:
  * **L1 (joint detection / signal-level)**: concatenate all TRPs into one
    ``K*N_BS`` receive array and MRC-combine - the information-lossless upper
    bound;
  * **L2 (post-equalisation symbol combining)**: each TRP equalises locally
    (MRC) and the central unit SNR-weights the symbol estimates - the classic
    "distributed combining" model.
  (L3, LLR soft combining, is bit-level and lives in the link-level phase;
  at the rate level L2 with phase alignment is its equivalent.)

Two estimation assumptions are supported, giving the bounds of the
"errors are compensable" question (uplink-specific advantage: each TRP can
measure its own error from its local pilots):

* ``estimate_errors=True``  (optimistic / realistic uplink): the combiner is
  computed from the *error-corrupted* channel, as if each TRP's local channel
  estimate absorbs its own receive error. Errors then only cause amplitude
  loss - the coherent gain is largely preserved.
* ``estimate_errors=False`` (pessimistic / uncompensated): the combiner is
  computed from the *clean* channel while the signal propagates through the
  errored channel, so relative phases are random and the coherent gain is
  destroyed - the same erosion as in the downlink story.

Signal model (rank-1 single stream, per subcarrier ``k``)::

    y_t[k] = g_t[k] * H_t[k] * x[k] + n_t[k]      t = 1..K
    g_t[k] = c_t * exp(-j 2 pi tau_t f_k)         receive timing + calibration
    H_t    : [N_BS x N_UE] uplink channel (reciprocity of the downlink model)

Combining (single stream):
    L1:  u = [g_1 u_1; ...; g_K u_K],  SINR = ||u||^2 / no          -> joint MRC
    L2:  xhat_t = v_t^H y_t (local MRC),  central weighted sum       -> SINR ~ sum_t gamma_t
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from .errors import TRPErrorModel
from .channels import build_downlink_channel


@dataclass
class UDMIMMetrics:
    """Per-batch scalar metrics for the uplink DMIMO link (one combiner)."""

    combiner: str              # "joint" | "symbol"
    rate_bpshz: float          # with receive errors
    rate_coherent_bpshz: float # no errors (g = 1) upper bound
    gain_lin: float            # mean combined power with errors
    gain_coherent_lin: float   # mean combined power, no errors
    gain_loss_db: float        # 10*log10(gain_coherent / gain)

    def as_dict(self) -> dict:
        return asdict(self)


class UDMIMOLink(nn.Module):
    """System-level single-user uplink DMIMO link with per-TRP receive errors.

    Parameters
    ----------
    num_trps : int
        Number of receiving TRPs.
    num_rx_ant : int
        Receive antennas per TRP (N_BS).
    num_ue_ant : int
        UE transmit antennas (N_UE).
    n_subcarriers : int
        Number of subcarriers.
    subcarrier_spacing : float
        Subcarrier spacing in Hz (drives the timing phase ramp).
    tau_seconds : sequence[float] | None
        Per-TRP receive timing offset (TRP 0 must be 0).
    cal_amp_error / cal_pha_error : float | None
        Per-receive-chain calibration error strengths (None = off).
    granularity : str
        Timing phase-ramp granularity (see ``dmimo.errors``).
    channel_kind / pathloss / trp_distances / carrier_frequency /
    cdl_model / tdl_model
        Per-TRP channel generation - reuses ``dmimo.channels``
        (``build_downlink_channel``); the uplink channel is its transpose
        (reciprocity). The error model is applied at the *receive* side.
    tx_mode : str
        UE transmit strategy (open-loop, no UE CSI): ``"equal"`` (fixed
        equal-gain vector), ``"random"`` (per-batch random unit-norm vector,
        open-loop transmit diversity) or ``"custom"`` (uses ``tx_weights``).
    tx_weights : sequence[float] | None
        Fixed unit-norm UE transmit vector (used by ``"equal"`` / ``"custom"``;
        equal-gain by default).
    """

    def __init__(self, num_trps=3, num_rx_ant=4, num_ue_ant=1, n_subcarriers=64,
                 subcarrier_spacing=30e3, tau_seconds=None, cal_amp_error=None,
                 cal_pha_error=None, granularity="SC", channel_kind="simple",
                 pathloss=False, trp_distances=(100.0, 200.0, 350.0),
                 carrier_frequency=3.5e9, delay_spread=100e-9,
                 cdl_model="A", tdl_model="C", tx_mode="equal", tx_weights=None):
        super().__init__()
        self.k = int(num_trps)
        self.nbs = int(num_rx_ant)      # N_BS per TRP
        self.nue = int(num_ue_ant)      # N_UE
        self.n = int(n_subcarriers)
        if tau_seconds is None:
            tau_seconds = [0.0] * self.k
        # Downlink per-TRP channel generator (Nt -> N_BS here); uplink = transpose.
        self.dl_channel = build_downlink_channel(
            channel_kind, num_trps=self.k, num_tx_ant=self.nbs, num_ue_ant=self.nue,
            n_subcarriers=self.n, subcarrier_spacing=subcarrier_spacing,
            carrier_frequency=carrier_frequency, pathloss=pathloss,
            trp_distances=trp_distances, delay_spread=delay_spread,
            cdl_model=cdl_model, tdl_model=tdl_model)
        # Receive-side error: per-TRP timing + per-receive-chain calibration.
        self.error = TRPErrorModel(subcarrier_spacing, self.n, tau_seconds,
                                   num_tx_ant=self.nbs, cal_amp_error=cal_amp_error,
                                   cal_pha_error=cal_pha_error, granularity=granularity)
        # UE transmit strategy (open-loop; no UE-side CSI).
        self.tx_mode = str(tx_mode).lower()
        if tx_weights is None:
            tx_weights = torch.ones(self.nue) / (self.nue ** 0.5)
        self._x0 = torch.as_tensor(list(tx_weights), dtype=torch.complex64)

    # ------------------------------------------------------------------
    def tx_vector(self, batch_size: int, device=None) -> torch.Tensor:
        """UE transmit vector ``x : [B, N_UE]`` (unit norm per batch sample)."""
        if self.tx_mode == "random":
            x = torch.randn(batch_size, self.nue, device=device) \
                + 1j * torch.randn(batch_size, self.nue, device=device)
            return x / (x.abs().square().sum(dim=1, keepdim=True).sqrt() + 1e-12)
        x = self._x0.to(device)
        return x.expand(batch_size, -1)   # "equal" / "custom"

    def channel_ul(self, batch_size: int, device=None) -> torch.Tensor:
        """Uplink channel ``h_ul : [B, K, N_BS, N_UE, N]`` (reciprocity)."""
        h_dl = self.dl_channel.sample(batch_size, device)   # [B, K, N_UE, N_BS, N]
        return h_dl.transpose(2, 3).contiguous()

    def error_ul(self, h_ul: torch.Tensor) -> torch.Tensor:
        """Apply receive-side errors -> ``h_ul_err : [B, K, N_BS, N_UE, N]``."""
        # TRPErrorModel works on [B, K, D, Nt, N]; calibrate the N_BS axis here.
        h = h_ul.transpose(2, 3)                            # [B, K, N_UE, N_BS, N]
        return self.error.apply(h).transpose(2, 3).contiguous()

    def _ue_channel(self, h_ul: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Effective channel incl. UE transmit vector: ``u_t[k] = H_t x``.

        ``h_ul : [B, K, N_BS, N_UE, N]``, ``x : [B, N_UE]``
        -> ``u : [B, K, N_BS, N]``.
        """
        return torch.einsum("bkdtn,bd->bkdn", h_ul, x)

    @staticmethod
    def _rate_from_sinr(sinr: torch.Tensor) -> torch.Tensor:
        """``log2(1 + SINR)`` averaged over subcarriers -> ``[B]``."""
        return torch.log2(1.0 + sinr.clamp(min=1e-12)).mean(dim=1)

    # ------------------------------------------------------------------
    def combine(self, h_ul: torch.Tensor, h_ul_err: torch.Tensor, no: float,
                combiner: str = "joint", estimate_errors: bool = True
                ) -> dict:
        """Compute combined SINR / power for one combining layer.

        Parameters
        ----------
        h_ul : [B, K, N_BS, N_UE, N] clean uplink channel
        h_ul_err : [B, K, N_BS, N_UE, N] error-corrupted channel
        no : float, noise variance per complex dimension
        combiner : "joint" (L1) | "symbol" (L2)
        estimate_errors : bool, combiner computed from the errored channel
            (True = local estimation absorbs the errors) or the clean channel
            (False = uncompensated, pessimistic).

        Returns
        -------
        dict with "sinr" [B, N], "power" [B, N] (combined signal power) and
        "noise" [B, N] (combined noise power).
        """
        B, K, Nbs, Nue, N = h_ul.shape
        x = self.tx_vector(B, h_ul.device)
        u = self._ue_channel(h_ul, x)      # clean effective channel  [B,K,Nbs,N]
        ue = self._ue_channel(h_ul_err, x)  # errored effective channel [B,K,Nbs,N]
        ref = ue if estimate_errors else u

        if combiner == "joint":
            # L1: concatenate all TRPs into one K*N_BS array, MRC.
            s = ref.reshape(B, K * Nbs, N)                       # combiner
            s = s / (s.abs().square().sum(dim=1, keepdim=True).sqrt() + 1e-12)
            ue2 = ue.reshape(B, K * Nbs, N)                      # errored signal vec
            sig = torch.einsum("bkn,bkn->bn", s.conj(), ue2)     # signal (errored)
            power = sig.abs().square()
            noise = no * s.abs().square().sum(dim=1)             # = no (unit comb)
        else:
            # L2: per-TRP local MRC -> symbol estimates xhat_t = a_t x + n_t
            # (a_t = estimated per-TRP amplitude), central amplitude-weighted
            # (scalar-MRC) combining:  sig = sum_t a_t * xhat_t.
            # SNR = sum_t a_t^2 / no == the optimal joint SNR (equal noise).
            v = ref / (ref.abs().square().sum(dim=2, keepdim=True).sqrt() + 1e-12)
            xhat = torch.einsum("bkdn,bkdn->bkn", v.conj(), ue)  # [B,K,N]
            a = ref.abs().square().sum(dim=2).sqrt()             # [B,K,N]
            sig = (a * xhat).sum(dim=1)                          # [B,N]
            power = sig.abs().square()
            noise = no * (a.square() * v.abs().square().sum(dim=2)).sum(dim=1)
        sinr = power / (noise + 1e-12)
        return {"sinr": sinr, "power": power, "noise": noise}

    def forward(self, batch_size: int, snr_db: float, combiner: str = "joint",
                estimate_errors: bool = True, device=None) -> UDMIMMetrics:
        """Run one batch and return scalar metrics for one combiner."""
        h_ul = self.channel_ul(batch_size, device)
        h_ul_err = self.error_ul(h_ul)
        no = 10.0 ** (-snr_db / 10.0)

        r = self.combine(h_ul, h_ul_err, no, combiner, estimate_errors)
        r0 = self.combine(h_ul, h_ul, no, combiner, True)   # coherent upper bound
        rate = self._rate_from_sinr(r["sinr"])
        rate0 = self._rate_from_sinr(r0["sinr"])
        gain = r["power"].mean(dim=1)
        gain0 = r0["power"].mean(dim=1)
        return UDMIMMetrics(
            combiner=combiner,
            rate_bpshz=float(rate.mean().item()),
            rate_coherent_bpshz=float(rate0.mean().item()),
            gain_lin=float(gain.mean().item()),
            gain_coherent_lin=float(gain0.mean().item()),
            gain_loss_db=float(10.0 * torch.log10(gain0.mean() / (gain.mean() + 1e-12)).item()),
        )


def build_ulink(num_trps=3, num_rx_ant=4, num_ue_ant=1, n_subcarriers=64,
                subcarrier_spacing=30e3, tau_seconds=None, cal_amp_error=None,
                cal_pha_error=None, granularity="SC", channel_kind="simple",
                pathloss=False, trp_distances=(100.0, 200.0, 350.0),
                carrier_frequency=3.5e9, delay_spread=100e-9,
                cdl_model="A", tdl_model="C", tx_mode="equal",
                tx_weights=None) -> UDMIMOLink:
    """Build a :class:`UDMIMOLink` from plain keyword arguments."""
    return UDMIMOLink(
        num_trps=num_trps, num_rx_ant=num_rx_ant, num_ue_ant=num_ue_ant,
        n_subcarriers=n_subcarriers, subcarrier_spacing=subcarrier_spacing,
        tau_seconds=tau_seconds, cal_amp_error=cal_amp_error,
        cal_pha_error=cal_pha_error, granularity=granularity,
        channel_kind=channel_kind, pathloss=pathloss,
        trp_distances=trp_distances, carrier_frequency=carrier_frequency,
        delay_spread=delay_spread, cdl_model=cdl_model, tdl_model=tdl_model,
        tx_mode=tx_mode, tx_weights=tx_weights,
    )


def evaluate_ulink(link: UDMIMOLink, batch_size: int, snr_db: float,
                   combiner: str = "joint", estimate_errors: bool = True,
                   device=None) -> UDMIMMetrics:
    """Run ``combiner`` on ``link`` over one batch."""
    return link(batch_size, snr_db, combiner=combiner,
                estimate_errors=estimate_errors, device=device)
