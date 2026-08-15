"""
Per-TRP imperfection model for the downlink DMIMO scenario, aligned with the
reference ``apply_interTRP_error`` implementation (MATLAB port).

Both errors are applied directly to the raw channel grid ``h : [B, K, D, Nt, N]``
(*kernel* per-TRP timing ramp first, then per-TX-antenna calibration):

  1. **Timing error** (时延). TRP 1 is the reference (zero offset); other TRPs get
     ``tau_t`` seconds. In frequency domain this is a per-subcarrier linear phase
     ramp ``exp(-j 2 pi scs * idx * tau_t)``, where ``idx`` depends on
     ``granularity`` ('SC' | 'RB' | 'SC_RB_granular') exactly as in the reference.
  2. **Calibration error** (校准). Per-TX-antenna complex scalar
     ``Wr = |1 + sqrt(cal_amp_error/2) randn| * exp(j sqrt(cal_pha_error) randn)``.
     TRP 1 (the clean reference) gets identity (no calibration error).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _idx_vec(n: int, granularity: str):
    """Per-subcarrier frequency index, matching ``apply_interTRP_error``."""
    if granularity == "RB":
        return [(k + 1) * 12 for k in range(n)]                  # (1..N)*12
    if granularity == "SC_RB_granular":
        return [((k // 12) * 12) + 12 for k in range(n)]         # RB-boundary step
    return [k + 1 for k in range(n)]                             # 1..N (SC)


class TimingError(nn.Module):
    """Relative timing offsets -> per-subcarrier phase ramps (grid-level).

    Parameters
    ----------
    subcarrier_spacing : float
        Sub-carrier spacing in Hz (e.g. ``30e3``).
    n_subcarriers : int
        Number of (used) subcarriers.
    tau_seconds : sequence[float]
        Per-TRP timing offset in seconds relative to TRP 1 (first entry 0).
    granularity : str
        ``'SC'`` | ``'RB'`` | ``'SC_RB_granular'`` (per reference).
    """

    def __init__(self, subcarrier_spacing=30e3, n_subcarriers=64,
                 tau_seconds=(0.0, 130e-9, 260e-9), granularity="SC"):
        super().__init__()
        self.d_f = float(subcarrier_spacing)
        self.n = int(n_subcarriers)
        self.granularity = granularity
        self.register_buffer("tau", torch.as_tensor(list(tau_seconds), dtype=torch.float32))
        self.register_buffer("_idx", torch.as_tensor(_idx_vec(self.n, granularity),
                                                     dtype=torch.float32))

    @property
    def num_trps(self) -> int:
        return self.tau.numel()

    def phase_factor(self, device=None) -> torch.Tensor:
        """Per-TRP, per-subcarrier phase ramp ``[K, N] = exp(-j 2 pi d_f idx tau)``."""
        tau = self.tau.to(device)
        idx = self._idx.to(device)
        phase = -2.0 * torch.pi * self.d_f * tau.unsqueeze(1) * idx.unsqueeze(0)
        return torch.exp(1j * phase)

    def offset(self, batch_size, device=None) -> torch.Tensor:
        """Broadcastable ``[1, K, 1, 1, N]`` timing ramp (for direct use)."""
        f = self.phase_factor(device)                 # [K, N]
        return f.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(batch_size, -1, 1, 1, -1)


class CalibrationError(nn.Module):
    """Random per-TX-antenna amplitude / phase calibration (TRP 1 = identity).

    Follows the reference: non-reference TRPs get each TX antenna scaled by
    ``|1 + sqrt(cal_amp_error/2)*randn| * exp(j sqrt(cal_pha_error)*randn)``.
    ``cal_amp_error`` / ``cal_pha_error`` are linear (variance-like) strengths;
    ``None`` (or 0) disables the corresponding part.
    """

    def __init__(self, num_trps=3, num_tx_ant=1, cal_amp_error=None, cal_pha_error=None):
        super().__init__()
        self.k = int(num_trps)
        self.nt = int(num_tx_ant)
        self.cal_amp_error = None if cal_amp_error is None else float(cal_amp_error)
        self.cal_pha_error = None if cal_pha_error is None else float(cal_pha_error)

    @property
    def num_trps(self) -> int:
        return self.k

    def offset(self, batch_size, device=None) -> torch.Tensor:
        """Per-TX-antenna calibration ``[B, K, Nt]`` (TRP 1 = ones)."""
        B = int(batch_size)
        amp = torch.ones(B, self.k, self.nt, device=device)
        pha = torch.zeros(B, self.k, self.nt, device=device)
        if self.cal_amp_error:
            amp[:, 1:, :] = torch.abs(
                1.0 + math.sqrt(self.cal_amp_error / 2.0) * torch.randn(B, self.k - 1, self.nt, device=device))
        if self.cal_pha_error:
            pha[:, 1:, :] = math.sqrt(self.cal_pha_error) * torch.randn(B, self.k - 1, self.nt, device=device)
        return amp * torch.exp(1j * pha)              # [B, K, Nt]


class TRPErrorModel(nn.Module):
    """Applies timing + calibration errors directly to the channel grid.

    ``h : [B, K, D, Nt, N]`` -> ``h_err`` (timing ramp then per-antenna calibration).
    """

    def __init__(self, subcarrier_spacing=30e3, n_subcarriers=64,
                 tau_seconds=(0.0, 130e-9, 260e-9), num_tx_ant=1,
                 cal_amp_error=None, cal_pha_error=None, granularity="SC"):
        super().__init__()
        self.timing = TimingError(subcarrier_spacing, n_subcarriers, tau_seconds, granularity)
        self.calibration = CalibrationError(self.timing.num_trps, num_tx_ant,
                                            cal_amp_error, cal_pha_error)

    @property
    def num_trps(self) -> int:
        return self.timing.num_trps

    @property
    def granularity(self) -> str:
        return self.timing.granularity

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        """Return the error-corrupted channel (timing then calibration)."""
        B, K, D, Nt, N = h.shape
        h = h * self.timing.phase_factor(h.device).view(1, K, 1, 1, N)  # timing
        E = self.calibration.offset(B, h.device)                        # [B,K,Nt]
        h = h * E.view(B, K, 1, Nt, 1)                                  # per-TX cal
        return h
