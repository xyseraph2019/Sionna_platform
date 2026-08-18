"""
CSI feedback quantization: model the UE -> BS PMI feedback link for any precoder.

The platform currently assumes the BS either has perfect CSI (MRT / CJT) or only
a Type I wideband PMI (codebook). This module adds the missing middle: the UE
computes the ideal *continuous* precoder from the measured channel, quantizes it
to a finite bit budget per subband, and the BS reconstructs and transmits with
the quantized ``w_q``.

Semantics (important)
---------------------
This wrapper quantizes the *output* of a precoder, which models two distinct
physical links - keep them apart when interpreting results:

* **UE -> BS PMI feedback** (3GPP sense): the UE sends a finite-bit
  representation of the precoder to the BS. This applies to *continuous*
  precoders such as MRT / CJT, whose ideal coefficients are otherwise assumed
  to be known perfectly at the BS.
* **Fronthaul signalling / fixed-point precision** (optional, opt-in): the BS
  (e.g. a central unit) sends the precoder coefficients to the TRPs over a
  capacity-limited fronthaul, or implements the precoder in fixed point.

**NN-PMI is NOT quantized by default**: it runs at the BS from the fed-back
Type I PMI and its output is generated locally, so there is no feedback to
quantize. Wrapping an NN precoder in :class:`QuantizedFeedback` (with
``ste=True`` for quantization-aware training) is only meaningful for the
fronthaul / precision story - the examples and configuration therefore exclude
``NN-PMI`` from output quantization by default.

Design
------
:class:`QuantizedFeedback` is a :class:`~dmimo.precoding.Precoder`-protocol
wrapper, so it plugs into the existing ``evaluate_many`` fairness framework
(shared channel / bits / errors across all precoders) without touching
``link.py`` / ``link_level.py``.

Quantizers
----------
* :class:`PhaseQuantizer` - constant-modulus phase quantization (Type-I/II
  style): the amplitude is retained (or optionally quantized with ``bits_amp``),
  the phase is uniformly quantized to ``bits_phase`` bits per complex
  coefficient.
* :class:`ScalarQuantizer` - real & imaginary parts are clipped to the peak
  amplitude and uniformly quantized to ``bits`` levels each.

Feedback granularity
--------------------
One representative coefficient per subband (the subband's centre subcarrier),
broadcast back to subcarriers - the same ``subband_size`` convention as
:class:`~dmimo.precoding.CJTPrecoder` and ``NNMixerPMI``.

Bit budget (per slot)::

    feedback_bits = K * Nt * r * S * b_per_coeff,   S = ceil(N / subband_size)

with ``b_per_coeff = bits_phase`` (phase) or ``2 * bits`` (I/Q).

Straight-through estimator
--------------------------
``ste=True`` quantizes in the forward pass but passes the gradient through
unchanged, so an NN precoder can be trained end-to-end *through* the quantizer
to learn quantization-robust precoding (used by ``examples/modelTrain.py``).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .precoding import Precoder, _normalize


def _uq(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform quantizer on [-1, 1] with ``2**bits`` levels (round-to-nearest)."""
    L = 2 ** int(bits)
    return torch.round(x.clamp(-1.0, 1.0) * (L - 1)) / (L - 1)


class PhaseQuantizer:
    """Constant-modulus phase quantization (per complex coefficient).

    Parameters
    ----------
    bits_phase : int
        Phase resolution in bits (2/3/4/6/8 -> 4/8/16/64/256 phase levels).
    bits_amp : int | None
        Amplitude resolution in bits; ``None`` keeps the continuous amplitude.
        When set, the amplitude is normalised per batch to its max and uniformly
        quantized (Type-II-style amplitude feedback).
    """

    def __init__(self, bits_phase: int = 4, bits_amp: Optional[int] = None):
        self.bits_phase = int(bits_phase)
        self.bits_amp = None if bits_amp is None else int(bits_amp)

    @property
    def bits_per_coeff(self) -> int:
        return self.bits_phase + (self.bits_amp or 0)

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        L = 2 ** self.bits_phase
        step = 2.0 * torch.pi / L
        phase = torch.round(torch.angle(w) / step) * step
        amp = torch.abs(w)
        if self.bits_amp:
            m = amp.amax().clamp_min(1e-12)
            amp = _uq(amp / m, self.bits_amp) * m
        return amp * torch.exp(1j * phase)


class ScalarQuantizer:
    """I/Q scalar quantization: clip to peak amplitude, quantize each part.

    Parameters
    ----------
    bits : int
        Bits per real (and per imaginary) part; total = ``2 * bits`` per
        complex coefficient.
    """

    def __init__(self, bits: int = 4):
        self.bits = int(bits)

    @property
    def bits_per_coeff(self) -> int:
        return 2 * self.bits

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        m = w.abs().amax().clamp_min(1e-12)
        return (_uq(w.real / m, self.bits) + 1j * _uq(w.imag / m, self.bits)) * m


class QuantizedFeedback:
    """Precoder-protocol wrapper: ``base(h) -> w -> quantize -> w_q``.

    Parameters
    ----------
    base_precoder : Precoder
        The continuous precoder to wrap (MRT / ZF / CJT / NN-PMI).
    quantizer : PhaseQuantizer | ScalarQuantizer
        The coefficient quantizer applied per subband representative.
    subband_size : int
        Subcarriers per feedback subband (one representative per subband).
    ste : bool
        Straight-through estimator: quantize in forward, identity gradient.

    Notes
    -----
    * Type I already *is* a quantized codebook (beam index + QPSK co-phasing);
      wrapping it again would be a double quantization, so it is left untouched
      and used as the fixed low-bit reference in comparisons.
    * ``precodes_from_errors`` is inherited from the base precoder (CJT
      observes/compensates errors; MRT / NN-PMI do not).
    """

    def __init__(self, base_precoder, quantizer, subband_size: int = 12,
                 ste: bool = False):
        self.base = base_precoder
        self.q = quantizer
        self.subband_size = int(subband_size)
        self.ste = bool(ste)

    @property
    def precodes_from_errors(self) -> bool:
        return getattr(self.base, "precodes_from_errors", False)

    def feedback_bits(self, num_trps: int, num_tx_ant: int, rank: int,
                      n_subcarriers: int) -> int:
        """Total feedback bits per slot: K * Nt * r * S * bits_per_coeff."""
        S = int(math.ceil(int(n_subcarriers) / self.subband_size))
        return int(num_trps * num_tx_ant * rank * S * self.q.bits_per_coeff)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """``h : [B, K, D, Nt, N]`` -> quantized precoder ``w_q : [B, K, Nt, r, N]``."""
        w = self.base(h)                       # [B, K, Nt, r, N]
        B, K, Nt, r, N = w.shape
        S = int(math.ceil(N / self.subband_size))
        # subband representative = centre subcarrier of each subband
        centers = torch.arange(S, device=w.device) * self.subband_size \
            + self.subband_size // 2
        centers = centers.clamp(max=N - 1).long()
        ws = w[..., centers]                   # [B, K, Nt, r, S]
        wq = self.q(ws)
        if self.ste:
            wq = ws + (wq - ws).detach()       # straight-through estimator
        sub = (torch.arange(N, device=w.device) // self.subband_size).long()
        wq = wq[..., sub]                      # broadcast back to subcarriers
        return _normalize(wq)


def make_quantized(base_precoder, quant: str = "phase", bits_phase: int = 4,
                   bits_amp: Optional[int] = None, bits_iq: int = 4,
                   subband_size: int = 12, ste: bool = False) -> QuantizedFeedback:
    """Build a :class:`QuantizedFeedback` wrapper from a quantizer name.

    ``quant``: ``"phase"`` -> :class:`PhaseQuantizer`, ``"iq"`` ->
    :class:`ScalarQuantizer`. Any other value raises ``ValueError``.
    """
    if quant == "phase":
        qz = PhaseQuantizer(bits_phase=bits_phase, bits_amp=bits_amp)
    elif quant == "iq":
        qz = ScalarQuantizer(bits=bits_iq)
    else:
        raise ValueError(f"unknown quantizer '{quant}' (expected 'phase' | 'iq')")
    return QuantizedFeedback(base_precoder, qz, subband_size=subband_size, ste=ste)
