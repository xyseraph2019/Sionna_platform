"""
Link-level uplink DMIMO: transmit *actual bits* from a single UE and decode at
K TRPs with three combining layers, reporting BLER / BER.

Building blocks (mirrors ``dmimo.link_level`` but for the uplink receive side):

* TX: 5G NR TB encoder (LDPC + CRC) -> QAM -> per-subcarrier symbols, spread
  over the UE antennas by the open-loop transmit vector ``x`` (no UE CSI).
* Channel: per-TRP uplink channel ``H_t`` (reciprocity of the downlink model)
  with receive-side timing / calibration errors ``g_t`` -> effective channel
  ``u_t[k] = g_t[k] * H_t[k] * x``; AWGN added per TRP.
* RX (three combining layers, ``combiner``):
  * ``"joint"`` (L1): concatenate all TRPs into one ``K*N_BS`` array and LMMSE
    equalise jointly - the information-lossless upper bound;
  * ``"symbol"`` (L2): per-TRP LMMSE equalisation, central amplitude-weighted
    (scalar-MRC) combining of the symbol estimates;
  * ``"llr"`` (L3): per-TRP equalisation + Demapper, central sum of LLRs.
* Error handling (``estimate_errors``): the equaliser / combiner is computed
  from the *error-corrupted* effective channel (True = each TRP's local
  estimation absorbs its own receive error; realistic uplink) or from the
  *clean* channel (False = uncompensated, errors break coherence).
* Channel estimation (``est_density``): with ``est_density > 0`` the UE sends a
  frequency-comb DMRS (pilot on ``1/est_density`` subcarriers, value 1) and each
  TRP estimates its own effective channel by LS (``h_hat = y_pilot``) with
  linear interpolation - the "absorption" becomes a real physical process with
  estimation noise and interpolation residual. ``est_density = 0`` keeps the
  perfect-CSI semantics (``estimate_errors=True`` uses the true errored
  channel).
* Error detection: 5G NR TB CRC (``use_crc=True``) or plain LDPC full-bit
  comparison.

Noise convention (same as ``dmimo.link_level``): ``no = 10**(-snr_db/10)`` is
the total AWGN variance per resource element, SNR = 1/no.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .uplink import UDMIMOLink


class ULinkLevelDMIMO:
    """Link-level (bit-processing) single-user uplink DMIMO evaluator.

    Parameters
    ----------
    link : UDMIMOLink
        Provides the uplink channel, receive-error model and UE transmit vector.
    qam_order : int
        QAM constellation size (16 -> 4 bits/symbol).
    code_rate : float
        Target transport-block code rate.
    rank : int
        Number of streams (must be 1 in this version - single stream).
    use_crc : bool
        Use the 5G NR TB encoder/decoder with TB CRC (True) or plain LDPC
        with full-bit comparison (False).
    est_density : float
        DMRS pilot density in (0, 1]: pilots on ``1/est_density`` subcarriers
        (``1.0`` = pilot on every subcarrier, no interpolation). ``0`` disables
        estimation (perfect-CSI semantics). Data subcarriers = N - #pilots.
    pilot_boost_db : float
        DMRS pilot power boost (dB) **on top of the density compensation**:
        a comb with density ``d`` concentrates the same total symbol power as
        a data symbol (per-RE power = ``1/d``), so ``boost_db`` is the extra
        power the pilot REs get beyond that. LS estimation noise scales as
        ``no * d / 10**(boost_db/10)``.
    device : str | None
    """

    def __init__(self, link: UDMIMOLink, qam_order: int = 16, code_rate: float = 0.5,
                 rank: int = 1, use_crc: bool = True, est_density: float = 0.0,
                 pilot_boost_db: float = 0.0, device: str = None):
        if int(rank) != 1:
            raise NotImplementedError("ULinkLevelDMIMO v1 supports rank=1 only")
        self.link = link
        self.rank = int(rank)
        self.qam_order = int(qam_order)
        self.bits_sym = int(math.log2(qam_order))
        self.est_density = float(est_density)
        # Pilot amplitude = density compensation (per-symbol power is kept
        # equal to a data symbol: a comb with density d concentrates the same
        # total power on d*N REs, i.e. RE power = 1/d) times the EXTRA boost.
        self.pilot_amp = math.sqrt(10.0 ** (pilot_boost_db / 10.0)) \
            / math.sqrt(max(self.est_density, 1e-6))
        self.pilot_mask = self._pilot_mask(int(link.n), self.est_density)
        self.pilot_idx = torch.nonzero(self.pilot_mask, as_tuple=False).squeeze(-1)
        self.n_data = int(link.n)                      # data uses ALL subcarriers
        self.n = self.n_data * self.rank * self.bits_sym
        self.k = int(round(self.n * code_rate))
        if self.k < 1:
            raise ValueError("Resulting TB too small (k<1); lower code_rate/rank.")
        self.use_crc = bool(use_crc)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        from sionna.phy.mapping import Mapper, Demapper
        from sionna.phy.nr import TBEncoder, TBDecoder

        self.mapper = Mapper("qam", self.bits_sym, device=device)
        self.demapper = Demapper("app", "qam", self.bits_sym, device=device)
        if use_crc:
            self.enc = TBEncoder(target_tb_size=self.k, num_coded_bits=self.n,
                                 target_coderate=code_rate,
                                 num_bits_per_symbol=self.bits_sym,
                                 num_layers=self.rank, channel_type="PDSCH",
                                 device=device)
            self.dec = TBDecoder(self.enc, device=device)
            self.tb_size = self.enc.tb_size
            self.crc_length = self.enc.tb_crc_encoder.crc_length
        else:
            self.enc = LDPC5GEncoder(self.k, self.n,
                                     num_bits_per_symbol=self.bits_sym, device=device)
            self.dec = LDPC5GDecoder(self.enc, device=device)
            self.tb_size = self.k
            self.crc_length = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _pilot_mask(n: int, density: float) -> torch.Tensor:
        """DMRS comb pilot mask in the *pilot symbol* (data occupies all
        subcarriers, time-multiplexed like real DMRS - no data loss).

        ``density <= 0`` -> no pilots (estimation disabled);
        ``density >= 1`` -> pilot on every subcarrier (no interpolation);
        otherwise every ``round(1/density)``-th subcarrier is a pilot.
        """
        m = torch.ones(n, dtype=torch.bool)
        if density <= 0.0:
            m[:] = False
        elif density < 1.0:
            step = max(int(round(1.0 / density)), 2)
            m[:] = False
            m[::step] = True
        return m

    def _ls_estimate(self, y_dmrs: torch.Tensor) -> torch.Tensor:
        """Per-TRP LS channel estimation from the (possibly boosted) comb DMRS.

        ``y_dmrs : [B, K, N_BS, N]`` (pilots only; zeros elsewhere) ->
        ``h_hat : [B, K, N_BS, N]``: LS at the pilot subcarriers
        (``= y / pilot_amp`` since the pilot value is ``pilot_amp``), linear
        interpolation elsewhere. Estimation noise ``no / pilot_amp**2`` enters
        the pilots, so the "absorption" is real: it carries the receive error
        *and* estimation noise (reduced by the density-compensated pilot power
        and the extra boost).
        """
        B, K, Nbs, N = y_dmrs.shape
        idx = self.pilot_idx.to(y_dmrs.device)
        hp = y_dmrs[..., idx] / self.pilot_amp               # [B,K,Nbs,P]

        def _lin(v: torch.Tensor) -> torch.Tensor:
            v = v.reshape(-1, 1, hp.shape[-1])
            out = F.interpolate(v, size=N, mode="linear", align_corners=False)
            return out.reshape(B, K, Nbs, N)

        h_hat = _lin(hp.real) + 1j * _lin(hp.imag)
        h_hat[..., idx] = hp                              # pin exact LS values
        return h_hat

    @staticmethod
    def _lmmse1(y: torch.Tensor, h: torch.Tensor, no: float):
        """Rank-1 LMMSE equalisation over the antenna dim.

        ``y, h : [B, ..., A, N]`` (A = receive antennas, possibly 1).
        Returns ``(x_hat, no_eff)`` with
        ``x_hat = h^H y / (|h|^2 + no)`` (per subcarrier), ``no_eff = no/(|h|^2+no)``.
        """
        hh = h.abs().square().sum(dim=-2)               # [B, ..., N]
        x_hat = torch.einsum("...an,...an->...n", h.conj(), y) / (hh + no)
        no_eff = no / (hh + no)
        return x_hat, no_eff

    def _receive(self, y: torch.Tensor, ue: torch.Tensor, ref: torch.Tensor,
                 no: float, combiner: str):
        """Combine ``y : [B, K, N_BS, N]`` into LLRs ``[B, n]``.

        ``ue`` = errored effective channel (signal propagates through it),
        ``ref`` = channel used for equalisation / combining (``ue`` when
        ``estimate_errors=True``, clean channel otherwise).
        """
        B, K, Nbs, N = y.shape
        if combiner == "joint":
            # L1: one big K*N_BS array, single LMMSE.
            y2 = y.reshape(B, K * Nbs, N)
            h2 = ref.reshape(B, K * Nbs, N)
            x_hat, no_eff = self._lmmse1(y2, h2, no)
            x_hat = x_hat.unsqueeze(1).unsqueeze(1)          # [B,1,1,N]
            no_eff = no_eff.unsqueeze(1).unsqueeze(1)
            llr = self.demapper(x_hat, no_eff)               # [B,1,1,N,bits]
        elif combiner == "symbol":
            # L2: per-TRP unbiased MRC with the *estimated* channel (ref),
            # central scalar-MRC with weights a_t = |ref_t|^2 (SNR-optimal).
            hh = ref.abs().square().sum(dim=-2)                # [B,K,N]
            x_t = torch.einsum("...an,...an->...n", ref.conj(), y) / (hh + 1e-12)
            no_t = no / (hh + 1e-12)
            w = hh / (hh.sum(dim=1, keepdim=True) + 1e-12)
            x_hat = (w * x_t).sum(dim=1)                       # [B,N]
            no_eff = (w.square() * no_t).sum(dim=1)            # [B,N]
            x_hat = x_hat.unsqueeze(1).unsqueeze(1)
            no_eff = no_eff.unsqueeze(1).unsqueeze(1)
            llr = self.demapper(x_hat, no_eff)                 # [B,1,1,N,bits]
        else:  # "llr"
            # L3: per-TRP LMMSE (with the estimated channel ref) + Demapper,
            # central sum of LLRs.
            x_t, no_t = self._lmmse1(y, ref, no)               # [B,K,N]
            x_t = x_t.unsqueeze(2).unsqueeze(2)                # [B,K,1,1,N]
            no_t = no_t.unsqueeze(2).unsqueeze(2)
            llr_t = self.demapper(x_t, no_t)                   # [B,K,1,1,N,bits]
            llr = llr_t.sum(dim=1)                             # [B,1,1,N,bits]
        return llr.reshape(B, self.n)

    def _build_signal(self, batch_size, snr_db, device):
        """TX + channel + (optional DMRS symbol) -> ``(y, ue, u, h_hat, bits)``.

        Data occupies ALL subcarriers (``y = ue*s + n``); when
        ``est_density > 0`` a separate DMRS symbol carries unit-power comb
        pilots (time-multiplexed, no data loss) and ``h_hat`` is the per-TRP LS
        estimate from it.
        """
        no = 10.0 ** (-snr_db / 10.0)
        h_ul = self.link.channel_ul(batch_size, device)
        h_ul_err = self.link.error_ul(h_ul)
        x = self.link.tx_vector(batch_size, device)
        u = self.link._ue_channel(h_ul, x)
        ue = self.link._ue_channel(h_ul_err, x)

        bits = torch.randint(0, 2, (batch_size, self.k), dtype=torch.float32, device=device)
        cw = self.enc(bits)
        s = self.mapper(cw.reshape(batch_size, 1, self.rank, self.n_data,
                                   self.bits_sym)).squeeze(-1)       # [B,1,1,N]
        noise = (torch.randn(batch_size, self.link.k, self.link.nbs, self.link.n,
                             device=device)
                 + 1j * torch.randn(batch_size, self.link.k, self.link.nbs,
                                    self.link.n, device=device)) * math.sqrt(no / 2.0)
        y = ue * s + noise                                          # [B,K,Nbs,N]

        h_hat = None
        if self.est_density > 0:
            # DMRS symbol: boosted unit-power pilots on pilot_idx, zeros
            # elsewhere. Noise is drawn INDEPENDENTLY from the data symbol
            # (realistic: DMRS and data occupy different symbols).
            pgrid = torch.zeros(batch_size, 1, 1, self.link.n,
                                dtype=torch.complex64, device=device)
            pgrid[..., self.pilot_idx.to(device)] = self.pilot_amp
            noise_dmrs = (torch.randn(batch_size, self.link.k, self.link.nbs,
                                      self.link.n, device=device)
                          + 1j * torch.randn(batch_size, self.link.k,
                                             self.link.nbs, self.link.n,
                                             device=device)) * math.sqrt(no / 2.0)
            y_dmrs = ue * pgrid + noise_dmrs
            h_hat = self._ls_estimate(y_dmrs)
        return y, ue, u, h_hat, bits

    def _ref(self, ue, u, h_hat, estimate_errors):
        """Combiner channel: LS estimate (real absorption) > errored (ideal
        absorption) > clean (uncompensated)."""
        if self.est_density > 0 and estimate_errors:
            return h_hat
        return ue if estimate_errors else u

    def block(self, batch_size: int, snr_db: float, combiner: str = "joint",
              estimate_errors: bool = True, device=None, seed=None):
        """Run one batch -> ``(bler, ber, info_bits)`` (inference only)."""
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            device = device or self.device
            y, ue, u, h_hat, bits = self._build_signal(batch_size, snr_db, device)
            no = 10.0 ** (-snr_db / 10.0)
            ref = self._ref(ue, u, h_hat, estimate_errors)
            llr = self._receive(y, ue, ref, no, combiner)
            return self._decode(llr, bits, batch_size, device)

    def _decode(self, llr, bits, batch_size, device):
        if self.use_crc:
            b_hat, crc_ok = self.dec(llr)
            bler = float((~crc_ok).float().mean().item())
        else:
            b_hat = self.dec(llr)[..., :self.k]
            bler = float((b_hat != bits).any(dim=-1).float().mean().item())
        ber = float((b_hat != bits).float().mean().item())
        return bler, ber, self.k

    def evaluate(self, batch_size: int, snr_db: float, configs: dict,
                 device=None, seed=None):
        """Evaluate several ``(combiner, estimate_errors)`` configs on the SAME
        channel / bits realization (fair comparison), like ``evaluate_many``.

        ``configs``: ``{name: (combiner, estimate_errors)}``.
        Returns ``{name: (bler, ber, k)}``.
        """
        if seed is not None:
            torch.manual_seed(seed)
        device = device or self.device
        no = 10.0 ** (-snr_db / 10.0)
        y, ue, u, h_hat, bits = self._build_signal(batch_size, snr_db, device)

        with torch.no_grad():
            out = {}
            for name, (combiner, est) in configs.items():
                ref = self._ref(ue, u, h_hat, est)
                llr = self._receive(y, ue, ref, no, combiner)
                bler, ber, k = self._decode(llr, bits, batch_size, device)
                out[name] = (bler, ber, k)
        return out

    def throughput_bps(self, bler: float, slot_duration: float = 0.5e-3) -> float:
        """Achievable throughput assuming BLER-errored blocks are lost."""
        return self.k * (1.0 - bler) / slot_duration
