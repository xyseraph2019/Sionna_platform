"""
Link-level DMIMO models in the style of the Sionna CDL tutorial
(``MIMO_OFDM_Transmissions_over_CDL``): one ``Block``-based model per direction
whose ``call(batch_size, ebno_db)`` returns the transmitted / received
information bits.

* :class:`DLModel` — downlink: K TRPs (each with its own CDL + timing /
  calibration error) transmit precoded streams to a single UE.
* :class:`ULModel` — uplink: a single UE transmits to K TRPs which combine
  their received signals (``"joint"`` / ``"symbol"`` / ``"llr"``).

Both share :class:`DMIMOPhyModel`, which owns the OFDM resource grid, DMRS
pilots, the 5G-NR TB (LDPC + CRC) FEC chain, LS channel estimation and LMMSE
equalization — the exact building blocks of the tutorial.

PHY conventions (aligned with the tutorial)
-------------------------------------------
* Noise is set via ``ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)``:
  the Eb/N0 axis accounts for the modulation order, the code rate and the
  pilot / nulled-carrier overhead of the resource grid.
* The channel is modeled in the *frequency domain*: the CIR is sampled once
  per OFDM symbol (symbol-rate sampling) and converted with
  ``cir_to_ofdm_channel`` (see :class:`dmimo.channels.DMIMOChannel`).
  ``domain="time"`` (``cir_to_time_channel`` + ``OFDMModulator`` /
  ``OFDMDemodulator``) is reserved as an extension point.
* ``perfect_csi=True`` uses the true effective channel; ``perfect_csi=False``
  uses LS channel estimation from the DMRS pilots with nearest-neighbour
  interpolation — both feeding an LMMSE equalizer.
* Downlink precoding happens in the frequency domain from the *effective*
  (non-nulled) subcarriers, per TRP and per OFDM symbol; the combining at the
  UE sums the per-TRP precoded channels (coherent joint transmission).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from sionna.phy import Block
from sionna.phy.channel import ApplyOFDMChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.nr import TBEncoder, TBDecoder
from sionna.phy.ofdm import (ResourceGrid, ResourceGridMapper, LSChannelEstimator,
                             LMMSEEqualizer, PilotPattern)
from sionna.phy.utils import ebnodb2no

from .channels import DMIMOChannel
from .errors import TRPErrorModel
from .precoding import IndependentMRT


def _effective_indices(fft_size, num_guard_carriers, dc_null) -> np.ndarray:
    """Effective (non-nulled) subcarrier indices of the full FFT grid."""
    n_left, n_right = int(num_guard_carriers[0]), int(num_guard_carriers[1])
    ind = np.arange(n_left, fft_size - n_right)
    if dc_null:
        ind = ind[ind != fft_size // 2]
    return ind


def _dense_pilot_pattern(num_streams, num_ofdm_symbols, fft_size, pilot_syms,
                         eff_indices, device):
    """Interleaved-comb pilot pattern with full per-stream subcarrier coverage.

    ``KroneckerPilotPattern`` gives each stream a frequency comb with spacing
    ``num_tx*num_streams_per_tx`` (= rank), so at rank >= 2 every stream has
    pilots on only a fraction of the subcarriers and ``LSChannelEstimator`` must
    interpolate in frequency; that interpolation bias is systematic and is *not*
    reflected in the returned ``err_var``, which pins the BLER to a high-SNR
    floor. This pattern alternates the comb offset per pilot symbol so that
    every subcarrier of every stream is a nonzero pilot in exactly one of the
    pilot symbols (the extra symbol is the physical cost of full per-stream
    coverage). Pilots are only placed on the *effective* subcarriers so that
    ``ResourceGrid.num_data_symbols`` stays consistent with the mapper's data
    positions (as with the Kronecker pattern).
    """
    num_tx = 1
    mask = np.zeros([num_tx, num_streams, num_ofdm_symbols, fft_size], dtype=bool)
    vals = np.zeros([num_tx, num_streams, num_ofdm_symbols, fft_size], dtype=np.complex64)
    for s in range(num_streams):
        for i, sym in enumerate(pilot_syms):
            mask[0, s, sym, eff_indices] = True      # reserve effective pilot REs
            parity = (s + i) % num_streams
            sub = eff_indices[parity % num_streams::num_streams]
            vals[0, s, sym, sub] = 1.0 + 0.0j        # interleaved comb (zeros elsewhere)
    pilots = np.stack([vals[0, s][mask[0, s]] for s in range(num_streams)])[None]
    return PilotPattern(mask=mask, pilots=pilots, normalize=False, device=device)


def _effective_mask(rg) -> torch.Tensor:
    """Boolean mask over the full FFT grid selecting the effective subcarriers."""
    ind = rg.effective_subcarrier_ind
    mask = torch.zeros(int(rg.fft_size), dtype=torch.bool)
    mask[ind] = True
    return mask


class DMIMOPhyModel(Block):
    """Shared link-level PHY chain (resource grid, FEC, estimation, equalization).

    Parameters
    ----------
    subcarrier_spacing : float
        Subcarrier spacing in Hz.
    fft_size : int
        FFT size (full subcarrier grid, incl. guard / DC carriers).
    num_guard_carriers : tuple[int, int]
        Number of nulled guard carriers to the left / right of the spectrum.
    dc_null : bool
        Null the DC subcarrier.
    n_symbols : int
        Number of OFDM symbols per slot.
    pilot_ofdm_symbol_indices : sequence[int]
        OFDM symbol indices reserved for DMRS pilots (front-loaded, 5G
        mapping type A: ``[2, 11]`` like the tutorial).
    pilot_boost_db : float
        DMRS pilot energy boost in dB relative to a data symbol.
    cyclic_prefix_length : int
        Cyclic-prefix length in samples (only affects Eb/N0 in the frequency
        domain; matters for the reserved time-domain extension).
    qam_order : int
        QAM constellation size (2 -> QPSK).
    code_rate : float
        Target transport-block code rate (info bits / coded bits).
    rank : int
        Number of layers (streams per transmitter).
    use_crc : bool
        Use the 5G NR transport-block encoder/decoder with TB CRC error
        detection; otherwise a plain LDPC code with full-bit comparison.
    perfect_csi : bool
        Perfect CSI at the receiver (true effective channel) instead of LS
        channel estimation from the DMRS pilots.
    device : str | None
    """

    def __init__(self, subcarrier_spacing=15e3, fft_size=76,
                 num_guard_carriers=(5, 6), dc_null=True, n_symbols=14,
                 pilot_ofdm_symbol_indices=(2, 11), pilot_boost_db=0.0,
                 cyclic_prefix_length=6, qam_order=4, code_rate=0.5, rank=1,
                 use_crc=True, perfect_csi=False, device=None):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device
        self.rank = int(rank)
        self.qam_order = int(qam_order)
        self.bits_sym = int(math.log2(qam_order))
        self.code_rate = float(code_rate)
        self.perfect_csi = bool(perfect_csi)
        self.use_crc = bool(use_crc)
        self.n_symbols = int(n_symbols)
        self.cyclic_prefix_length = int(cyclic_prefix_length)
        self.pilot_boost_db = float(pilot_boost_db)
        self.pilot_syms = [int(i) % self.n_symbols for i in pilot_ofdm_symbol_indices]
        for i in self.pilot_syms:
            if not (0 <= i < self.n_symbols):
                raise ValueError(f"pilot symbol index {i} out of range [0, {self.n_symbols}).")

        # ---- OFDM resource grid with DMRS pilots ---------------------------
        # The interleaved-comb pattern (full per-stream subcarrier coverage)
        # is only used in the legacy flat-grid configuration (no guard / DC
        # carriers): with guard/DC carriers Sionna 2.0.1's type-grid data count
        # disagrees with ``num_data_symbols`` for custom patterns, so the
        # tutorial-style Kronecker pattern is used there (it is consistent).
        legacy_flat = list(num_guard_carriers) == [0, 0] and not dc_null
        if self.rank >= 2 and len(self.pilot_syms) >= 2 and legacy_flat:
            eff = _effective_indices(int(fft_size), num_guard_carriers, dc_null)
            pp = _dense_pilot_pattern(self.rank, self.n_symbols, int(fft_size),
                                      self.pilot_syms, eff, device)
            self.rg = ResourceGrid(
                num_ofdm_symbols=self.n_symbols, fft_size=int(fft_size),
                subcarrier_spacing=subcarrier_spacing, num_tx=1,
                num_streams_per_tx=self.rank,
                cyclic_prefix_length=self.cyclic_prefix_length,
                num_guard_carriers=list(num_guard_carriers), dc_null=dc_null,
                pilot_pattern=pp, device=device)
        else:
            self.rg = ResourceGrid(
                num_ofdm_symbols=self.n_symbols, fft_size=int(fft_size),
                subcarrier_spacing=subcarrier_spacing, num_tx=1,
                num_streams_per_tx=self.rank,
                cyclic_prefix_length=self.cyclic_prefix_length,
                num_guard_carriers=list(num_guard_carriers), dc_null=dc_null,
                pilot_pattern="kronecker",
                pilot_ofdm_symbol_indices=self.pilot_syms, device=device)
        if self.pilot_boost_db != 0.0:
            amp = math.sqrt(10.0 ** (self.pilot_boost_db / 10.0))
            self.rg.pilot_pattern.pilots.mul_(amp)

        self.effective_mask = _effective_mask(self.rg)   # [fft] bool
        self.n_eff = int(self.effective_mask.sum().item())

        # ---- Sionna PHY blocks ---------------------------------------------
        self.sm = StreamManagement(np.array([[1]]), self.rank)
        self.rg_mapper = ResourceGridMapper(self.rg, device=device)
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="nn",
                                         device=device)
        self.lmmse = LMMSEEqualizer(self.rg, self.sm, device=device)
        self.channel_apply = ApplyOFDMChannel(add_awgn=True, device=device)

        # ---- data / coded bit budget ----------------------------------------
        # ``rg.num_data_symbols`` is the number of data REs **per stream** on
        # the effective subcarriers (pilots and nulled carriers excluded).
        self.n_data_sym = int(self.rg.num_data_symbols)
        self.n = self.n_data_sym * self.rank * self.bits_sym
        self.k = int(round(self.n * self.code_rate))
        if self.k < 1:
            raise ValueError("Resulting TB too small (k<1); lower code_rate/rank.")

        # ---- TX / RX blocks -------------------------------------------------
        self.mapper = Mapper("qam", self.bits_sym, device=device)
        self.demapper = Demapper("app", "qam", self.bits_sym, device=device)
        if use_crc:
            self.encoder = TBEncoder(target_tb_size=self.k, num_coded_bits=self.n,
                                     target_coderate=self.code_rate,
                                     num_bits_per_symbol=self.bits_sym,
                                     num_layers=self.rank, channel_type="PDSCH",
                                     device=device)
            self.decoder = TBDecoder(self.encoder, device=device)
            self.tb_size = self.encoder.tb_size
            self.crc_length = self.encoder.tb_crc_encoder.crc_length
        else:
            self.encoder = LDPC5GEncoder(self.k, self.n,
                                         num_bits_per_symbol=self.bits_sym,
                                         device=device)
            self.decoder = LDPC5GDecoder(self.encoder, device=device)
            self.tb_size = self.k
            self.crc_length = 0

    # ------------------------------------------------------------------
    # Fairness interface: sample one realization once, evaluate several
    # configurations (precoders / combiners) on the very same tensors.
    # ------------------------------------------------------------------
    def sample_realization(self, batch_size, device=None) -> dict:
        """Sample channel + information bits; overridden by the subclasses."""
        raise NotImplementedError

    def block_from_realization(self, real: dict, ebno_db: float, device=None,
                               **config):
        """Evaluate one configuration on a pre-sampled realization.

        Returns ``(bler, ber, k)``.
        """
        bits, b_hat, crc_ok = self._forward_bits(real, ebno_db, device, **config)
        return self._bler_ber(bits, b_hat, crc_ok)

    def call(self, batch_size: int, ebno_db: float, device=None, **config):
        """Tutorial-style entry: ``(b, b_hat)`` for one batch (inference)."""
        with torch.no_grad():
            real = self.sample_realization(batch_size, device)
            bits, b_hat, _crc = self._forward_bits(real, ebno_db, device, **config)
            return bits, b_hat

    # ------------------------------------------------------------------
    def _bler_ber(self, bits, b_hat, crc_ok=None):
        """BLER / BER from transmitted and decoded information bits."""
        if self.use_crc and crc_ok is not None:
            bler = float((~crc_ok).float().mean().item())
        else:
            bler = float((b_hat != bits).any(dim=-1).float().mean().item())
        ber = float((b_hat != bits).float().mean().item())
        return bler, ber, self.k

    def _decode(self, llr: torch.Tensor, bits: torch.Tensor):
        """Decode LLRs -> ``(b_hat, crc_ok)`` (``crc_ok`` None for plain LDPC)."""
        if self.use_crc:
            b_hat, crc_ok = self.decoder(llr)
            return b_hat[..., :self.k], crc_ok
        return self.decoder(llr)[..., :self.k], None

    def _embed_full(self, h_eff: torch.Tensor) -> torch.Tensor:
        """Embed an effective-subcarrier channel into the full FFT grid."""
        B, nr, nra, nt, nta, nsym, _neff = h_eff.shape
        out = torch.zeros(B, nr, nra, nt, nta, nsym, self.rg.fft_size,
                          dtype=h_eff.dtype, device=h_eff.device)
        out[..., self.effective_mask.to(h_eff.device)] = h_eff
        return out


class DLModel(DMIMOPhyModel):
    """Downlink DMIMO: K TRPs -> 1 UE (coherent joint transmission).

    In addition to the :class:`DMIMOPhyModel` parameters:

    Parameters
    ----------
    num_trps : int
        Number of transmitting TRPs (K).
    num_bs_ant : int
        Antennas per TRP (must be even for CDL: dual-polarised array).
    num_ue_ant : int
        UE receive antennas (also the number of streams for ``rank``).
    channel_kind : str
        ``"simple" | "cdl" | "tdl" | "uma" | "umi"`` (see DMIMOChannel).
    cdl_model / tdl_model : str
        3GPP 38.901 model index (CDL: A..E).
    delay_spread : float
        Nominal delay spread [s].
    speed : float
        UE speed [m/s] (channel aging; 0 = static).
    pathloss, trp_distances, carrier_frequency, shadow_fading, bs_height
        Per-TRP coverage settings (see DMIMOChannel).
    tau_seconds / cal_amp_error / cal_pha_error / granularity
        Per-TRP timing / calibration error model (see dmimo.errors).
    precoder : Precoder | None
        Default precoder used when none is passed to ``call`` / ``block_...``.
    """

    def __init__(self, num_trps=3, num_bs_ant=8, num_ue_ant=4,
                 channel_kind="cdl", cdl_model="A", tdl_model="C",
                 delay_spread=100e-9, speed=0.0, pathloss=False,
                 trp_distances=(100.0, 200.0, 350.0), carrier_frequency=2.6e9,
                 tau_seconds=None, cal_amp_error=None, cal_pha_error=None,
                 granularity="SC", shadow_fading=True, bs_height=25.0,
                 precoder=None, **phy):
        rank = int(phy.pop("rank", min(int(num_ue_ant), 4)))
        super().__init__(rank=rank, **phy)
        self.num_trps = int(num_trps)
        self.num_bs_ant = int(num_bs_ant)
        self.num_ue_ant = int(num_ue_ant)
        if tau_seconds is None:
            tau_seconds = [0.0] * self.num_trps
        if len(tau_seconds) != self.num_trps:
            raise ValueError(
                f"tau_seconds has {len(tau_seconds)} entries but num_trps="
                f"{self.num_trps} (one timing offset per TRP is required).")
        self.channel = DMIMOChannel(
            kind=channel_kind, num_trps=self.num_trps,
            num_bs_ant=self.num_bs_ant, num_ue_ant=self.num_ue_ant,
            fft_size=self.rg.fft_size,
            subcarrier_spacing=self.rg.subcarrier_spacing,
            carrier_frequency=carrier_frequency, delay_spread=delay_spread,
            cdl_model=cdl_model, tdl_model=tdl_model, speed=speed,
            pathloss=pathloss, trp_distances=trp_distances,
            shadow_fading=shadow_fading, bs_height=bs_height, device=self._device)
        self.error = TRPErrorModel(self.rg.subcarrier_spacing, self.n_eff,
                                   tau_seconds, num_tx_ant=self.num_bs_ant,
                                   cal_amp_error=cal_amp_error,
                                   cal_pha_error=cal_pha_error,
                                   granularity=granularity)
        self.precoder = precoder or IndependentMRT(rank=self.rank)

    # ------------------------------------------------------------------
    def sample_realization(self, batch_size, device=None) -> dict:
        device = device or self._device
        h_grid = self.channel.sample(batch_size, self.n_symbols,
                                     self.rg.ofdm_symbol_duration, device)
        h_eff = h_grid[..., self.effective_mask.to(h_grid.device)]
        h_err_eff = self._apply_errors(h_eff)
        bits = torch.randint(0, 2, (batch_size, self.k),
                             dtype=torch.float32, device=device)
        return {"h_eff": h_eff, "h_err_eff": h_err_eff, "bits": bits}

    def _apply_errors(self, h_eff: torch.Tensor) -> torch.Tensor:
        """Per-TRP timing / calibration errors on the effective subcarriers."""
        syms = []
        for sym in range(self.n_symbols):
            hs = h_eff[:, 0, :, :, :, sym, :].permute(0, 2, 1, 3, 4)  # [B,K,D,Nt,n_eff]
            he = self.error.apply(hs)                                  # errors
            he = he.permute(0, 2, 1, 3, 4)                             # [B,D,K,Nt,n_eff]
            syms.append(he.unsqueeze(1).unsqueeze(5))                  # [B,1,D,K,Nt,1,n_eff]
        return torch.cat(syms, dim=5)                                  # [B,1,D,K,Nt,n_sym,n_eff]

    def _precode(self, precoder, h_eff: torch.Tensor, h_err_eff: torch.Tensor):
        """Per-symbol, per-TRP frequency-domain precoding on effective subcarriers.

        Returns ``(w, H_eff)`` with ``w : [B,K,Nt,rank,n_sym,n_eff]`` and
        ``H_eff : [B,1,D,rank,n_sym,n_eff]`` (coherent sum over the TRPs of the
        *errored* channels, i.e. the actual transmission channel).
        """
        w_syms, H_syms = [], []
        for sym in range(self.n_symbols):
            hc = h_eff[:, 0, :, :, :, sym, :].permute(0, 2, 1, 3, 4)    # clean
            he = h_err_eff[:, 0, :, :, :, sym, :].permute(0, 2, 1, 3, 4)
            src = he if getattr(precoder, "precodes_from_errors", False) else hc
            w = precoder(src)                                          # [B,K,Nt,r,n_eff]
            H = torch.einsum("bkdan,bkaln->bdln", he, w)               # [B,D,r,n_eff]
            w_syms.append(w)
            H_syms.append(H)
        w = torch.stack(w_syms, dim=-2)                                # [B,K,Nt,r,n_sym,n_eff]
        H_eff = torch.stack(H_syms, dim=-2).unsqueeze(1)               # [B,1,D,r,n_sym,n_eff]
        return w, H_eff

    def _forward_bits(self, real: dict, ebno_db: float, device=None,
                      precoder=None):
        device = device or self._device
        precoder = precoder or self.precoder
        batch_size = real["bits"].shape[0]
        no = float(ebnodb2no(ebno_db, self.bits_sym, self.code_rate, self.rg))

        _, H_eff = self._precode(precoder, real["h_eff"], real["h_err_eff"])
        H_eff = H_eff.unsqueeze(3)                       # [B,1,D,1,rank,n_sym,n_eff]
        bits = real["bits"]
        cw = self.encoder(bits)
        cw_r = cw.reshape(batch_size, 1, self.rank, self.n_data_sym, self.bits_sym)
        s = self.mapper(cw_r).squeeze(-1)                    # [B,1,rank,n_data]
        x_rg = self.rg_mapper(s)                             # [B,1,rank,n_sym,fft]

        H_eff_full = self._embed_full(H_eff)                 # [B,1,D,1,rank,n_sym,fft]
        y = self.channel_apply(x_rg, H_eff_full, no)               # [B,1,D,n_sym,fft]

        if self.perfect_csi:
            h_hat, err_var = H_eff, 0.0
        else:
            h_hat, err_var = self.ls_est(y, no)
        x_hat, no_eff = self.lmmse(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        llr = llr.reshape(batch_size, self.n)
        b_hat, crc_ok = self._decode(llr, bits)
        return bits, b_hat, crc_ok


class ULModel(DMIMOPhyModel):
    """Uplink DMIMO: 1 UE -> K TRPs (rank-1, distributed combining).

    In addition to the :class:`DMIMOPhyModel` parameters:

    Parameters
    ----------
    num_trps : int
        Number of receiving TRPs (K).
    num_bs_ant : int
        Receive antennas per TRP.
    num_ue_ant : int
        UE transmit antennas.
    channel_kind / cdl_model / tdl_model / delay_spread / speed / pathloss /
    trp_distances / carrier_frequency / shadow_fading / bs_height
        Channel settings (the uplink channel is the transpose of the downlink
        one, reciprocity; errors are receive-side per TRP).
    tau_seconds / cal_amp_error / cal_pha_error / granularity
        Per-TRP receive timing / calibration errors.
    tx_mode : str
        UE open-loop transmit strategy: ``"equal"`` (fixed equal-gain vector),
        ``"random"`` (per-batch random unit-norm vector).
    combiner : str
        ``"joint"`` (L1, all TRPs equalised jointly) | ``"symbol"`` (L2,
        per-TRP LMMSE + amplitude-weighted combining) | ``"llr"`` (L3,
        per-TRP LMMSE + central sum of LLRs).
    estimate_errors : bool
        Equalization channel absorbs the receive errors (True: local channel
        estimate / errored channel) or not (False: clean channel, pessimistic).
    """

    def __init__(self, num_trps=3, num_bs_ant=8, num_ue_ant=4,
                 channel_kind="cdl", cdl_model="A", tdl_model="C",
                 delay_spread=100e-9, speed=0.0, pathloss=False,
                 trp_distances=(100.0, 200.0, 350.0), carrier_frequency=2.6e9,
                 tau_seconds=None, cal_amp_error=None, cal_pha_error=None,
                 granularity="SC", shadow_fading=True, bs_height=25.0,
                 tx_mode="equal", combiner="joint", estimate_errors=True,
                 **phy):
        phy.pop("rank", None)                     # UL model is rank-1
        super().__init__(rank=1, **phy)
        self.num_trps = int(num_trps)
        self.num_bs_ant = int(num_bs_ant)
        self.num_ue_ant = int(num_ue_ant)
        if tau_seconds is None:
            tau_seconds = [0.0] * self.num_trps
        if len(tau_seconds) != self.num_trps:
            raise ValueError(
                f"tau_seconds has {len(tau_seconds)} entries but num_trps="
                f"{self.num_trps} (one timing offset per TRP is required).")
        self.channel = DMIMOChannel(
            kind=channel_kind, num_trps=self.num_trps,
            num_bs_ant=self.num_bs_ant, num_ue_ant=self.num_ue_ant,
            fft_size=self.rg.fft_size,
            subcarrier_spacing=self.rg.subcarrier_spacing,
            carrier_frequency=carrier_frequency, delay_spread=delay_spread,
            cdl_model=cdl_model, tdl_model=tdl_model, speed=speed,
            pathloss=pathloss, trp_distances=trp_distances,
            shadow_fading=shadow_fading, bs_height=bs_height, device=self._device)
        # Receive-side error model: each TRP's receive chain carries its own
        # timing / calibration error (the BS antenna axis plays the "TX" role
        # of dmimo.errors, mirroring the downlink model).
        self.error = TRPErrorModel(self.rg.subcarrier_spacing, self.n_eff,
                                   tau_seconds, num_tx_ant=self.num_bs_ant,
                                   cal_amp_error=cal_amp_error,
                                   cal_pha_error=cal_pha_error,
                                   granularity=granularity)
        self.tx_mode = str(tx_mode).lower()
        self.combiner = str(combiner).lower()
        self.estimate_errors = bool(estimate_errors)

        # Data RE indices for the manual per-TRP combining layers. The pilot
        # pattern mask is already defined on the *effective* subcarriers
        # ([n_sym, n_eff]); data REs = non-pilot effective REs.
        pmask = torch.as_tensor(self.rg.pilot_pattern.mask[0, 0]).bool()  # [n_sym, n_eff]
        self.data_re_idx = (~pmask).reshape(-1).nonzero(
            as_tuple=False).squeeze(-1)
        self.n_data = int(self.data_re_idx.numel())
        assert self.n_data == self.n_data_sym, \
            (f"data RE mismatch: manual {self.n_data} vs "
             f"resource grid {self.n_data_sym}")

    # ------------------------------------------------------------------
    def _tx_vector(self, batch_size, device):
        """UE open-loop transmit vector ``x : [B, num_ue_ant]`` (unit norm)."""
        if self.tx_mode == "random":
            x = torch.randn(batch_size, self.num_ue_ant, device=device) \
                + 1j * torch.randn(batch_size, self.num_ue_ant, device=device)
            return x / (x.abs().square().sum(dim=1, keepdim=True).sqrt() + 1e-12)
        x = torch.ones(batch_size, self.num_ue_ant, dtype=torch.complex64,
                       device=device) / (self.num_ue_ant ** 0.5)
        return x

    def sample_realization(self, batch_size, device=None) -> dict:
        device = device or self._device
        h_ul = self.channel.sample_uplink(batch_size, self.n_symbols,
                                          self.rg.ofdm_symbol_duration, device)
        # h_ul: [B, K, Nbs, 1, D, n_sym, fft]
        h_eff = h_ul[..., self.effective_mask.to(h_ul.device)]   # [B,K,Nbs,1,D,n_sym,n_eff]
        h_err_eff = self._apply_errors(h_eff)                    # receive errors
        x = self._tx_vector(batch_size, device)                  # [B, D]
        # effective per-TRP channel u_t = H_t x  ->  [B,K,Nbs,n_sym,n_eff]
        u = torch.einsum("bkndse,bd->bknse", h_eff.squeeze(3), x)
        ue = torch.einsum("bkndse,bd->bknse", h_err_eff.squeeze(3), x)
        bits = torch.randint(0, 2, (batch_size, self.k),
                             dtype=torch.float32, device=device)
        return {"h_eff": h_eff, "h_err_eff": h_err_eff,
                "u": u, "ue": ue, "x": x, "bits": bits}

    def _apply_errors(self, h_eff: torch.Tensor) -> torch.Tensor:
        """Per-TRP receive timing / calibration errors (effective subcarriers)."""
        syms = []
        for sym in range(self.n_symbols):
            hs = h_eff[:, :, :, 0, :, sym, :]                    # [B,K,Nbs,D,n_eff]
            hs = hs.transpose(2, 3)                              # [B,K,D,Nbs,n_eff]
            he = self.error.apply(hs)                            # errors
            he = he.transpose(2, 3).contiguous()                 # [B,K,Nbs,D,n_eff]
            syms.append(he.unsqueeze(3).unsqueeze(5))            # [B,K,Nbs,1,D,1,n_eff]
        return torch.cat(syms, dim=5)                            # [B,K,Nbs,1,D,n_sym,n_eff]

    def _receive(self, y: torch.Tensor, ref: torch.Tensor, no: float, combiner: str):
        """Combine per-TRP received grids into LLRs ``[B, n]``.

        ``y : [B, K, Nbs, n_sym, fft]`` (AWGN already added), ``ref`` the
        equalization channel on the effective grid ``[B, K, Nbs, n_sym, n_eff]``
        (LS estimate / errored / clean effective channel). Only the non-pilot
        REs are equalised.
        """
        B, K, Nbs, nsym, fft = y.shape
        idx = self.data_re_idx.to(y.device)
        y_eff = y[..., self.effective_mask.to(y.device)]         # [B,K,Nbs,n_sym,n_eff]
        y_d = y_eff.reshape(B, K, Nbs, -1)[..., idx]             # [B,K,Nbs,n_data]
        ref_d = ref.reshape(B, K, Nbs, -1)[..., idx]
        if combiner == "joint":
            y2 = y_d.reshape(B, K * Nbs, -1)
            h2 = ref_d.reshape(B, K * Nbs, -1)
            hh = h2.abs().square().sum(dim=1, keepdim=True)     # [B,1,n_data]
            # keep the einsum result 3D so the division broadcasts over the
            # antenna axis only (a 2D [B,N] / 3D [B,1,N] would give [B,B,N])
            x_hat = (torch.einsum("ban,ban->bn", h2.conj(), y2)
                     .unsqueeze(1) / (hh + no))                 # [B,1,n_data]
            no_eff = no / (hh + no)
        elif combiner == "symbol":
            hh = ref_d.abs().square().sum(dim=2)                 # [B,K,n_data]
            x_t = torch.einsum("...an,...an->...n", ref_d.conj(), y_d) / (hh + 1e-12)
            no_t = no / (hh + 1e-12)
            w = hh / (hh.sum(dim=1, keepdim=True) + 1e-12)
            x_hat = (w * x_t).sum(dim=1)
            no_eff = (w.square() * no_t).sum(dim=1)
        else:  # "llr"
            hh = ref_d.abs().square().sum(dim=2, keepdim=True)     # [B,K,1,n_data]
            x_t = (torch.einsum("...an,...an->...n", ref_d.conj(), y_d)
                   .unsqueeze(2) / (hh + no))                      # [B,K,1,n_data]
            no_t = no / (hh + no)                                  # [B,K,1,n_data]
            llr_t = self.demapper(x_t.unsqueeze(2),
                                  no_t.unsqueeze(2))               # [B,K,1,1,n_data]
            llr = llr_t.sum(dim=1).reshape(B, self.n)
            return llr
        llr = self.demapper(x_hat.unsqueeze(1).unsqueeze(1),
                            no_eff.unsqueeze(1).unsqueeze(1))
        return llr.reshape(B, self.n)

    def _forward_bits(self, real: dict, ebno_db: float, device=None,
                      combiner=None, estimate_errors=None):
        device = device or self._device
        combiner = (combiner or self.combiner).lower()
        if estimate_errors is None:
            estimate_errors = self.estimate_errors
        batch_size = real["bits"].shape[0]
        no = float(ebnodb2no(ebno_db, self.bits_sym, self.code_rate, self.rg))

        u = real["u"].unsqueeze(3).unsqueeze(4)    # [B,K,Nbs,1,1,n_sym,n_eff] clean
        ue = real["ue"].unsqueeze(3).unsqueeze(4)  # errored
        bits = real["bits"]
        cw = self.encoder(bits)
        cw_r = cw.reshape(batch_size, 1, 1, self.n_data_sym, self.bits_sym)
        s = self.mapper(cw_r).squeeze(-1)            # [B,1,1,n_data]
        x_rg = self.rg_mapper(s)                     # [B,1,1,n_sym,fft]

        ue_full = self._embed_full(ue)               # [B,K,Nbs,1,1,n_sym,fft]
        y = self.channel_apply(x_rg, ue_full, no)          # [B,K,Nbs,n_sym,fft]

        if self.perfect_csi:
            h_hat = ue                                # perfect CSI of errored channel
        else:
            h_hat, _err_var = self.ls_est(y, no)      # [B,K,Nbs,1,1,n_sym,n_eff]
        ref = h_hat if estimate_errors else u         # equalization channel (7D grid)
        llr = self._receive(y, ref, no, combiner)
        b_hat, crc_ok = self._decode(llr, bits)
        return bits, b_hat, crc_ok
