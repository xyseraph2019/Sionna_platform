"""
Downlink per-TRP channel generators for the DMIMO scenario.

Channel sources:

  * ``"simple"`` — i.i.d. circular-complex Rayleigh ``CN(0,1)`` (fast, for the
    mechanism demo and quick NN iteration);
  * ``"uma"`` / ``"umi"`` — Sionna 3GPP system-level model with **multiple BSs
    (TRPs) + one UE in downlink**. Includes per-TRP **pathloss / shadow fading**
    (coverage), antenna-array spatial correlation and multipath frequency
    selectivity. Pathloss can be switched off (then each TRP's channel is
    normalised to unit energy);
  * ``"cdl"`` — one Sionna ``CDL`` per TRP (clustered delay line, spatial
    correlation via ``PanelArray``); per-TRP pathloss added externally;
  * ``"tdl"`` — one Sionna ``TDL`` per TRP (frequency selectivity only).

All generators produce ``h : [B, K, N_t, N]`` (batch, TRP, antennas, subcarrier)
and are differentiable / batched.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _pathloss_gain(distances, carrier_frequency=3.5e9):
    """Per-TRP large-scale gain ``beta = 10^(-PL/10)`` from distance.

    Uses a standard urban-macro pathloss ``PL = 128.1 + 37.6 log10(d_km)`` dB
    (COST-231 Hata-like). Only used for CDL / TDL where Sionna does not model
    pathloss itself.
    """
    d_km = torch.as_tensor(list(distances), dtype=torch.float32) / 1000.0
    pl_db = 128.1 + 37.6 * torch.log10(d_km.clamp(min=1e-3))
    return 10.0 ** (-pl_db / 10.0)  # linear power gain


def _resource_grid(n_subcarriers, subcarrier_spacing):
    from sionna.phy.ofdm import ResourceGrid

    return ResourceGrid(subcarrier_spacing=subcarrier_spacing, num_ofdm_symbols=1,
                        fft_size=int(n_subcarriers), cyclic_prefix_length=0,
                        num_guard_carriers=[0, 0])


def _panel(num_elements, carrier_frequency, device=None):
    from sionna.phy.channel.tr38901 import PanelArray

    return PanelArray(num_rows_per_panel=1, num_cols_per_panel=int(num_elements),
                      polarization="single", polarization_type="V",
                      antenna_pattern="omni", carrier_frequency=carrier_frequency,
                      device=device)


def _panel_dual(num_ant_per_pol, carrier_frequency, device=None):
    """Dual-polarised ULA: ``num_ant_per_pol`` columns x 2 pol = 2*P antennas.

    Used for the TRP arrays so the Type I codebook's co-phasing between the two
    polarisation groups is well defined (``[v; phi*v]``).
    """
    from sionna.phy.channel.tr38901 import PanelArray

    return PanelArray(num_rows_per_panel=1, num_cols_per_panel=int(num_ant_per_pol),
                      polarization="dual", polarization_type="VH",
                      antenna_pattern="omni", carrier_frequency=carrier_frequency,
                      device=device)


class DownlinkChannel(nn.Module):
    """``"simple"`` i.i.d. Rayleigh per TRP (fast baseline)."""

    def __init__(self, num_trps=3, num_tx_ant=4, num_ue_ant=1, n_subcarriers=64, beta=None):
        super().__init__()
        self.k = int(num_trps)
        self.nt = int(num_tx_ant)
        self.d = int(num_ue_ant)
        self.n = int(n_subcarriers)
        if beta is None:
            beta = [1.0] * self.k
        self.register_buffer("beta", torch.as_tensor(list(beta), dtype=torch.float32))

    def sample(self, batch_size, device=None):
        # h : [B, K, D, Nt, N]
        real = torch.randn(batch_size, self.k, self.d, self.nt, self.n, device=device)
        imag = torch.randn(batch_size, self.k, self.d, self.nt, self.n, device=device)
        h = (real + 1j * imag) / (2.0 ** 0.5)
        return h * self.beta.to(device).view(1, self.k, 1, 1, 1)


class SionnaDownlinkChannel(nn.Module):
    """Per-TRP downlink channels from Sionna UMa/UMi/CDL/TDL.

    Parameters
    ----------
    kind : str
        ``"uma" | "umi" | "cdl" | "tdl"``.
    num_trps, num_tx_ant, n_subcarriers : int
        Geometry.
    subcarrier_spacing, carrier_frequency : float
        OFDM / carrier parameters (Hz).
    pathloss : bool
        Enable per-TRP pathloss (``True``) or normalise each TRP to unit energy
        (``False``).
    trp_distances : sequence[float]
        Per-TRP distance in metres (used by UMa/UMi topology or the external
        CDL/TDL pathloss model).
    """

    def __init__(self, kind="uma", num_trps=3, num_tx_ant=4, num_ue_ant=1,
                 n_subcarriers=64, subcarrier_spacing=30e3, carrier_frequency=3.5e9,
                 pathloss=True, trp_distances=(100.0, 200.0, 350.0), shadow_fading=True,
                 delay_spread=100e-9, cdl_model="A", tdl_model="C",
                 bs_height=25.0):
        super().__init__()
        kind = kind.lower()
        assert kind in ("uma", "umi", "cdl", "tdl"), f"unknown kind {kind}"
        self.kind = kind
        self.k = int(num_trps)
        self.nt = int(num_tx_ant)
        self.d = int(num_ue_ant)
        self.n = int(n_subcarriers)
        self.pathloss = bool(pathloss)
        self.subcarrier_spacing = subcarrier_spacing
        self.carrier_frequency = carrier_frequency
        self.trp_distances = [float(d) for d in trp_distances]
        self._bs_height = float(bs_height)
        self.shadow_fading = bool(shadow_fading)
        self._rg = _resource_grid(self.n, subcarrier_spacing)
        self._last_B = None   # last topology batch size (UMa/UMi rebuild on change)

        from sionna.phy.channel import OFDMChannel

        if kind in ("uma", "umi"):
            self._rebuild_sionna_model(self._rg)
            self._models = None
        else:
            from sionna.phy.channel.tr38901 import CDL, TDL

            self._ofdm = []
            self._models = []
            for _t in range(self.k):
                if kind == "cdl":
                    bs = _panel_dual(self.nt // 2, carrier_frequency)  # dual-pol TRP
                    ut = _panel(self.d, carrier_frequency)             # UE array
                    model = CDL(model=cdl_model, delay_spread=delay_spread,
                                carrier_frequency=carrier_frequency,
                                ut_array=ut, bs_array=bs, direction="downlink")
                else:
                    model = TDL(model=tdl_model, delay_spread=delay_spread,
                                carrier_frequency=carrier_frequency,
                                min_speed=0.0, max_speed=0.0,
                                num_rx_ant=self.d, num_tx_ant=self.nt)
                self._models.append(model)
                self._ofdm.append(OFDMChannel(channel_model=model, resource_grid=self._rg,
                                              normalize_channel=(not pathloss),
                                              return_channel=True))
            # external pathloss for CDL/TDL (Sionna does not include it)
            self.register_buffer("beta", _pathloss_gain(self.trp_distances, carrier_frequency))
            self._model = None

    def _rebuild_sionna_model(self, rg):
        """(Re)build the UMa/UMi model + OFDM channel (batch-size-safe).

        Sionna's ``set_topology`` refuses to change the batch size without
        ``reset_topology()``, and ``torch._dynamo.reset()`` (called by the Sionna
        reset) crashes on Windows/GBK locales, so we rebuild the model when the
        batch size changes.
        """
        from sionna.phy.channel import OFDMChannel
        from sionna.phy.channel.tr38901 import UMa, UMi

        cls = UMa if self.kind == "uma" else UMi
        bs = _panel_dual(self.nt // 2, self.carrier_frequency)  # dual-pol TRP: 2P antennas
        ut = _panel(self.d, self.carrier_frequency)             # UE array: D antennas
        self._model = cls(carrier_frequency=self.carrier_frequency, o2i_model="low",
                          ut_array=ut, bs_array=bs, direction="downlink",
                          enable_pathloss=self.pathloss,
                          enable_shadow_fading=self.shadow_fading)
        self._ofdm = OFDMChannel(channel_model=self._model, resource_grid=self._rg,
                                 normalize_channel=(not self.pathloss),
                                 return_channel=True)

    def _set_topology(self, batch_size):
        B = int(batch_size)
        if self._last_B is not None and self._last_B != B:
            # rebuild so Sionna accepts the new batch size (see _rebuild_sionna_model)
            self._rebuild_sionna_model(self._rg)
        self._last_B = B
        model = self._model
        K = self.k
        dist = torch.as_tensor(self.trp_distances, dtype=torch.float32)
        bs_loc = torch.stack([dist, torch.zeros_like(dist),
                              torch.full_like(dist, self._bs_height)], dim=1)  # [K,3]
        bs_loc = bs_loc.unsqueeze(0).expand(B, K, 3).contiguous()
        ut_loc = torch.zeros(B, 1, 3, dtype=torch.float32)
        ut_loc[:, 0, 2] = 1.5
        bo = torch.zeros(B, K, 3)
        bo[:, :, 0] = 3.14159265358979  # arrays point toward +x (UE at origin)
        uo = torch.zeros(B, 1, 3)
        vel = torch.zeros(B, 1, 3)
        in_state = torch.zeros(B, 1, dtype=torch.bool)
        model.set_topology(ut_loc, bs_loc, uo, bo, vel, in_state, los=False)

    def _normalize_coverage(self, ht):
        """Scale so the strongest TRP has unit mean power (keeps coverage ratios).

        With pathloss ON the absolute channel magnitudes are tiny; normalising by
        the strongest TRP lets the SNR refer to that TRP while preserving the
        relative per-TRP coverage gradient. Stores ``self._ref_power`` (per-TRP
        relative power, strongest = 1).
        """
        ref = ht.abs().square().mean(dim=(0, 2, 3, 4))   # per-TRP mean power [K]
        ref = ref.detach()
        self._ref_power = (ref / (ref.max() + 1e-12)).cpu()
        return ht / (ref.max().sqrt() + 1e-12)

    def sample(self, batch_size, device=None):
        B = int(batch_size)
        xf = torch.ones(B, 1, 1, 1, 1, self.n, dtype=torch.complex64, device=device)
        if self.kind in ("uma", "umi"):
            self._set_topology(B)
            _, h = self._ofdm(xf, torch.tensor(1.0))
            # h: [B, num_rx=1, num_rx_ant=D, num_tx=K, num_tx_ant, nsym=1, nsc]
            ht = h[:, 0, :, :, :, 0, :]                  # [B, D, K, Nt, N]
            ht = ht.permute(0, 2, 1, 3, 4).contiguous()  # [B, K, D, Nt, N]
            if self.pathloss:
                ht = self._normalize_coverage(ht)
            return ht
        # CDL / TDL: one model per TRP
        parts = []
        for t, ofdm in enumerate(self._ofdm):
            _, h = ofdm(xf, torch.tensor(1.0))
            # h: [B, num_rx=1, num_rx_ant=D, num_tx=1, num_tx_ant, nsym=1, nsc]
            # h[:,0,:,0,:,0,:] is [B, D, Nt, N]; add the TRP axis -> [B,1,D,Nt,N]
            parts.append(h[:, 0, :, 0, :, 0, :].unsqueeze(1))
        ht = torch.cat(parts, dim=1)                     # [B, K, D, Nt, N]
        if self.pathloss:
            ht = ht * self.beta.to(ht.device).view(1, self.k, 1, 1, 1).sqrt()
            ht = self._normalize_coverage(ht)
        return ht.contiguous()


def build_downlink_channel(kind="uma", **kw):
    """Factory: ``"simple"`` or a Sionna ``kind``."""
    if kind == "simple":
        return DownlinkChannel(num_trps=kw.get("num_trps", 3),
                               num_tx_ant=kw.get("num_tx_ant", 4),
                               num_ue_ant=kw.get("num_ue_ant", 1),
                               n_subcarriers=kw.get("n_subcarriers", 64),
                               beta=kw.get("beta"))
    # `beta` only applies to the "simple" channel; drop it for Sionna builders.
    kw.pop("beta", None)
    return SionnaDownlinkChannel(kind=kind, **kw)


# ---------------------------------------------------------------------------
# Link-level grid channel interface (Sionna channel-tensor convention)
# ---------------------------------------------------------------------------

class DMIMOChannel(nn.Module):
    """Frequency-domain grid channel in Sionna's channel-tensor convention.

    Produces the *downlink* channel
    ``h : [B, num_rx=1, num_ue_ant, num_tx=K, num_bs_ant, n_sym, fft]``
    (1 UE, K TRPs). The uplink channel is obtained by reciprocity
    (``h.permute(0, 3, 4, 1, 2, 5, 6)`` → ``[B, K, num_bs_ant, 1, num_ue_ant,
    n_sym, fft]``).

    Channel sources (``kind``):

      * ``"simple"`` — i.i.d. Rayleigh, frequency-flat, static across symbols
        (fast verification baseline);
      * ``"cdl"`` — one Sionna ``CDL`` per TRP; the CIR is sampled once per
        OFDM symbol (symbol-rate sampling) and converted with
        ``cir_to_ofdm_channel`` — the Sionna CDL tutorial pipeline. A nonzero
        ``speed`` makes the channel vary across symbols (channel aging);
      * ``"tdl"`` — one Sionna ``TDL`` per TRP (same CIR pipeline);
      * ``"uma"`` / ``"umi"`` — Sionna system-level model with K TRPs + 1 UE
        (static across symbols; topology rebuilt when the batch size changes).

    Per-TRP pathloss (``pathloss=True``) is applied for ``cdl``/``tdl`` via an
    external large-scale model, and the channel is normalised so the strongest
    TRP has unit mean power (coverage ratios preserved).
    """

    def __init__(self, kind="cdl", num_trps=3, num_bs_ant=8, num_ue_ant=4,
                 fft_size=76, subcarrier_spacing=30e3, carrier_frequency=3.5e9,
                 delay_spread=100e-9, cdl_model="A", tdl_model="C", speed=0.0,
                 pathloss=False, trp_distances=(100.0, 200.0, 350.0),
                 shadow_fading=True, bs_height=25.0, beta=None, device=None):
        super().__init__()
        kind = kind.lower()
        assert kind in ("simple", "cdl", "tdl", "uma", "umi"), f"unknown kind {kind}"
        self.kind = kind
        self.k = int(num_trps)
        self.nt = int(num_bs_ant)
        self.d = int(num_ue_ant)
        self.n = int(fft_size)
        self.speed = float(speed)
        self.pathloss = bool(pathloss)
        self.subcarrier_spacing = float(subcarrier_spacing)
        self.carrier_frequency = float(carrier_frequency)
        self.delay_spread = float(delay_spread)
        self.trp_distances = [float(x) for x in trp_distances]
        self.shadow_fading = bool(shadow_fading)
        self._bs_height = float(bs_height)
        self.device = device

        if kind == "simple":
            if beta is None:
                beta = [1.0] * self.k
            self.register_buffer("beta", torch.as_tensor(list(beta), dtype=torch.float32))
            self._models = None
        elif kind in ("uma", "umi"):
            self._models = SionnaDownlinkChannel(
                kind=kind, num_trps=self.k, num_tx_ant=self.nt, num_ue_ant=self.d,
                n_subcarriers=self.n, subcarrier_spacing=self.subcarrier_spacing,
                carrier_frequency=self.carrier_frequency, pathloss=self.pathloss,
                trp_distances=self.trp_distances, shadow_fading=self.shadow_fading,
                delay_spread=self.delay_spread, cdl_model=cdl_model,
                tdl_model=tdl_model, bs_height=self._bs_height)
        else:
            from sionna.phy.channel.tr38901 import CDL, TDL

            self._models = []
            for _t in range(self.k):
                if kind == "cdl":
                    if self.nt % 2:
                        raise ValueError("CDL requires an even number of BS "
                                         "antennas (dual-pol array).")
                    bs = _panel_dual(self.nt // 2, self.carrier_frequency, device)
                    ut = _panel(self.d, self.carrier_frequency, device)
                    model = CDL(model=cdl_model, delay_spread=self.delay_spread,
                                carrier_frequency=self.carrier_frequency,
                                ut_array=ut, bs_array=bs, direction="downlink",
                                min_speed=self.speed, max_speed=self.speed,
                                device=device)
                else:
                    model = TDL(model=tdl_model, delay_spread=self.delay_spread,
                                carrier_frequency=self.carrier_frequency,
                                min_speed=self.speed, max_speed=self.speed,
                                num_rx_ant=self.d, num_tx_ant=self.nt,
                                device=device)
                self._models.append(model)
            if self.pathloss:
                self.register_buffer(
                    "beta", _pathloss_gain(self.trp_distances, self.carrier_frequency))

    @property
    def frequencies(self) -> torch.Tensor:
        """Subcarrier frequencies of the full FFT grid (incl. guard / DC carriers)."""
        from sionna.phy.channel import subcarrier_frequencies

        return subcarrier_frequencies(self.n, self.subcarrier_spacing, device=self.device)

    def sample(self, batch_size, num_ofdm_symbols, ofdm_symbol_duration, device=None):
        """Downlink grid channel ``h : [B, 1, D, K, Nt, n_sym, fft]``.

        Parameters
        ----------
        batch_size : int
        num_ofdm_symbols : int
            Number of OFDM symbols (time samples taken once per symbol).
        ofdm_symbol_duration : float
            Seconds per OFDM symbol incl. cyclic prefix (defines the CIR
            sampling frequency ``1/ofdm_symbol_duration``).
        device : str | None
        """
        B = int(batch_size)
        n_sym = int(num_ofdm_symbols)
        device = device or self.device
        if self.kind == "simple":
            real = torch.randn(B, 1, self.d, self.k, self.nt, 1, self.n, device=device)
            imag = torch.randn(B, 1, self.d, self.k, self.nt, 1, self.n, device=device)
            h = (real + 1j * imag) / (2.0 ** 0.5)
            h = h * self.beta.to(device).view(1, 1, 1, self.k, 1, 1, 1)
            return h.expand(B, 1, self.d, self.k, self.nt, n_sym, self.n).contiguous()
        if self.kind in ("uma", "umi"):
            h = self._models.sample(B, device)          # [B, K, D, Nt, N] static
            h = h.permute(0, 2, 1, 3, 4).unsqueeze(5)   # [B, D, K, Nt, 1, N]
            h = h.unsqueeze(1)                           # [B, 1, D, K, Nt, 1, N]
            return h.expand(B, 1, self.d, self.k, self.nt, n_sym, self.n).contiguous()
        # CDL / TDL: CIR sampled once per OFDM symbol -> channel frequency response.
        from sionna.phy.channel import cir_to_ofdm_channel

        f = self.frequencies.to(device)
        parts = []
        for t, model in enumerate(self._models):
            a, tau = model(B, n_sym, 1.0 / float(ofdm_symbol_duration))
            # a: [B, 1, D, 1, Nt, P, n_sym]; tau: [B, 1, 1, P]
            ht = cir_to_ofdm_channel(f, a, tau, normalize=True)   # [B,1,D,1,Nt,n_sym,N]
            parts.append(ht)
        h = torch.cat(parts, dim=3)                      # [B,1,D,K,Nt,n_sym,N]
        if self.pathloss:
            b = self.beta.to(h.device).view(1, 1, 1, self.k, 1, 1, 1)
            h = h * b.sqrt()
            ref = h.abs().square().mean(dim=(0, 2, 4, 5, 6))   # per-TRP mean power
            h = h / (ref.max().sqrt() + 1e-12)                 # strongest TRP -> unit
        return h.contiguous()

    def sample_uplink(self, batch_size, num_ofdm_symbols, ofdm_symbol_duration,
                      device=None) -> torch.Tensor:
        """Uplink grid channel ``h_ul : [B, K, Nt, 1, D, n_sym, fft]`` (reciprocity)."""
        h = self.sample(batch_size, num_ofdm_symbols, ofdm_symbol_duration, device)
        return h.permute(0, 3, 4, 1, 2, 5, 6).contiguous()

