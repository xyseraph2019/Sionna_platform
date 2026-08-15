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


def _panel(num_elements, carrier_frequency):
    from sionna.phy.channel.tr38901 import PanelArray

    return PanelArray(num_rows_per_panel=1, num_cols_per_panel=int(num_elements),
                      polarization="single", polarization_type="V",
                      antenna_pattern="omni", carrier_frequency=carrier_frequency)


def _panel_dual(num_ant_per_pol, carrier_frequency):
    """Dual-polarised ULA: ``num_ant_per_pol`` columns x 2 pol = 2*P antennas.

    Used for the TRP arrays so the Type I codebook's co-phasing between the two
    polarisation groups is well defined (``[v; phi*v]``).
    """
    from sionna.phy.channel.tr38901 import PanelArray

    return PanelArray(num_rows_per_panel=1, num_cols_per_panel=int(num_ant_per_pol),
                      polarization="dual", polarization_type="VH",
                      antenna_pattern="omni", carrier_frequency=carrier_frequency)


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
                self._ofdm.append(OFDMChannel(channel_model=model, resource_grid=rg,
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
        ref = ht.abs().square().mean(dim=(0, 1, 3, 4))   # per-TRP mean power [K]
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
            parts.append(h[:, 0, :, 0, :, 0, :].permute(0, 2, 1, 3).unsqueeze(1))  # [B,1,D,Nt,N]
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

