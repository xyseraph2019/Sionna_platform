"""
Lightweight component registry for the generic 5G link-level platform.

Components (channels, channel estimators, MIMO detectors, transport-block
decoders, transmitters) are looked up by name and built through a callable
``builder``. Built-in components are registered here; advanced users register
their own (e.g. a custom channel estimator) at the *surface level* and then
select it from a YAML config, without touching the orchestration core.

Kinds and builder signatures (see ``components/base.py``):

  channel     : (ChannelConfig, resource_grid, device, num_tx_ant, num_rx_ant)
                -> (ofdm_channel, system_level_model_or_None)
  estimator   : (pusch_transmitter, device) -> estimator | None | "perfect"
  detector    : (pusch_transmitter, device) -> OFDMDetector | None
  decoder     : (pusch_transmitter, device) -> TBDecoder | None
  transmitter : (CarrierConfig, PUSCHConfig, TBConfig, device) -> Transmitter
"""
from __future__ import annotations

_REGS: dict = {"channel": {}, "estimator": {}, "detector": {}, "decoder": {}, "transmitter": {}}


def register(kind: str, name: str, builder) -> None:
    """Register a component ``builder`` under ``kind/name``."""
    if kind not in _REGS:
        raise ValueError(f"Unknown registry kind {kind!r}; known: {sorted(_REGS)}")
    _REGS[kind][name] = builder


def has(kind: str, name: str) -> bool:
    return name in _REGS.get(kind, {})


def build(kind: str, name, *args, **kwargs):
    """Build a registered component by name (throw if unknown)."""
    if kind not in _REGS or name not in _REGS[kind]:
        known = sorted(_REGS.get(kind, {}))
        raise ValueError(f"Unknown {kind} '{name}'. Registered: {known}")
    return _REGS[kind][name](*args, **kwargs)


def names(kind: str) -> list:
    return sorted(_REGS.get(kind, {}))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def panel_array(num_elements: int, carrier_frequency: float, device: str = "cpu"):
    """Single-panel, single-polarisation TR38.901 ``PanelArray``."""
    from sionna.phy.channel.tr38901 import PanelArray

    return PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=int(num_elements),
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
        device=device,
    )


def _scs(cfg):
    return cfg.max_speed if cfg.max_speed is not None else cfg.min_speed


# ---------------------------------------------------------------------------
# Built-in channels: builder(cfg, resource_grid, device, tx_ant, rx_ant)
# -> (ofdm_channel_or_None, system_level_model_or_None)
# ---------------------------------------------------------------------------
def _ch_awgn(cfg, rg, device, tx_ant, rx_ant):
    return None, None


def _ch_tdl(cfg, rg, device, tx_ant, rx_ant):
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.channel import OFDMChannel

    model = TDL(model=cfg.model, delay_spread=cfg.delay_spread,
                carrier_frequency=cfg.carrier_frequency,
                min_speed=cfg.min_speed, max_speed=_scs(cfg),
                num_rx_ant=rx_ant, num_tx_ant=tx_ant, device=device)
    ofdm = OFDMChannel(channel_model=model, resource_grid=rg,
                       normalize_channel=cfg.normalize_channel,
                       return_channel=True, device=device)
    return ofdm, None


def _ch_cdl(cfg, rg, device, tx_ant, rx_ant):
    from sionna.phy.channel.tr38901 import CDL
    from sionna.phy.channel import OFDMChannel

    ut = panel_array(tx_ant, cfg.carrier_frequency, device)
    bs = panel_array(rx_ant, cfg.carrier_frequency, device)
    model = CDL(model=cfg.model, delay_spread=cfg.delay_spread,
                carrier_frequency=cfg.carrier_frequency,
                ut_array=ut, bs_array=bs, direction="uplink",
                min_speed=cfg.min_speed, max_speed=_scs(cfg), device=device)
    ofdm = OFDMChannel(channel_model=model, resource_grid=rg,
                       normalize_channel=cfg.normalize_channel,
                       return_channel=True, device=device)
    return ofdm, None


def _make_uma_umi(cfg, rg, device, tx_ant, rx_ant, cls):
    from sionna.phy.channel import OFDMChannel

    ut = panel_array(tx_ant, cfg.carrier_frequency, device)
    bs = panel_array(rx_ant, cfg.carrier_frequency, device)
    sl = cls(carrier_frequency=cfg.carrier_frequency, o2i_model=cfg.o2i_model,
             ut_array=ut, bs_array=bs, direction="uplink",
             enable_pathloss=False, enable_shadow_fading=False, device=device)
    ofdm = OFDMChannel(channel_model=sl, resource_grid=rg, normalize_channel=True,
                       return_channel=True, device=device)
    return ofdm, sl


def _ch_uma(cfg, rg, device, tx_ant, rx_ant):
    from sionna.phy.channel.tr38901 import UMa

    return _make_uma_umi(cfg, rg, device, tx_ant, rx_ant, UMa)


def _ch_umi(cfg, rg, device, tx_ant, rx_ant):
    from sionna.phy.channel.tr38901 import UMi

    return _make_uma_umi(cfg, rg, device, tx_ant, rx_ant, UMi)


register("channel", "awgn", _ch_awgn)
register("channel", "tdl", _ch_tdl)
register("channel", "cdl", _ch_cdl)
register("channel", "uma", _ch_uma)
register("channel", "umi", _ch_umi)

# ---------------------------------------------------------------------------
# Built-in channel estimators: builder(pusch_transmitter, device) -> estimator
#   None      -> Sionna default PUSCH LS channel estimator (linear interp.)
#   "perfect" -> Sionna perfect-CSI mode (uses h input)
# ---------------------------------------------------------------------------
register("estimator", "ls", lambda tx, device="cpu": None)
register("estimator", "perfect", lambda tx, device="cpu": "perfect")

# ---------------------------------------------------------------------------
# Built-in MIMO detectors: builder(pusch_transmitter, device) -> detector
#   None      -> Sionna default LinearDetector (LMMSE)
# ---------------------------------------------------------------------------
register("detector", "lmmse", lambda tx, device="cpu": None)


def _det_zf(tx, device="cpu"):
    from sionna.phy.ofdm.detection import LinearDetector
    from sionna.phy.mimo import StreamManagement
    import numpy as np

    sm = StreamManagement(np.ones([1, tx._num_tx], bool), tx._num_layers)
    return LinearDetector("zf", "bit", "maxlog", tx.resource_grid, sm, "qam",
                          tx._num_bits_per_symbol,
                          precision=None, device=device)


register("detector", "zf", _det_zf)

# ---------------------------------------------------------------------------
# Built-in TB decoders: builder(pusch_transmitter, device) -> decoder (None=default)
# ---------------------------------------------------------------------------
register("decoder", "default", lambda tx, device="cpu": None)