"""
Configuration dataclasses and YAML loader for the 5G link-level simulation platform.

These classes map 1-to-1 onto the Sionna ``sionna.phy.nr`` building blocks
(``CarrierConfig``, ``PUSCHConfig``, ``TBConfig``) while exposing a clean,
serialisable representation that can be defined in a YAML file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union

import yaml


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve a device specifier to a concrete device string.

    ``"auto"`` (or ``None``/empty) selects ``cuda:0`` when a CUDA GPU is
    available, otherwise ``cpu``. A literal ``"cpu"`` / ``"cuda:0"`` / ...
    is returned unchanged.
    """
    if device is None or str(device).strip().lower() in ("", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"
    return str(device)


@dataclass
class CarrierConfig:
    """OFDM / NR carrier configuration (maps to ``sionna.phy.nr.CarrierConfig``).

    Attributes
    ----------
    subcarrier_spacing : float
        Subcarrier spacing in kHz (15, 30, 60, 120). Drives slot duration.
    n_size_grid : int
        Number of resource blocks in the carrier grid.
    n_cell_id : int
        Physical cell ID.
    slot_number, frame_number : int
        Slot / frame indices.
    """

    subcarrier_spacing: float = 15.0  # kHz
    n_size_grid: int = 4
    n_cell_id: int = 1
    slot_number: int = 0
    frame_number: int = 0

    @property
    def slot_duration(self) -> float:
        """Slot duration in seconds for the configured subcarrier spacing."""
        # NR slot duration = 1 ms at 15 kHz, halved each time SCS doubles.
        return 0.001 * (15000.0 / (self.subcarrier_spacing * 1000.0))


@dataclass
class TBConfig:
    """Transport-block configuration (maps to ``sionna.phy.nr.TBConfig``)."""

    mcs_index: int = 14
    mcs_table: int = 1
    channel_type: str = "PUSCH"
    n_id: Optional[int] = None


@dataclass
class PUSCHConfig:
    """PUSCH (uplink physical layer) configuration.

    Attributes
    ----------
    num_layers : int
        Number of MIMO spatial layers / streams.
    num_antenna_ports : int
        Number of transmit antenna ports at the UE.
    mapping_type : str
        DMRS mapping type, "A" or "B".
    symbol_start, symbol_length : int
        Time-domain PUSCH allocation within a slot, e.g. (0, 14).
    precoding : str
        "non-codebook" or "codebook".
    transform_precoding : bool
        Whether DFT-spread OFDM (single carrier) is used.

    Notes
    -----
    ``num_layers`` is the number of MIMO spatial streams.
    ``num_antenna_ports`` is the number of UE transmit antenna ports.
    The platform currently models a single-UE physical link, so there is no
    ``num_ut`` parameter.
    """

    num_layers: int = 1
    num_antenna_ports: int = 1
    mapping_type: str = "A"
    symbol_start: int = 0
    symbol_length: int = 14
    precoding: str = "non-codebook"
    transform_precoding: bool = False


@dataclass
class ChannelConfig:
    """Physical propagation channel configuration.

    Attributes
    ----------
    channel_type : str
        One of ``"awgn" | "tdl" | "cdl" | "uma" | "umi"`` or a custom registered
        channel name.
    model : str
        TDL/CDL model id, e.g. "A", "B", "C", "D", "E" or "CDL-A"...
    delay_spread : float
        RMS delay spread in seconds.
    carrier_frequency : float
        Carrier frequency in Hz.
    min_speed, max_speed : float
        UE velocity range in m/s (Doppler / channel aging).
    o2i_model : str
        Outdoor-to-indoor pathloss model for uma/umi ("low"|"high").
    ut_distance, ut_height : float
        UE position used by UMa/UMi topology.
    normalize_channel : bool
        Whether to normalise the frequency-domain channel to unit energy.
        For TDL/CDL this is optional (default False).
        For UMa/UMi the platform currently always normalises, so this field is
        ignored for those channel types.
    num_rx_ant : int | None
        Number of BS receive antennas; ``None`` means use the PUSCH antenna-port
        count.
    """

    channel_type: str = "awgn"  # awgn | tdl | cdl | uma | umi | <custom>
    model: str = "C"
    delay_spread: float = 100e-9
    carrier_frequency: float = 3.5e9
    min_speed: float = 0.0
    max_speed: Optional[float] = None
    o2i_model: str = "low"  # outdoor-to-indoor pathloss model for uma/umi ("low"|"high")
    ut_distance: float = 100.0  # horizontal BS-UT distance [m] for uma/umi topology
    ut_height: float = 1.5  # UT height [m] for uma/umi topology
    normalize_channel: bool = False
    num_rx_ant: Optional[int] = None  # BS receive antennas; None -> use PUSCH antenna ports


@dataclass
class ReceiverConfig:
    """Receiver algorithm selection (registry names, see ``registry.py``).

    Attributes
    ----------
    channel_estimator : str
        ``"ls"`` (default), ``"perfect"`` (needs a fading channel that provides
        ``h``), or any registered custom estimator name.
    mimo_detector : str
        ``"lmmse"`` (default), ``"zf"``, or any registered detector name.
    tb_decoder : str
        ``"default"`` or any registered decoder name.
    """

    channel_estimator: str = "ls"
    mimo_detector: str = "lmmse"
    tb_decoder: str = "default"


@dataclass
class SimConfig:
    """Top-level simulation control.

    Attributes
    ----------
    carrier, pusch, tb, channel, receiver : section configs
    snr_db : list[float]
        SNR points in dB to sweep (noise variance derived from these).
    num_trials : int
        Number of transport blocks per SNR point (Monte Carlo trials).
    batch_size : int
        Inner vectorisation batch (keep <= num_trials).
    device : str
        "cpu" or "cuda:0".
    return_crc_status : bool
        Whether the receiver returns the TB CRC status.
    seed : int
        Global random seed for reproducibility.
    """

    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    pusch: PUSCHConfig = field(default_factory=PUSCHConfig)
    tb: TBConfig = field(default_factory=TBConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)

    snr_db: Optional[List[float]] = None       # explicit SNR list (dB); None -> snr_start/stop/step
    snr_start_db: float = 0.0                  # SNR sweep range (dB)
    snr_stop_db: float = 10.0
    snr_step_db: float = 5.0
    num_trials: int = 200
    batch_size: int = 32
    device: str = "auto"  # "auto" selects cuda:0 if available, else cpu
    return_crc_status: bool = True
    seed: int = 7

    @property
    def snr_grid(self) -> List[float]:
        """SNR sweep points in dB.

        Uses the explicit ``snr_db`` list when given; otherwise builds the
        arithmetic range ``snr_start_db, +snr_step_db, ..., <= snr_stop_db``.
        """
        if self.snr_db is not None:
            return list(self.snr_db)
        if self.snr_step_db <= 0:
            raise ValueError("snr_step_db must be > 0.")
        n = max(int(math.floor((self.snr_stop_db - self.snr_start_db) / self.snr_step_db + 1e-9)) + 1, 1)
        return [round(self.snr_start_db + i * self.snr_step_db, 6) for i in range(n)]


# --------------------------------------------------------------------------
# YAML (de)serialisation
# --------------------------------------------------------------------------
def _coerce(v, target):
    """Coerce a YAML value to an annotated numeric type if possible.

    Some YAML parsers load scientific-notation floats without a decimal
    point (e.g. ``3.5e9``, ``100e-9``) as strings; this normalises them.

    ``target`` may be the actual type or (due to postponed annotations) its
    string name, e.g. ``float`` or ``'float'``.
    """
    is_float = target in (float, "float") or target == float
    is_int = target in (int, "int") or target == int
    if is_float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    if is_int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return v
    return v


def _from_dict(cls, data):
    if not isinstance(data, dict):
        return cls()
    kwargs = {}
    for name, f in cls.__dataclass_fields__.items():
        if name in data and data[name] is not None:
            kwargs[name] = _coerce(data[name], f.type)
    return cls(**kwargs)


def load_config(path: str) -> SimConfig:
    """Load a :class:`SimConfig` from a YAML file.

    Nested section names follow the dataclass attribute names
    (``carrier``, ``pusch``, ``tb``, ``channel``).
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = SimConfig()
    # Plain top-level list may represent snr_db etc.
    cfg.snr_db = raw.get("snr_db", cfg.snr_db)
    cfg.snr_start_db = raw.get("snr_start_db", cfg.snr_start_db)
    cfg.snr_stop_db = raw.get("snr_stop_db", cfg.snr_stop_db)
    cfg.snr_step_db = raw.get("snr_step_db", cfg.snr_step_db)
    cfg.num_trials = raw.get("num_trials", cfg.num_trials)
    cfg.batch_size = raw.get("batch_size", cfg.batch_size)
    cfg.device = raw.get("device", cfg.device)
    cfg.return_crc_status = raw.get("return_crc_status", cfg.return_crc_status)
    cfg.seed = raw.get("seed", cfg.seed)

    cfg.carrier = _from_dict(CarrierConfig, raw.get("carrier", {}))
    cfg.pusch = _from_dict(PUSCHConfig, raw.get("pusch", {}))
    cfg.tb = _from_dict(TBConfig, raw.get("tb", {}))
    cfg.channel = _from_dict(ChannelConfig, raw.get("channel", {}))
    cfg.receiver = _from_dict(ReceiverConfig, raw.get("receiver", {}))
    return cfg


def save_config(cfg: SimConfig, path: str) -> None:
    """Serialise a :class:`SimConfig` to a YAML file."""
    section = {
        "snr_db": list(cfg.snr_grid),
        "snr_start_db": cfg.snr_start_db,
        "snr_stop_db": cfg.snr_stop_db,
        "snr_step_db": cfg.snr_step_db,
        "num_trials": cfg.num_trials,
        "batch_size": cfg.batch_size,
        "device": cfg.device,
        "return_crc_status": cfg.return_crc_status,
        "seed": cfg.seed,
        "carrier": asdict(cfg.carrier),
        "pusch": asdict(cfg.pusch),
        "tb": asdict(cfg.tb),
        "channel": asdict(cfg.channel),
        "receiver": asdict(cfg.receiver),
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(section, fh, sort_keys=False, allow_unicode=True)

