"""
sionna5g - A 5G NR link-level simulation platform built on NVIDIA Sionna (PyTorch backend).

Modules
-------
config          : Configuration dataclasses + YAML loader for scenarios.
transmitter     : 5G NR PUSCH transmitter wrapper (NF API).
channel         : AWGN / TDL / CDL physical channel models.
receiver        : 5G NR PUSCH receiver wrapper (configurable LS/perfect estimator + LMMSE/ZF detector + LDPC decoder).
simulator       : High-level LinkSimulator orchestrating TX -> channel -> RX (dependency-injectable).
metrics         : BLER / BER / throughput computation.
link_adaptation : MCS / CQI selection driven by measured BLER vs. SNR.
plotter         : Convenience plotting helpers (BLER/BER/throughput curves).
registry        : Component registry for pluggable channels / estimators / detectors / decoders.
components      : Protocol contracts for the swappable components.
"""

from .config import (
    CarrierConfig,
    PUSCHConfig,
    TBConfig,
    ChannelConfig,
    SimConfig,
    load_config,
    save_config,
)
from .transmitter import build_pusch_config, PUSCHTransmitterWrapper
from .channel import ChannelModelWrapper, build_channel
from .receiver import PUSCHReceiverWrapper
from .simulator import LinkSimulator
from .metrics import LinkMetrics, evaluate_link
from .link_adaptation import MCSConfig, select_mcs_for_snr, LinkAdaptation
from .plotter import plot_link_performance

__all__ = [
    "CarrierConfig",
    "PUSCHConfig",
    "TBConfig",
    "ChannelConfig",
    "SimConfig",
    "load_config",
    "save_config",
    "build_pusch_config",
    "PUSCHTransmitterWrapper",
    "ChannelModelWrapper",
    "build_channel",
    "PUSCHReceiverWrapper",
    "LinkSimulator",
    "LinkMetrics",
    "evaluate_link",
    "MCSConfig",
    "select_mcs_for_snr",
    "LinkAdaptation",
    "plot_link_performance",
]

__version__ = "0.1.0"
