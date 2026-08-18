"""YAML config support for the DMIMO platforms (link-level and system-level)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml


@dataclass
class DMIMOConfig:
    channel_kind: str = "uma"                 # simple|tdl|cdl|uma|umi
    cdl_model: str = "A"                  # CDL 模型：A/B/C/D/E
    tdl_model: str = "C"                  # TDL 模型：A/B/C/D/E
    num_trps: int = 3
    num_tx_ant: int = 4
    num_ue_ant: int = 1
    n_subcarriers: int = 64
    subcarrier_spacing_khz: float = 30.0
    carrier_frequency: float = 3.5e9
    pathloss: bool = False
    trp_distances_m: Tuple[float, ...] = (100.0, 200.0, 350.0)
    tau_ns: Tuple[float, ...] = (0.0, 130.0, 260.0)
    cal_amp_error: float = 0.1                # None -> no calibration error
    cal_pha_error: float = 0.1
    granularity: str = "SC"                   # SC|RB|SC_RB_granular
    subband_size: int = 12                    # CJT 与 NN-PMI 统一预编码粒度（子载波/子带）
    # ---- P3: CSI 反馈量化（有限比特反馈，none=理想 CSI 上界）----
    feedback_quant: str = "none"              # none | phase | iq
    feedback_bits_phase: int = 4              # phase 量化：每复系数相位比特（2/3/4/6/8）
    feedback_bits_amp: Optional[int] = None   # phase 量化：振幅比特（None=保留连续振幅）
    feedback_bits_iq: int = 4                 # iq 量化：每实/虚部比特
    feedback_ste: bool = False                # 直通估计器（NN-PMI 抗量化训练用）
    feedback_subband_size: Optional[int] = None  # 反馈子带大小；None -> 复用 subband_size
    rank: int = 1
    qam_order: int = 16
    code_rate: float = 0.5
    use_channel_estimation: bool = False      # P1: DMRS-based LS channel estimation
    est_density: float = 0.5                  # P1: DMRS comb density in the DMRS symbol (pilots/subcarrier)
    pilot_boost_db: float = 0.0               # P1: DMRS pilot energy boost (dB) vs a data symbol
    n_symbols: int = 14                       # P1: OFDM symbols per slot (time-frequency grid)
    dmrs_symbol: int = 2                      # P1: front-loaded pilot-only DMRS symbol index
    num_dmrs_symbols: int = 1                 # P1: number of pilot-only DMRS symbols
    use_crc: bool = True                      # P2: 5G NR TB encoder/decoder + TB CRC BLER
    precoder: str = "all"                     # mrt|cjt|type1|all|nn
    nn_pmi_ckpt: Optional[str] = None         # path to trained NN-PMI model; None -> NN-PMI disabled
    snr_db: Optional[List[float]] = None      # explicit SNR list (dB); None -> snr_start/stop/step
    snr_start_db: float = -24.0               # SNR sweep range (dB)
    snr_stop_db: float = -2.0
    snr_step_db: float = 2.0
    num_trials: int = 256
    num_mc_batches: int = 1       # 每个 SNR 下跑的蒙特卡洛批次数（每批 num_trials 个 TB）
    device: str = "auto"
    seed: int = 0

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

    @property
    def tau_seconds(self) -> Tuple[float, ...]:
        return tuple(t * 1e-9 for t in self.tau_ns)

    def build_link(self):
        """Build a :class:`~dmimo.link.DMIMODownlink` from this config."""
        from .link import build_link

        return build_link(num_trps=self.num_trps, num_tx_ant=self.num_tx_ant,
                          num_ue_ant=self.num_ue_ant, n_subcarriers=self.n_subcarriers,
                          subcarrier_spacing=self.subcarrier_spacing_khz * 1e3,
                          tau_seconds=self.tau_seconds,
                          cal_amp_error=self.cal_amp_error, cal_pha_error=self.cal_pha_error,
                          granularity=self.granularity, channel_kind=self.channel_kind,
                          pathloss=self.pathloss, trp_distances=self.trp_distances_m,
                          carrier_frequency=self.carrier_frequency,
                           cdl_model=self.cdl_model, tdl_model=self.tdl_model)


def _coerce(name, v):
    if name in ("trp_distances_m", "tau_ns"):
        try:
            return tuple(float(x) for x in v)
        except (TypeError, ValueError):
            return v
    if name == "snr_db":
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return v
    if name in ("num_trps", "num_tx_ant", "num_ue_ant", "n_subcarriers", "qam_order", "rank",
                "n_symbols", "dmrs_symbol", "num_dmrs_symbols", "num_trials",
                "num_mc_batches", "seed", "subband_size",
                "feedback_bits_phase", "feedback_bits_iq", "feedback_subband_size"):
        try:
            return int(v)
        except (TypeError, ValueError):
            return v
    if name in ("subcarrier_spacing_khz", "carrier_frequency", "cal_amp_error", "cal_pha_error",
                "code_rate", "est_density", "pilot_boost_db",
                "snr_start_db", "snr_stop_db", "snr_step_db"):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    if name in ("use_channel_estimation", "use_crc", "feedback_ste"):
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)
    return v


def load_dmimo_config(path: str) -> DMIMOConfig:
    """Load a :class:`DMIMOConfig` from a YAML file (numeric fields coerced)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = DMIMOConfig()
    for name, f in DMIMOConfig.__dataclass_fields__.items():
        if name in raw and raw[name] is not None:
            setattr(cfg, name, _coerce(name, raw[name]))
    return cfg


def scenario_tag(num_trps, rank, n_subcarriers, qam_order, code_rate,
                 channel_kind="uma", est=False, num_dmrs_symbols=1,
                 err=False, subband_size=None, pathloss=False,
                 feedback: Optional[str] = None) -> str:
    """Compact scenario tag for result-file names (filename == metadata).

    Example (3 TRPs, rank 2, 240 SC, 16-QAM, rate 0.5, UMa, LS estimation with
    2 DMRS symbols, calibration errors on)::

        '3trp_rank2_240sc_qam16_r050_uma_est2dmrs_err'

    Every result file (BLER figure, NN model checkpoint, dataset) should embed
    this tag so its scenario is self-describing.

    ``feedback``: CSI feedback quantizer label, e.g. ``"phase4"`` / ``"iq4"``
    (appended as ``_fb<feedback>``); ``None`` means no quantization.
    """
    tag = (f"{int(num_trps)}trp_rank{int(rank)}_{int(n_subcarriers)}sc_"
           f"qam{int(qam_order)}_r{float(code_rate):.2f}").replace(".", "")
    tag += f"_{channel_kind}"
    if pathloss:
        tag += "_pl"
    if est:
        tag += f"_est{int(num_dmrs_symbols)}dmrs"
    if err:
        tag += "_err"
    if subband_size:
        tag += f"_sub{int(subband_size)}"
    if feedback:
        tag += f"_fb{feedback}"
    return tag
