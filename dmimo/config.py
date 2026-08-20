"""YAML config support for the DMIMO link-level platform (DLModel / ULModel)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import yaml


@dataclass
class DMIMOConfig:
    # ---- 信道与拓扑 ----
    channel_kind: str = "cdl"                 # simple|tdl|cdl|uma|umi
    cdl_model: str = "A"                  # CDL 模型：A/B/C/D/E
    tdl_model: str = "C"                  # TDL 模型：A/B/C/D/E
    num_trps: int = 3
    num_tx_ant: int = 8                   # 每 TRP 天线数（CDL 需偶数，双极化）
    num_ue_ant: int = 4                   # UE 天线数（>= rank）
    subcarrier_spacing_khz: float = 15.0
    carrier_frequency: float = 2.6e9
    pathloss: bool = False
    trp_distances_m: Tuple[float, ...] = (100.0, 200.0, 350.0)
    speed: float = 0.0                    # UE 速度 m/s（信道老化；0=静态）
    delay_spread_ns: float = 100.0        # CDL/TDL 名义时延扩展 [ns]

    # ---- 每 TRP 误差（时延 / 校准）----
    tau_ns: Tuple[float, ...] = (0.0, 130.0, 260.0)
    cal_amp_error: float = 0.1            # None -> no calibration error
    cal_pha_error: float = 0.1
    granularity: str = "SC"               # SC|RB|SC_RB_granular
    subband_size: int = 12                # CJT 预编码子带粒度（子载波）

    # ---- 链路与编码 ----
    rank: int = 1
    qam_order: int = 4                    # 星座点数（4=QPSK）
    code_rate: float = 0.5
    use_crc: bool = True                  # 5G NR TB encoder/decoder + TB CRC BLER
    precoder: str = "all"                 # mrt|cjt|type1|all

    # ---- 资源栅格与 DMRS ----
    n_symbols: int = 14                   # OFDM 符号数（slot）
    fft_size: int = 76                    # 全 FFT 子载波数（含 guard/DC）
    num_guard_carriers: Tuple[int, int] = (5, 6)   # 左右 guard 子载波数
    dc_null: bool = True                  # DC 子载波置零
    cyclic_prefix_length: int = 6         # CP 长度（频域建模仅影响 Eb/N0 折算）
    pilot_ofdm_symbol_indices: Tuple[int, ...] = (2, 11)   # DMRS 导频符号索引
    pilot_boost_db: float = 0.0           # DMRS 导频能量提升（dB）
    perfect_csi: bool = False             # 接收端完美 CSI（否则 DMRS LS 估计）

    # ---- UL 专属 ----
    combiner: str = "joint"               # joint|symbol|llr
    estimate_errors: bool = True          # 均衡信道是否吸收接收误差

    # ---- 建模域 ----
    domain: str = "freq"                  # freq | time（time 为预留扩展点）

    # ---- Eb/N0 扫描与 MC ----
    ebno_db: Optional[List[float]] = None # 显式 Eb/N0 列表；None -> ebno_start/stop/step
    ebno_start_db: float = -5.0
    ebno_stop_db: float = 19.0
    ebno_step_db: float = 4.0
    num_trials: int = 256                 # 每 MC 批的 TB 数（batch_size）
    num_mc_batches: int = 5               # 每 SNR 的 MC 迭代数上限（配合提前停止）
    target_bler: float = 1e-3             # sim_ber 提前停止目标 BLER
    num_target_block_errors: int = 1000   # sim_ber 提前停止块错误数
    device: str = "auto"
    seed: int = 0

    @property
    def ebno_grid(self) -> List[float]:
        """Eb/N0 sweep points in dB (default ``-5 .. 19``, step 4, like the
        Sionna CDL tutorial)."""
        if self.ebno_db is not None:
            return list(self.ebno_db)
        if self.ebno_step_db <= 0:
            raise ValueError("ebno_step_db must be > 0.")
        n = max(int(math.floor((self.ebno_stop_db - self.ebno_start_db) / self.ebno_step_db + 1e-9)) + 1, 1)
        return [round(self.ebno_start_db + i * self.ebno_step_db, 6) for i in range(n)]

    @property
    def tau_seconds(self) -> Tuple[float, ...]:
        return tuple(t * 1e-9 for t in self.tau_ns)

    @property
    def delay_spread(self) -> float:
        return self.delay_spread_ns * 1e-9

    @property
    def pilot_symbols(self) -> Tuple[int, ...]:
        """DMRS pilot symbol indices (modulo ``n_symbols``)."""
        return tuple(int(i) % self.n_symbols for i in self.pilot_ofdm_symbol_indices)

    # ------------------------------------------------------------------
    # Link-level models (Sionna Block style, see dmimo.model)
    # ------------------------------------------------------------------
    def build_dl_model(self, device: Optional[str] = None):
        """Build a :class:`~dmimo.model.DLModel` from this config."""
        from .model import DLModel

        return DLModel(
            num_trps=self.num_trps, num_bs_ant=self.num_tx_ant,
            num_ue_ant=self.num_ue_ant, channel_kind=self.channel_kind,
            cdl_model=self.cdl_model, tdl_model=self.tdl_model,
            delay_spread=self.delay_spread, speed=self.speed,
            pathloss=self.pathloss, trp_distances=self.trp_distances_m,
            carrier_frequency=self.carrier_frequency,
            tau_seconds=self.tau_seconds, cal_amp_error=self.cal_amp_error,
            cal_pha_error=self.cal_pha_error, granularity=self.granularity,
            subcarrier_spacing=self.subcarrier_spacing_khz * 1e3,
            fft_size=self.fft_size, num_guard_carriers=self.num_guard_carriers,
            dc_null=self.dc_null, n_symbols=self.n_symbols,
            pilot_ofdm_symbol_indices=self.pilot_symbols,
            pilot_boost_db=self.pilot_boost_db,
            cyclic_prefix_length=self.cyclic_prefix_length,
            qam_order=self.qam_order, code_rate=self.code_rate,
            rank=self.rank, use_crc=self.use_crc,
            perfect_csi=self.perfect_csi, device=device)

    def build_ul_model(self, device: Optional[str] = None):
        """Build a :class:`~dmimo.model.ULModel` from this config."""
        from .model import ULModel

        return ULModel(
            num_trps=self.num_trps, num_bs_ant=self.num_tx_ant,
            num_ue_ant=self.num_ue_ant, channel_kind=self.channel_kind,
            cdl_model=self.cdl_model, tdl_model=self.tdl_model,
            delay_spread=self.delay_spread, speed=self.speed,
            pathloss=self.pathloss, trp_distances=self.trp_distances_m,
            carrier_frequency=self.carrier_frequency,
            tau_seconds=self.tau_seconds, cal_amp_error=self.cal_amp_error,
            cal_pha_error=self.cal_pha_error, granularity=self.granularity,
            subcarrier_spacing=self.subcarrier_spacing_khz * 1e3,
            fft_size=self.fft_size, num_guard_carriers=self.num_guard_carriers,
            dc_null=self.dc_null, n_symbols=self.n_symbols,
            pilot_ofdm_symbol_indices=self.pilot_symbols,
            pilot_boost_db=self.pilot_boost_db,
            cyclic_prefix_length=self.cyclic_prefix_length,
            qam_order=self.qam_order, code_rate=self.code_rate,
            use_crc=self.use_crc, perfect_csi=self.perfect_csi,
            combiner=self.combiner, estimate_errors=self.estimate_errors,
            device=device)


def _coerce(name, v):
    if name in ("trp_distances_m", "tau_ns", "pilot_ofdm_symbol_indices", "num_guard_carriers"):
        try:
            return tuple(float(x) if name in ("trp_distances_m", "tau_ns") else int(x) for x in v)
        except (TypeError, ValueError):
            return v
    if name == "ebno_db":
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return v
    if name in ("num_trps", "num_tx_ant", "num_ue_ant", "qam_order", "rank",
                "n_symbols", "num_trials", "num_mc_batches", "seed", "subband_size",
                "fft_size", "cyclic_prefix_length", "num_target_block_errors"):
        try:
            return int(v)
        except (TypeError, ValueError):
            return v
    if name in ("subcarrier_spacing_khz", "carrier_frequency", "cal_amp_error", "cal_pha_error",
                "code_rate", "pilot_boost_db", "speed", "delay_spread_ns", "target_bler",
                "ebno_start_db", "ebno_stop_db", "ebno_step_db"):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    if name in ("use_crc", "perfect_csi", "dc_null", "estimate_errors", "pathloss"):
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
                 err=False, subband_size=None, pathloss=False) -> str:
    """Compact scenario tag for result-file names (filename == metadata).

    Example (3 TRPs, rank 2, 76 SC, QPSK, rate 0.5, CDL, LS estimation with
    2 DMRS symbols, calibration errors on)::

        '3trp_rank2_76sc_qam4_r050_cdl_est2dmrs_err'

    Every result file (BLER figure / CSV / JSON) should embed this tag so its
    scenario is self-describing.
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
    return tag
