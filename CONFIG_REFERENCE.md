# 配置参考

本文说明 YAML 配置中每个参数的作用、由哪个模块消费。配置文件：`configs/dmimo_linklevel.yaml`
（下行，`examples/dmimo_linklevel_cfg.py` 使用）、`configs/dmimo_ul_linklevel.yaml`
（上行，`examples/udmimo_linklevel.py` 使用）。加载与字段定义见 `dmimo/config.py`
（`DMIMOConfig` / `load_dmimo_config`）。

## 1. 信道与拓扑

| 字段 | 作用 | 使用方 |
|---|---|---|
| `channel_kind` | `simple` / `tdl` / `cdl` / `uma` / `umi` | `DMIMOChannel` |
| `cdl_model` | CDL 模型：A/B/C/D/E，`channel_kind=cdl` 时生效 | `DMIMOChannel`（每 TRP 一个 CDL） |
| `tdl_model` | TDL 模型：A/B/C/D/E，`channel_kind=tdl` 时生效 | `DMIMOChannel` |
| `num_trps` | TRP 数量（K） | `DLModel` / `ULModel` |
| `num_tx_ant` | 每 TRP 天线数（下行发射 / 上行接收；CDL 需偶数，双极化） | `DMIMOChannel`、预编码 |
| `num_ue_ant` | UE 天线数（下行接收 / 上行发射，>= rank） | `DMIMOChannel`、预编码 |
| `subcarrier_spacing_khz` | 子载波间隔（kHz） | 资源栅格、时延相位斜坡、CDL 符号时长 |
| `carrier_frequency` | 载频（Hz） | CDL/TDL 天线阵列、多普勒 |
| `pathloss` | 是否开启路损（CDL/TDL 外部大尺度模型 + 最强 TRP 归一化） | `DMIMOChannel` |
| `trp_distances_m` | 各 TRP 到 UE 距离（m） | 路损、UMa/UMi 拓扑 |
| `speed` | UE 速度 m/s（CDL 逐符号信道老化；0=静态） | `DMIMOChannel` |
| `delay_spread_ns` | CDL/TDL 名义时延扩展 [ns] | `DMIMOChannel` |

## 2. 每 TRP 误差

| 字段 | 作用 | 使用方 |
|---|---|---|
| `tau_ns` | 各 TRP 时延（ns），TRP0 必须为 0，长度必须等于 `num_trps` | `TRPErrorModel`（逐子载波相位斜坡） |
| `cal_amp_error` | 校准幅度误差强度，`None` 表示关闭 | `CalibrationError` |
| `cal_pha_error` | 校准相位误差强度，`None` 表示关闭 | `CalibrationError` |
| `granularity` | 时延相位斜坡粒度：`SC` / `RB` / `SC_RB_granular` | `TimingError` |
| `subband_size` | 子带大小（子载波数），CJT 预编码粒度 | `CJTPrecoder` |

## 3. 链路与编码

| 字段 | 作用 | 使用方 |
|---|---|---|
| `rank` | 层数/流数（DL）；UL 固定 rank=1 | `DLModel` / `ULModel` |
| `qam_order` | QAM 阶数（4=QPSK，16=16QAM...） | `DMIMOPhyModel`（Mapper/Demapper） |
| `code_rate` | 目标码率 | `TBEncoder` / `LDPC5GEncoder` |
| `use_crc` | 是否使用 5G NR TB CRC 检错（`false` = 裸 LDPC 整比特比对） | `DMIMOPhyModel` |
| `precoder` | 下行预编码选择：`mrt` / `cjt` / `type1` / `all` | `dmimo_linklevel_cfg.py` |

## 4. 资源栅格与 CSI

| 字段 | 作用 | 使用方 |
|---|---|---|
| `n_symbols` | 每 slot OFDM 符号数 | `DMIMOPhyModel` 资源栅格 |
| `fft_size` | 全 FFT 子载波数（含 guard/DC），如 76 | `DMIMOPhyModel` 资源栅格、`DMIMOChannel` |
| `num_guard_carriers` | 左右 guard 子载波数，如 `[5, 6]` | `DMIMOPhyModel` 资源栅格 |
| `dc_null` | DC 子载波置零 | `DMIMOPhyModel` 资源栅格 |
| `cyclic_prefix_length` | CP 长度（频域建模仅影响 Eb/N0 折算；`time` 域为预留扩展点） | `DMIMOPhyModel` |
| `pilot_ofdm_symbol_indices` | DMRS 导频符号索引，如 `[2, 11]`（映射类型 A） | `DMIMOPhyModel`（Kronecker 导频） |
| `pilot_boost_db` | DMRS 导频能量提升（dB） | `DMIMOPhyModel` |
| `perfect_csi` | 接收端完美 CSI（真实有效信道）；`false` = DMRS LS 估计（NN 插值） | `DLModel` / `ULModel` |

## 5. UL 专属

| 字段 | 作用 | 使用方 |
|---|---|---|
| `combiner` | 合并层级：`joint`（L1 联合检测）/ `symbol`（L2 均衡后合并）/ `llr`（L3 LLR 合并）；驱动中可用 `all` | `ULModel` |
| `estimate_errors` | 均衡信道是否吸收接收误差（`false` = 用干净信道，悲观/未补偿） | `ULModel` |

## 6. 建模域

| 字段 | 作用 | 使用方 |
|---|---|---|
| `domain` | 建模域：`freq`（本期实现）；`time`（`cir_to_time_channel` + OFDM 调制/解调）为预留扩展点 | `DMIMOPhyModel` |

## 7. Eb/N0 扫描与 MC

| 字段 | 作用 | 使用方 |
|---|---|---|
| `ebno_db` | 显式 Eb/N0 列表（dB）；`None` 时用起止步长自动生成 | `DMIMOConfig.ebno_grid` |
| `ebno_start_db` / `ebno_stop_db` / `ebno_step_db` | 等间距 Eb/N0 扫描范围（默认 -5..19 step 4，示例 Notebook 风格） | `DMIMOConfig.ebno_grid` |
| `num_trials` | 每 MC 批的 TB 数（batch_size） | `sim_ber_many` |
| `num_mc_batches` | 每 SNR 点 MC 迭代数上限（配合 target-BLER 提前停止） | `sim_ber_many` |
| `target_bler` | sim_ber 提前停止目标 BLER（默认 1e-3） | `sim_ber_many` |
| `num_target_block_errors` | sim_ber 提前停止块错误数（默认 1000） | `sim_ber_many` |
| `device` | `cpu` / `cuda:0` / `auto` | 示例脚本 |
| `seed` | 随机种子 | 示例脚本 |

> 链路级 BLER 曲线使用 **Eb/N0 轴**（`ebnodb2no` 计入码率/调制/导频/置零开销）；
> 落盘 CSV/JSON 同时给出每点的 SNR（`1/no`）。同一 SNR 点所有预编码器/合并器
> 共享同一份信道/比特/噪声（`sim_ber_many` 公平性，满足 AGENTS.md §3）。

## 8. 已清理/明确的点

- 移除了与链路级重构无关的模块：`sionna5g/`（PUSCH 平台）、`dmimo/link.py` /
  `uplink.py`（系统级 rate 模型）、`dmimo/nn_pmi.py` / `modelDesign.py` /
  `model_complexity.py` / `feedback.py`（NN-PMI 与 CSI 反馈量化）、
  `dmimo/experiment.py`（已废弃 shim）及对应示例与配置。
- 遗留字段 `use_channel_estimation` / `est_density` / `dmrs_symbol` /
  `num_dmrs_symbols` / `feedback_*` / `nn_pmi_ckpt` / `snr_*` 已随删除一并移除，
  统一由 `perfect_csi` + `pilot_ofdm_symbol_indices` + `ebno_*` 取代。
- `scenario_tag` 不再包含反馈量化标签。
