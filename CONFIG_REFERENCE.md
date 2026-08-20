# 配置参考

本文说明 YAML 配置中每个参数的作用、由哪个模块消费，以及哪些字段是“同一个东西的两种写法”。

## 1. 通用 5G PUSCH 平台：`sionna5g/config.py`

### 顶层 `SimConfig`

| 字段 | 作用 | 使用方 |
|---|---|---|
| `snr_db` | 显式 SNR 点列表（dB） | `SimConfig.snr_grid` |
| `snr_start_db` / `snr_stop_db` / `snr_step_db` | 等间距 SNR 扫描范围；与 `snr_db` 二选一，`snr_db` 优先 | `SimConfig.snr_grid` |
| `num_trials` | 每个 SNR 点的蒙特卡洛 TB 数 | `LinkSimulator.run_curve` |
| `batch_size` | 每个前向批次的 TB 数，`num_trials` 会按它分片 | `LinkSimulator.run_snr` |
| `device` | `cpu` / `cuda:0` / `auto` | `resolve_device`，贯穿 TX/Channel/RX |
| `seed` | 全局随机种子 | 各示例脚本 |
| `return_crc_status` | 接收机是否返回 TB CRC 状态 | `PUSCHReceiverWrapper` |

> `snr_db` 和 `snr_start_db/stop/step` 不是重复配置，它们是“显式列表”和“等间距扫描”两种表达方式。

### `carrier`

| 字段 | 作用 | 使用方 |
|---|---|---|
| `subcarrier_spacing` | 子载波间隔（kHz），影响 slot 时长 | `CarrierConfig.slot_duration`、Sionna `CarrierConfig` |
| `n_size_grid` | RB 数 | Sionna `CarrierConfig` |
| `n_cell_id` | 物理小区 ID | Sionna `CarrierConfig` |
| `slot_number` / `frame_number` | 时隙/帧号 | Sionna `CarrierConfig` |

### `pusch`

| 字段 | 作用 | 使用方 |
|---|---|---|
| `num_layers` | 空间层数/流数 | Sionna `PUSCHConfig` |
| `num_antenna_ports` | UE 发射天线端口数 | Sionna `PUSCHConfig`、信道 TX 天线数 |
| `mapping_type` | DMRS 映射类型 A/B | Sionna `PUSCHConfig` |
| `symbol_start` / `symbol_length` | PUSCH 时域符号分配 | Sionna `PUSCHConfig` |
| `precoding` | `non-codebook` / `codebook` | Sionna `PUSCHConfig` |
| `transform_precoding` | 是否使用 DFT-s-OFDM | Sionna `PUSCHConfig` |

> 已移除未使用的 `num_ut`。当前平台只建模单用户物理链路。

### `tb`

| 字段 | 作用 | 使用方 |
|---|---|---|
| `mcs_index` | MCS 索引 | Sionna `TBConfig` |
| `mcs_table` | MCS 表 | Sionna `TBConfig` |
| `channel_type` | `PUSCH` / `PDSCH` | Sionna `TBConfig` |
| `n_id` | 可选加扰 ID | Sionna `TBConfig` |

### `channel`

| 字段 | 作用 | 使用方 |
|---|---|---|
| `channel_type` | `awgn` / `tdl` / `cdl` / `uma` / `umi` / 自定义注册名 | `ChannelModelWrapper` |
| `model` | TDL/CDL 模型名，如 `C`、`CDL-A` | Sionna `TDL` / `CDL` |
| `delay_spread` | RMS 时延扩展 | Sionna `TDL` / `CDL` |
| `carrier_frequency` | 载频（Hz） | Sionna 信道/天线阵列 |
| `min_speed` / `max_speed` | UE 速度范围，驱动多普勒 | Sionna `TDL` / `CDL` |
| `o2i_model` | UMa/UMi 的 O2I 模型 `low`/`high` | Sionna `UMa` / `UMi` |
| `ut_distance` / `ut_height` | UMa/UMi 拓扑中的 UE 位置 | `ChannelModelWrapper._set_topology` |
| `normalize_channel` | 是否归一化频域信道；TDL/CDL 可配，UMa/UMi 固定归一化 | `ChannelModelWrapper` |
| `num_rx_ant` | 基站接收天线数；`None` 时等于 `pusch.num_antenna_ports` | `LinkSimulator` |

> `num_rx_ant` 不是重复配置：当你不关心接收天线数时省略它，平台会自动取发射天线端口数。

### `receiver`

| 字段 | 作用 | 使用方 |
|---|---|---|
| `channel_estimator` | `ls` / `perfect` / 自定义注册名 | `PUSCHReceiverWrapper` |
| `mimo_detector` | `lmmse` / `zf` / 自定义注册名 | `PUSCHReceiverWrapper` |
| `tb_decoder` | `default` / 自定义注册名 | `PUSCHReceiverWrapper` |

> 已移除未接线的 `num_demod_iter`。如果以后要开放 LDPC 外环迭代，应先在 `PUSCHReceiverWrapper` 中真正接入再放回配置。

---

## 2. DMIMO 配置：`dmimo/config.py`

### 信道与拓扑

| 字段 | 作用 | 使用方 |
|---|---|---|
| `channel_kind` | `simple` / `tdl` / `cdl` / `uma` / `umi` | `build_downlink_channel` |
| `cdl_model` | CDL 模型：A/B/C/D/E，`channel_kind=cdl` 时生效 | `SionnaDownlinkChannel` |
| `tdl_model` | TDL 模型：A/B/C/D/E，`channel_kind=tdl` 时生效 | `SionnaDownlinkChannel` |
| `num_trps` | TRP 数量 | 所有 DMIMO 模块 |
| `num_tx_ant` | 每个 TRP 的发射天线数 | 信道、预编码 |
| `num_ue_ant` | UE 接收天线数 | 信道、预编码 |
| `n_subcarriers` | 子载波数 | 信道、链路级资源栅格 |
| `subcarrier_spacing_khz` | 子载波间隔（kHz） | 时延相位斜坡、资源栅格 |
| `carrier_frequency` | 载频（Hz） | 信道/天线阵列 |
| `pathloss` | 是否开启路损 | Sionna 信道 |
| `trp_distances_m` | 各 TRP 到 UE 的距离（m） | UMa/UMi 拓扑、CDL/TDL 外部路损 |

### 误差与预编码

| 字段 | 作用 | 使用方 |
|---|---|---|
| `tau_ns` | 各 TRP 时延（ns），TRP0 必须为 0 | `TimingError` |
| `cal_amp_error` | 校准幅度误差强度，`None` 表示关闭 | `CalibrationError` |
| `cal_pha_error` | 校准相位误差强度，`None` 表示关闭 | `CalibrationError` |
| `granularity` | 时延相位斜坡粒度：`SC` / `RB` / `SC_RB_granular` | `TimingError` |
| `subband_size` | 子带大小（子载波数），CJT 与 NN-PMI 统一使用 | `CJTPrecoder`、`NNMixerPMI` |
| `precoder` | `mrt` / `cjt` / `type1` / `all` / `nn` | 示例脚本选择要跑的预编码器 |

### CSI 反馈量化（P3：有限比特反馈）

| 字段 | 作用 | 使用方 |
|---|---|---|
| `feedback_quant` | `none` / `phase` / `iq`，是否启用反馈量化（`none` = 理想 CSI 上界） | `QuantizedFeedback` / 示例脚本 |
| `feedback_bits_phase` | phase 量化每复系数的相位比特数（2/3/4/6/8） | `PhaseQuantizer` |
| `feedback_bits_amp` | phase 量化振幅比特数，`null` = 保留连续振幅 | `PhaseQuantizer` |
| `feedback_bits_iq` | iq 量化每实/虚部比特数（每复系数总开销 = 2×） | `ScalarQuantizer` |
| `feedback_ste` | 直通估计器开关（NN-PMI 抗量化训练用） | `QuantizedFeedback` |
| `feedback_subband_size` | 反馈子带大小（子载波），`null` = 复用 `subband_size` | `QuantizedFeedback` |

> 反馈开销 = `num_trps × num_tx_ant × rank × S × b`，其中 `S = ceil(n_subcarriers / feedback_subband_size)`，`b` 为每复系数比特（phase 型 = `feedback_bits_phase`，iq 型 = `2 × feedback_bits_iq`）。
> **语义**：本模块量化的是预编码器*输出*，对应 UE→BS 的 PMI 反馈（MRT/CJT 等连续预编码）或 fronthaul 系数信令/定点精度（可选场景）。**NN-PMI 默认不套输出量化**——它在基站本地由反馈回来的 Type I PMI 生成、输出无需回传，只作为不量化基准参与对比；如要做 fronthaul 故事，需显式包 `QuantizedFeedback(ste=True)` 并配 `feedback_ste` 训练。
> Type I 码本本身就是量化预编码（beam 索引 + QPSK 共相位），不做双重包装，作为固定低比特参考。

### 链路级参数（DLModel / ULModel，Sionna Block 风格，`dmimo/model.py`）

| 字段 | 作用 | 使用方 |
|---|---|---|
| `rank` | 层数/流数（DL）；UL 固定 rank=1 | `DLModel` / `ULModel` |
| `qam_order` | QAM 阶数（4=QPSK，16=16QAM...） | `DMIMOPhyModel` |
| `code_rate` | 目标码率 | `TBEncoder` / `LDPC5GEncoder` |
| `perfect_csi` | 接收端完美 CSI（真值有效信道）；`false` = DMRS LS 估计（NN 插值） | `DLModel` / `ULModel` |
| `fft_size` | 全 FFT 子载波数（含 guard/DC），如 76 | `DMIMOPhyModel` 资源栅格、`DMIMOChannel` |
| `num_guard_carriers` | 左右 guard 子载波数，如 `[5, 6]` | `DMIMOPhyModel` 资源栅格 |
| `dc_null` | DC 子载波置零 | `DMIMOPhyModel` 资源栅格 |
| `cyclic_prefix_length` | CP 长度（频域建模仅影响 Eb/N0 折算；`time` 域为预留扩展点） | `DMIMOPhyModel` |
| `pilot_ofdm_symbol_indices` | DMRS 导频符号索引，如 `[2, 11]`（映射类型 A）；优先于 `dmrs_symbol`+`num_dmrs_symbols` | `DMIMOPhyModel` |
| `pilot_boost_db` | DMRS 导频能量提升（dB） | `DMIMOPhyModel` |
| `n_symbols` | 每 slot OFDM 符号数 | `DMIMOPhyModel` |
| `use_crc` | 是否使用 TB CRC 检错 | `DMIMOPhyModel` |
| `speed` | UE 速度 m/s（CDL 逐符号信道老化；0=静态） | `DMIMOChannel` |
| `delay_spread_ns` | CDL/TDL 名义时延扩展 [ns] | `DMIMOChannel` |
| `combiner` | UL 合并层级：`joint` / `symbol` / `llr`（或 `all`） | `ULModel` |
| `estimate_errors` | UL 均衡信道是否吸收接收误差（`false` = 用干净信道，悲观） | `ULModel` |
| `domain` | 建模域：`freq`（本期）；`time` 为预留扩展点 | `DMIMOPhyModel` |
| `nn_pmi_ckpt` | NN-PMI 模型 checkpoint 路径 | `dmimo_linklevel_cfg.py` |

> 旧字段 `use_channel_estimation` / `est_density` / `dmrs_symbol` / `num_dmrs_symbols`
> 由 `perfect_csi` + `pilot_ofdm_symbol_indices` 取代（`est_density` 语义被 Kronecker
> 导频全子载波覆盖取代）；保留读取以兼容旧 YAML，`pilot_symbols` 属性优先用显式索引。

### SNR / Eb-N0 与运行

| 字段 | 作用 | 使用方 |
|---|---|---|
| `snr_db` / `snr_start_db` / `snr_stop_db` / `snr_step_db` | SNR 扫描（系统级 rate 模型用），二选一 | `DMIMOConfig.snr_grid` |
| `ebno_db` / `ebno_start_db` / `ebno_stop_db` / `ebno_step_db` | Eb/N0 扫描（链路级 BLER 用，默认 -5..19 step 4，示例 Notebook 风格），二选一 | `DMIMOConfig.ebno_grid` → `sim_ber_many` |
| `num_trials` | 每 MC 批的 TB 数（batch_size） | `sim_ber_many` |
| `num_mc_batches` | 每 SNR 点 MC 迭代数上限（配合 target-BLER 提前停止） | `sim_ber_many` |
| `target_bler` | sim_ber 提前停止目标 BLER（默认 1e-3） | `sim_ber_many` |
| `num_target_block_errors` | sim_ber 提前停止块错误数（默认 1000） | `sim_ber_many` |
| `device` | `cpu` / `cuda:0` / `auto` | 示例脚本 |
| `seed` | 随机种子 | 示例脚本 |

> 链路级 BLER 曲线使用 **Eb/N0 轴**（`ebnodb2no` 计入码率/调制/导频/置零开销）；
> 落盘 CSV/JSON 同时给出每点的 SNR（`1/no`）。同一 SNR 点所有预编码器/合并器
> 共享同一份信道/比特/噪声（`sim_ber_many` 公平性，满足 AGENTS.md §3）。

---

## 3. 已清理/明确的点

- 删除 `PUSCHConfig.num_ut`：单用户链路不需要。
- 删除 `ReceiverConfig.num_demod_iter`：原先未实际接入接收机。
- `ChannelConfig.channel_type` 文档已补全 `uma` / `umi`。
- `ChannelConfig.normalize_channel` 说明 UMa/UMi 固定归一化，避免误解。
- 内置信道/接收机构建不再强制经过 `registry` 间接层，代码路径更直白。
- 示例新增 `examples/common.py`，统一处理根路径导入。
