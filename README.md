# DMIMO 链路级仿真平台（基于 NVIDIA Sionna / PyTorch）

基于 [NVIDIA Sionna](https://nvlabs.github.io/sionna/)（**PyTorch** 后端）的
**DMIMO（分布式 MIMO）链路级仿真器**：以 Sionna 官方教程
`tutorials/phy/MIMO_OFDM_Transmissions_over_CDL.ipynb` 的结构，用
`sionna.phy.Block` 积木搭建 3GPP 38.901 **CDL** 信道上的 MIMO OFDM 点对点链路，
评估 **BLER / BER**，支持上行与下行、完美/LS CSI、多 TRP 相干合并与每 TRP 时延/校准误差。

> **Sionna 版本**：面向 **Sionna 2.x**（PyTorch 后端），在 `pytorch2` conda 环境
> （Python 3.11，`C:\Users\xyseraph\anaconda3\envs\pytorch2\python.exe`）对照
> **Sionna 2.0.1** 验证。**不依赖 TensorFlow**。

## 1. 环境搭建

```powershell
conda activate pytorch2
python -m pip install sionna==2.0.1 numpy scipy pyyaml matplotlib
```

## 2. 目录结构

```
D:\Platform\
├── dmimo\                     # 链路级模型包
│   ├── model.py               # DMIMOPhyModel(Block) + DLModel / ULModel
│   ├── channels.py            # DMIMOChannel：CIR 驱动的频域网格信道（CDL/TDL/simple/uma/umi）
│   ├── precoding.py           # MRT / ZF / CJT / Type I 预编码
│   ├── errors.py              # 每 TRP 时延（相位斜坡）+ 校准误差模型
│   ├── config.py              # DMIMOConfig + YAML 加载 + scenario_tag
│   ├── sim.py                 # sim_ber_many（公平共享 + 提前停止 + 进度）
│   ├── results.py             # save_curves（PNG/CSV/JSON，Eb/N0 与 SNR 双轴）
│   └── __init__.py
├── configs\
│   ├── dmimo_linklevel.yaml   # 下行链路级场景
│   └── dmimo_ul_linklevel.yaml# 上行链路级场景
├── examples\
│   ├── dmimo_linklevel.py     # 下行 BLER（CLI）
│   ├── dmimo_linklevel_cfg.py # 下行 BLER（YAML 驱动，三变体对比）
│   ├── udmimo_linklevel.py    # 上行 BLER（三层合并）
│   └── common.py              # 示例共用路径引导
├── tutorials\                 # Sionna 官方教程 Notebook（phy/sys，供参考）
├── tools\git_push.ps1         # 用本地凭证推送的 git 工具
├── CONFIG_REFERENCE.md        # 配置参数参考
├── PROJECT_NOTES.md           # 开发经验记录
└── requirements.txt
```

## 3. 快速开始

```powershell
# 下行 BLER（YAML 驱动，三变体：单 TRP / 3TRP 相干 / 3TRP+误差）
python examples\dmimo_linklevel_cfg.py --config configs\dmimo_linklevel.yaml

# 下行 BLER（CLI，快速改参数）
python examples\dmimo_linklevel.py --channel cdl --rank 2 --ebno-start -5 --ebno-stop 15

# 上行 BLER（joint / symbol / llr 三层合并）
python examples\udmimo_linklevel.py --config configs\dmimo_ul_linklevel.yaml
```

每个驱动都会打印进度（Eb/N0、迭代、已运行时间/ETA、运行中 BLER），并在
`out/dmimo/` 下保存 **PNG（BLER vs Eb/N0，次轴 SNR）+ CSV（双轴数值）+
JSON（场景元信息 + 曲线）**。

## 4. 信号链（示例 Notebook 风格）

```
下行（DLModel）：b -> TB/LDPC 编码 -> QAM -> ResourceGridMapper(+DMRS 导频)
  -> 每 TRP 频域预编码（MRT/CJT/Type I）-> UE 相干合并 H_eff = Σ_t H_t W_t
  -> ApplyOFDMChannel + AWGN -> [完美 CSI | LS 估计] -> LMMSE 均衡
  -> Demapper -> 译码 -> b_hat（BLER/BER）

上行（ULModel）：b -> 编码/映射 -> 1 UE 开环发射（等增益/随机向量）
  -> K TRP 接收（每 TRP 自身时延/校准误差）-> joint/symbol/llr 合并
  -> Demapper -> 译码 -> b_hat
```

## 5. PHY 约定

- **Eb/N0 轴**：`ebnodb2no(ebno_db, num_bits_per_symbol, coderate, resource_grid)`
  把 Eb/N0 折算为每 RE 噪声方差，计入码率、调制、导频与置零子载波开销；落盘同时给
  出对应 SNR（`1/no`）。
- **CIR 驱动信道**：`DMIMOChannel` 逐 OFDM 符号采样 CDL/TDL 的 CIR
  （`a, tau`）并经 `cir_to_ofdm_channel` 转频域；`speed` 产生逐符号信道老化；
  `simple` 为频域平坦特例；`domain="time"`（`cir_to_time_channel` + OFDM 调制/解调）
  为预留扩展点。
- **公平性**：`sim_ber_many` 在每个 MC 批只采样一次信道/比特/噪声，所有预编码器
  （或合并器配置）共用这份数据，并支持 target-BLER 提前停止（sim_ber 规则）。
- **CSI**：`perfect_csi=true` 用真实有效信道；`false` 用 DMRS 导频 +
  `LSChannelEstimator`（最近邻插值）+ `LMMSEEqualizer`。

## 6. 注意事项

- 仅物理层（L1）：无 HARQ、无调度/高层协议建模。
- **Sionna 2.0.1 的自定义导频模式在 guard/DC 下与 `num_data_symbols` 计数不一致**，
  因此教程风格配置（guard/DC，如 `fft_size: 76` + `[5,6]` guard）一律用 Kronecker
  导频；rank≥2 的交错梳状导频（全子载波覆盖）仅用于无 guard/DC 的旧配置。
- CDL 要求每 TRP 天线数为偶数（双极化阵列）。
- 长任务（完整 BLER 扫描）建议用 CUDA（`device: "auto"` 自动检测）。
