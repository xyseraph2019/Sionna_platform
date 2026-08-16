# 5G 链路级仿真平台（基于 NVIDIA Sionna / PyTorch）

一个基于 [NVIDIA Sionna](https://nvlabs.github.io/sionna/)（**PyTorch** 后端）的模块化
**5G NR 链路级仿真器**。它建模单用户 PUSCH（上行）物理链路，在可配置的载波、MIMO、
传播信道场景下评估 **BLER / BER / 吞吐量**。

> **Sionna 版本**：本平台面向 **Sionna 2.x**（PyTorch 后端），在 `pytorch2`
> conda 环境（Python 3.11）中对照 **Sionna 2.0.1** 验证。**不依赖 TensorFlow**。
> 如需 GPU 加速，需要 NVIDIA GPU + 支持 CUDA 的 PyTorch（见第 3 节）。

## 1. 环境搭建

```powershell
conda activate pytorch2
python -m pip install sionna==2.0.1 pyyaml matplotlib
```

也可在任意 Python 3.9–3.12 环境安装 Sionna 的 PyTorch 版：

```powershell
python -m pip install "sionna[torch]"
```

## 2. 目录结构

```
D:\Platform\
├── sionna5g\                    # 仿真器包
│   ├── config.py                # 配置 dataclass + YAML 加载 + device 解析
│   ├── transmitter.py           # 5G NR PUSCH 发射机（频域）
│   ├── channel.py               # 信道包装（AWGN / TDL / CDL / UMa / UMi / 自定义）
│   ├── receiver.py              # PUSCH 接收机（LS/perfect 估计 + LMMSE/ZF 检测 + LDPC + CRC）
│   ├── simulator.py             # LinkSimulator 编排（支持依赖注入）
│   ├── metrics.py               # BLER / BER / 吞吐量
│   ├── link_adaptation.py       # 基于 BLER 的 MCS / CQI 选择
│   ├── plotter.py               # 结果绘图
│   ├── registry.py              # 组件注册表（信道/估计/检测/译码/发射 可插拔）
│   └── components/base.py       # [已废弃] 仅向后兼容，不再被核心引用
├── configs\                     # YAML 场景
│   ├── awgn_qpsk.yaml
│   ├── tdl_c_16qam.yaml
│   ├── tdl_b_64qam_mimo.yaml
│   └── custom_rx.yaml           # 自定义接收机示例
├── examples\
│   ├── bler_curves.py           # 完整 SNR 扫描 + 绘图
│   ├── single_point.py          # 单 SNR 快速检查
│   ├── run_scenarios.py         # 分组对比批量场景驱动
│   ├── custom_components.py     # 自定义信道/估计/检测（推荐）
│   ├── plugins\                 # [已废弃] 旧插件目录，仅向后兼容
│   │   ├── custom_rx.py         # 旧版，建议改用 custom_components
│   │   ├── custom_channel.py    # 旧版，建议改用 custom_components
│   │   └── compare_receivers.py # 端到端接收机对比 demo
│   └── common.py                # 示例共用路径引导
├── run_simulation.py            # 命令行入口
└── requirements.txt
```

## 3. CPU / GPU

默认 `device: auto`：检测到 CUDA GPU 则用 `cuda:0`，否则 `cpu`。

- **改 YAML**（对该场景生效）：在 `configs\*.yaml` 中设 `device: cuda:0` 或 `device: cpu`。
- **命令行覆盖**（任意场景）：
  ```powershell
  python run_simulation.py --config configs\tdl_c_16qam.yaml --device cuda:0
  ```
`device` 会贯穿发射、信道、接收等所有阶段，所有张量都放在指定设备上。

检查 GPU 是否可用：

```powershell
& "C:\Users\xyseraph\anaconda3\envs\pytorch2\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 4. 快速开始

```powershell
python run_simulation.py --config configs\awgn_qpsk.yaml
python run_simulation.py --config configs\tdl_c_16qam.yaml --plot out\tdl_c.png
python run_simulation.py --config configs\tdl_b_64qam_mimo.yaml --device cuda:0
python examples\run_scenarios.py --trials 500          # 分组对比（信道/调制/MIMO/城市/接收机/自定义）
```

## 5. 信号链

```
比特 -> LDPC 编码 -> 速率匹配 -> 加扰 -> QAM -> 层映射
      -> DMRS + 资源栅格 -> [码本预编码] -> OFDM（频域栅格）
      -> 信道（AWGN | TDL | CDL | UMa | UMi | 自定义）-> + AWGN(SNR)
      -> LS/perfect 信道估计 -> LMMSE/ZF/MF 检测 -> 解扰/速率匹配
      -> LDPC 译码 -> CRC 校验 -> BLER / BER / 吞吐量
```

所有处理均由 Sionna 的 `sionna.phy.nr`（PUSCH 收发）、`sionna.phy.channel`
（AWGN、TDL、CDL、UMa/UMi、OFDMChannel）与 `sionna.phy.mimo`（LMMSE/ZF）提供。

## 6. 场景配置

配置段：`carrier`、`pusch`、`tb`、`channel`、`receiver`。

- `carrier`：`subcarrier_spacing`（kHz）、`n_size_grid`、`n_cell_id`。
- `pusch`：`num_layers`、`num_antenna_ports`、`mapping_type`、`symbol_start`/`symbol_length`、`precoding`。
- `tb`：`mcs_index`、`mcs_table`。
- `channel`：`channel_type`（`awgn`|`tdl`|`cdl`|`uma`|`umi` 或**自定义注册名**）、
  `model`、`delay_spread`、`carrier_frequency`、`min_speed`/`max_speed`、`o2i_model`、
  `normalize_channel`、`num_rx_ant`（基站接收天线，默认等于 PUSCH 天线端口数）。
- `receiver`：按注册名选择接收算法：
  - `channel_estimator`：`"ls"`（默认）| `"perfect"` | 自定义，
  - `mimo_detector`：`"lmmse"`（默认）| `"zf"` | 自定义，
  - `tb_decoder`：`"default"` | 自定义，
  - `num_demod_iter`：可选的外环迭代次数。

顶层字段：`snr_db`、`num_trials`、`batch_size`、`device`、`seed`。

## 7. 通用平台：如何做"新算法 / 新场景"（只改表层代码）

平台是一层薄的编排层，可替换部件（信道、信道估计、MIMO 检测、TB 译码、发射机）
按名字在 `sionna5g/registry` 中查找、由注册函数构建。**加新算法/新场景时核心
（`simulator.py`、`receiver.py`、`run_scenarios.py`）不用动**，只需两步。

### 方式一：纯配置切换（已有内置组件，零代码）

在 YAML 的 `receiver:` / `channel:` 段直接选名字即可：

```yaml
receiver:
  channel_estimator: "perfect"   # ls | perfect | <自定义名>
  mimo_detector: "lmmse"         # lmmse | zf | <自定义名>
```

### 方式二：注册自定义组件（一个文件）

写一个小模块，调用 `register` 注册，运行前 `import` 一下，然后在配置里按名字选。

**注册函数（builder）的签名**：

- `channel`：    `builder(cfg, resource_grid, device, num_tx_ant, num_rx_ant) -> (ofdm, sl_model)`
- `estimator`：  `builder(pusch_transmitter, device) -> estimator | None | "perfect"`
- `detector`：   `builder(pusch_transmitter, device) -> OFDMDetector | None`
- `decoder`：    `builder(pusch_transmitter, device) -> TBDecoder | None`
- `transmitter`：`builder(carrier, pusch, tb, device) -> Transmitter`

**例 1：加一个自定义信道**（"平坦瑞利"，见 `examples/plugins/custom_channel.py`）

```python
from sionna5g import registry
from sionna.phy.channel import OFDMChannel

def flat_rayleigh(cfg, resource_grid, device, num_tx_ant, num_rx_ant):
    model = FlatRayleigh(num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant, device=device)
    ofdm = OFDMChannel(channel_model=model, resource_grid=resource_grid,
                       normalize_channel=True, return_channel=True, device=device)
    return ofdm, None  # (OFDM 信道, 系统级模型或 None)

registry.register("channel", "flat_rayleigh", flat_rayleigh)
```

随后配置里 `channel.channel_type: "flat_rayleigh"` 即生效。

**例 2：加一个自定义 MIMO 检测器**（匹配滤波 MF，见 `examples/plugins/custom_rx.py`）

```python
from sionna.phy.ofdm.detection import LinearDetector
from sionna.phy.mimo import StreamManagement
import numpy as np

def mf_detector(pusch_transmitter, device="cpu"):
    sm = StreamManagement(np.ones([1, pusch_transmitter._num_tx], bool),
                          pusch_transmitter._num_layers)
    return LinearDetector("mf", "bit", "maxlog", pusch_transmitter.resource_grid,
                          sm, "qam", pusch_transmitter._num_bits_per_symbol, device=device)

registry.register("detector", "mf", mf_detector)
```

然后配置里 `receiver.mimo_detector: "mf"` 即生效。

### 依赖注入（整体换接收机/信道）

`LinkSimulator(cfg, components={"tx": ..., "channel": ..., "rx": ...})` 接受预构建组件；
未提供的部件仍按 `cfg` 构建。这样无需改模拟器即可跑一个完全自定义的接收机。

### 端到端示例

```powershell
python examples\plugins\compare_receivers.py --trials 200
```

同一 TDL-C 2×2 链路在 LS+LMMSE、LS+ZF、LS+MF、LSavg+LMMSE、Perfect+LMMSE 以及
自定义 `flat_rayleigh` 信道下的对比结果输出到 `out/rxcompare/`。

## 8. 注意事项

- 仅物理层（L1）：无 HARQ、调度或高层协议建模（那些属于 `sionna.sys`）。
- 信道估计默认 LS + 线性插值，检测默认 LMMSE（max-log）；均可在 `receiver:` 配置段按名切换。
- 吞吐量假设 CRC 通过的传输块无差错交付；时隙时长按配置的子载波间隔计算。
- 低 SNR / 高 MCS 处于 LDPC 硬失败区，BLER 需要足够的蒙特卡洛试验次数（`num_trials`）。
- UMa / UMi 为便于在同一 SNR 轴比较，关闭了路损与阴影衰落，并把频域信道归一化。

## 9. 下行 DMIMO 建模（多 TRP 预编码 + 误差）

`dmimo/` 是一个**独立、向量化、可微**的下行分布式 MIMO（DMIMO）链路模型，用于研究
"各 TRP 独立预编码，但被 TRP 间时延/校准误差吃掉相干合并增益" 的机制，并为后续
**神经网络预编码优化**提供数据与接口。

**信号模型（每子载波 k）**，K 个 TRP 协作服务单个 UE：

```
z[k] = Σ_t  g_t[k] · (w_t[k]ᴴ h_t[k]),      g_t[k] = c_t · exp(-j2π τ_t f_k)
```

- `h_t[k]`：TRP t → UE 的下行信道（每 TRP 有 `N_t` 根天线）；
- `w_t[k]`：TRP t 的预编码，**独立按自身信道**生成（MRT/ZF 基线，不感知其他 TRP/误差）；
- `g_t[k]`：TRP t 的误差标量——
  - **时延误差 τ_t**：以 TRP1 为基准（τ₁=0），其他 TRP 取 0 / 130 / 260 ns，
    表现为逐子载波线性相位斜坡 `exp(-j2π τ_t f_k)`；
  - **校准误差（幅度、相位各以 -10 dB 强度随机）**：
    `A_t = 1 + δA_t, δA_t~N(0,σ_A²)`（σ_A=10^(‑10/20)≈0.316），
    `φ_t~N(0,σ_φ²)`（σ_φ≈0.316 rad≈18°），`c_t = A_t e^{jφ_t}`；
- 接收功率 `gain=|z|²`，UE 速率 `R = mean_k log2(1 + gain/no)`。

**核心机制**：无误差时跨 TRP 相干叠加，合并增益 ≈ K² 倍单 TRP 功率；有时延/校准误差后
相位随机化 → 变为非相干叠加，增益被“吃掉”。用 `gain_loss_dB` 和速率量化。

**目录**
- `dmimo/errors.py`：时延（逐子载波相位斜坡）+ 校准（幅值/相位，-10 dB 强度随机）误差模型；
- `dmimo/channels.py`：每 TRP 下行信道生成，**可选 `simple`/`tdl`/`cdl`/`uma`/`umi`**，
  路损可开关（默认开，按最强 TRP 归一化以体现覆盖梯度）；
- `dmimo/precoding.py`：独立 MRT/ZF 基线 + **CJT 联合预编码**（相干联合传输，逐子载波 SVD）+ **Type I 码本**（双极化，wideband/subband）+ `Precoder` 接口（后续 NN 预编码器注入点）；
- `dmimo/link.py`：`DMIMODownlink` 合并多 TRP、施加误差、输出 SINR/速率/增益损耗；同时提供 `build_link`/`evaluate_precoder`/`generate_dataset`/`save_dataset`；
- `dmimo/experiment.py`：[已废弃] 仅向后兼容，新代码请使用 `dmimo.link`。

**运行示例**（打印增益损耗随 τ 的变化，并保存训练用数据集）：

```powershell
python examples\dmimo_demo.py --batch 2048 --snr 10                      # 机制对比（simple）
python examples\dmimo_demo.py --channel uma --coverage --batch 1024      # 用 UMa 信道 + 覆盖(路损)展示
```

`--channel` 可选 `simple | tdl | cdl | uma | umi`；`--coverage` 额外打印 UMa 开路损时各 TRP 的覆盖梯度。

### Type I 码本（wideband / subband）

TRP 用**双极化**阵列（`Nt=2P`），Type I 码本预编码 `w[k][:,l]=(1/√2)[v_l; φ_l[k]v_l]`：
- 波束 `v_l`（l=1..rank）由 Sionna `grid_of_beams_dft_ula(P, O)` 的 DFT GoB **wideband** 选出（
  rank≥2 用正交波束 m, m+P, m+2P, ...）；
- 共相位 `φ_l∈{1,j,-1,-j}`（QPSK）可选 **wideband**（整带一个）或 **subband**（每 subband 一个）。
- 支持 **rank 1–4**（需 UE 多天线，`num_ue_ant=rank`），多流速率用 `log2 det(I+H^H H/no)`。

```powershell
python examples\dmimo_type1.py --batch 2048 --nsc 64        # rank-1 wideband/subband
python examples\dmimo_rank_compare.py --batch 4096          # rank 1-4 x {MRT,TypeI} x {无误差,带误差}
```

实测（双极化 UMa，K=3，4 天线 UE，SNR=10dB，rank 1–4）：
| rank | MRT 无误差 | MRT 带误差 | TypeI 无误差 | TypeI 带误差 | 误差损失(MRT) | 误差损失(TypeI) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 8.96 | 8.37 | 7.45 | 7.44 | 0.59 | ≈0 |
| 2 | 16.91 | 15.79 | 12.64 | 12.66 | 1.12 | ≈0 |
| 3 | 23.93 | 22.31 | 16.60 | 16.67 | 1.62 | ≈0 |
| 4 | 29.97 | 27.79 | 19.13 | 19.19 | 2.19 | ≈0 |

结论：速率随 rank 增长（多流）；**MRT ≫ Type I**（码本量化损失随 rank 增大，rank4 差约 10.8 bps/Hz）；
**误差对 MRT 影响随 rank 增大**（0.59→2.19 bps/Hz，流越多对跨 TRP 非相干越敏感），
而 **Type I 因码本本身较粗，几乎不受所建模误差影响（≈0）**——粗量化预编码对 inter-TRP 时延/校准误差不敏感。

**CJT（相干联合传输）基线**（`CJTPrecoder`）：把 K 个 TRP 信道拼接成联合信道 `H_joint=[H_1..H_K]`，做逐子载波联合 SVD 取前 r 个右奇异向量为联合预编码，再按 TRP 分块、每 TRP 各自归一化（独立功率预算）。它体现“跨 TRP 相干联合优化”的上界：rank-1 优于独立 MRT（9.52 vs 8.94 bps/Hz），且**误差会真实吃掉 CJT 增益**。`examples/dmimo_rank_compare.py` 已含 CJT 列。

> **CJT rank≥2 布局 bug 已修复**：旧实现 `Vh[..., :r, :].permute(0,3,1,2)` 产出 `[B, M, N, r]`（与注释的 `[B, M, r, N]` 不符），随后 `reshape(B, K, Nt, r, N)` 会把子载波/层索引错位打乱——rank-1 层维为 1 不受影响，rank≥2 时有效信道变得非平滑，接收端 LS 估计完全失效（估计 MSE≈100%，链路级 BLER 锁死 1.0）。已改为 `permute(0,3,2,1)` 并**沿子载波做 SVD 相位连续化**（标准 eigen-beam 相位跟踪），rank-2 CJT 有效功率 101→196（真正超过 MRT），估计 MSE 100%→~5%，链路级 BLER 从 1.0 降到 ~0.1（完美 CSI 时 0 dB 即 BLER=0）。

### 链路级 DMIMO（BLER 曲线）`dmimo/link_level.py`

`LinkLevelDMIMO` 在 multi-TRP 下行上做**真实比特级传输**：5G NR **TB 编码（LDPC+CRC+加扰/交织/速率匹配）** → **QAM** 调制 → 逐子载波过有效信道 `H_eff[k]=Σ_t h_err_t[k]W_t[k]`（预编码用干净信道、合并用带误差信道）→ **LMMSE** 均衡 → **Demapper** → **TB 译码** → **TB CRC 检错** → **BLER/BER/吞吐**。它复用 `dmimo` 的信道/预编码/误差构件，输出的是链路级 BLER 曲线（而非系统级 Shannon 速率）。

**P1 — DMRS 类导频 + LS 信道估计**（`use_channel_estimation`）：传输采用 **Sionna 标准 PHY 积木**（对照官方教程 `MIMO_OFDM_Transmissions_over_CDL`）：`ResourceGrid`（`n_symbols` 个 OFDM 符号 × `n_subcarriers` 个子载波）+ **`KroneckerPilotPattern` 自动生成单位功率随机 QPSK 导频**（`pilot_ofdm_symbol_indices=[dmrs_symbol]` 指定前置 DMRS 专用符号，全部子载波、每流正交序列）→ `ResourceGridMapper` 放置数据+导频 → `ApplyOFDMChannel(add_awgn=True)` 过有效信道加噪 → **`LSChannelEstimator`**（完美 CSI 则直接用真实信道）→ **`LMMSEEqualizer`** → Demapper → 译码。**导频与数据分符号**：DMRS 专用符号只承载导频，其余符号全承载数据。**噪声按 Sionna 规范**：`no=10^(-snr/10)` 为每 RE 总噪声方差（SNR=1/no）。`est_density` 参数保留但已被 Kronecker 全子载波导频取代；`pilot_boost_db` 为 DMRS 能量提升。

**P2 — 真实 TB 处理 + CRC 检错**（`use_crc`，默认开）：使用 Sionna `TBEncoder/TBDecoder`（`channel_type="PDSCH"`），含 TB CRC-24A、可选 CB 分割/CRC、加扰、交织、速率匹配；**BLER 由 TB CRC 判定**（BER 仍按解码比特比对供参考）。`use_crc=false` 退回裸 LDPC + 整比特比对，便于 A/B 对比估计/检错带来的真实性差距。

```powershell
python examples\dmimo_linklevel.py --batch 512
```

实测（16-QAM，码率 1/2，K=3，UMa，单 UE；BLER-vs-SNR）：
| SNR(dB) | 单TRP MRT | 3TRP MRT相干 | 3TRP MRT+误差 | 3TRP CJT+误差 |
|---|---:|---:|---:|---:|
| -10 | 1.00 | 0.85 | 1.00 | 1.00 |
| -8 | 1.00 | 0.09 | 0.96 | 0.93 |
| -6 | 1.00 | 0.00 | 0.45 | 0.39 |
| -4 | 1.00 | 0.00 | 0.03 | 0.03 |
| -2 | 1.00 | 0.00 | 0.00 | 0.00 |

结论：3 TRP 相干把瀑布区从 ~-2 dB 推到 ~-10~-8 dB（**DMIMO 相干增益**）；**误差把它拉回 ~-6~-2 dB（吃掉约 4 dB 相干增益）**；CJT 与 MRT 在 rank-1 相干下相当，且同样被误差影响。曲线图存于 `out/dmimo/linklevel_bler.png`。

**扩展（rank 1–4、Type I 宽带码本、误差可调）**：`examples/dmimo_linklevel.py` 支持
`--rank`（1–4）、`--precoder mrt|cjt|type1|all`、`--cal-amp-error`、`--cal-pha-error`、
`--tau-ns 0,130,260`、`--granularity SC|RB|SC_RB_granular`、`--est-density`（导频密度，
`0`=完美 CSI，`>0` 开启 LS 估计）、`--no-crc`（退回裸 LDPC 整比特比对）、
`--snr-start/--snr-stop/--snr-step`（SNR 扫描范围）、`--sweep`（打印
rank 1–4 的 SNR@10% BLER 表）。Type I 用宽带码本（`subband_size=nsc`，整带一个波束+co-phasing）。

```powershell
python examples\dmimo_linklevel.py --rank 4 --precoder type1 --cal-amp-error 0.2 --sweep
```

**YAML 配置驱动**：DMIMO 也支持配置文件（`dmimo/config.py` 的 `DMIMOConfig`/`load_dmimo_config`），
示例 `configs/dmimo_linklevel.yaml` 里写全部参数（信道/天线/rank/调制/码率/误差/SNR 范围/试验数），脚本读取运行：

```powershell
python examples\dmimo_linklevel_cfg.py --config configs\dmimo_linklevel.yaml
```
**SNR 扫描用"起止+步长"三段配置**（改范围只动三个数）：`snr_start_db / snr_stop_db / snr_step_db`，
自动生成 `start, +step, ..., <=stop` 的等距点；若想要任意 SNR 点，仍可直接写 `snr_db: [..]` 列表
覆盖。5G 平台（`run_simulation.py`）同样支持该写法（另有 `--snr-start/--snr-stop/--snr-step`
命令行覆盖）。


对比**按预编码分组、组内公平**（`{预编码} 单TRP / 3TRP相干 / 3TRP+误差`），避免“拿 MRT 单 TRP 和码本多 TRP 比”的混淆。同预编码下 **3TRP 相干 ≥ 单TRP**，误差再吃掉一部分增益（如 rank-2 时 MRT 10% 点：单TRP −11 dB、3TRP 相干 −21 dB、3TRP+误差 −19 dB）。

SNR@10% BLER（链路级，16-QAM，rate 1/2，K=3，UMa）：
| rank | MRT 相干 | MRT+误差 | CJT+误差 | TypeI-宽带+误差 |
|---|---:|---:|---:|---:|
| 1 | −6.47 | −2.79 | −2.97 | 5.47 |
| 2 | −3.83 | −2.22 | 4.65 | 9.30 |
| 3 | −2.48 | −0.16 | 1.89 | —(>10dB) |
| 4 | −0.25 | 1.55 | 7.52 | —(>10dB) |

> 注：上表 rank≥2 的 CJT 数值受**已修复的预编码布局 bug**（见上方 `CJTPrecoder` 说明）影响偏悲观；
> 修复后 CJT 有效功率超过 MRT（64 天线 rank-2 场景 ~196 vs ~176），rank-1 不受影响。

解读：误差吃掉约 3.7 dB 相干增益（rank-1）；rank 升高时“整 TB”BLER 在同等每层 SNR 下更易出错
（TB 符号数随 rank 增长），故 SNR@10% 上移；Type I 宽带码本在高 rank 明显弱于连续预编码。

**CJT/Type I 修复后实测（64 天线、rank-2、16QAM、rate 1/2、UMa、Sionna 标准管线 LS、TB-CRC；`configs/dmimo_linklevel.yaml`）**：
修复前 CJT 因布局 bug + 手写估计插值残差在估计下 BLER 锁死 ~1.0；改用 Sionna 积木
（Kronecker 全子载波导频 + `LSChannelEstimator` + `LMMSEEqualizer`）后 LS 与完美 CSI 收敛一致，
无插值底板，10 dB 下 CJT/Type I 均 ~0.04–0.12（完美 CSI 时 0 dB 即 BLER=0）：

| SNR(dB) | CJT 单TRP | CJT 3TRP相干 | CJT 3TRP+误差 | TypeI 3TRP相干 | TypeI 3TRP+误差 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.05 | 0.09 | 0.11 | 0.61 | 0.64 |
| 6 | 0.10 | 0.11 | 0.09 | 0.06 | 0.06 |
| 10 | 0.06 | 0.06 | 0.13 | 0.04 | 0.04 |

高 SNR 底板 ~0.2 来自 **LS 估计残差**（每流锚点间距 6 个子载波的线性插值，CJT/Type I 有效信道
变化快于 MRT）；`est_density=0.5`（每流 4 SC）可把底板降到 ~0.1。

**估计/检错带来的真实性差距**：开启 P1（`use_channel_estimation`）后，导频开销在**时域**（DMRS 专用符号，Kronecker 全子载波导频），数据子载波不损失；
低 SNR 下 LS 估计噪声抬高瀑布区（相对完美 CSI 约 1–2 dB），高 SNR 下 LS 与完美 CSI 收敛一致——
**无插值残差底板**（DMRS 符号每个子载波都是导频，逐子载波估计），CJT/Type I 不再有之前手写
估计时的 ~0.2 高 SNR 底板。开启 P2 后 BLER 由 TB CRC 判定（含 CRC 开销与加扰/交织），与
整比特比对在瀑布区基本一致。当前模型仍是“有效信道层面”的链路仿真：**尚无完整 OFDM 波形
（无 IFFT/CP）、无 HARQ、无 PMI/CSI 反馈**——这些是后续真实性补齐点。

**Type I 在 rank-2/64 天线场景"偏弱"是真实码本限制，不是 bug**：单宽带 DFT 波束在 NLOS UMa
富散射信道上只捕获一个主方向（有效功率 ~5 vs MRT ~175，约 15 dB 差距），这是单波束码本的
量化/方向限制；加上 LS 估计残差，10 dB 下 BLER ≈0.2（完美 CSI 时 10 dB 可到 0）。

示例输出（SNR=10 dB，K=3，4 天线/TRP，64 子载波，Δf=30 kHz）：

| 配置 | rate(bps/Hz) | 合并增益(dB) | 相对相干增益损耗(dB) |
|---|---:|---:|---:|
| 单 TRP | 5.18 | 6.02 | 0 |
| 3 TRP 无误差（相干） | 8.38 | 15.39 | 0 |
| 3 TRP 时延 0/130/260 ns | 8.18 | 14.82 | 0.57 |
| 3 TRP 时延 0/390/520 ns | 7.21 | 12.91 | 2.48 |

> 说明：时延损耗随子载波数/带宽或 τ 增大而增大；上面的数值是默认 64 子载波 @30 kHz 的结果。
> 误差口径：预编码用**干净信道**、合并用**带误差信道**；校准误差按参考实现逐 TX 天线（TRP0 干净），
> 强度经 `cal_amp_error` / `cal_pha_error`（线性方差，`None`=无误差）调整，时延粒度经 `granularity`。

**为 NN 预编码优化预留的接口**：自定义预编码器只需满足
`precoder(h: [B,K,N_t,N]) -> w: [B,K,N_t,N]`，即可用 `dmimo.experiment.evaluate_precoder`
对比其速率 vs 独立 MRT 基线；数据集已包含 (信道, 误差, 基线指标)，供离线训练。


