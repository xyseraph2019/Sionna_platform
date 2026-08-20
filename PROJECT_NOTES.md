# 项目开发笔记 / Lessons Learned

> 这份文件用于记录本项目开发过程中总结的经验和约定。
> 以后无论是人还是 AI 继续维护这个仓库，都应该先读一下这里。

## 1. 工作环境限制

- 早期会话中 AI 无法执行终端命令（Windows 下无 shell 通道），只能读改代码、
  由用户本机运行回传报错。
- **更新**：后续会话已具备 pwsh 通道，可运行 Python（项目运行环境为 conda
  `pytorch2`：`C:\Users\xyseraph\anaconda3\envs\pytorch2\python.exe`，
  Python 3.11 + torch 2.13.0+cu126 + sionna 2.0.1，有 CUDA）。验证用 `--device cpu`，
  正式跑仿真用 CUDA。

## 2. 项目结构约定

- `D:\Platform` 是项目根目录。
- `dmimo/`：DMIMO 链路级模型包（DLModel / ULModel，Sionna Block 风格）。
- `configs/`：YAML 场景配置（dmimo_linklevel.yaml / dmimo_ul_linklevel.yaml）。
- `examples/`：可运行示例（三个链路级驱动）。
- `tutorials/`：Sionna 官方教程 Notebook（phy/sys，参考用）。
- `out/`：运行输出，默认不纳入版本管理。

## 3. 配置原则

- 每个配置参数必须明确“给谁用”。
- 不允许“定义了但没接入”的死参数。
- 新参数需要同步更新：
  - `dmimo/config.py`
  - 对应 YAML
  - `CONFIG_REFERENCE.md`
  - 使用该参数的脚本

## 4. 仿真公平性原则

- 对比多个 PMI/预编码器时，必须保证：
  - 同一 SNR 下使用同一份信道
  - 同一份信息比特
  - 同一份噪声/误差
- 推荐使用 `dmimo.sim.sim_ber_many(model, ebno_db, configs, ...)` 的结构：
  - 每个 MC 批只采样一次 channel + bits（`model.sample_realization`）
  - 所有 precoder / combiner 配置共用这份数据（`model.block_from_realization`）
  - `on_batch` 回调提供逐批进度；target-BLER 提前停止（sim_ber 规则）
  - 每 SNR 点可多批累积（`num_mc_batches`）

## 5. 进度与结果输出

- 长任务必须显示：
  - 当前 SNR / 总 SNR
  - 当前 batch / 总 batch
  - 已运行时间 / ETA
  - 当前平均 BLER、sum rate、post-SNR
- 结果不能只画图，必须保存：
  - PNG 图
  - CSV（逐 SNR 曲线）
  - JSON（场景元信息 + 曲线）

## 6. 信道模型使用约定

- `channel_kind` 支持：
  - `simple`：自研 i.i.d. Rayleigh，快速验证
  - `tdl`：Sionna TDL
  - `cdl`：Sionna CDL
  - `uma` / `umi`：Sionna 系统级信道
- CDL/TDL 模型通过 `cdl_model` / `tdl_model` 选择：
  - CDL-A/B/C 通常为 NLOS
  - CDL-D/E 通常为 LOS
- UMa/UMi 默认关闭路损和阴影，便于同一 SNR 轴比较。

## 7. 已修复的坑

- `dmimo/channels.py`
  - CDL/TDL 的 `h` 维度顺序必须是 `[B, K, D, Nt, N]`
  - `_normalize_coverage()` 求每个 TRP 平均功率时维度应保留 K
- `dmimo/channels.py`
  - `SionnaDownlinkChannel` 中 `rg` 应改为 `self._rg`
- 多 PMI 对比不要各自独立随机采样，必须共享 channel/bits。

## 8. 版本管理

- 项目使用 Git 管理，远程为 GitHub。
- `out/`、`*.whl`、`*.pt` 等生成物不入库。
- 重构建议小步提交，不要一次性大改。

## 9. 后续维护建议

- 增加自动化冒烟测试，至少覆盖：
  - simple / tdl / cdl / uma / umi
  - rank 1 / 2 / 3 / 4
  - 有误差 / 无误差
  - 不同 Eb/N0
- 进度/保存逻辑已统一到 `dmimo/sim.py`（sim_ber_many）与 `dmimo/results.py`
  （save_curves：PNG + CSV + JSON，Eb/N0 与 SNR 双轴），后续示例只负责组装配置。
- 多 PMI 对比的公平采样已内置在 `sim_ber_many`，新脚本直接复用即可。

## 10. 链路级重构记录（Sionna Block 风格，2026）

以示例 Notebook `tutorials/phy/MIMO_OFDM_Transmissions_over_CDL.ipynb` 的结构
重构 UL/DL DMIMO 链路级代码：

- 新增 `dmimo/model.py`：`DMIMOPhyModel(Block)` 基类 + `DLModel` / `ULModel`。
  - 入口 `call(batch_size, ebno_db) -> (b, b_hat)`（教程风格）；
  - 噪声用 `ebnodb2no`（Eb/N0 轴，计入码率/调制/导频/置零开销）；
  - 信道 CIR 驱动：`dmimo/channels.py::DMIMOChannel` 逐 OFDM 符号采样
    CDL/TDL CIR → `cir_to_ofdm_channel`（`simple` 为频域平坦特例）；
    `domain="time"` 为预留扩展点（`cir_to_time_channel` + 调制/解调）；
  - DL：每 TRP 频域预编码（MRT/CJT/TypeI/NN-PMI/量化反馈），UE 相干合并；
    UL：K TRP 接收，joint / symbol / llr 三层合并，`estimate_errors` 语义保留。
- 新增 `dmimo/sim.py`：`sim_ber_many`（公平共享 + 提前停止 + 进度）、`sim_ber_curve`。
- 新增 `dmimo/results.py`：`save_curves` / `print_curve_table`。
- 删除旧 `dmimo/link_level.py` / `dmimo/uplink_level.py`（直接替换，调用点已同步更新）。
- 系统级 rate 模型（`dmimo/link.py` / `dmimo/uplink.py`）保持不变。
- 注意事项：
  - `ResourceGrid` 的自定义导频模式在 guard/DC 下与 `num_data_symbols` 计数不一致
    （Sionna 2.0.1），故 rank>=2 的交错梳状导频仅用于无 guard/DC 的旧配置，
    教程风格配置（guard/DC）一律用 Kronecker 导频（计数一致）。
  - `Block` 自带只读 `device` 属性，模型内部设备存 `_device`。
  - 误差模型 `TRPErrorModel` 的 `tau_seconds` 长度必须等于 `num_trps`（已校验）。
