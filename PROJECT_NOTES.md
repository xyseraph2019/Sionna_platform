# 项目开发笔记 / Lessons Learned

> 这份文件用于记录本项目开发过程中总结的经验和约定。
> 以后无论是人还是 AI 继续维护这个仓库，都应该先读一下这里。

## 1. 工作环境限制

- 当前对话环境中，AI 无法执行终端命令（Windows 下无 shell 通道）。
- AI 能做的事：
  - 阅读代码
  - 修改代码
  - 给出运行/调试/重构方案
- 需要用户在本机运行并回传报错。

## 2. 项目结构约定

- `D:\Platform` 是项目根目录。
- `sionna5g/`：通用 5G PUSCH 链路级平台。
- `dmimo/`：DMIMO 下行研究模块。
- `configs/`：YAML 场景配置。
- `examples/`：可运行示例。
- `out/`：运行输出，默认不纳入版本管理。

## 3. 配置原则

- 每个配置参数必须明确“给谁用”。
- 不允许“定义了但没接入”的死参数。
- 新参数需要同步更新：
  - `dmimo/config.py` 或 `sionna5g/config.py`
  - 对应 YAML
  - `CONFIG_REFERENCE.md`
  - 使用该参数的脚本

## 4. 仿真公平性原则

- 对比多个 PMI/预编码器时，必须保证：
  - 同一 SNR 下使用同一份信道
  - 同一份信息比特
  - 同一份噪声/误差
- 推荐使用 `LinkLevelDMIMO.evaluate_many(..., on_batch=...)` 的结构：
  - 每个 SNR 只采样一次 channel + bits
  - 所有 precoder 共用这份数据
  - 可多 batch 累积，例如 `num_mc_batches`

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
  - 不同 SNR
- 统一 `dmimo_linklevel.py` 和 `dmimo_linklevel_cfg.py` 的进度/保存逻辑。
- 将多 PMI 对比抽成公共模块，避免两个脚本重复实现。
