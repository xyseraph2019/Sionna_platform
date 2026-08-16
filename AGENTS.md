# AI 协作规则（AGENTS.md）

> 本文件是给 AI 助手 / 后续维护者看的项目级规则。
> 在修改本仓库代码前，请先阅读本文件以及 `README.md`、`CONFIG_REFERENCE.md`、`PROJECT_NOTES.md`。

## 1. 工作方式

- 修改代码前先确认项目结构和当前入口。
- 改动尽量小步、可回滚，避免一次性大重构。
- 不删除当前可运行的功能；如果必须废弃，保留向后兼容 shim 并标记 deprecated。

## 2. 配置规则

- 每个配置参数必须明确“给谁用”。
- 不允许加入“定义了但没接入”的死参数。
- 新增/修改配置时必须同步：
  - 对应 `config.py`
  - 对应 YAML
  - `CONFIG_REFERENCE.md`
  - 使用该参数的脚本/文档

## 3. 仿真公平性

- 对比多个 PMI / 预编码器时，同一 SNR 下必须使用同一份：
  - 信道
  - 信息比特
  - 噪声/误差
- 优先使用 `LinkLevelDMIMO.evaluate_many()`，避免每个 precoder 各自独立随机采样。

## 4. 输出与可观测性

- 长任务必须提供进度：
  - 当前 SNR / 总 SNR
  - 当前 batch / 总 batch
  - 已运行时间 / ETA
  - 当前平均 BLER、sum rate、post-SNR
- 运行结果必须保存：
  - 图（PNG）
  - 数值（CSV）
  - 场景元信息（JSON）

## 5. 文件管理

- 不修改 `out/` 下的已有结果，除非用户明确要求。
- 生成物不入 Git：
  - `out/`
  - `*.whl`
  - `*.pt`
  - `*.pth`
  - 日志文件
- 每次修改后，在回复中列出改动文件清单。

## 6. 信道模型约定

- `channel_kind` 支持：
  - `simple`：快速验证用 i.i.d. Rayleigh
  - `tdl`：Sionna TDL
  - `cdl`：Sionna CDL
  - `uma` / `umi`：Sionna 系统级信道
- CDL/TDL 模型通过 `cdl_model` / `tdl_model` 配置。
- UMa/UMi 默认关闭路损/阴影，便于同一 SNR 轴比较；如需覆盖差异，用 `pathloss: true`。

## 7. 版本管理

- 遵循现有 Git 工作流。
- 每个逻辑改动尽量独立提交。
- 提交信息建议：
  - `feat:` 新功能
  - `fix:` 修复
  - `refactor:` 重构
  - `docs:` 文档
  - `test:` 测试
