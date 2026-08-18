"""NN-PMI 模型复杂度测试：当前场景配置下的 MACs / FLOPs / 参数量。

在当前场景配置（默认 ``configs/dmimo_linklevel.yaml``）下，按与训练/推理
完全一致的实例化口径构建 MLP-Mixer subband-PMI 网络（有 checkpoint 时直接
加载真实架构），统计：

  * 网络核心（``MLPMixerSubbandPMI``）的 MACs / FLOPs / 参数量（hook 实测）；
  * ``NNMixerPMI`` 封装全链路（含 Type I wideband-PMI 特征提取）的估算 FLOPs；
  * 按 batch 换算，以及一次完整 BLER 评估（全部 SNR x 链路 x MC batch）里
    NN-PMI 的总计算量。

Run::

    python examples\\model_complexity.py
    python examples\\model_complexity.py --batch 128 --config configs\\dmimo_linklevel.yaml
"""
import argparse
import json
import math
import os
import sys

import common  # noqa: E402
import torch

from dmimo.config import load_dmimo_config, scenario_tag
from dmimo.model_complexity import count_macs, human_count
from dmimo import NNMixerPMI, load_model

# 与 examples/modelTrain.py 的架构常量保持一致（无 checkpoint 时使用）
MIXER_BLOCKS = 2
MIXER_HIDDEN = 64
OVERSMPL = 4


def auto_ckpt_path(c) -> str:
    """与 examples/dmimo_linklevel_cfg.py 相同的自动 checkpoint 路径规则。"""
    tag = scenario_tag(
        c.num_trps, c.rank, c.n_subcarriers, c.qam_order, c.code_rate,
        c.channel_kind,
        est=c.use_channel_estimation,
        num_dmrs_symbols=c.num_dmrs_symbols,
        err=c.cal_amp_error is not None or c.cal_pha_error is not None,
        subband_size=c.subband_size,
    )
    return os.path.join(common.ROOT, "out", "dmimo", "model", f"nn_pmi_mixer_{tag}.pt")


def frontend_macs_per_sample(c, oversmpl=OVERSMPL) -> int:
    """Type I wideband-PMI 特征提取的解析 MAC 估算（不含网络）。

    ``wideband_pmi`` 的主成本是两个 einsum（两个极化组分别投影到 DFT 波束
    格）::

        pp = einsum("mp,bkdpn->bkdmn", V.conj(), hp)   # [M,P] x [B,K,D,P,N]
        pm = einsum("mp,bkdpn->bkdmn", V.conj(), hm)

    每个 einsum 的乘加数 = B * K * D * M * N * P，其中 P = Nt/2, M = P*O。
    beam_power / argmax / 拼接等其余运算 < 1%，忽略。
    """
    P = c.num_tx_ant // 2
    M = P * oversmpl
    per_batch = 2 * c.num_trps * c.num_ue_ant * M * c.n_subcarriers * P
    return per_batch  # batch=1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.path.join("configs", "dmimo_linklevel.yaml"))
    p.add_argument("--batch", type=int, default=128,
                   help="batch size for the measurement / reporting")
    p.add_argument("--device", default=None, help="cpu (default) / cuda:0")
    p.add_argument("--out", default=None,
                   help="JSON 输出路径（默认 out/dmimo/model/complexity_<tag>.json）")
    a = p.parse_args()

    c = load_dmimo_config(a.config)
    dev = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    num_subbands = int(math.ceil(c.n_subcarriers / c.subband_size))
    ckpt = auto_ckpt_path(c)

    # ---- 构建/加载模型（与 dmimo_linklevel_cfg.py 同口径）--------------------
    if os.path.exists(ckpt):
        model, meta = load_model(ckpt, device=dev)
        src = f"checkpoint {os.path.basename(ckpt)}"
        blocks, hidden = model.net.blocks, model.net.hidden
    else:
        model = NNMixerPMI(rank=c.rank, oversmpl=OVERSMPL,
                           subband_size=c.subband_size,
                           n_subcarriers=c.n_subcarriers,
                           num_ant=c.num_tx_ant, num_trps=c.num_trps,
                           blocks=MIXER_BLOCKS, hidden=MIXER_HIDDEN, device=dev)
        src = "default arch (blocks=2, hidden=64, no checkpoint found)"
        blocks, hidden = MIXER_BLOCKS, MIXER_HIDDEN
        meta = {}

    P2 = c.num_tx_ant                      # 2P 天线
    B = int(a.batch)

    # 网络输入 = Type I wideband PMI [B, K, 2P, rank]（复数）
    w_wb = torch.randn(B, c.num_trps, P2, c.rank, dtype=torch.complex64, device=dev)

    rep = count_macs(model.net, w_wb, batch_size=B, device=dev, unit="M")
    fe_macs = frontend_macs_per_sample(c)              # 每样本（batch=1）
    wrapper_macs = rep.macs + fe_macs                  # 全链路估算

    # 一次完整 BLER 评估（dmimo_linklevel_cfg 口径）里 NN-PMI 的总 FLOPs：
    # 每个 SNR 点 x 3 条链路（单TRP/3TRP相干/3TRP+err）x num_mc_batches 个 batch
    snrs = c.snr_grid
    nn_calls = len(snrs) * 3 * c.num_mc_batches * B
    run_flops = wrapper_macs * 2 * nn_calls

    # 统一以 M 为单位显示（M MACs / M FLOPs）
    M = "M"
    h = lambda v: human_count(v, M)  # noqa: E731

    tag = scenario_tag(c.num_trps, c.rank, c.n_subcarriers, c.qam_order,
                       c.code_rate, c.channel_kind, est=c.use_channel_estimation,
                       num_dmrs_symbols=c.num_dmrs_symbols,
                       err=c.cal_amp_error is not None or c.cal_pha_error is not None,
                       subband_size=c.subband_size)

    print(f"== NN-PMI 模型复杂度（scenario: {tag}）==")
    print(f"  模型来源: {src}")
    print(f"  架构: num_ant(2P)={P2} rank={c.rank} K={c.num_trps} "
          f"S={num_subbands} (n_sc={c.n_subcarriers}, subband={c.subband_size}) "
          f"blocks={blocks} hidden={hidden}")
    print(f"  device={dev}  batch={B}\n")

    print("-- 网络核心 MLPMixerSubbandPMI（hook 实测）--")
    print(rep)
    print("  逐层 MACs:")
    for name, macs in rep.per_module:
        print(f"    {name:<28} {h(macs // max(B, 1))} MACs/sample"
              f" = {h(2 * (macs // max(B, 1)))} FLOPs/sample")

    print("\n-- NNMixerPMI 封装全链路（网络 + wideband-PMI 特征提取）--")
    print(f"  wideband-PMI 特征提取: {h(fe_macs)} MACs/sample "
          f"(解析估算)")
    print(f"  全链路: {h(wrapper_macs)} MACs/sample = "
          f"{h(wrapper_macs * 2)} FLOPs/sample")
    print(f"  batch={B}: {h(wrapper_macs * B * 2)} FLOPs")

    print(f"\n-- 一次完整 BLER 评估的 NN-PMI 总计算量 --")
    print(f"  SNR 点 {len(snrs)} x 3 链路 x {c.num_mc_batches} MC-batch "
          f"x batch {B} = {nn_calls:,} 次 NN 前向")
    print(f"  总 FLOPs = {h(run_flops)}")

    report = {
        "scenario_tag": tag,
        "model_source": src,
        "arch": dict(num_tx_ant=c.num_tx_ant, rank=c.rank, num_trps=c.num_trps,
                     n_subcarriers=c.n_subcarriers, subband_size=c.subband_size,
                     num_subbands=num_subbands, blocks=blocks, hidden=hidden),
        "batch_size": B,
        "unit": "M",   # 以下 *_m 字段均以 M（百万）为单位
        "net": rep.as_dict(),
        "wideband_pmi_macs_per_sample": fe_macs,
        "wrapper_macs_per_sample": wrapper_macs,
        "wrapper_flops_per_sample": wrapper_macs * 2,
        "full_bler_run": {
            "nn_forward_calls": nn_calls,
            "total_flops": run_flops,
        },
        "summary_m": {
            "net_macs_m": rep.macs / 1e6,
            "net_flops_m": rep.flops / 1e6,
            "net_flops_batch_m": rep.flops_batch / 1e6,
            "wideband_pmi_macs_m": fe_macs / 1e6,
            "wrapper_macs_m": wrapper_macs / 1e6,
            "wrapper_flops_m": wrapper_macs * 2 / 1e6,
            "wrapper_flops_batch_m": wrapper_macs * 2 * B / 1e6,
            "full_bler_run_flops_m": run_flops / 1e6,
        },
        "checkpoint_meta": {k: v for k, v in meta.items()
                            if not isinstance(v, (list, dict))},
    }

    out = a.out or os.path.join(common.ROOT, "out", "dmimo", "model",
                                f"complexity_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
