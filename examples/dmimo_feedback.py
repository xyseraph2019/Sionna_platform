"""CSI 反馈量化扫描：速率（和可选 BLER）vs 反馈比特预算。

对连续预编码（MRT / CJT / NN-PMI）施加有限比特反馈量化（phase 或 iq），
扫描不同比特预算，与理想 CSI 上界（完美 MRT/CJT）和 Type I 码本基线在
同一共享信道/误差框架下对比（``evaluate_precoder`` / ``evaluate_many`` 语义，
保证公平性）。

Run::

    python examples\\dmimo_feedback.py                        # phase 量化 2/3/4/6/8 bit
    python examples\\dmimo_feedback.py --quant iq --bits 2,3,4
    python examples\\dmimo_feedback.py --bler                  # 额外跑链路级 BLER（3TRP+err @ --snr-db）

输出到 ``out/dmimo/fb/<tag>_<quant>/``：rate_*.csv / .json / .png，可选 bler_*.csv。
"""
import argparse
import json
import os
import sys

import common  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dmimo.config import load_dmimo_config, scenario_tag  # noqa: E402
from dmimo import (  # noqa: E402
    build_link,
    IndependentMRT,
    CJTPrecoder,
    TypeICodebook,
    QuantizedFeedback,
    PhaseQuantizer,
    ScalarQuantizer,
)


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.path.join("configs", "dmimo_linklevel.yaml"))
    p.add_argument("--quant", default="phase", help="phase | iq")
    p.add_argument("--bits", default="2,3,4,6,8", help="比特预算列表（逗号分隔）")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--snr-db", type=float, default=0.0, help="速率评估 SNR (dB)")
    p.add_argument("--bler", action="store_true", help="额外跑链路级 BLER（3TRP+err @ --snr-db）")
    p.add_argument("--out", default=None, help="输出目录（默认 out/dmimo/fb/<tag>_<quant>）")
    a = p.parse_args()

    c = load_dmimo_config(a.config)
    torch.manual_seed(c.seed)
    dev = "cuda:0" if torch.cuda.is_available() and c.device in ("auto", "cuda:0") else "cpu"
    bits_list = [int(b) for b in a.bits.split(",") if b.strip()]
    if not bits_list:
        print("no bits given")
        return 1

    def mk(num_trps, err):
        base = dict(num_tx_ant=c.num_tx_ant, num_ue_ant=c.num_ue_ant,
                    n_subcarriers=c.n_subcarriers,
                    subcarrier_spacing=c.subcarrier_spacing_khz * 1e3,
                    channel_kind=c.channel_kind, pathloss=c.pathloss,
                    trp_distances=[c.trp_distances_m[0]] if num_trps == 1 else list(c.trp_distances_m),
                    granularity=c.granularity, carrier_frequency=c.carrier_frequency,
                    cdl_model=c.cdl_model, tdl_model=c.tdl_model)
        if err and num_trps > 1:
            base.update(tau_seconds=c.tau_seconds, cal_amp_error=c.cal_amp_error,
                        cal_pha_error=c.cal_pha_error)
        else:
            base.update(tau_seconds=[0.0] * num_trps, cal_amp_error=None, cal_pha_error=None)
        return build_link(num_trps=num_trps, **base)

    l0 = mk(3, False)   # 3TRP 相干（无误差）
    le = mk(3, True)    # 3TRP + 误差

    # ---- 连续预编码（将被量化）：UE 把理想预编码经 b bit 反馈回 BS ----
    base_factories = {
        "MRT": lambda: IndependentMRT(rank=c.rank),
        "CJT": lambda: CJTPrecoder(rank=c.rank, subband_size=c.subband_size),
    }

    # ---- 参考线（不量化）----
    # NN-PMI 在 BS 本地由反馈回来的 Type I PMI 生成，输出不需要回传，因此不套
    # 输出量化器（输出量化只对应 fronthaul 系数信令 / 定点精度场景，见
    # dmimo/feedback.py 的语义说明）。这里把 NN-PMI 作为不量化的参考线参与对比。
    refs = {
        "MRT-perfect": lambda: IndependentMRT(rank=c.rank),
        "CJT-perfect": lambda: CJTPrecoder(rank=c.rank, subband_size=c.subband_size),
        "TypeI-wide": lambda: TypeICodebook(rank=c.rank, subband_size=c.n_subcarriers),
    }
    ckpt = auto_ckpt_path(c)
    if os.path.exists(ckpt):
        from dmimo import load_model

        _nn, _meta = load_model(ckpt, device=dev)
        refs["NN-PMI"] = lambda: _nn
        print(f"  loaded NN-PMI (reference, no output quantization): {os.path.basename(ckpt)}")
    else:
        print(f"  [warn] NN-PMI checkpoint not found: {ckpt} -> skip NN-PMI")

    fb_sub = c.feedback_subband_size or c.subband_size

    def make_quantizer(b):
        if a.quant == "phase":
            return PhaseQuantizer(bits_phase=b)
        return ScalarQuantizer(bits=b)

    def rate_shared(link, pcs):
        """同一份信道/误差/噪声上评估所有 precoder（evaluate_many 的速率版）。"""
        h = link.channel.sample(a.batch, dev)
        h_err = link.error.apply(h)
        no = 10.0 ** (-a.snr_db / 10.0)
        out = {}
        for n, pc in pcs.items():
            w = pc(h_err) if getattr(pc, "precodes_from_errors", False) else pc(h)
            out[n] = float(link._rate(link._combine(h_err, w), no).mean().item())
        return out

    # ================= 系统级速率 vs 比特 =================
    print(f"== CSI 反馈量化扫描（{a.quant}, SNR={a.snr_db} dB, batch={a.batch}, "
          f"K={c.num_trps}, rank={c.rank}, subband={fb_sub}）==")
    rows = []
    for b in bits_list:
        qz = make_quantizer(b)
        qpcs = {n: QuantizedFeedback(f(), qz, subband_size=fb_sub)
                for n, f in base_factories.items()}
        fb_bits = qpcs[next(iter(qpcs))].feedback_bits(
            c.num_trps, c.num_tx_ant, c.rank, c.n_subcarriers)
        row = {"bits": b, "fb_bits": fb_bits}
        row.update({f"{n}_coh": v for n, v in rate_shared(l0, qpcs).items()})
        row.update({f"{n}_err": v for n, v in rate_shared(le, qpcs).items()})
        rows.append(row)
        print(f"  bits={b:>2d}  fb={fb_bits:>6d} bit/slot  "
              + "  ".join(f"{n}={row[n+'_err']:.2f}(err)/{row[n+'_coh']:.2f}(coh)"
                          for n in qpcs))

    ref_pcs = {n: f() for n, f in refs.items()}   # 参考线实例（不量化）
    ref_rates = {"coh": rate_shared(l0, ref_pcs), "err": rate_shared(le, ref_pcs)}
    print("  参考（不量化）: " + "  ".join(
        f"{n}={ref_rates['err'][n]:.2f}(err)/{ref_rates['coh'][n]:.2f}(coh)"
        for n in refs))

    # ---- 保存 ----
    tag = scenario_tag(c.num_trps, c.rank, c.n_subcarriers, c.qam_order,
                       c.code_rate, c.channel_kind, est=c.use_channel_estimation,
                       num_dmrs_symbols=c.num_dmrs_symbols,
                       err=c.cal_amp_error is not None or c.cal_pha_error is not None,
                       subband_size=c.subband_size, feedback=a.quant)
    out_dir = a.out or os.path.join(common.ROOT, "out", "dmimo", "fb", f"{tag}")
    os.makedirs(out_dir, exist_ok=True)

    import csv

    rate_csv = os.path.join(out_dir, f"rate_{tag}.csv")
    with open(rate_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["bits", "fb_bits"] + [f"{n}_{s}" for n in base_factories for s in ("coh", "err")]
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get("bits"), r.get("fb_bits")] + [r.get(col) for col in cols[2:]])
    print("Saved ->", rate_csv)

    # ---- 画图：两个面板（相干 / +误差）----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, suf, title in ((axes[0], "coh", "3TRP coherent"),
                           (axes[1], "err", "3TRP + timing/cal errors")):
        for n in base_factories:
            ax.plot([r["bits"] for r in rows], [r[f"{n}_{suf}"] for r in rows],
                    "o-", label=f"{n} ({a.quant})")
        for n in refs:
            label = n.replace("-perfect", " (perfect)")
            ax.axhline(ref_rates[suf][n], ls="--", lw=1.2, label=label)
        ax.set_xlabel("feedback bits per coefficient")
        ax.set_ylabel("rate (bps/Hz)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"CSI feedback quantization: rate vs bits ({tag})")
    fig.tight_layout()
    rate_png = os.path.join(out_dir, f"rate_{tag}.png")
    fig.savefig(rate_png, dpi=150)
    print("Saved ->", rate_png)

    rate_json = os.path.join(out_dir, f"rate_{tag}.json")
    with open(rate_json, "w", encoding="utf-8") as fh:
        json.dump({"scenario_tag": tag, "quant": a.quant, "snr_db": a.snr_db,
                   "batch": a.batch, "subband_size": fb_sub,
                   "rows": rows, "references": ref_rates}, fh, indent=2, ensure_ascii=False)
    print("Saved ->", rate_json)

    # ================= 链路级 BLER（可选）=================
    if a.bler:
        from dmimo import LinkLevelDMIMO

        ll = LinkLevelDMIMO(le, qam_order=c.qam_order, code_rate=c.code_rate,
                            rank=c.rank, use_channel_estimation=c.use_channel_estimation,
                            est_density=c.est_density, use_crc=c.use_crc,
                            pilot_boost_db=c.pilot_boost_db, n_symbols=c.n_symbols,
                            dmrs_symbol=c.dmrs_symbol,
                            num_dmrs_symbols=c.num_dmrs_symbols, device=dev)
        print(f"\n== 链路级 BLER @ {a.snr_db} dB（evaluate_many 共享信道/比特）==")
        bler_rows = []
        for b in bits_list:
            qz = make_quantizer(b)
            qpcs = {n: QuantizedFeedback(f(), qz, subband_size=fb_sub)
                    for n, f in base_factories.items()}
            res = ll.evaluate_many(a.batch, a.snr_db, qpcs)
            row = {"bits": b}
            for n in qpcs:
                row[n] = res[n][0]
            bler_rows.append(row)
            print(f"  bits={b:>2d}  " + "  ".join(f"{n}={row[n]:.4f}" for n in qpcs))
        ref_res = ll.evaluate_many(a.batch, a.snr_db, ref_pcs)
        print("  参考: " + "  ".join(f"{n}={ref_res[n][0]:.4f}" for n in refs))

        bler_csv = os.path.join(out_dir, f"bler_{tag}.csv")
        with open(bler_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["bits"] + list(base_factories))
            for r in bler_rows:
                w.writerow([r["bits"]] + [r[n] for n in base_factories])
        print("Saved ->", bler_csv)
        bler_json = os.path.join(out_dir, f"bler_{tag}.json")
        with open(bler_json, "w", encoding="utf-8") as fh:
            json.dump({"scenario_tag": tag, "quant": a.quant, "snr_db": a.snr_db,
                       "rows": bler_rows, "references": {k: v[0] for k, v in ref_res.items()}},
                      fh, indent=2, ensure_ascii=False)
        print("Saved ->", bler_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
