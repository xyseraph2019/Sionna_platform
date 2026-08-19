"""上行 DMIMO 链路级（BLER）扫描：三层合并 × 误差处理假设。

单 UE 发射真实比特（5G NR TB：LDPC+CRC+QAM），K 个 TRP 联合接收，
逐 SNR 出 BLER 曲线：

* 合并层级：joint（L1 联合检测）/ symbol（L2 均衡后符号合并）/ llr（L3 LLR 合并）；
* 误差处理：estimate_errors=True（本地估计吸收接收误差）与 False（未补偿）；
* 同一 SNR 点所有配置共享同一份信道/比特/噪声（evaluate 公平性）。

Run::

    python examples\\udmimo_linklevel.py --channel simple --batch 128
    python examples\\udmimo_linklevel.py --channel uma --nsc 240 --snr-start -8 --snr-stop 8
    python examples\\udmimo_linklevel.py --combiner llr --no-crc
"""
import argparse
import csv
import json
import os
import sys
import time

import common  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dmimo import build_ulink, ULinkLevelDMIMO  # noqa: E402


def snr_range(start, stop, step):
    n = int((stop - start) / step + 1e-9) + 1
    return [round(start + i * step, 6) for i in range(max(n, 1))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", default="simple", help="simple|tdl|cdl|uma|umi")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--trps", type=int, default=3)
    p.add_argument("--num-rx-ant", type=int, default=4)
    p.add_argument("--num-ue-ant", type=int, default=1)
    p.add_argument("--nsc", type=int, default=64)
    p.add_argument("--tau-ns", default="0,130,260")
    p.add_argument("--cal-amp-error", type=float, default=0.1)
    p.add_argument("--cal-pha-error", type=float, default=0.1)
    p.add_argument("--qam", type=int, default=16)
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--combiner", default="all", help="all|joint|symbol|llr")
    p.add_argument("--est", default="both", help="true|false|both")
    p.add_argument("--no-crc", action="store_true")
    p.add_argument("--tx-mode", default="equal", help="equal|random")
    p.add_argument("--est-density", type=float, default=0.0,
                   help="DMRS 导频密度 (0,1]；0=perfect CSI（est 用理想带误差信道）")
    p.add_argument("--pilot-boost-db", type=float, default=0.0,
                   help="DMRS 导频功率提升 (dB)，估计噪声按 no/10^(boost/10) 下降")
    p.add_argument("--snr-start", type=float, default=-4.0)
    p.add_argument("--snr-stop", type=float, default=12.0)
    p.add_argument("--snr-step", type=float, default=2.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    K = a.trps
    tau = [float(x) * 1e-9 for x in a.tau_ns.split(",")]
    tau = [tau[i % len(tau)] for i in range(K)]

    link = build_ulink(
        num_trps=K, num_rx_ant=a.num_rx_ant, num_ue_ant=a.num_ue_ant,
        n_subcarriers=a.nsc, subcarrier_spacing=30e3,
        tau_seconds=tau, cal_amp_error=a.cal_amp_error, cal_pha_error=a.cal_pha_error,
        channel_kind=a.channel, pathloss=False, tx_mode=a.tx_mode,
        trp_distances=(100.0, 200.0, 350.0, 500.0, 700.0, 900.0)[:K],
        carrier_frequency=3.5e9)

    ll = ULinkLevelDMIMO(link, qam_order=a.qam, code_rate=a.rate, rank=1,
                         use_crc=not a.no_crc, est_density=a.est_density,
                         pilot_boost_db=a.pilot_boost_db, device=dev)

    combiners = {"all": ["joint", "symbol", "llr"],
                 "joint": ["joint"], "symbol": ["symbol"], "llr": ["llr"]}[a.combiner]
    ests = {"both": [True, False], "true": [True], "false": [False]}[a.est]
    configs = {f"{c}({ 'est' if e else 'noest' })": (c, e)
               for c in combiners for e in ests}

    snrs = snr_range(a.snr_start, a.snr_stop, a.snr_step)
    pilot_frac = ll.pilot_idx.numel() / ll.link.n
    print(f"== 上行 DMIMO 链路级（{a.channel}, K={K}, N_BS={a.num_rx_ant}, "
          f"N_UE={a.num_ue_ant}, n_sc={a.nsc}, qam={a.qam}, rate={a.rate}, "
          f"crc={not a.no_crc}, tx={a.tx_mode}）==")
    print(f"  导频: est_density={a.est_density} -> DMRS 符号梳状 {ll.pilot_idx.numel()}/{ll.link.n} "
          f"子载波（{pilot_frac:.0%}），boost={a.pilot_boost_db} dB，"
          f"数据占满 {ll.n_data} SC（TB k={ll.k}）")
    print(f"  tau(ns)={a.tau_ns[:min(len(a.tau_ns), 20)]}... cal={a.cal_amp_error}/{a.cal_pha_error}")
    print("  配置: " + ", ".join(configs))

    curves = {name: [] for name in configs}
    t0 = time.time()
    for i, s in enumerate(snrs):
        res = ll.evaluate(a.batch, s, configs, seed=i)
        for name, (bler, ber, k) in res.items():
            curves[name].append(bler)
        el = time.time() - t0
        eta = el / (i + 1) * (len(snrs) - i - 1)
        print(f"  [SNR {i + 1:2d}/{len(snrs)}] {s:6.1f} dB  "
              + "  ".join(f"{n}={curves[n][-1]:.3f}" for n in configs)
              + f"  ({el:5.1f}s, ETA {eta:5.1f}s)", flush=True)

    print("\n  SNR   | " + " | ".join(f"{n:>14}" for n in configs))
    for i, s in enumerate(snrs):
        print(f"  {s:5.1f} | " + " | ".join(f"{curves[n][i]:14.4f}" for n in configs))

    # ---- 保存 ----
    tag = (f"{a.channel}_k{K}_nbs{a.num_rx_ant}_nue{a.num_ue_ant}_"
           f"{a.nsc}sc_qam{a.qam}_r{a.rate:.2f}").replace(".", "")
    out_dir = a.out or os.path.join(common.ROOT, "out", "dmimo", "ul_link", tag)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bler.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["snr_db"] + list(configs))
        for i, s in enumerate(snrs):
            w.writerow([s] + [curves[n][i] for n in configs])
    print("Saved ->", csv_path)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n in configs:
        ls = "-" if "est)" in n else "--"
        ax.plot(snrs, curves[n], "o", ls=ls, ms=4, label=n)
    ax.axhline(0.1, color="grey", ls=":", lw=1, label="10% BLER")
    ax.set_yscale("log"); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BLER")
    ax.set_title(f"Uplink DMIMO BLER ({tag})")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(out_dir, "bler.png")
    fig.savefig(png, dpi=150)
    print("Saved ->", png)

    with open(os.path.join(out_dir, "bler.json"), "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "channel": a.channel, "num_trps": K,
                   "num_rx_ant": a.num_rx_ant, "num_ue_ant": a.num_ue_ant,
                   "n_subcarriers": a.nsc, "qam_order": a.qam, "code_rate": a.rate,
                   "est_density": a.est_density,
                   "tau_ns": a.tau_ns, "cal_amp_error": a.cal_amp_error,
                   "cal_pha_error": a.cal_pha_error, "tx_mode": a.tx_mode,
                   "snr_db": snrs, "bler": curves}, fh, indent=2, ensure_ascii=False)
    print("Saved ->", os.path.join(out_dir, "bler.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
