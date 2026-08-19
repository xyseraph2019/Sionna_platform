"""上行 DMIMO 机制 demo：多 TRP 联合接收的合并增益与误差损耗。

单用户 -> K 个 TRP 联合接收（系统级速率域）：

* 两种合并层级：joint（L1 联合检测，信号级拼接 MRC）与 symbol（L2 每 TRP
  本地 MRC + 中心幅度加权合并）；
* 两种误差处理假设：estimate_errors=True（本地估计吸收误差，乐观/真实上行）
  与 False（未补偿，误差破坏相干，悲观——与下行故事同构）；
* 对称性自检：无误差时相干合并增益应随 K 增长（joint ~K×、symbol ~K²×，
  功率口径），速率则两种合并都趋于最优 Σ‖u_t‖²/no。

Run::

    python examples\\udmimo_demo.py                          # simple 信道快速验证
    python examples\\udmimo_demo.py --channel uma --batch 512
    python examples\\udmimo_demo.py --trps 1,2,3,6 --tau-ns 0,130,260
"""
import argparse
import json
import os
import sys

import common  # noqa: E402
import torch  # noqa: E402

from dmimo import build_ulink  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", default="simple", help="simple|tdl|cdl|uma|umi")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--trps", default="1,2,3,6", help="TRP 数量列表")
    p.add_argument("--num-rx-ant", type=int, default=4, help="每 TRP 接收天线 N_BS")
    p.add_argument("--num-ue-ant", type=int, default=1, help="UE 发射天线 N_UE")
    p.add_argument("--nsc", type=int, default=64)
    p.add_argument("--tau-ns", default="0,130,260", help="各 TRP 接收时延 (ns)")
    p.add_argument("--cal-amp-error", type=float, default=0.1)
    p.add_argument("--cal-pha-error", type=float, default=0.1)
    p.add_argument("--out", default=None, help="输出目录（默认 out/dmimo/ul）")
    a = p.parse_args()

    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    trps = [int(k) for k in a.trps.split(",") if k.strip()]
    tau = [float(x) * 1e-9 for x in a.tau_ns.split(",")]

    DIST = (100.0, 200.0, 350.0, 500.0, 700.0, 900.0)

    def mk(k):
        # 每 TRP 的接收时延按索引取前 k 个（TRP0=0）；K=1 时无误差。
        # 时延列表不足 k 个时循环扩展（如 0,130,260,0,130,260,...）。
        t = [tau[i % len(tau)] for i in range(k)] if k > 1 else [0.0]
        return build_ulink(
            num_trps=k, num_rx_ant=a.num_rx_ant, num_ue_ant=a.num_ue_ant,
            n_subcarriers=a.nsc, subcarrier_spacing=30e3,
            tau_seconds=t, cal_amp_error=(a.cal_amp_error if k > 1 else None),
            cal_pha_error=(a.cal_pha_error if k > 1 else None),
            channel_kind=a.channel, pathloss=False,
            trp_distances=DIST[:k],
            carrier_frequency=3.5e9)

    print(f"== 上行 DMIMO 机制（{a.channel}, batch={a.batch}, SNR={a.snr} dB, "
          f"N_BS={a.num_rx_ant}/TRP, N_UE={a.num_ue_ant}, n_sc={a.nsc}）==")
    print(f"  接收时延(ns)={a.tau_ns}  校准 amp={a.cal_amp_error} pha={a.cal_pha_error}")

    rows = []
    coherent_single = {}   # K=1 的相干 gain（功率口径），用于增益比
    for k in trps:
        link = mk(k).to(dev)
        g_single = None
        for combiner in ("joint", "symbol"):
            m_est = link(a.batch, a.snr, combiner=combiner, estimate_errors=True, device=dev)
            m_noest = link(a.batch, a.snr, combiner=combiner, estimate_errors=False, device=dev)
            if k == 1:
                coherent_single[combiner] = m_est.gain_coherent_lin
            ratio = m_est.gain_coherent_lin / (coherent_single[combiner] + 1e-12)
            rows.append({
                "k": k, "combiner": combiner,
                "rate_coh": m_est.rate_coherent_bpshz,
                "rate_err_est": m_est.rate_bpshz,
                "rate_err_noest": m_noest.rate_bpshz,
                "gain_loss_db_est": m_est.gain_loss_db,
                "gain_loss_db_noest": m_noest.gain_loss_db,
                "gain_ratio_vs_single": float(ratio),
            })
            print(f"  K={k:>2d} {combiner:<7s} "
                  f"rate: coh={m_est.rate_coherent_bpshz:6.2f}  "
                  f"err(est)={m_est.rate_bpshz:6.2f}  err(noest)={m_noest.rate_bpshz:6.2f}  "
                  f"| 增益损耗: est={m_est.gain_loss_db:5.2f}dB  noest={m_noest.gain_loss_db:5.2f}dB  "
                  f"| 相干增益比(K=1)={ratio:5.1f}x")

    # ---- 保存 ----
    out_dir = a.out or os.path.join(common.ROOT, "out", "dmimo", "ul", a.channel)
    os.makedirs(out_dir, exist_ok=True)
    import csv

    tag = f"{a.channel}_nbs{a.num_rx_ant}_nue{a.num_ue_ant}"
    csv_path = os.path.join(out_dir, f"udmimo_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved ->", csv_path)
    json_path = os.path.join(out_dir, f"udmimo_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"channel": a.channel, "batch": a.batch, "snr_db": a.snr,
                   "tau_ns": a.tau_ns, "cal_amp_error": a.cal_amp_error,
                   "cal_pha_error": a.cal_pha_error,
                   "num_rx_ant": a.num_rx_ant, "num_ue_ant": a.num_ue_ant,
                   "rows": rows}, fh, indent=2, ensure_ascii=False)
    print("Saved ->", json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
