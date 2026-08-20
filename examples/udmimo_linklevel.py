"""上行 DMIMO 链路级（BLER）扫描：三层合并 × 误差处理假设（Sionna Block 风格）。

单 UE 发射真实比特（5G NR TB：LDPC+CRC+QAM），K 个 TRP 联合接收，
逐 Eb/N0 出 BLER 曲线（:class:`dmimo.model.ULModel` + :func:`dmimo.sim.sim_ber_many`）：

* 合并层级：joint（L1 联合检测）/ symbol（L2 均衡后符号合并）/ llr（L3 LLR 合并）；
* 误差处理：estimate_errors=True（本地估计吸收接收误差）与 False（未补偿）；
* 同一 SNR 点所有配置共享同一份信道/比特/噪声（sim_ber_many 公平性）。

Run::

    python examples\\udmimo_linklevel.py --config configs\\dmimo_ul_linklevel.yaml
    python examples\\udmimo_linklevel.py --channel simple --batch 128 --ebno-start -10
    python examples\\udmimo_linklevel.py --combiner llr --est false
"""
import argparse
import os
import sys
import time

import common  # noqa: E402
import torch  # noqa: E402

from dmimo import ULModel, sim_ber_many, save_curves, print_curve_table
from dmimo.config import load_dmimo_config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/dmimo_ul_linklevel.yaml")
    p.add_argument("--out", default=None)
    p.add_argument("--channel", default=None, help="simple|tdl|cdl|uma|umi")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--ebno-start", type=float, default=None)
    p.add_argument("--ebno-stop", type=float, default=None)
    p.add_argument("--ebno-step", type=float, default=None)
    p.add_argument("--combiner", default=None, help="all|joint|symbol|llr")
    p.add_argument("--est", default=None, help="true|false|both")
    a = p.parse_args()

    c = load_dmimo_config(a.config)
    if a.channel:
        c.channel_kind = a.channel
    if a.batch:
        c.num_trials = a.batch
    if a.ebno_start is not None:
        c.ebno_start_db = a.ebno_start
    if a.ebno_stop is not None:
        c.ebno_stop_db = a.ebno_stop
    if a.ebno_step is not None:
        c.ebno_step_db = a.ebno_step
    torch.manual_seed(c.seed)
    dev = "cuda:0" if torch.cuda.is_available() and c.device in ("auto", "cuda:0") \
        else "cpu"

    model = c.build_ul_model(device=dev)
    combiners = {"all": ["joint", "symbol", "llr"],
                 "joint": ["joint"], "symbol": ["symbol"], "llr": ["llr"]} \
        .get((a.combiner or c.combiner).lower(), [(a.combiner or c.combiner).lower()])
    ests = {"true": [True], "false": [False], "both": [True, False]} \
        .get((a.est or "").lower(), [c.estimate_errors])
    configs = {f"{cb}({'est' if e else 'noest'})": {"combiner": cb, "estimate_errors": e}
               for cb in combiners for e in ests}

    print(f"== 上行 DMIMO 链路级（{c.channel_kind}{c.cdl_model}, K={c.num_trps}, "
          f"N_BS={c.num_tx_ant}, N_UE={c.num_ue_ant}, "
          f"qam={c.qam_order}, rate={c.code_rate}, "
          f"crc={c.use_crc}）==")
    print(f"  栅格: {c.n_symbols} 符号 x {c.fft_size} SC（有效 {model.n_eff}，"
          f"数据 RE {model.n_data}），DMRS 符号 {list(c.pilot_symbols)}")
    print(f"  tau(ns)={list(c.tau_ns)} cal={c.cal_amp_error}/{c.cal_pha_error}")
    print("  配置: " + ", ".join(configs))

    ebno_list = c.ebno_grid
    curves = {name: [] for name in configs}
    bers = {name: [] for name in configs}
    snr_list = []
    t0 = time.time()
    for i, ebno in enumerate(ebno_list):
        res = sim_ber_many(
            model, ebno, configs, batch_size=c.num_trials,
            max_mc_iter=c.num_mc_batches,
            num_target_block_errors=c.num_target_block_errors,
            target_bler=c.target_bler, seed=c.seed + i, device=dev,
            verbose=False,
            on_batch=lambda it, tot, eb, stats: print(
                f"  [Eb/N0 {i + 1:2d}/{len(ebno_list)} {eb:6.1f}dB iter {it:2d}/{tot}] "
                + "  ".join(f"{n}={s['bler']:.3f}" for n, s in stats.items()),
                flush=True))
        for name, (bler, ber, _k) in res.items():
            curves[name].append(bler)
            bers[name].append(ber)
        from sionna.phy.utils import ebnodb2no
        no = float(ebnodb2no(ebno, model.bits_sym, model.code_rate, model.rg))
        snr_list.append(-10.0 * torch.log10(torch.tensor(no)).item())
        el = time.time() - t0
        eta = el / (i + 1) * (len(ebno_list) - i - 1)
        print(f"    ({el:5.1f}s elapsed, ETA {eta:5.1f}s)", flush=True)

    final = {"ebno_db": ebno_list, "snr_db": snr_list,
             "curves": curves, "ber": bers, "k": model.k}
    print("\n===== BLER =====")
    print_curve_table(final)

    tag = (f"{c.channel_kind}{c.cdl_model}_k{c.num_trps}_nbs{c.num_tx_ant}_"
           f"nue{c.num_ue_ant}_{c.fft_size}sc_qam{c.qam_order}_"
           f"r{c.code_rate:.2f}").replace(".", "")
    meta = {"direction": "uplink", "channel_kind": c.channel_kind,
            "cdl_model": c.cdl_model, "num_trps": c.num_trps,
            "num_rx_ant": c.num_tx_ant, "num_ue_ant": c.num_ue_ant,
            "qam_order": c.qam_order, "code_rate": c.code_rate,
            "tau_ns": list(c.tau_ns), "cal_amp_error": c.cal_amp_error,
            "cal_pha_error": c.cal_pha_error,
            "title": f"UL DMIMO BLER ({tag})"}
    out = a.out or os.path.join(common.ROOT, "out", "dmimo", "ul_link", tag, "bler.png")
    paths = save_curves(out, final, meta)
    print("Saved ->", paths["png"])
    print("Saved ->", paths["csv"])
    print("Saved ->", paths["json"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
