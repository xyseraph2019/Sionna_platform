"""YAML-driven link-level DMIMO BLER evaluation.
Run: python examples\\dmimo_linklevel_cfg.py --config configs\\dmimo_linklevel.yaml
"""
import os, sys, argparse, time


def _stamp(path):
    """Append a timestamp so a rerun does not overwrite the previous figure."""
    root, ext = os.path.splitext(path)
    return f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"

import common  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from dmimo.config import load_dmimo_config, scenario_tag
from dmimo import LinkLevelDMIMO
from dmimo import CJTPrecoder, IndependentMRT, TypeICodebook





def _auto_nn_pmi_ckpt(c):
    """Build the expected NN-PMI checkpoint path from the current scenario."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dmimo_linklevel.yaml")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    c = load_dmimo_config(a.config)
    torch.manual_seed(c.seed)
    dev = "cuda:0" if torch.cuda.is_available() and c.device in ("auto", "cuda:0") else "cpu"

    def mk(num_trps, err):
        base = dict(num_tx_ant=c.num_tx_ant, num_ue_ant=c.num_ue_ant,
                    n_subcarriers=c.n_subcarriers, subcarrier_spacing=c.subcarrier_spacing_khz * 1e3,
                    channel_kind=c.channel_kind, pathloss=c.pathloss,
                    trp_distances=[c.trp_distances_m[0]] if num_trps == 1 else list(c.trp_distances_m),
                    granularity=c.granularity, carrier_frequency=c.carrier_frequency)
        from dmimo import build_link
        if err and num_trps > 1:
            base.update(tau_seconds=c.tau_seconds, cal_amp_error=c.cal_amp_error, cal_pha_error=c.cal_pha_error)
        else:
            base.update(tau_seconds=[0.0] * num_trps, cal_amp_error=None, cal_pha_error=None)
        return build_link(num_trps=num_trps, **base)

    precoders = {#"MRT": lambda: IndependentMRT(rank=c.rank),
                 "CJT": lambda: CJTPrecoder(rank=c.rank, subband_size=c.subband_size),
                 "TypeI-wide": lambda: TypeICodebook(rank=c.rank, subband_size=c.n_subcarriers),
                 }
    # 自动按当前场景选择 NN-PMI 模型；YAML 里可以写具体路径，或写 "auto"/空。
    if c.nn_pmi_ckpt in (None, "", "auto"):
        c.nn_pmi_ckpt = _auto_nn_pmi_ckpt(c)

    if c.nn_pmi_ckpt and os.path.exists(c.nn_pmi_ckpt):
        from dmimo import load_model

        _nn, _nn_meta = load_model(c.nn_pmi_ckpt, device=dev)
        precoders["NN-PMI"] = lambda: _nn
        print(f"  loaded NN-PMI: {c.nn_pmi_ckpt}")
        print(f"    meta: tag={_nn_meta.get('tag')}  val_loss={_nn_meta.get('val_loss')}")
    elif "nn" in c.precoder.lower() or "nnpmi" in c.precoder.lower():
        print(f"  [warning] NN-PMI requested but checkpoint not found: {c.nn_pmi_ckpt}")
        print("  -> skip NN-PMI")
    sel = list(precoders) if c.precoder.lower() == "all" else \
        [{"mrt": "MRT", "cjt": "CJT", "type1": "TypeI-wide", "typei": "TypeI-wide",
          "nn": "NN-PMI", "nnpmi": "NN-PMI"}.get(c.precoder.lower(), c.precoder)]
    return _run(a, c, mk, precoders, sel, dev)
def _run(a, c, mk, precoders, sel, dev):
    tag = scenario_tag(c.num_trps, c.rank, c.n_subcarriers, c.qam_order,
                       c.code_rate, c.channel_kind, est=c.use_channel_estimation,
                       num_dmrs_symbols=c.num_dmrs_symbols,
                       err=c.cal_amp_error is not None or c.cal_pha_error is not None)
    kw = dict(use_channel_estimation=c.use_channel_estimation,
              est_density=c.est_density, use_crc=c.use_crc,
              pilot_boost_db=c.pilot_boost_db, n_symbols=c.n_symbols,
              dmrs_symbol=c.dmrs_symbol, num_dmrs_symbols=c.num_dmrs_symbols)
    l1 = LinkLevelDMIMO(mk(1, False), qam_order=c.qam_order, code_rate=c.code_rate, rank=c.rank, device=dev, **kw)
    l0 = LinkLevelDMIMO(mk(3, False), qam_order=c.qam_order, code_rate=c.code_rate, rank=c.rank, device=dev, **kw)
    le = LinkLevelDMIMO(mk(3, True), qam_order=c.qam_order, code_rate=c.code_rate, rank=c.rank, device=dev, **kw)
    est = (f"LS(d={c.est_density}, pilot={l0.num_pilots}/{l0.n_data} SC)"
           if c.use_channel_estimation else "perfect CSI")
    crc = f"TB-CRC{l0.crc_length}" if c.use_crc else "full-bit compare"
    print(f"rank={c.rank} k/n={l0.k}/{l0.n} qam={c.qam_order} rate={c.code_rate} "
          f"channel={c.channel_kind} tau_ns={list(c.tau_ns)} cal={c.cal_amp_error}/{c.cal_pha_error}")
    print(f"  CSI={est}  detect={crc}  grid={l0.n_symbols}sym x {l0.n_data}sc "
          f"(data {l0.n_data_sym} sym)  tb_size={l0.tb_size}")
    curves = {}
    snr = c.snr_grid
    pcs = {name: precoders[name]() for name in sel}
    t_start = time.time()
    total_snr = len(snr)

    for i, s in enumerate(snr):
        # One channel/bits realization per SNR is shared by all precoders.
        res1 = l1.evaluate_many(c.num_trials, s, pcs, seed=c.seed + i, num_batches=c.num_mc_batches)
        res0 = l0.evaluate_many(c.num_trials, s, pcs, seed=c.seed + i, num_batches=c.num_mc_batches)
        rese = le.evaluate_many(c.num_trials, s, pcs, seed=c.seed + i, num_batches=c.num_mc_batches)
        for name in sel:
            curves.setdefault(f"{name} singleTRP", []).append(res1[name][0])
            curves.setdefault(f"{name} 3TRP coherent", []).append(res0[name][0])
            curves.setdefault(f"{name} 3TRP+err", []).append(rese[name][0])
        elapsed = time.time() - t_start
        avg = elapsed / (i + 1)
        eta = avg * (total_snr - i - 1)
        print(
            f"  [SNR {i + 1:2d}/{total_snr}] {s:6.1f} dB done "
            f"({elapsed:7.1f}s elapsed, ETA {eta:7.1f}s)",
            flush=True,
        )

    print("  SNR   | " + " | ".join(f"{n:>20}" for n in curves))
    for i, s in enumerate(snr):
        print(f"  {s:5.1f} | " + " | ".join(f"{curves[n][i]:20.3f}" for n in curves))

    fig, ax = plt.subplots(figsize=(10, 6))
    for n, ys in curves.items():
        ax.plot(snr, ys, "o-", label=n)
    ax.axhline(0.1, color="grey", ls="--", lw=1, label="10% BLER")
    ax.set_yscale("log"); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BLER")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    out = a.out or _stamp(os.path.join(common.ROOT,
                                       "out", "dmimo",
                                       f"linklevel_{tag}.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Saved ->", out)
    # ---- 保存数值结果（CSV / JSON），方便后续画图或表格分析 ----
    import csv
    import json

    csv_path = os.path.splitext(out)[0] + ".csv"
    json_path = os.path.splitext(out)[0] + ".json"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["snr_db"] + list(curves.keys()))
        for i, s in enumerate(snr):
            writer.writerow([s] + [curves[n][i] for n in curves])

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "scenario_tag": tag,
                "num_trps": c.num_trps,
                "rank": c.rank,
                "n_subcarriers": c.n_subcarriers,
                "qam_order": c.qam_order,
                "code_rate": c.code_rate,
                "channel_kind": c.channel_kind,
                "num_trials": c.num_trials,
                "num_mc_batches": c.num_mc_batches,
                "snr_db": snr,
                "bler": curves,
            },
            fh,
            indent=2,
        )

    print("Saved CSV ->", csv_path)
    print("Saved JSON ->", json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

