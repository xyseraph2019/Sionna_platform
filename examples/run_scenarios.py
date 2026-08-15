"""
Curated 5G link-level comparison sweeps.

Instead of dumping every scenario into one crowded figure, this driver runs a
small set of *strongly contrasting* scenarios, grouped by theme, and draws one
meaningful figure per group:

  Group A - channel selectivity      : SISO 16-QAM over AWGN / TDL-D / TDL-C / CDL-A
  Group B - modulation (spectral eff.): SISO QPSK / 16-QAM / 64-QAM over AWGN
  Group C - MIMO spatial multiplexing : TDL-C 1x1 / 2x2 / 4x4 (16-QAM & 64-QAM)
  Group D - urban spatial models      : UMa / UMi, SISO and 2x2 MIMO

Outputs (under ``out/results``):

  scenarios/<name>.csv|json   per-scenario per-SNR tables
  figures/<key>.png           one two-panel (BLER + throughput) figure per group
  summary.md                  compact, theme-oriented report

Each SNR point uses ``TRIALS`` Monte-Carlo transport blocks (default 1500) and a
larger inner batch (``BATCH``, default 128) so the GPU is well utilised and the
curves are smooth. The SNR grids use a 1 dB step so steep waterfalls are sampled
finely.

Run::  python examples/run_scenarios.py [--trials 1500] [--batch 128] [--groups a b c d]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

import common  # noqa: E402

from sionna5g.config import (  # noqa: E402
    SimConfig, CarrierConfig, PUSCHConfig, TBConfig, ChannelConfig, ReceiverConfig,
)
from sionna5g.simulator import LinkSimulator  # noqa: E402
from sionna5g.metrics import (  # noqa: E402
    save_metrics_csv, save_metrics_json, bler_10db,
)
from sionna5g.plotter import plot_group_comparison  # noqa: E402


def _stamp(path):
    """Append a timestamp so a rerun does not overwrite the previous figure."""
    root, ext = os.path.splitext(path)
    return f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"


ROOT = common.ROOT
OUT = os.path.join(ROOT, "out", "results")
SCEN_DIR = os.path.join(OUT, "scenarios")
FIG_DIR = os.path.join(OUT, "figures")

TRIALS = 1500  # default Monte-Carlo transport blocks per SNR point
BATCH = 128    # default inner-vectorisation batch (GPU utilisation)


def _db1(start: int, end: int) -> list:
    """1 dB-spaced integer SNR grid from ``start`` to ``end`` inclusive."""
    return list(range(start, end + 1))


# name -> (channel_type, model, mcs, layers, ports, rx_ant, snr_grid, label)
SCENARIOS = {
    # --- Group A: channel selectivity (SISO 16-QAM MCS12) -----------------
    "awgn_16qam_siso":  ("awgn", "-", 12, 1, 1, 1, _db1(4, 16), "AWGN"),
    "tdl_d_16qam_siso": ("tdl", "D", 12, 1, 1, 1, _db1(0, 14), "TDL-D"),
    "tdl_c_16qam_siso": ("tdl", "C", 12, 1, 1, 1, _db1(-2, 16), "TDL-C"),
    "cdla_16qam_siso":  ("cdl", "A", 12, 1, 1, 1, _db1(0, 18), "CDL-A"),
    # --- Group B: modulation (AWGN SISO) -----------------------------------
    "awgn_qpsk_siso":   ("awgn", "-", 6, 1, 1, 1, _db1(-4, 8), "QPSK MCS6"),
    "awgn_16qam_siso_b":("awgn", "-", 12, 1, 1, 1, _db1(4, 16), "16-QAM MCS12"),
    "awgn_64qam_siso":  ("awgn", "-", 20, 1, 1, 1, _db1(6, 20), "64-QAM MCS20"),
    # --- Group C: MIMO spatial multiplexing (TDL-C) ------------------------
    "tdl_c_16qam_mimo2x2": ("tdl", "C", 12, 2, 2, 2, _db1(-2, 16), "16-QAM 2x2"),
    "tdl_c_16qam_mimo4x4": ("tdl", "C", 12, 4, 4, 4, _db1(0, 16), "16-QAM 4x4"),
    "tdl_c_64qam_mimo4x4": ("tdl", "C", 20, 4, 4, 4, _db1(8, 22), "64-QAM 4x4"),
    # --- Group D: urban spatial models (UMa / UMi) -------------------------
    "uma_16qam_siso":   ("uma", "-", 12, 1, 1, 1, _db1(0, 18), "UMa 1x1"),
    "uma_16qam_mimo2x2":("uma", "-", 12, 2, 2, 2, _db1(0, 22), "UMa 2x2"),
    "umi_16qam_siso":   ("umi", "-", 12, 1, 1, 1, _db1(0, 16), "UMi 1x1"),
    "umi_16qam_mimo2x2":("umi", "-", 12, 2, 2, 2, _db1(0, 18), "UMi 2x2"),
    # --- Group E: receiver algorithms (same TDL-C 2x2 link, MCS12) ---------
    "rx_ls_lmmse":   ("tdl", "C", 12, 2, 2, 2, _db1(0, 18), "LS + LMMSE"),
    "rx_ls_zf":      ("tdl", "C", 12, 2, 2, 2, _db1(0, 18), "LS + ZF"),
    "rx_ls_mf":      ("tdl", "C", 12, 2, 2, 2, _db1(0, 18), "LS + MF"),
    "rx_lsavg_lmmse":("tdl", "C", 12, 2, 2, 2, _db1(0, 18), "LSavg + LMMSE"),
    "rx_perfect_lmmse":("tdl", "C", 12, 2, 2, 2, _db1(0, 18), "Perfect + LMMSE"),
    # --- Group F: custom (user-registered) channel -------------------------
    "ch_flat":       ("flat_rayleigh", "-", 12, 2, 2, 2, _db1(0, 20), "FlatRayleigh 2x2"),
}

# Optional per-scenario receiver selection (registry names); others use default.
RECEIVERS = {
    "rx_ls_lmmse":     ReceiverConfig("ls", "lmmse"),
    "rx_ls_zf":        ReceiverConfig("ls", "zf"),
    "rx_ls_mf":        ReceiverConfig("ls", "mf"),
    "rx_lsavg_lmmse":  ReceiverConfig("ls_avg", "lmmse"),
    "rx_perfect_lmmse":ReceiverConfig("perfect", "lmmse"),
}
# (group key, title, one-line meaning, [scenario names])
GROUPS = [
    ("a_channel", "Channel selectivity vs BLER (SISO 16-QAM)",
     "How the channel (AWGN -> TDL-D -> TDL-C -> CDL-A, increasing selectivity) shifts the BLER waterfall to the right",
     ["awgn_16qam_siso", "tdl_d_16qam_siso", "tdl_c_16qam_siso", "cdla_16qam_siso"]),
    ("b_modulation", "Modulation order vs BLER / throughput (AWGN SISO)",
     "How much extra SNR higher-order modulation needs for the same 10% BLER, and the throughput it buys",
     ["awgn_qpsk_siso", "awgn_16qam_siso_b", "awgn_64qam_siso"]),
    ("c_mimo", "MIMO spatial multiplexing (TDL-C)",
     "Throughput (and required SNR) scaling from 1x1 -> 2x2 -> 4x4",
     ["tdl_c_16qam_siso", "tdl_c_16qam_mimo2x2", "tdl_c_16qam_mimo4x4", "tdl_c_64qam_mimo4x4"]),
    ("d_urban", "Urban spatial models: UMa vs UMi (SISO & 2x2)",
     "Multi-antenna support and BLER/throughput behaviour of macro vs micro urban channels",
     ["uma_16qam_siso", "uma_16qam_mimo2x2", "umi_16qam_siso", "umi_16qam_mimo2x2"]),
    ("e_receiver", "Receiver algorithms (TDL-C 2x2, MCS12)",
     "Effect of channel-estimation & detection choices (LS/LSavg/Perfect x LMMSE/ZF/MF) on BLER",
     ["rx_ls_lmmse", "rx_ls_zf", "rx_ls_mf", "rx_lsavg_lmmse", "rx_perfect_lmmse"]),
    ("f_custom_ch", "Custom flat-Rayleigh channel (registered by user)",
     "A user-registered propagation channel selected by channel_type",
     ["ch_flat"]),
]


def build_cfg(ch_type, model, mcs, layers, ports, rx_ant, snr, batch=BATCH, seed=7) -> SimConfig:
    """Construct a :class:`SimConfig` for a scenario row."""
    cfg = SimConfig(
        carrier=CarrierConfig(subcarrier_spacing=30.0, n_size_grid=8 if ports >= 4 else 4, n_cell_id=1),
        pusch=PUSCHConfig(
            num_layers=layers,
            num_antenna_ports=ports,
            mapping_type="A",
            symbol_start=0,
            symbol_length=14,
            precoding="non-codebook",
        ),
        tb=TBConfig(mcs_index=mcs, mcs_table=1),
        channel=ChannelConfig(
            channel_type=ch_type,
            model=model,
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            min_speed=0.0,
            max_speed=3.0 if ch_type in ("tdl", "cdl") else None,
            o2i_model="low",
            ut_distance=100.0,
            ut_height=1.5,
            num_rx_ant=rx_ant,
        ),
        snr_db=list(snr),
        num_trials=TRIALS,
        batch_size=int(batch),
        device="auto",
        return_crc_status=True,
        seed=seed,
    )
    # UMa/UMi freeze their topology buffers at the first batch, so make the
    # trial count an exact multiple of the mini-batch size.
    cfg.num_trials = int(((cfg.num_trials + cfg.batch_size - 1) // cfg.batch_size) * cfg.batch_size)
    return cfg


def run_all(trials: int = TRIALS, batch: int = BATCH, groups: str | None = None) -> None:
    os.makedirs(SCEN_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    selected_groups = [g.strip() for g in groups.split(",")] if groups else [g[0][0] for g in GROUPS]

    # Register the example custom components (ls_avg / mf / flat_rayleigh).
    import examples.custom_components  # noqa: F401

    # ---- 1. Run every scenario once (reused across groups) ----
    results = {}
    names = sorted(SCENARIOS)
    for i, name in enumerate(names):
        ch_type, model, mcs, layers, ports, rx_ant, snr, label = SCENARIOS[name]
        cfg = build_cfg(ch_type, model, mcs, layers, ports, rx_ant, snr, batch=batch)
        cfg.receiver = RECEIVERS.get(name, ReceiverConfig())
        cfg.num_trials = int(((trials + cfg.batch_size - 1) // cfg.batch_size) * cfg.batch_size)
        torch.manual_seed(cfg.seed)
        print(f"[{i+1:02d}/{len(names)}] {label:12s} {ch_type}/{model} MCS{mcs} {layers}L x{ports}", flush=True)
        sim = LinkSimulator(cfg)
        res = sim.run_curve()
        results[name] = res
        save_metrics_csv(res, os.path.join(SCEN_DIR, name + ".csv"))
        save_metrics_json(res, os.path.join(SCEN_DIR, name + ".json"))
        b10 = bler_10db(res)
        print(f"        10%-BLER @ {('-' if b10 is None else round(b10, 2))} dB, "
              f"max TP {max(m.throughput_bps for m in res)/1e6:.2f} Mbps", flush=True)

    # ---- 2. One comparison figure per group + bilingual summary ----
    # Chinese group title / meaning keyed by group letter.
    TITLE_ZH = {
        "a": "信道频率选择性（SISO 16-QAM）",
        "b": "调制阶数 / 频谱效率（AWGN SISO）",
        "c": "MIMO 空分复用（TDL-C）",
        "d": "城市空间模型：UMa vs UMi（SISO 与 2×2）",
        "e": "接收机算法对比（TDL-C 2×2, MCS12）",
        "f": "自定义平坦瑞利信道（用户注册）",
    }
    MEANING_ZH = {
        "a": "不同信道（AWGN → TDL-D → TDL-C → CDL-A，频率选择性递增）使 BLER 瀑布曲线逐步右移",
        "b": "高阶调制达到相同 10% BLER 所需额外 SNR，以及其带来的吞吐提升",
        "c": "吞吐与所需 SNR 随天线数 1×1 → 2×2 → 4×4 的扩展",
        "d": "宏小区(UMa)/微小区(UMi)信道在多天线下的 BLER 与吞吐表现",
        "e": "信道估计（LS/LSavg/Perfect）与检测（LMMSE/ZF/MF）选择对 BLER 的影响",
        "f": "用户通过注册表新增的传播信道，由 channel_type 直接选用",
    }

    def _table(scen_names, zh_header: bool) -> list:
        if zh_header:
            lines = ["| 场景 | 10% BLER(dB) | 最大吞吐(Mbps) |", "|---|---:|---:|"]
        else:
            lines = ["| Scenario | 10%-BLER (dB) | Max TP (Mbps) |", "|---|---:|---:|"]
        for n in scen_names:
            label = SCENARIOS[n][-1]
            b10 = bler_10db(results[n])
            lines.append(
                f"| {label} | {('-' if b10 is None else round(b10, 2))} | "
                f"{max(m.throughput_bps for m in results[n])/1e6:.2f} |"
            )
        return lines

    def _group_block(lang: str, group, zh: bool) -> list:
        key, title, meaning, scen_names = group
        lines = []
        if zh:
            lines += [f"## 分组 {key[0].upper()} — {TITLE_ZH[key[0]]}", f"**要点：** {MEANING_ZH[key[0]]}", ""]
            lines += _table(scen_names, zh_header=True)
        else:
            lines += [f"## Group {key[0].upper()} — {title}", f"**Point:** {meaning}", ""]
            lines += _table(scen_names, zh_header=False)
        return lines + [""]

    # One two-panel comparison figure per group.
    for group in GROUPS:
        if group[0][0] not in selected_groups:
            continue
        key, title, _, scen_names = group
        plot_group_comparison(
            title, [(SCENARIOS[n][-1], results[n]) for n in scen_names],
            save_path=_stamp(os.path.join(FIG_DIR, key + ".png")),
        )

    # --- Chinese block (first) ---
    zh_lines = [
        "# 5G 链路级仿真 — 主题对比总结（中文）",
        "",
        f"每 SNR 点蒙特卡洛试验次数：`{trials}`；内层并行 batch：`{batch}`（GPU 利用率更高，曲线更平滑）",
        "SNR 网格步进：1 dB（瀑布区采样更细）。",
        "BER：所有传输比特（含失败块）的原始误码率。",
        "UMa / UMi：已关闭路损与阴影衰落，频域信道归一化，便于在同一 SNR 轴比较。",
        "",
    ]
    for group in GROUPS:
        if group[0][0] not in selected_groups:
            continue
        zh_lines += _group_block("zh", group, zh=True)
    zh_lines.append("分组图见 `out/results/figures/`；逐 SNR 表格见 `out/results/scenarios/`。")

    # --- English block (second) ---
    en_lines = [
        "---",
        "",
        "# 5G Link-Level Simulation — Themed Comparison Summary (English)",
        "",
        f"Monte-Carlo trials per SNR point: `{trials}`; inner batch: `{batch}`",
        "SNR grid step: 1 dB.",
        "BER = raw bit-error rate over ALL transmitted bits.",
        "UMa / UMi: pathloss & shadow fading disabled, unit-energy frequency channel.",
        "",
    ]
    for group in GROUPS:
        if group[0][0] not in selected_groups:
            continue
        en_lines += _group_block("en", group, zh=False)
    en_lines.append("Figures -> `out/results/figures/`; per-SNR tables -> `out/results/scenarios/`.")

    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(zh_lines + en_lines) + "\n")
    print("\nDone. Figures ->", FIG_DIR)


def main() -> int:
    p = argparse.ArgumentParser(description="Curated 5G comparison sweeps.")
    p.add_argument("--trials", type=int, default=TRIALS, help="Trials per SNR point.")
    p.add_argument("--batch", type=int, default=BATCH, help="Inner vectorisation batch.")
    p.add_argument("--groups", type=str, default=None,
                   help="Comma-separated group letters to run, e.g. 'a,c' (default: all).")
    args = p.parse_args()
    run_all(trials=args.trials, batch=args.batch, groups=args.groups)
    return 0


if __name__ == "__main__":
    sys.exit(main())
