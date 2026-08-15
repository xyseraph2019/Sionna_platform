"""
Command-line entry point for the 5G link-level simulation platform.

Examples
--------
Run a scenario defined by a YAML config and print the metric table::

    python run_simulation.py --config configs/awgn_qpsk.yaml

Add the ``--plot`` flag to also render BLER/BER/throughput curves to a file::

    python run_simulation.py --config configs/tdl_c_16qam.yaml --plot out/tdl_c.png

Run link adaptation (MCS selection across the SNR range) with ``--adapt``::

    python run_simulation.py --config configs/tdl_c_16qam.yaml --adapt --plot-mcs out/mcs.png

Use ``--device cuda:0`` for GPU acceleration if available.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# Make the ``sionna5g`` package importable when running this script in-place.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sionna5g.config import load_config, resolve_device
from sionna5g.simulator import LinkSimulator
from sionna5g.link_adaptation import LinkAdaptation
from sionna5g.plotter import plot_link_performance, plot_mcs_selection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5G NR link-level simulator built on NVIDIA Sionna (PyTorch)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "awgn_qpsk.yaml"),
        help="Path to a YAML scenario config.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda:0 / auto (default; picks cuda if available)")
    parser.add_argument("--trials", type=int, default=None, help="Override num_trials.")
    parser.add_argument(
        "--snr",
        type=float,
        nargs="+",
        default=None,
        help="Override the SNR list (dB).",
    )
    parser.add_argument(
        "--snr-start", type=float, default=None, help="SNR sweep start (dB)."
    )
    parser.add_argument(
        "--snr-stop", type=float, default=None, help="SNR sweep stop (dB)."
    )
    parser.add_argument(
        "--snr-step", type=float, default=None, help="SNR sweep step (dB)."
    )
    parser.add_argument(
        "--plot", type=str, default=None, help="Save link-performance plot to a path."
    )
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Run link adaptation (MCS selection) across the SNR range.",
    )
    parser.add_argument(
        "--plot-mcs", type=str, default=None, help="Save MCS-selection plot to a path."
    )
    parser.add_argument(
        "--target-bler", type=float, default=0.1, help="BLER target for adaptation."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg.device = resolve_device(args.device)
    else:
        cfg.device = resolve_device(cfg.device)
    if args.trials:
        cfg.num_trials = args.trials
    if args.snr:
        cfg.snr_db = list(args.snr)
    if args.snr_start is not None or args.snr_stop is not None or args.snr_step is not None:
        cfg.snr_db = None  # range flags take precedence over a fixed YAML list
    if args.snr_start is not None:
        cfg.snr_start_db = args.snr_start
    if args.snr_stop is not None:
        cfg.snr_stop_db = args.snr_stop
    if args.snr_step is not None:
        cfg.snr_step_db = args.snr_step

    torch.manual_seed(cfg.seed)
    print(f"Loaded config: {args.config}")
    print(f"  device={cfg.device}  channel={cfg.channel.channel_type} model={cfg.channel.model} "
          f"layers={cfg.pusch.num_layers} ant={cfg.pusch.num_antenna_ports} "
          f"mcs={cfg.tb.mcs_index} snr={cfg.snr_grid}")

    if not args.adapt:
        sim = LinkSimulator(cfg)
        results = sim.run_curve()
        print("\n===== Link-level metrics (BLER / BER / Throughput) =====")
        for m in results:
            print(" ", m)
        if args.plot:
            os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
            plot_link_performance(
                results,
                title=(f"{cfg.channel.channel_type.upper()} / {cfg.channel.model} "
                       f"/ MCS{cfg.tb.mcs_index} / {cfg.pusch.num_layers}L"),
                save_path=args.plot,
            )
            print(f"Plot saved to {args.plot}")
    else:
        adapt = LinkAdaptation(cfg, target_bler=args.target_bler)
        selections = adapt.select_curve(cfg.snr_grid)
        print("\n===== Link adaptation (MCS / CQI selection) =====")
        print(f"{'SNR(dB)':>8} {'MCS':>4} {'CQI':>4} {'BLER':>10} {'Throughput(Mbps)':>18}")
        for s in selections:
            print(
                f"{s.snr_db:8.2f} {s.mcs_index:4d} {s.cqi:4d} "
                f"{s.bler:10.3e} {s.throughput_bps/1e6:18.3f}"
            )
        if args.plot_mcs:
            os.makedirs(os.path.dirname(args.plot_mcs) or ".", exist_ok=True)
            plot_mcs_selection(selections, save_path=args.plot_mcs)
            print(f"MCS plot saved to {args.plot_mcs}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
