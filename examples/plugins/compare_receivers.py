"""
Demo: the generic platform's plug-in seams end-to-end.

Registers custom components (examples/plugins/custom_rx.py, custom_channel.py),
then uses both of the two extension surfaces:

  1. config-driven receiver selection: run identical TDL-C 2x2 links but with
     different (estimator, detector) combos and compare BLER;
  2. dependency injection + custom channel: build a sim with the injected
     custom ``flat_rayleigh`` channel.

Run::  python examples/plugins/compare_receivers.py [--trials 200]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from examples import common

import torch

from sionna5g.config import (SimConfig, CarrierConfig, PUSCHConfig, TBConfig,
                             ChannelConfig, ReceiverConfig)
from sionna5g.simulator import LinkSimulator
from sionna5g.metrics import bler_10db, save_metrics_csv
from sionna5g.plotter import plot_link_performance

# ---- import the flat custom-components module so names exist in the registry ----
import examples.custom_components  # noqa: F401  (registers ls_avg, mf, flat_rayleigh)



def _stamp(path):
    """Append a timestamp so a rerun does not overwrite the previous figure."""
    root, ext = os.path.splitext(path)
    return f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"



def build_tdl_cfg(trials=200) -> SimConfig:
    return SimConfig(
        carrier=CarrierConfig(subcarrier_spacing=30.0, n_size_grid=4, n_cell_id=1),
        pusch=PUSCHConfig(num_layers=2, num_antenna_ports=2, mapping_type="A",
                          symbol_start=0, symbol_length=14, precoding="non-codebook"),
        tb=TBConfig(mcs_index=12, mcs_table=1),
        channel=ChannelConfig(channel_type="tdl", model="C", delay_spread=100e-9,
                              carrier_frequency=3.5e9, min_speed=0.0, max_speed=1.0,
                              num_rx_ant=2),
        snr_db=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        num_trials=trials, batch_size=128, device="auto", return_crc_status=True, seed=11,
    )


def run_receiver_variants(trials):
    variants = {
        "LS + LMMSE":        ReceiverConfig("ls",        "lmmse"),
        "LS + ZF":           ReceiverConfig("ls",        "zf"),
        "LS + MF":           ReceiverConfig("ls",        "mf"),
        "LSavg + LMMSE":     ReceiverConfig("ls_avg",    "lmmse"),
        "Perfect + LMMSE":   ReceiverConfig("perfect",   "lmmse"),
    }
    out = {}
    for label, recv in variants.items():
        cfg = build_tdl_cfg(trials)
        cfg.receiver = recv
        torch.manual_seed(cfg.seed)
        sim = LinkSimulator(cfg)
        res = sim.run_curve()
        out[label] = res
        b10 = bler_10db(res)
        print(f"  {label:16s} 10%-BLER @ {('-' if b10 is None else round(b10, 2))} dB")
    return out


def run_custom_channel(trials):
    cfg = build_tdl_cfg(trials)
    cfg.channel = ChannelConfig(channel_type="flat_rayleigh", model="-", delay_spread=100e-9,
                                carrier_frequency=3.5e9, num_rx_ant=2)
    cfg.snr_db = [0, 4, 8, 12, 16, 20]
    torch.manual_seed(cfg.seed)
    sim = LinkSimulator(cfg)
    res = sim.run_curve()
    b10 = bler_10db(res)
    print(f"  FlatRayleigh custom channel  10%-BLER @ {('-' if b10 is None else round(b10, 2))} dB")
    return {"FlatRayleigh (custom)": res}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=200)
    args = p.parse_args()

    out_dir = os.path.join(common.ROOT, "out", "rxcompare")
    os.makedirs(out_dir, exist_ok=True)

    print("== Receiver algorithm comparison (TDL-C 2x2, MCS12) ==")
    curves = run_receiver_variants(args.trials)

    print("\n== Custom channel (flat Rayleigh) via registry ==")
    curves.update(run_custom_channel(args.trials))

    # Save overlay figure + per-variant CSVs.
    for label, res in curves.items():
        save_metrics_csv(res, os.path.join(out_dir, label.replace(" ", "_").replace("+", "_") + ".csv"))
    from sionna5g.plotter import plot_bler_overlay
    plot_bler_overlay(list(curves.items()),
                      title="Receiver algorithm comparison (TDL-C 2x2, MCS12)",
                      save_path=_stamp(os.path.join(out_dir, "receiver_compare.png")))
    print("\nSaved to", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())