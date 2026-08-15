"""
Preliminary UMa-channel comparison of three DMIMO cases.

  Case 1 : single TRP with MRT precoding (near TRP)  -> DMIMO comparison baseline
  Case 2 : 3 TRP with precoding, NO error            -> DMIMO theoretical upper bound
  Case 3 : 3 TRP with precoding (independent per TRP) + modeled timing/calibration
           errors                                    -> realistic (error-eroded) performance

Run:  python examples/dmimo_compare.py [--batch 1024] [--coverage | --no-coverage]
"""
from __future__ import annotations

import argparse
import os
import sys

import common  # noqa: E402

import torch

from dmimo.experiment import build_link, evaluate_precoder
from dmimo.precoding import IndependentMRT

TAU = (0.0, 130e-9, 260e-9)
DIST = (100.0, 200.0, 350.0)
SNRS = [0, 5, 10, 15, 20]


def run_table(batch, snr_list, nt, nsc, pathloss):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    mrt = IndependentMRT()
    rows = []
    for snr in snr_list:
        # Case 1: single near TRP, MRT precoding
        m1 = evaluate_precoder(build_link(num_trps=1, num_tx_ant=nt, n_subcarriers=nsc,
                                          channel_kind="uma", pathloss=pathloss,
                                          trp_distances=[DIST[0]]), batch, snr, mrt, dev)
        # Case 2: 3 TRP coherent (no error) = upper bound
        m2 = evaluate_precoder(build_link(num_trps=3, num_tx_ant=nt, n_subcarriers=nsc,
                                          channel_kind="uma", pathloss=pathloss,
                                          trp_distances=DIST, tau_seconds=(0, 0, 0)),
                               batch, snr, mrt, dev)
        # Case 3: 3 TRP + modeled errors
        m3 = evaluate_precoder(build_link(num_trps=3, num_tx_ant=nt, n_subcarriers=nsc,
                                          channel_kind="uma", pathloss=pathloss,
                                          trp_distances=DIST, tau_seconds=TAU,
                                          cal_amp_error=0.1, cal_pha_error=0.1),
                               batch, snr, mrt, dev)
        rows.append((snr, m1.rate_bpshz, m2.rate_bpshz, m3.rate_bpshz,
                     m2.rate_bpshz - m1.rate_bpshz,   # DMIMO gain vs single TRP
                     m2.rate_bpshz - m3.rate_bpshz,   # rate lost to errors
                     m3.gain_loss_db))                # combined-gain loss (dB)
    return rows


def print_table(rows, pathloss):
    tag = "pathloss ON (coverage)" if pathloss else "pathloss OFF (unit energy)"
    print(f"\n=== UMa channel, {tag} ===")
    print("  SNR  | singleTRP | 3TRP coherent(UB) | 3TRP+errors | DMIMO gain | err rate-loss | gain-loss")
    print("  dB   |  bps/Hz   |      bps/Hz        |   bps/Hz    |   bps/Hz    |    bps/Hz     |    dB")
    for (snr, r1, r2, r3, gain, eloss, gl) in rows:
        print(f"  {snr:4d} | {r1:8.3f} | {r2:18.3f} | {r3:10.3f} | {gain:9.3f} | {eloss:11.3f} | {gl:6.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--nt", type=int, default=4)
    p.add_argument("--nsc", type=int, default=64)
    p.add_argument("--coverage", action="store_true",
                   help="pathloss ON (default is OFF unless this flag is given)")
    args = p.parse_args()
    torch.manual_seed(0)

    for pathloss in (True, False):
        if pathloss and not args.coverage:
            continue  # only run coverage table when requested
        rows = run_table(args.batch, SNRS, args.nt, args.nsc, pathloss)
        print_table(rows, pathloss)
    print("\n(3TRP+errors uses per-TRP timing offsets 0/130/260 ns and amplitude+phase calibration -10 dB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
