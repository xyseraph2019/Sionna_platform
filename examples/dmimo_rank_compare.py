"""
Compare Type I codebook rank 1-4 vs continuous MRT, with and without the
modeled inter-TRP channel errors, on a dual-polarised UMa downlink DMIMO channel
with a 4-antenna UE.

Run:  python examples/dmimo_rank_compare.py [--batch 2048] [--snr 10]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dmimo.experiment import build_link, evaluate_precoder
from dmimo.precoding import IndependentMRT, TypeICodebook, CJTPrecoder

TAU = (0.0, 130e-9, 260e-9)
DIST = (100.0, 200.0, 350.0)
RANKS = [1, 2, 3, 4]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--nsc", type=int, default=64)
    args = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    def mk(err):
        return build_link(num_trps=3, num_tx_ant=4, num_ue_ant=4, n_subcarriers=args.nsc,
                          channel_kind="uma", pathloss=False, trp_distances=DIST,
                          tau_seconds=(TAU if err else (0.0, 0.0, 0.0)),
                          cal_amp_error=(0.1 if err else None),
                          cal_pha_error=(0.1 if err else None))

    l0, le = mk(False), mk(True)

    print(f"== DMIMO rank comparison (dual-pol UMa, K=3, 4-ant UE, SNR={args.snr} dB) ==")
    print("  rank | MRT no-err | MRT +err | CJT no-err | CJT +err | TI no-err | TI +err | err-loss MRT | err-loss CJT | err-loss TI")
    for r in RANKS:
        m0 = evaluate_precoder(l0, args.batch, args.snr, IndependentMRT(rank=r), dev)
        me = evaluate_precoder(le, args.batch, args.snr, IndependentMRT(rank=r), dev)
        c0 = evaluate_precoder(l0, args.batch, args.snr, CJTPrecoder(rank=r), dev)
        ce = evaluate_precoder(le, args.batch, args.snr, CJTPrecoder(rank=r), dev)
        t0 = evaluate_precoder(l0, args.batch, args.snr, TypeICodebook(rank=r, num_subbands=8), dev)
        te = evaluate_precoder(le, args.batch, args.snr, TypeICodebook(rank=r, num_subbands=8), dev)
        print(f"  {r:4d} | {m0.rate_bpshz:9.3f} | {me.rate_bpshz:8.3f} | "
              f"{c0.rate_bpshz:9.3f} | {ce.rate_bpshz:8.3f} | "
              f"{t0.rate_bpshz:9.3f} | {te.rate_bpshz:8.3f} | "
              f"{m0.rate_bpshz-me.rate_bpshz:11.3f} | {c0.rate_bpshz-ce.rate_bpshz:10.3f} | "
              f"{t0.rate_bpshz-te.rate_bpshz:10.3f}")

    print("\n(no-err = coherent upper bound; +err = per-TRP timing 0/130/260ns + cal amp/phase -10dB)")
    print("rate in bps/Hz; all values at rank r layers over a 4-antenna UE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
