"""
Compare 3GPP Type I codebook precoding (wideband vs subband) against continuous
MRT on a dual-polarised UMa downlink DMIMO channel.

Each TRP: dual-pol array (2P antennas). Type I rank-1 uses a DFT beam (wideband)
and a QPSK co-phasing between the two polarisation groups, chosen either wideband
(one for the band) or per-subband.

Run:  python examples/dmimo_type1.py [--batch 1024] [--nsc 64]
"""
from __future__ import annotations

import argparse
import os
import sys

import common  # noqa: E402

import torch

from dmimo import build_link, evaluate_precoder
from dmimo import IndependentMRT, TypeICodebook

TAU = (0.0, 130e-9, 260e-9)
DIST = (100.0, 200.0, 350.0)
SNRS = [0, 5, 10, 15, 20]


def link(nsc, **kw):
    return build_link(num_trps=3, num_tx_ant=4, n_subcarriers=nsc,
                      channel_kind="uma", pathloss=False, trp_distances=DIST,
                      tau_seconds=TAU, cal_amp_error=0.1, cal_pha_error=0.1, **kw)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--nsc", type=int, default=64)
    args = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    mrt = IndependentMRT()
    l = link(args.nsc)

    print("== Type I codebook vs continuous MRT (dual-pol UMa, K=3, pathloss OFF) ==")
    print("  SNR  |   MRT   | TypeI-wide | TypeI-subband(8) | sub-win | codebook-gap")
    for snr in SNRS:
        m_mrt = evaluate_precoder(l, args.batch, snr, mrt, dev)
        m_w = evaluate_precoder(l, args.batch, snr, TypeICodebook(subband_size=args.nsc), dev)
        m_s = evaluate_precoder(l, args.batch, snr, TypeICodebook(num_subbands=8), dev)
        print(f"  {snr:4d} | {m_mrt.rate_bpshz:6.3f} | {m_w.rate_bpshz:11.3f} | "
              f"{m_s.rate_bpshz:15.3f} | {m_s.rate_bpshz-m_w.rate_bpshz:+7.3f} | "
              f"{m_mrt.rate_bpshz-m_s.rate_bpshz:12.3f}")

    print("\n== subband gain vs number of subbands (SNR=10 dB) ==")
    print("  #subbands | TypeI rate | sub-win over wideband")
    wide = evaluate_precoder(l, args.batch, 10.0, TypeICodebook(subband_size=args.nsc), dev)
    print(f"  wideband  | {wide.rate_bpshz:9.3f} | 0.000")
    for ns in (2, 4, 8, 16, 32):
        m = evaluate_precoder(l, args.batch, 10.0, TypeICodebook(num_subbands=ns), dev)
        print(f"  {ns:9d} | {m.rate_bpshz:9.3f} | {m.rate_bpshz-wide.rate_bpshz:+7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
