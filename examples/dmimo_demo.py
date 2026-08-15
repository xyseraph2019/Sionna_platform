"""
Demonstrate the DMIMO downlink with Sionna channel models (simple / TDL / CDL /
UMa) and per-TRP timing + calibration errors.

Run::

    python examples/dmimo_demo.py --batch 2048 --snr 10

Shows:
  A. error-erodes-coherent-gain mechanism (pathloss OFF, unit-energy channels);
  B. per-TRP coverage when pathloss is ON (UMa, TRPs at different distances);
  C. how the channel model (simple / TDL / CDL / UMa) changes the error impact.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import common  # noqa: E402

import torch

from dmimo.experiment import build_link, evaluate_precoder
from dmimo.precoding import IndependentMRT

TAU = (0.0, 130e-9, 260e-9)
DIST = (100.0, 200.0, 350.0)


def fmt(m):
    return (f"rate={m.rate_bpshz:6.3f} bps/Hz  gain={10*math.log10(m.gain_lin):6.2f} dB  "
            f"gain(coh)={10*math.log10(m.gain_coherent_lin):6.2f} dB  loss={m.gain_loss_db:6.2f} dB")


def run_mechanism(batch, snr, nt, nsc, ch):
    """Section A: pathloss off, single TRP vs coherent vs timing vs cal vs both."""
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    mrt = IndependentMRT()
    m1 = evaluate_precoder(build_link(num_trps=1, num_tx_ant=nt, n_subcarriers=nsc,
                                      channel_kind=ch, pathloss=False, trp_distances=[100.0]),
                           batch, snr, mrt, dev)
    m0 = evaluate_precoder(build_link(num_trps=3, num_tx_ant=nt, n_subcarriers=nsc,
                                      channel_kind=ch, pathloss=False, trp_distances=DIST),
                           batch, snr, mrt, dev)
    mT = evaluate_precoder(build_link(num_trps=3, num_tx_ant=nt, n_subcarriers=nsc,
                                      channel_kind=ch, pathloss=False, trp_distances=DIST, tau_seconds=TAU),
                           batch, snr, mrt, dev)
    mC = evaluate_precoder(build_link(num_trps=3, num_tx_ant=nt, n_subcarriers=nsc,
                                      channel_kind=ch, pathloss=False, trp_distances=DIST,
                                      cal_amp_error=0.1, cal_pha_error=0.1),
                           batch, snr, mrt, dev)
    mTC = evaluate_precoder(build_link(num_trps=3, num_tx_ant=nt, n_subcarriers=nsc,
                                       channel_kind=ch, pathloss=False, trp_distances=DIST,
                                       tau_seconds=TAU, cal_amp_error=0.1, cal_pha_error=0.1),
                            batch, snr, mrt, dev)
    print(f"  [channel={ch}]")
    print(f"    single TRP                  {fmt(m1)}")
    print(f"    3 TRP coherent              {fmt(m0)}")
    print(f"    3 TRP timing 0/130/260ns    {fmt(mT)}")
    print(f"    3 TRP cal amp+phase -10dB   {fmt(mC)}")
    print(f"    3 TRP timing+cal            {fmt(mTC)}")
    print(f"    coherent ~ {m0.gain_coherent_lin/m1.gain_coherent_lin:4.2f}x single; "
          f"gain eaten: timing {mT.gain_loss_db:.2f}, cal {mC.gain_loss_db:.2f}, "
          f"combined {mTC.gain_loss_db:.2f} dB\n")


def run_coverage(batch, snr, nt, nsc):
    """Section B: UMa pathloss ON -> per-TRP coverage + rate."""
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    mrt = IndependentMRT()
    print("  [UMa, pathloss ON, TRP distances 100/200/350 m]")
    for tau in [(0.0, 0.0, 0.0), TAU]:
        link = build_link(num_trps=3, num_tx_ant=nt, n_subcarriers=nsc,
                          channel_kind="uma", pathloss=True, trp_distances=DIST,
                          tau_seconds=tau)
        tens = link.forward_tensors(batch, snr, mrt, dev)
        p_trp = tens["channel"].abs().square().mean(dim=(0, 2, 3))  # per TRP, over Nt, N & batch
        lbl = "coherent" if tau[1] == 0.0 else "timing 0/130/260"
        print(f"    [{lbl:16s}] per-TRP mean |h|^2 = "
              f"{['%.4g' % v.item() for v in p_trp]}; "
              f"rate={tens['rate'].mean().item():6.3f} bps/Hz")
    return dev


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--nt", type=int, default=4)
    p.add_argument("--nsc", type=int, default=64)
    p.add_argument("--channel", type=str, default="simple",
                   help="simple | tdl | cdl | uma | umi")
    p.add_argument("--coverage", action="store_true", help="also show UMa coverage")
    args = p.parse_args()

    torch.manual_seed(0)
    print("== A. mechanism (pathloss OFF) ==", flush=True)
    run_mechanism(args.batch, args.snr, args.nt, args.nsc, args.channel)

    print("== C. channel-model impact on error loss (timing+cal, pathloss OFF) ==", flush=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    mrt = IndependentMRT()
    for ch in ("simple", "tdl", "cdl", "uma"):
        m = evaluate_precoder(build_link(num_trps=3, num_tx_ant=args.nt, n_subcarriers=args.nsc,
                                         channel_kind=ch, pathloss=False, trp_distances=DIST,
                                         tau_seconds=TAU, cal_amp_error=0.1, cal_pha_error=0.1),
                              args.batch, args.snr, mrt, dev)
        print(f"  {ch:6s}  rate={m.rate_bpshz:6.3f}  loss={m.gain_loss_db:6.2f} dB  "
              f"coh_gain={10*math.log10(m.gain_coherent_lin):6.2f} dB")

    if args.coverage:
        print("\n== B. UMa coverage (pathloss ON) ==", flush=True)
        run_coverage(args.batch, args.snr, args.nt, args.nsc)

    out_dir = os.path.join(common.ROOT, "out", "dmimo")
    os.makedirs(out_dir, exist_ok=True)
    link = build_link(num_trps=3, num_tx_ant=args.nt, n_subcarriers=args.nsc,
                      channel_kind=args.channel, pathloss=(args.channel != "simple"),
                      trp_distances=DIST, tau_seconds=TAU, cal_amp_error=0.1, cal_pha_error=0.1)
    from dmimo.experiment import generate_dataset, save_dataset
    save_dataset(generate_dataset(link, args.batch, args.snr, num_batches=1, device=dev),
                 os.path.join(out_dir, "dmimo_dataset.pt"))
    print(f"\nSaved training-ready dataset -> out/dmimo/dmimo_dataset.pt (channel={args.channel})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

