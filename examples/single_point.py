"""
Quick single-point evaluation: transmit a small batch over an AWGN QPSK link
at a fixed SNR and report BLER / BER / throughput.

Run::

    python examples/single_point.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from sionna5g.config import load_config
from sionna5g.simulator import LinkSimulator


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "configs", "awgn_qpsk.yaml"))
    torch.manual_seed(cfg.seed)

    sim = LinkSimulator(cfg)
    print(
        f"Info bits / transport block: {sim.info_bits_per_tb}, "
        f"slot duration: {cfg.carrier.slot_duration*1e3:.3f} ms"
    )

    for snr in [0.0, 2.0, 4.0, 6.0, 8.0]:
        m = sim.run_snr(snr, num_trials=100)
        print(" ", m)


if __name__ == "__main__":
    main()
