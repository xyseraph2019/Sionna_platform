"""
End-to-end example: run a full BLER / BER / throughput SNR sweep on a
TDL-C 16-QAM scenario and save the resulting curves.

Run from the platform root (``D:\\Platform``)::

    python examples/bler_curves.py

Requires the ``pytorch2`` conda environment (Sionna PyTorch backend) or any
environment where  ``import sionna`` works.
"""
import os
import sys
import time

# Allow in-place import of the ``sionna5g`` package.
import common  # noqa: E402

import torch

from sionna5g.config import load_config
from sionna5g.simulator import LinkSimulator
from sionna5g.plotter import plot_link_performance


def _stamp(path):
    """Append a timestamp so a rerun does not overwrite the previous figure."""
    root, ext = os.path.splitext(path)
    return f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"



def main():
    root = common.ROOT
    cfg = load_config(os.path.join(root, "configs", "tdl_c_16qam.yaml"))
    torch.manual_seed(cfg.seed)

    sim = LinkSimulator(cfg)

    # Use a coarse grid for a quick demo; increase for smoother curves.
    snr_grid = list(range(-2, 20, 2))
    results = sim.run_curve(snr_grid, num_trials=80)

    print("\n===== TDL-C 16-QAM BLER/BER/Throughput =====")
    for m in results:
        print(" ", m)

    out = _stamp(os.path.join(root, "out", "tdl_c_16qam_bler.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_link_performance(
        results,
        title="TDL-C / 16-QAM / MCS12 / 1-layer",
        save_path=out,
    )
    print(f"\nFigure saved to: {out}")


if __name__ == "__main__":
    main()
