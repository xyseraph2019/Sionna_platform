"""
Plotting helpers for link-level simulation results.

Produces the standard BLER / BER / throughput-vs-SNR figures used to report
physical-layer performance. A non-interactive matplotlib backend is used so
the scripts also work headless; call ``plt.show()`` (or save figures) from the
example scripts.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (works headless)

import matplotlib.pyplot as plt

from .metrics import LinkMetrics
from .link_adaptation import MCSConfig


def plot_link_performance(
    metrics: Sequence[LinkMetrics],
    title: str = "5G NR Link-Level Performance",
    save_path: Optional[str] = None,
    figsize=(10, 4),
):
    """Plot BLER, BER and throughput vs SNR.

    Parameters
    ----------
    metrics : sequence of LinkMetrics (one per SNR point).
    title : plot title.
    save_path : if given, the figure is saved to this path.
    """
    snr = [m.snr_db for m in metrics]
    bler = [m.bler for m in metrics]
    ber = [m.ber for m in metrics]
    tp = [m.throughput_bps / 1e6 for m in metrics]  # Mbps

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title)

    axes[0].plot(snr, bler, "o-", label="BLER")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("BLER")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    if any(v > 0 for v in bler):
        axes[0].set_yscale("log")

    axes[1].plot(snr, ber, "s-", label="BER")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("BER")
    axes[1].grid(True, which="both", alpha=0.3)
    nonzero_ber = [v for v in ber if v > 0]
    if nonzero_ber:
        axes[1].set_yscale("log")
        axes[1].set_ylim(bottom=max(min(nonzero_ber), 1e-8))
    else:
        axes[1].set_ylim(bottom=1e-8)
    axes[1].legend()

    axes[2].plot(snr, tp, "^-", label="Throughput")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("Throughput (Mbps)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_bler_overlay(
    curves,
    title: str = "BLER vs SNR (multi-scenario)",
    save_path: Optional[str] = None,
    figsize=(10, 6),
):
    """Plot the BLER-vs-SNR curve of several scenarios on a single log axis.

    Parameters
    ----------
    curves : iterable of (label, List[LinkMetrics]).
        Each entry draws one BLER curve labelled with ``label``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)
    for label, metrics in curves:
        snr = [m.snr_db for m in metrics]
        bler = [m.bler for m in metrics]
        ax.plot(snr, bler, "o-", label=label)
    ax.axhline(0.1, color="grey", linestyle="--", linewidth=1, label="BLER=10%")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BLER")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(1e-4, min([m.bler for _, m in curves for m in m] or [1e-4]) * 0.5))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_group_comparison(
    title: str,
    curves,
    save_path: Optional[str] = None,
    figsize=(12, 5),
):
    """Two-panel comparison for one themed group of scenarios.

    Left panel: BLER vs SNR (log scale, with the 10% operating line).
    Right panel: throughput vs SNR. ``curves`` is a sequence of
    (label, List[LinkMetrics]).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)
    for label, metrics in curves:
        snr = [m.snr_db for m in metrics]
        bler = [m.bler for m in metrics]
        tp = [m.throughput_bps / 1e6 for m in metrics]
        axes[0].plot(snr, bler, "o-", label=label)
        axes[1].plot(snr, tp, "o-", label=label)
    axes[0].axhline(0.1, color="grey", ls="--", lw=1, label="10% BLER")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("BLER")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Throughput (Mbps)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_mcs_selection(selections: Sequence[MCSConfig], save_path: Optional[str] = None):
    """Plot the selected MCS / CQI and resulting throughput vs SNR."""
    snr = [s.snr_db for s in selections]
    mcs = [s.mcs_index for s in selections]
    cqi = [s.cqi for s in selections]
    tp = [s.throughput_bps / 1e6 for s in selections]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle("Link Adaptation: MCS / CQI selection")

    axes[0].step(snr, mcs, where="mid")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Selected MCS index")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(snr, cqi, where="mid", color="tab:green")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("CQI (1-15)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(snr, tp, "o-")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("Throughput (Mbps)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
