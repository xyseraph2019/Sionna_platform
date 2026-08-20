"""
Result persistence for link-level DMIMO runs: one helper writes the PNG figure,
the CSV table and the JSON scenario metadata together, so every run leaves the
three artifacts required by the project conventions (AGENTS.md §4).

Curves layout::

    curves = {
        "ebno_db": [ ... ],            # Eb/N0 axis (dB)
        "snr_db":  [ ... ],            # per-data-RE SNR = 1/no (dB)
        "curves":  {name: [bler, ...]},
        "ber":     {name: [ber, ...]},
        "k": int,
    }

The PNG figure plots BLER vs Eb/N0 (log scale) with a secondary x-axis showing
the corresponding SNR.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def print_curve_table(curves, keys=None):
    """Print the BLER table (rows = Eb/N0 points, columns = configurations)."""
    names = keys or list(curves["curves"])
    print("  Eb/N0(dB) | " + " | ".join(f"{n:>16}" for n in names))
    for i, ebno in enumerate(curves["ebno_db"]):
        print(f"  {ebno:8.2f} | "
              + " | ".join(f"{curves['curves'][n][i]:16.4f}" for n in names))


def save_curves(out_path, curves, meta=None):
    """Save BLER figure (PNG), numerical table (CSV) and scenario JSON.

    Parameters
    ----------
    out_path : str
        Target figure path; the CSV / JSON are written next to it with the
        same basename.
    curves : dict
        See module docstring.
    meta : dict | None
        Scenario metadata embedded in the JSON (scenario_tag, parameters...).

    Returns
    -------
    dict[str, str]
        ``{"png": ..., "csv": ..., "json": ...}`` written file paths.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    root, _ext = os.path.splitext(out_path)
    csv_path = root + ".csv"
    json_path = root + ".json"

    names = list(curves["curves"])
    ebno = curves["ebno_db"]
    snr = curves["snr_db"]

    # ---- CSV (dual axis: Eb/N0 and SNR) -------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ebno_db", "snr_db"] + [f"{n}_bler" for n in names]
                        + [f"{n}_ber" for n in names])
        for i in range(len(ebno)):
            writer.writerow([ebno[i], snr[i]]
                            + [curves["curves"][n][i] for n in names]
                            + [curves["ber"][n][i] for n in names])

    # ---- JSON (metadata + curves) -------------------------------------------
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta or {}, "curves": curves}, fh, indent=2,
                  ensure_ascii=False)

    # ---- PNG (BLER vs Eb/N0, secondary axis SNR) ----------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n in names:
        ax.plot(ebno, curves["curves"][n], "o-", ms=4, label=n)
    ax.axhline(0.1, color="grey", ls=":", lw=1, label="10% BLER")
    ax.set_yscale("log")
    vmin = min((curves["curves"][n][i] for n in names
                for i in range(len(ebno))), default=1.0)
    ax.set_ylim(bottom=1e-4 if vmin <= 0 else max(vmin * 0.5, 1e-6))
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("BLER")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{np.interp(t, ebno, snr):.0f}" for t in ticks])
    ax2.set_xlabel("SNR (dB)")
    if meta and meta.get("title"):
        ax.set_title(meta["title"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {"png": out_path, "csv": csv_path, "json": json_path}
