"""
Example plugin: register a *custom receiver algorithm* at the surface level.

Run this (or ``import`` it) once, then select the new component from a YAML
``receiver:`` section by name:

  channel_estimator: ls_avg     # custom: LS + time-averaged interpolation
  mimo_detector:     mf         # custom: matched-filter (maximal-ratio) detector

No core code needs to change: the receiver resolves these names through the
:mod:`sionna5g.registry`.
"""
from __future__ import annotations

from sionna5g import registry
from sionna.phy.nr import PUSCHLSChannelEstimator


def ls_time_averaged(transmitter, device="cpu"):
    """LS channel estimator with linear + time-averaged interpolation."""
    return PUSCHLSChannelEstimator(
        transmitter.resource_grid,
        transmitter._dmrs_length,
        transmitter._dmrs_additional_position,
        transmitter._num_cdm_groups_without_data,
        interpolation_type="lin_time_avg",
        device=device,
    )


def mf_detector(transmitter, device="cpu"):
    """Matched-filter (maximal-ratio combining) MIMO detector."""
    from sionna.phy.ofdm.detection import LinearDetector
    from sionna.phy.mimo import StreamManagement
    import numpy as np

    sm = StreamManagement(np.ones([1, transmitter._num_tx], bool), transmitter._num_layers)
    return LinearDetector("mf", "bit", "maxlog", transmitter.resource_grid, sm, "qam",
                          transmitter._num_bits_per_symbol, device=device)


# Register the custom components (idempotent).
registry.register("estimator", "ls_avg", ls_time_averaged)
registry.register("detector", "mf", mf_detector)

if __name__ == "__main__":
    print("Registered custom receiver components:")
    print("  estimators:", registry.names("estimator"))
    print("  detectors :", registry.names("detector"))