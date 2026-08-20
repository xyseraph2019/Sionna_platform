
"""DMIMO downlink modelling package.

Main entry points are re-exported here so examples can use::

    from dmimo import build_link, evaluate_precoder, DMIMODownlink
    from dmimo import LinkLevelDMIMO
"""
from .link import (
    DMIMMetrics,
    DMIMODownlink,
    build_link,
    evaluate_precoder,
    generate_dataset,
    save_dataset,
    load_dataset,
)
from .model import DMIMOPhyModel, DLModel, ULModel
from .sim import sim_ber_many, sim_ber_curve
from .results import save_curves, print_curve_table
from .modelDesign import MLPMixerSubbandPMI
from .nn_pmi import (
    NNMixerPMI,
    expand_subband_to_subcarriers,
    load_model,
    save_model,
    wideband_pmi,
)
from .precoding import (
    Precoder,
    IndependentMRT,
    IndependentZF,
    CJTPrecoder,
    TypeICodebook,
)
from .feedback import (
    PhaseQuantizer,
    ScalarQuantizer,
    QuantizedFeedback,
    make_quantized,
)
from .uplink import (
    UDMIMMetrics,
    UDMIMOLink,
    build_ulink,
    evaluate_ulink,
)

__all__ = [
    "DMIMMetrics",
    "DMIMODownlink",
    "build_link",
    "evaluate_precoder",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
    # ---- link-level models (Sionna Block style) ----
    "DMIMOPhyModel",
    "DLModel",
    "ULModel",
    "sim_ber_many",
    "sim_ber_curve",
    "save_curves",
    "print_curve_table",
    # ---- machine-learning / precoding / feedback ----
    "MLPMixerSubbandPMI",
    "NNMixerPMI",
    "wideband_pmi",
    "expand_subband_to_subcarriers",
    "save_model",
    "load_model",
    "Precoder",
    "IndependentMRT",
    "IndependentZF",
    "CJTPrecoder",
    "TypeICodebook",
    "PhaseQuantizer",
    "ScalarQuantizer",
    "QuantizedFeedback",
    "make_quantized",
    # ---- system-level (rate) uplink ----
    "UDMIMMetrics",
    "UDMIMOLink",
    "build_ulink",
    "evaluate_ulink",
]
