
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
from .link_level import LinkLevelDMIMO
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

__all__ = [
    "DMIMMetrics",
    "DMIMODownlink",
    "build_link",
    "evaluate_precoder",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
    "LinkLevelDMIMO",
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
]
