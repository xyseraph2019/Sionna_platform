
"""DMIMO link-level modelling package (Sionna Block style).

Entry points::

    from dmimo import DLModel, ULModel, sim_ber_many, sim_ber_curve
    from dmimo import save_curves, print_curve_table
    from dmimo import IndependentMRT, IndependentZF, CJTPrecoder, TypeICodebook
"""
from .model import DMIMOPhyModel, DLModel, ULModel
from .sim import sim_ber_many, sim_ber_curve
from .results import save_curves, print_curve_table
from .precoding import (
    Precoder,
    IndependentMRT,
    IndependentZF,
    CJTPrecoder,
    TypeICodebook,
)

__all__ = [
    # ---- link-level models (Sionna Block style) ----
    "DMIMOPhyModel",
    "DLModel",
    "ULModel",
    "sim_ber_many",
    "sim_ber_curve",
    "save_curves",
    "print_curve_table",
    # ---- precoding ----
    "Precoder",
    "IndependentMRT",
    "IndependentZF",
    "CJTPrecoder",
    "TypeICodebook",
]
