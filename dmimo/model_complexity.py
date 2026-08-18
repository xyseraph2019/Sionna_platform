"""
Model complexity estimation: MACs / FLOPs / parameter count.

A tiny, dependency-free counter. It registers forward hooks on the
multiplication-heavy layers of any ``torch.nn.Module`` (``nn.Linear``,
``nn.Conv1d``, ``nn.Conv2d``) and accumulates multiply-accumulate (MAC)
counts. Everything else (SiLU / LayerNorm / additions / reshapes / the final
per-antenna normalisation) is ignored — for the MLP-Mixer subband-PMI network
those amount to <1% of the total.

Conventions
-----------
* ``macs``  : number of multiply-accumulate operations (the "FLOPs" convention
  used by thop / ptflops — one MAC = one multiply + one add).
* ``flops`` : ``2 * macs`` (the convention used by NVIDIA / DeepSpeed style
  profilers).
* Counts are reported **per sample** (batch = 1) and, for convenience, also
  scaled to the measured batch size.

Usage
-----
>>> from dmimo.model_complexity import count_macs
>>> rep = count_macs(model.net, example_input)
>>> print(rep)            # human-readable summary
>>> rep.flops             # FLOPs per sample (2 * macs)
>>> rep.as_dict()         # JSON-serialisable
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class ComplexityReport:
    """Result of one complexity measurement.

    Attributes
    ----------
    macs : int
        Multiply-accumulates per sample (batch = 1).
    params : int
        Number of parameters of the measured model.
    input_shape : tuple[int, ...]
        Shape of the example input used (including the batch dim).
    batch_size : int
        Batch size the measurement was run at.
    per_module : list[(name, macs)]
        Per-layer MAC counts, sorted descending.
    """

    macs: int
    params: int
    input_shape: Tuple[int, ...]
    batch_size: int
    per_module: List[Tuple[str, int]] = field(default_factory=list)
    unit: str = "auto"   # display unit for __str__: "auto" | "K" | "M" | "G" | "T"

    @property
    def flops(self) -> int:
        """FLOPs per sample (``2 * macs``, NVIDIA / DeepSpeed convention)."""
        return 2 * self.macs

    @property
    def macs_batch(self) -> int:
        """MACs for the measured batch."""
        return self.macs * self.batch_size

    @property
    def flops_batch(self) -> int:
        """FLOPs for the measured batch."""
        return self.flops * self.batch_size

    def as_dict(self) -> dict:
        return {
            "macs": self.macs,
            "flops": self.flops,
            "macs_batch": self.macs_batch,
            "flops_batch": self.flops_batch,
            "params": self.params,
            "batch_size": self.batch_size,
            "input_shape": list(self.input_shape),
            "per_module": [(n, m) for n, m in self.per_module],
        }

    def __str__(self) -> str:
        h = lambda v: human_count(v, self.unit)  # noqa: E731
        return (
            f"MACs : {h(self.macs)} per sample"
            f" ({h(self.macs_batch)} / batch={self.batch_size})\n"
            f"FLOPs: {h(self.flops)} per sample"
            f" ({h(self.flops_batch)} / batch={self.batch_size})\n"
            f"Params: {human_count(self.params)}"
        )


_UNITS = ("", "K", "M", "G", "T")


def human_count(n: Union[int, float], unit: str = "auto") -> str:
    """Format a count, optionally pinned to one unit.

    Parameters
    ----------
    n : int | float
        The raw count (MACs / FLOPs / params).
    unit : str
        ``"auto"`` picks K / M / G / T by magnitude; ``"K"`` / ``"M"`` / ``"G"`` /
        ``"T"`` forces that unit (e.g. ``human_count(2_013_265_920, "M")``
        -> ``"2013.27M"``).

    Examples
    --------
    >>> human_count(15_728_640)            # '15.73M'  (auto)
    >>> human_count(15_728_640, "M")       # '15.73M'
    >>> human_count(2_013_265_920, "M")    # '2013.27M'
    """
    n = float(n)
    if unit and unit != "auto" and unit in _UNITS:
        idx = _UNITS.index(unit)
        value = n / (1000.0 ** idx)
        return f"{value:,.2f}{unit}"
    # auto: scale to the largest unit that keeps |n| >= 1 (raw below 1000)
    for u in _UNITS:
        if abs(n) < 1000.0 or u == "T":
            return f"{n:,.2f}{u}" if u else f"{n:,.0f}"
        n /= 1000.0
    return f"{n:,.2f}"  # unreachable


def _layer_macs(module: nn.Module, in_shape: Tuple[int, ...],
                out_shape: Tuple[int, ...]) -> int:
    """MACs of one forward through ``module`` given input/output shapes."""
    if isinstance(module, nn.Linear):
        # per element (last dim excluded): in_features x out_features
        n = 1
        for s in in_shape[:-1]:
            n *= int(s)
        return n * module.in_features * module.out_features
    if isinstance(module, nn.Conv2d):
        # [B, C_in, H, W] -> [B, C_out, H_out, W_out]
        n = 1
        for s in out_shape[1:]:
            n *= int(s)
        k = int(module.kernel_size[0]) * int(module.kernel_size[1])
        return n * module.in_channels * k
    if isinstance(module, nn.Conv1d):
        # [B, C_in, L] -> [B, C_out, L_out]
        n = 1
        for s in out_shape[1:]:
            n *= int(s)
        return n * module.in_channels * int(module.kernel_size[0])
    return 0  # anything else: ignored (counted as ~0)


def count_macs(model: nn.Module, example_input, batch_size: Optional[int] = None,
               device: str = "cpu", unit: str = "auto") -> ComplexityReport:
    """Measure MACs / FLOPs / params of ``model`` by forward hooks.

    Parameters
    ----------
    model : nn.Module
        The model to measure (e.g. ``MLPMixerSubbandPMI`` or ``NNMixerPMI.net``).
    example_input : tensor or tuple/list of tensors
        One sample-shaped input; the first dim is treated as the batch dim.
    batch_size : int | None
        Batch size to report; defaults to ``example_input.shape[0]``.
    device : str
        Device to run the measurement on (default ``cpu``).
    unit : str
        Display unit for the report's ``__str__`` (``"auto"`` | ``"K"`` |
        ``"M"`` | ``"G"`` | ``"T"``). Raw counts in ``as_dict()`` are always
        exact integers regardless of this.

    Notes
    -----
    * The model is put in ``eval()`` for the measurement and its original
      training mode is restored afterwards.
    * Only ``nn.Linear`` / ``nn.Conv1d`` / ``nn.Conv2d`` layers are counted;
      see the module docstring.
    """
    was_training = model.training
    model.eval()

    counts: dict = {}
    handles = []

    def make_hook(name: str):
        def _hook(module, inp, out):
            i = inp[0] if isinstance(inp, (tuple, list)) else inp
            o = out[0] if isinstance(out, (tuple, list)) else out
            counts[id(module)] = (name, _layer_macs(module, tuple(i.shape), tuple(o.shape)))
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            handles.append(module.register_forward_hook(
                make_hook(name or module.__class__.__name__)))

    inputs = example_input if isinstance(example_input, (tuple, list)) else (example_input,)
    inputs = tuple(t.to(device) if torch.is_tensor(t) else t for t in inputs)
    with torch.no_grad():
        model(*inputs)

    for h in handles:
        h.remove()
    if was_training:
        model.train()

    if batch_size is None:
        batch_size = int(inputs[0].shape[0]) if torch.is_tensor(inputs[0]) else 1
    total = sum(m for _, m in counts.values())
    per_module = sorted(((n, m) for n, m in counts.values()), key=lambda t: -t[1])
    params = sum(p.numel() for p in model.parameters())
    return ComplexityReport(
        macs=total // max(batch_size, 1),
        params=params,
        input_shape=tuple(int(s) for s in inputs[0].shape),
        batch_size=batch_size,
        per_module=per_module,
        unit=unit,
    )
