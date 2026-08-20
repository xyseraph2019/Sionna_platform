"""
Monte-Carlo BLER / BER evaluation in the style of ``sionna.phy.utils.sim_ber``,
extended with the project's *fairness* requirement: several configurations
(precoders / combiners) are evaluated on the very same channel / information
bits / noise realizations, and the run reports progress (SNR, batch, elapsed,
ETA, running BLER) like the legacy ``evaluate_many``.

Typical usage (tutorial-style experiment dicts)::

    from dmimo.sim import sim_ber_many

    dl = DLModel(...)
    configs = {
        "MRT":  {"precoder": IndependentMRT(rank=2)},
        "CJT":  {"precoder": CJTPrecoder(rank=2)},
    }
    res = sim_ber_many(dl, 6.0, configs, batch_size=256,
                       num_target_block_errors=1000, on_batch=progress)
"""
from __future__ import annotations

import time

import torch


def sim_ber_many(model, ebno_db, configs, batch_size=256, max_mc_iter=100,
                 num_target_block_errors=1000, target_bler=1e-3,
                 num_mc_batches=1, seed=None, on_batch=None, device=None,
                 verbose=True):
    """Evaluate several configurations on *shared* Monte-Carlo realizations.

    For every MC batch one channel / error / information-bit realization is
    sampled once and re-used by all ``configs`` (each entry is a dict of
    keyword arguments for ``model.block_from_realization`` — e.g.
    ``{"precoder": ...}`` for :class:`~dmimo.model.DLModel` or
    ``{"combiner": "joint", "estimate_errors": True}`` for
    :class:`~dmimo.model.ULModel`).

    The MC loop stops per SNR point when every configuration has either seen
    ``num_target_block_errors`` block errors or reaches an estimated BLER at or
    below ``target_bler`` (with a minimum sample count), or when
    ``max_mc_iter`` iterations are exhausted — the ``sim_ber`` early-stopping
    rule, applied to the slowest configuration.

    Parameters
    ----------
    model : DMIMOPhyModel
    ebno_db : float
        Eb/N0 in dB (converted to the AWGN variance inside the model).
    configs : dict[str, dict]
        Configuration name -> kwargs for ``model.block_from_realization``.
    batch_size : int
        Transport blocks per MC batch (per configuration).
    max_mc_iter : int
        Upper bound on the number of MC iterations per SNR point.
    num_target_block_errors : int
        Early-stopping threshold on accumulated block errors.
    target_bler : float
        Early-stopping BLER target.
    num_mc_batches : int
        Independent batches drawn per iteration (accumulated before checking
        the stopping rule).
    seed : int | None
        ``torch.manual_seed`` before the run.
    on_batch : callable | None
        ``on_batch(iter_idx, total_iter, ebno_db, stats)`` with
        ``stats[name] = {"bler": ..., "ber": ..., "blocks": ...}`` (running
        averages), called after each iteration.
    device : str | None
    verbose : bool
        Print one progress line per iteration when ``on_batch`` is not given.

    Returns
    -------
    dict[str, tuple[float, float, int]]
        ``{name: (bler, ber, k)}`` accumulated over the whole run.
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = device or getattr(model, "_device", None)
    acc = {
        name: {"blocks": 0, "block_errors": 0.0, "bit_errors": 0.0, "bits": 0}
        for name in configs
    }
    t0 = time.time()
    total_iter = int(max_mc_iter)
    for it in range(total_iter):
        for _ in range(int(num_mc_batches)):
            real = model.sample_realization(int(batch_size), device)
            for name, cfg in configs.items():
                bler, ber, k = model.block_from_realization(
                    real, ebno_db, device=device, **cfg)
                a = acc[name]
                a["blocks"] += int(batch_size)
                a["block_errors"] += bler * batch_size
                a["bits"] += int(batch_size) * k
                a["bit_errors"] += ber * batch_size * k

        # ---- running stats for the progress callback ------------------------
        stats = {}
        for name, a in acc.items():
            stats[name] = {
                "bler": a["block_errors"] / max(a["blocks"], 1),
                "ber": a["bit_errors"] / max(a["bits"], 1),
                "blocks": a["blocks"],
            }
        if on_batch is not None:
            on_batch(it + 1, total_iter, ebno_db, stats)
        elif verbose:
            items = "  ".join(f"{n}:BLER={s['bler']:.4f}" for n, s in stats.items())
            print(f"  [Eb/N0 {ebno_db:6.2f} dB iter {it + 1:3d}/{total_iter}] {items} "
                  f"({time.time() - t0:5.1f}s)", flush=True)

        # ---- early stopping (slowest configuration decides) -----------------
        done = True
        for a in acc.values():
            bler_est = a["block_errors"] / max(a["blocks"], 1)
            if a["block_errors"] < num_target_block_errors and \
                    not (bler_est <= target_bler and a["blocks"] >= 100):
                done = False
                break
        if done:
            break

    return {
        name: (a["block_errors"] / max(a["blocks"], 1),
               a["bit_errors"] / max(a["bits"], 1),
               model.k)
        for name, a in acc.items()
    }


def sim_ber_curve(model, ebno_db_list, configs, batch_size=256, max_mc_iter=100,
                  num_target_block_errors=1000, target_bler=1e-3,
                  num_mc_batches=1, seed=None, on_snr=None, device=None,
                  verbose=True):
    """Run :func:`sim_ber_many` over an Eb/N0 sweep.

    Returns ``{"ebno_db": [...], "snr_db": [...], "curves": {name: [bler, ...]},
    "ber": {name: [...]}, "k": int}`` where ``snr_db`` is the per-data-RE SNR
    (``1/no``) corresponding to each Eb/N0 point.
    """
    ebno_list = list(ebno_db_list)
    curves = {name: [] for name in configs}
    bers = {name: [] for name in configs}
    snr_list = []
    from sionna.phy.utils import ebnodb2no

    for i, ebno in enumerate(ebno_list):
        res = sim_ber_many(model, ebno, configs, batch_size=batch_size,
                           max_mc_iter=max_mc_iter,
                           num_target_block_errors=num_target_block_errors,
                           target_bler=target_bler,
                           num_mc_batches=num_mc_batches,
                           seed=None if seed is None else seed + i,
                           on_batch=on_snr, device=device, verbose=verbose)
        for name, (bler, ber, _k) in res.items():
            curves[name].append(bler)
            bers[name].append(ber)
        no = float(ebnodb2no(ebno, model.bits_sym, model.code_rate, model.rg))
        snr_list.append(-10.0 * (torch.log10(torch.tensor(no)).item()))
    return {"ebno_db": ebno_list, "snr_db": snr_list,
            "curves": curves, "ber": bers, "k": model.k}
