"""YAML-driven link-level DL DMIMO BLER evaluation (Sionna Block style).

Builds a :class:`dmimo.model.DLModel` from the config, evaluates the selected
precoders on *shared* Monte-Carlo realizations per Eb/N0 point
(:func:`dmimo.sim.sim_ber_many`), and saves the PNG / CSV / JSON artifacts via
:func:`dmimo.results.save_curves`.

Run::

    python examples\\dmimo_linklevel_cfg.py --config configs\\dmimo_linklevel.yaml
"""
import argparse
import os
import sys
import time

import common  # noqa: E402
import torch  # noqa: E402

from dmimo import (DLModel, IndependentMRT, CJTPrecoder, TypeICodebook,
                   sim_ber_many, save_curves, print_curve_table)
from dmimo.config import load_dmimo_config, scenario_tag


def _stamp(path):
    """Append a timestamp so a rerun does not overwrite the previous figure."""
    root, ext = os.path.splitext(path)
    return f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"


def _build_model(c, num_trps, with_err, device):
    """Build a DLModel for one link variant (single-TRP / coherent / +errors)."""
    base = dict(num_trps=num_trps,
                channel_kind=c.channel_kind, cdl_model=c.cdl_model,
                tdl_model=c.tdl_model, speed=c.speed, pathloss=c.pathloss,
                trp_distances=[c.trp_distances_m[0]] if num_trps == 1
                              else list(c.trp_distances_m),
                granularity=c.granularity,
                subcarrier_spacing=c.subcarrier_spacing_khz * 1e3,
                fft_size=c.fft_size, num_guard_carriers=c.num_guard_carriers,
                dc_null=c.dc_null, n_symbols=c.n_symbols,
                pilot_ofdm_symbol_indices=c.pilot_symbols,
                pilot_boost_db=c.pilot_boost_db,
                cyclic_prefix_length=c.cyclic_prefix_length,
                qam_order=c.qam_order, code_rate=c.code_rate,
                rank=c.rank, use_crc=c.use_crc,
                perfect_csi=c.perfect_csi, device=device)
    if with_err and num_trps > 1:
        base.update(tau_seconds=list(c.tau_seconds)[:num_trps],
                    cal_amp_error=c.cal_amp_error,
                    cal_pha_error=c.cal_pha_error)
    else:
        base.update(tau_seconds=[0.0] * num_trps,
                    cal_amp_error=None, cal_pha_error=None)
    return DLModel(**base)


def _make_precoders(c, n_eff, device):
    """Precoder factories from the config (MRT / CJT / Type I).

    ``n_eff``: number of effective subcarriers (used as the Type I wideband
    subband size, i.e. one beam + co-phasing over the whole band).
    """
    precoders = {
        "MRT": lambda: IndependentMRT(rank=c.rank),
        "CJT": lambda: CJTPrecoder(rank=c.rank, subband_size=c.subband_size),
        "TypeI-wide": lambda: TypeICodebook(rank=c.rank, subband_size=n_eff),
    }
    # Selection.
    sel = list(precoders) if c.precoder.lower() == "all" else \
        [{"mrt": "MRT", "cjt": "CJT", "type1": "TypeI-wide",
          "typei": "TypeI-wide"}.get(c.precoder.lower(), c.precoder)]
    return {name: precoders[name] for name in sel}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/dmimo_linklevel.yaml")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    c = load_dmimo_config(a.config)
    torch.manual_seed(c.seed)
    dev = "cuda:0" if torch.cuda.is_available() and c.device in ("auto", "cuda:0") \
        else "cpu"

    m1 = _build_model(c, 1, False, dev)
    m0 = _build_model(c, c.num_trps, False, dev)
    me = _build_model(c, c.num_trps, True, dev)
    models = {"singleTRP": m1, "3TRP-coherent": m0, "3TRP+err": me}
    pcs = _make_precoders(c, me.n_eff, dev)

    est = "perfect CSI" if c.perfect_csi else \
        f"LS ({len(c.pilot_symbols)} DMRS symbols)"
    crc = f"TB-CRC{me.crc_length}" if c.use_crc else "full-bit compare"
    print(f"== DL DMIMO link-level (Block style): channel={c.channel_kind}"
          f"{c.cdl_model}  {c.num_trps}TRP/{c.num_tx_ant}ant "
          f"rank={c.rank} QAM{c.qam_order} rate={c.code_rate} ==")
    print(f"  grid={c.n_symbols}sym x {c.fft_size}sc (eff {me.n_eff}, "
          f"data {me.n_data_sym}/stream)  k={me.k} n={me.n}")
    print(f"  CSI={est}  detect={crc}  tau_ns={list(c.tau_ns)} "
          f"cal={c.cal_amp_error}/{c.cal_pha_error}")

    ebno_list = c.ebno_grid
    variants = {vname: {pc: [] for pc in pcs} for vname in models}
    snr_list = []
    t_start = time.time()

    for i, ebno in enumerate(ebno_list):
        print(f"\n  [Eb/N0 {i + 1:2d}/{len(ebno_list)}] {ebno:6.1f} dB",
              flush=True)
        for vname, model in models.items():
            # All precoders share the same channel / bits / noise realizations.
            res = sim_ber_many(
                model, ebno,
                {pc: {"precoder": pcs[pc]()} for pc in pcs},
                batch_size=c.num_trials, max_mc_iter=c.num_mc_batches,
                num_target_block_errors=c.num_target_block_errors,
                target_bler=c.target_bler, seed=c.seed + i, device=dev,
                verbose=False,
                on_batch=lambda it, tot, eb, stats, vname=vname: print(
                    f"    [{vname} iter {it:2d}/{tot}] "
                    + "  ".join(f"{n}:BLER={s['bler']:.4f}" for n, s in stats.items()),
                    flush=True),
            )
            for pc, (bler, ber, _k) in res.items():
                variants[vname][pc].append(bler)
        from sionna.phy.utils import ebnodb2no
        no = float(ebnodb2no(ebno, me.bits_sym, me.code_rate, me.rg))
        snr_list.append(-10.0 * torch.log10(torch.tensor(no)).item())
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(ebno_list) - i - 1)
        print(f"    ({elapsed:6.1f}s elapsed, ETA {eta:6.1f}s)", flush=True)

    # Final table: every precoder x link variant.
    curves = {f"{pc} {vname}": variants[vname][pc]
              for vname in models for pc in pcs}
    final = {"ebno_db": ebno_list, "snr_db": snr_list,
             "curves": curves, "ber": {n: [0.0] * len(ebno_list) for n in curves},
             "k": me.k}
    print("\n===== BLER (shared realizations per variant) =====")
    print_curve_table(final)

    tag = scenario_tag(c.num_trps, c.rank, c.fft_size, c.qam_order, c.code_rate,
                       c.channel_kind, est=not c.perfect_csi,
                       num_dmrs_symbols=len(c.pilot_symbols),
                       err=c.cal_amp_error is not None or c.cal_pha_error is not None)
    meta = {"scenario_tag": tag, "direction": "downlink", "num_trps": c.num_trps,
            "num_tx_ant": c.num_tx_ant, "num_ue_ant": c.num_ue_ant,
            "rank": c.rank, "qam_order": c.qam_order, "code_rate": c.code_rate,
            "channel_kind": c.channel_kind, "cdl_model": c.cdl_model,
            "perfect_csi": c.perfect_csi, "tau_ns": list(c.tau_ns),
            "cal_amp_error": c.cal_amp_error, "cal_pha_error": c.cal_pha_error,
            "title": f"DL DMIMO BLER ({tag})"}
    out = a.out or _stamp(os.path.join(common.ROOT, "out", "dmimo",
                                       f"linklevel_{tag}.png"))
    paths = save_curves(out, final, meta)
    print("Saved ->", paths["png"])
    print("Saved ->", paths["csv"])
    print("Saved ->", paths["json"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
