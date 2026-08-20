"""Link-level DL DMIMO (rank 1-4) CLI driver (Sionna Block style).

Precoded MRT/CJT/TypeI / NN-PMI transmission over per-TRP CDL channels with
configurable timing/calibration errors, DMRS LS channel estimation and 5G NR
TB-CRC error detection. BLER curves on the Eb/N0 axis, all precoders sharing
the same realizations per point (:func:`dmimo.sim.sim_ber_many`).

Run::

    python examples\\dmimo_linklevel.py --rank 2 --channel cdl --ebno-start -5 --ebno-stop 15
"""
import argparse
import os
import sys
import time

import common  # noqa: E402
import torch  # noqa: E402

from dmimo import (DLModel, IndependentMRT, CJTPrecoder, TypeICodebook,
                   sim_ber_many, save_curves, print_curve_table)


def _stamp(path):
    root, ext = os.path.splitext(path)
    return f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--channel", default="cdl", help="simple|tdl|cdl|uma|umi")
    p.add_argument("--cdl-model", default="C")
    p.add_argument("--trps", type=int, default=3)
    p.add_argument("--bs-ant", type=int, default=32)
    p.add_argument("--ue-ant", type=int, default=4)
    p.add_argument("--qam", type=int, default=4)
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--precoder", default="all", help="mrt|cjt|type1|all")
    p.add_argument("--cal-amp-error", type=float, default=0.1)
    p.add_argument("--cal-pha-error", type=float, default=0.1)
    p.add_argument("--tau-ns", default="0,130,260")
    p.add_argument("--granularity", default="SC", help="SC|RB|SC_RB_granular")
    p.add_argument("--no-crc", action="store_true")
    p.add_argument("--perfect-csi", action="store_true")
    p.add_argument("--pilot-boost-db", type=float, default=0.0)
    p.add_argument("--n-symbols", type=int, default=14)
    p.add_argument("--pilot-symbols", default="2,11")
    p.add_argument("--fft-size", type=int, default=76)
    p.add_argument("--ebno-start", type=float, default=-5.0)
    p.add_argument("--ebno-stop", type=float, default=19.0)
    p.add_argument("--ebno-step", type=float, default=4.0)
    p.add_argument("--mc-iter", type=int, default=5)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tau = [float(x) * 1e-9 for x in a.tau_ns.split(",")]
    pilots = [int(x) for x in a.pilot_symbols.split(",")]

    def mk(num_trps, err):
        base = dict(num_trps=num_trps, num_bs_ant=a.bs_ant, num_ue_ant=a.ue_ant,
                    channel_kind=a.channel, cdl_model=a.cdl_model,
                    delay_spread=100e-9, speed=0.0, pathloss=False,
                    trp_distances=[100.0] if num_trps == 1 else [100.0, 200.0, 350.0],
                    granularity=a.granularity, subcarrier_spacing=15e3,
                    fft_size=a.fft_size, num_guard_carriers=(5, 6), dc_null=True,
                    n_symbols=a.n_symbols,
                    pilot_ofdm_symbol_indices=pilots,
                    pilot_boost_db=a.pilot_boost_db, cyclic_prefix_length=6,
                    qam_order=a.qam, code_rate=a.rate, rank=a.rank,
                    use_crc=not a.no_crc, perfect_csi=a.perfect_csi, device=dev)
        if err and num_trps > 1:
            base.update(tau_seconds=tau[:num_trps], cal_amp_error=a.cal_amp_error,
                        cal_pha_error=a.cal_pha_error)
        else:
            base.update(tau_seconds=[0.0] * num_trps,
                        cal_amp_error=None, cal_pha_error=None)
        return DLModel(**base)

    precoders = {
        "MRT": lambda: IndependentMRT(rank=a.rank),
        "CJT": lambda: CJTPrecoder(rank=a.rank),
        "TypeI-wide": lambda: TypeICodebook(rank=a.rank),
    }
    sel = list(precoders) if a.precoder.lower() == "all" else \
        [{"mrt": "MRT", "cjt": "CJT", "type1": "TypeI-wide",
          "typei": "TypeI-wide"}.get(a.precoder.lower(), a.precoder)]
    pcs = {name: precoders[name] for name in sel}

    models = {"singleTRP": mk(1, False),
              "3TRP-coherent": mk(a.trps, False),
              "3TRP+err": mk(a.trps, True)}
    me = models["3TRP+err"]
    print(f"== DL DMIMO link-level: {a.channel}{a.cdl_model} {a.trps}TRP/{a.bs_ant}ant "
          f"rank={a.rank} QAM{a.qam} rate={a.rate} ==")
    print(f"  grid={a.n_symbols}sym x {a.fft_size}sc (eff {me.n_eff}) "
          f"k={me.k} n={me.n}  CSI={'perfect' if a.perfect_csi else 'LS'}")
    print(f"  tau(ns)={a.tau_ns} cal={a.cal_amp_error}/{a.cal_pha_error}")

    n = int((a.ebno_stop - a.ebno_start) / a.ebno_step + 1e-9) + 1
    ebno_list = [round(a.ebno_start + i * a.ebno_step, 6) for i in range(max(n, 1))]
    variants = {vname: {pc: [] for pc in pcs} for vname in models}
    snr_list = []
    t0 = time.time()
    for i, ebno in enumerate(ebno_list):
        print(f"\n  [Eb/N0 {i + 1:2d}/{len(ebno_list)}] {ebno:6.1f} dB", flush=True)
        for vname, model in models.items():
            res = sim_ber_many(model, ebno,
                               {pc: {"precoder": pcs[pc]()} for pc in pcs},
                               batch_size=a.batch, max_mc_iter=a.mc_iter,
                               num_target_block_errors=1000, target_bler=1e-3,
                               seed=i, device=dev, verbose=False,
                               on_batch=lambda it, tot, eb, stats, vname=vname: print(
                                   f"    [{vname} iter {it:2d}/{tot}] "
                                   + "  ".join(f"{n}={s['bler']:.3f}"
                                               for n, s in stats.items()),
                                   flush=True))
            for pc, (bler, _ber, _k) in res.items():
                variants[vname][pc].append(bler)
        from sionna.phy.utils import ebnodb2no
        no = float(ebnodb2no(ebno, me.bits_sym, me.code_rate, me.rg))
        snr_list.append(-10.0 * torch.log10(torch.tensor(no)).item())
        el = time.time() - t0
        print(f"    ({el:5.1f}s, ETA {el / (i + 1) * (len(ebno_list) - i - 1):5.1f}s)",
              flush=True)

    curves = {f"{pc} {vname}": variants[vname][pc]
              for vname in models for pc in pcs}
    final = {"ebno_db": ebno_list, "snr_db": snr_list,
             "curves": curves, "ber": {n: [0.0] * len(ebno_list) for n in curves},
             "k": me.k}
    print("\n===== BLER =====")
    print_curve_table(final)

    tag = (f"{a.channel}{a.cdl_model}_{a.trps}trp_rank{a.rank}_{a.fft_size}sc_"
           f"qam{a.qam}_r{a.rate:.2f}").replace(".", "")
    meta = {"direction": "downlink", "channel_kind": a.channel,
            "cdl_model": a.cdl_model, "num_trps": a.trps,
            "rank": a.rank, "qam_order": a.qam, "code_rate": a.rate,
            "perfect_csi": a.perfect_csi, "tau_ns": a.tau_ns,
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
