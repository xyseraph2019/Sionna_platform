"""Link-level DMIMO (rank 1-4): real LDPC+QAM over multi-TRP downlink, MRT/CJT/
TypeI-wideband precoding, configurable timing/calibration errors, DMRS-like LS
channel estimation (P1) and 5G NR TB-CRC error detection (P2).
Run: python examples\\dmimo_linklevel.py --rank 2 --sweep
"""
import os, sys, argparse, time


def _stamp(path):
    """Append a timestamp so a rerun does not overwrite the previous figure."""
    root, ext = os.path.splitext(path)
    return f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from dmimo.experiment import build_link
from dmimo.link_level import LinkLevelDMIMO
from dmimo.precoding import IndependentMRT, CJTPrecoder, TypeICodebook

DIST = (100., 200., 350.)

def snr_range(start, stop, step):
    """Arithmetic SNR grid (dB): start, +step, ..., <= stop."""
    n = int((stop - start) / step + 1e-9) + 1
    return [round(start + i * step, 6) for i in range(max(n, 1))]

def snr_at_bler(bler, snrs, target=0.1):
    ps = pb = None
    for s, b in zip(snrs, bler):
        if ps is not None and pb > target >= b:
            return ps + (pb - target) / (pb - b) * (s - ps)
        ps, pb = s, b
    return None
def curve(ll, pc, snrs, batch):
    return [ll.block(batch, s, pc)[0] for s in snrs]



def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--nsc", type=int, default=64)
    p.add_argument("--qam", type=int, default=16)
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--precoder", type=str, default="all", help="mrt|cjt|type1|all")
    p.add_argument("--cal-amp-error", type=float, default=0.1)
    p.add_argument("--cal-pha-error", type=float, default=0.1)
    p.add_argument("--tau-ns", type=str, default="0,130,260")
    p.add_argument("--granularity", type=str, default="SC", help="SC|RB|SC_RB_granular")
    p.add_argument("--est-density", type=float, default=0.0,
                   help="pilot density 0<d<1 enables DMRS-like LS estimation (0=perfect CSI)")
    p.add_argument("--no-crc", action="store_true", help="plain LDPC + full-bit BLER instead of TB CRC")
    p.add_argument("--pilot-boost-db", type=float, default=0.0, help="DMRS pilot energy boost (dB)")
    p.add_argument("--n-symbols", type=int, default=14, help="OFDM symbols per slot")
    p.add_argument("--dmrs-symbol", type=int, default=2, help="pilot-only DMRS symbol index")
    p.add_argument("--num-dmrs-symbols", type=int, default=1, help="number of pilot-only DMRS symbols")
    p.add_argument("--snr-start", type=float, default=-14.0, help="SNR sweep start (dB)")
    p.add_argument("--snr-stop", type=float, default=8.0, help="SNR sweep stop (dB)")
    p.add_argument("--snr-step", type=float, default=2.0, help="SNR sweep step (dB)")
    p.add_argument("--sweep", action="store_true", help="rank1-4 SNR@10%% table")
    args = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tau = [float(x) * 1e-9 for x in args.tau_ns.split(",")]
    args.rank = min(args.rank, 4)
    args.snr_db = snr_range(args.snr_start, args.snr_stop, args.snr_step)

    def mk(num_trps, err, ue_ant):
        return build_link(num_trps=num_trps, num_tx_ant=4, num_ue_ant=ue_ant,
                          n_subcarriers=args.nsc, channel_kind="uma", pathloss=False,
                          trp_distances=[DIST[0]] if num_trps == 1 else DIST,
                          tau_seconds=(tau if (err and num_trps > 1) else [0.0] * num_trps),
                          cal_amp_error=(args.cal_amp_error if err else None),
                          cal_pha_error=(args.cal_pha_error if err else None),
                          granularity=args.granularity)
    precoders = {
        "MRT": lambda r: IndependentMRT(rank=r),
        "CJT": lambda r: CJTPrecoder(rank=r),
        "TypeI-wide": lambda r: TypeICodebook(rank=r, subband_size=args.nsc),
    }
    sel = list(precoders) if args.precoder.lower() == "all" else \
        [{"mrt": "MRT", "cjt": "CJT", "type1": "TypeI-wide", "typei": "TypeI-wide"}.get(args.precoder.lower(), args.precoder)]
    return _run(args, mk, precoders, sel, dev)

def _run(args, mk, precoders, sel, dev):
    r = args.rank
    kw = dict(use_channel_estimation=args.est_density > 0.0,
              est_density=max(args.est_density, 0.25), use_crc=not args.no_crc,
              pilot_boost_db=args.pilot_boost_db, n_symbols=args.n_symbols,
              dmrs_symbol=args.dmrs_symbol, num_dmrs_symbols=args.num_dmrs_symbols)
    l1 = LinkLevelDMIMO(mk(1, False, r), qam_order=args.qam, code_rate=args.rate, rank=r, device=dev, **kw)
    l0 = LinkLevelDMIMO(mk(3, False, r), qam_order=args.qam, code_rate=args.rate, rank=r, device=dev, **kw)
    le = LinkLevelDMIMO(mk(3, True, r), qam_order=args.qam, code_rate=args.rate, rank=r, device=dev, **kw)
    est = (f"LS(d={args.est_density}, pilot={l0.num_pilots}/{l0.n_data})"
           if args.est_density > 0.0 else "perfect CSI")
    print(f"rank={r} k/n={l0.k}/{l0.n} QAM={args.qam} rate={args.rate} "
          f"tau(ns)={args.tau_ns} cal_amp={args.cal_amp_error} cal_pha={args.cal_pha_error}")
    print(f"  CSI={est}  detect={'TB-CRC' if not args.no_crc else 'full-bit compare'}")
    curves = {}
    snrs = args.snr_db
    for name in sel:
        pc = precoders[name](r)
        curves[f"{name} singleTRP"] = curve(l1, pc, snrs, args.batch)
        curves[f"{name} 3TRP coherent"] = curve(l0, pc, snrs, args.batch)
        curves[f"{name} 3TRP+err"] = curve(le, pc, snrs, args.batch)
    print("  SNR   | " + " | ".join(f"{n:>20}" for n in curves))
    for i, s in enumerate(snrs):
        print(f"  {s:4g} | " + " | ".join(f"{curves[n][i]:20.3f}" for n in curves))

    fig, ax = plt.subplots(figsize=(10, 6))
    for n, ys in curves.items():
        ax.plot(snrs, ys, "o-", label=n)
    ax.axhline(0.1, color="grey", ls="--", lw=1, label="10% BLER")
    ax.set_yscale("log"); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BLER")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "dmimo")
    os.makedirs(out_dir, exist_ok=True)
    out = _stamp(os.path.join(out_dir, f"linklevel_bler_rank{r}.png"))
    fig.savefig(out, dpi=150)
    print("Saved BLER figure ->", out)

    if args.sweep:
        coarse = [-10, -6, -2, 2, 6, 10]
        print("== SNR@10% BLER across rank 1-4 ==")
        print("  rank | " + " | ".join(f"{n:>18}" for n in ["MRTcoh", "MRT+err", "CJT+err", "TypeI-wide+err"]))
        for rr in range(1, 5):
            L0 = LinkLevelDMIMO(mk(3, False, rr), qam_order=args.qam, code_rate=args.rate, rank=rr, device=dev, **kw)
            Le = LinkLevelDMIMO(mk(3, True, rr), qam_order=args.qam, code_rate=args.rate, rank=rr, device=dev, **kw)
            fm = lambda v: "-" if v is None else f"{v:6.2f}"
            mc = snr_at_bler(curve(L0, IndependentMRT(rank=rr), coarse, 128), coarse)
            row = [mc] + [snr_at_bler(curve(Le, precoders[n](rr), coarse, 128), coarse) for n in ["MRT", "CJT", "TypeI-wide"]]
            print(f"  {rr:4d} | " + " | ".join(f"{fm(v):>18}" for v in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())

