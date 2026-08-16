"""modelTrain.py — flat training of the MLP-Mixer subband-PMI precoder.

Structure follows ``modelTrain_mixer.py``: system constants at module level, a
modular loss, and data loading / model init / optimizer init / training /
testing all implemented flat inside ``main()`` (1-2 nesting levels, no hidden
closures).

Contract
--------
* **Model input** is the Type I **wideband PMI**, computed explicitly from the
  *error-free* channel ``h_clean`` (:func:`dmimo.nn_pmi.wideband_pmi`) and fed
  to the model at the first level: ``w_sub = model(W_pmi)``.
* **Loss** is the negative achievable rate on the *error-corrupted* channel
  ``h_err = link.error.apply(h_clean)`` (the DMIMO scenario: TX precodes from
  clean CSI, transmission runs through the errors).

Run::

    python examples\\modelTrain.py --config configs\\dmimo_linklevel.yaml
"""
import argparse
import csv
import math
import os
import sys
import time

import common  # noqa: E402
import torch

from dmimo.config import load_dmimo_config, scenario_tag
from dmimo import build_link
from dmimo import (
    MLPMixerSubbandPMI,
    NNMixerPMI,
    expand_subband_to_subcarriers,
    save_model,
    wideband_pmi,
)
from dmimo import CJTPrecoder, TypeICodebook

# =====================================================================
# System parameters (edit here or override via --flags)
# =====================================================================
DEVICE = "cuda:0"
SUBBAND_SIZE = 12          # subcarriers per subband
EPOCHS = 60
STEPS_PER_EPOCH = 16       # gradient steps per epoch
BATCH_SIZE = 128           # channel samples per gradient step
LR = 1e-3
SNR_DB = 0.0               # fixed training SNR (rate loss)
EARLY_STOP_PATIENCE = 8    # stop after N epochs without val improvement
MIXER_BLOCKS = 2
MIXER_HIDDEN = 64
VAL_STEPS = 4
CJT_EVERY = 10             # recompute the CJT baseline rate every N epochs (cached otherwise)
SEED = 0


def rate_loss(w_sub, h_err, link, snr_db, subband_size):
    """Modular loss: negative achievable rate on the errored channel.

    ``w_sub : [B, K, 2P, r, S]`` (model output) -> scalar ``-mean rate``.
    ``link`` provides ``_combine`` / ``_rate`` (DMIMODownlink).
    """
    w = expand_subband_to_subcarriers(w_sub, link.n_subcarriers, subband_size)
    h_eff = link._combine(h_err, w)          # [B, D, r, N] transmission channel
    no = 10.0 ** (-snr_db / 10.0)
    return -link._rate(h_eff, no).mean()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dmimo_linklevel.yaml")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--snr-db", type=float, default=SNR_DB)
    p.add_argument("--subband-size", type=int, default=None,
                   help="subcarriers per subband (default: config `subband_size`)")
    p.add_argument("--early-stop", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--cjt-every", type=int, default=CJT_EVERY,
                   help="recompute CJT baseline rate every N epochs (cached otherwise)")
    p.add_argument("--out", default=None,
                   help="model output path (default: out/dmimo/model/nn_pmi_mixer_<tag>.pt)")
    p.add_argument("--device", default=DEVICE)
    a = p.parse_args()

    torch.manual_seed(SEED)
    dev = a.device
    c = load_dmimo_config(a.config)
    subband_size = a.subband_size or c.subband_size

    # ---- 1. data loading: channel source (same geometry / errors as scenario) --
    link = build_link(num_trps=c.num_trps, num_tx_ant=c.num_tx_ant,
                      num_ue_ant=c.num_ue_ant, n_subcarriers=c.n_subcarriers,
                      subcarrier_spacing=c.subcarrier_spacing_khz * 1e3,
                      tau_seconds=c.tau_seconds,
                      cal_amp_error=c.cal_amp_error, cal_pha_error=c.cal_pha_error,
                      granularity=c.granularity, channel_kind=c.channel_kind,
                      pathloss=c.pathloss, trp_distances=c.trp_distances_m,
                      carrier_frequency=c.carrier_frequency)

    # ---- 2. model init ---------------------------------------------------------
    num_subbands = int(math.ceil(c.n_subcarriers / subband_size))
    model = MLPMixerSubbandPMI(num_ant=c.num_tx_ant, rank=c.rank,
                               num_subbands=num_subbands, num_trps=c.num_trps,
                               blocks=MIXER_BLOCKS, hidden=MIXER_HIDDEN)
    model.to(dev)

    # ---- 3. optimizer init -----------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)

    # ---- 4. training -----------------------------------------------------------
    type1 = TypeICodebook(rank=c.rank, subband_size=c.n_subcarriers)   # baseline 1
    cjt = CJTPrecoder(rank=c.rank, subband_size=subband_size)          # baseline 2 (same granularity)
    no = 10.0 ** (-a.snr_db / 10.0)
    hist = {"train_loss": [], "train_rate": [], "val_loss": [], "val_rate": [],
            "val_type1_rate": [], "val_cjt_rate": [], "gain_over_type1": [],
            "gap_to_cjt": [], "cjt_computed": [], "epoch_time": []}
    best_val, stale = None, 0
    cached_cjt = None
    for epoch in range(a.epochs):
        start = time.time()
        compute_cjt = (a.cjt_every >= 1) and (epoch % a.cjt_every == 0)
        model.train()
        train_loss = 0.0
        for step in range(a.steps_per_epoch):
            h_clean = link.channel.sample(a.batch_size, dev)   # data (clean)
            h_err = link.error.apply(h_clean)                  # errored (for loss)
            w_pmi, _ = wideband_pmi(h_clean, c.rank)           # PMI from CLEAN h
            w_sub = model(w_pmi)                               # model input = PMI
            loss = rate_loss(w_sub, h_err, link, a.snr_db, subband_size)  # modular loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / a.steps_per_epoch
        hist["train_loss"].append(avg_train_loss)
        hist["train_rate"].append(-avg_train_loss)             # bps/Hz

        # ---- 5. testing --------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            type1_rate = 0.0
            cjt_rate = 0.0 if compute_cjt else 0.0
            for _ in range(VAL_STEPS):
                h_clean = link.channel.sample(a.batch_size, dev)
                h_err = link.error.apply(h_clean)
                w_pmi, _ = wideband_pmi(h_clean, c.rank)
                w_sub = model(w_pmi)
                val_loss += rate_loss(w_sub, h_err, link, a.snr_db, subband_size).item()
                # Type I-wideband baseline rate on the same errored channel
                w_t = type1(h_clean)
                h_eff_t = link._combine(h_err, w_t)
                type1_rate += link._rate(h_eff_t, no).mean().item()
                # CJT baseline (SVD upper bound): recomputed every `cjt_every` epochs
                if compute_cjt:
                    w_c = cjt(h_clean)
                    h_eff_c = link._combine(h_err, w_c)
                    cjt_rate += link._rate(h_eff_c, no).mean().item()
            val_loss /= VAL_STEPS
            type1_rate /= VAL_STEPS
            if compute_cjt:
                cached_cjt = cjt_rate / VAL_STEPS
            cjt_rate = cached_cjt
        hist["val_loss"].append(val_loss)
        hist["val_rate"].append(-val_loss)
        hist["val_type1_rate"].append(type1_rate)
        hist["val_cjt_rate"].append(cjt_rate)
        hist["cjt_computed"].append(bool(compute_cjt))
        hist["gain_over_type1"].append(-val_loss - type1_rate) # bps/Hz over TypeI
        hist["gap_to_cjt"].append(-val_loss - cjt_rate)        # bps/Hz vs CJT (<0 below)
        hist["epoch_time"].append(time.time() - start)

        cjt_mark = "" if compute_cjt else "*"   # * = cached from last CJT epoch
        print(f"Epoch {epoch:3d}/{a.epochs}  TrainLoss: {avg_train_loss:.4f} "
              f"TrainRate: {hist['train_rate'][-1]:.4f}  TestLoss: {val_loss:.4f} "
              f"TestRate: {hist['val_rate'][-1]:.4f}  TypeI: {type1_rate:.4f} "
              f"CJT: {cjt_rate:.4f}{cjt_mark}  "
              f"GainOverTypeI: {hist['gain_over_type1'][-1]:+.4f} "
              f"GapToCJT: {hist['gap_to_cjt'][-1]:+.4f}  "
              f"Time: {hist['epoch_time'][-1]:.2f}s")

        # early stop + track best
        if best_val is None or val_loss < best_val - 1e-6:
            best_val, stale = val_loss, 0
        else:
            stale += 1
            if stale >= a.early_stop:
                print(f"Early stop at epoch {epoch} (best TestLoss {best_val:.4f})")
                break

    # ---- 6. save model ---------------------------------------------------------
    err = c.cal_amp_error is not None or c.cal_pha_error is not None
    tag = scenario_tag(c.num_trps, c.rank, c.n_subcarriers, c.qam_order,
                       c.code_rate, c.channel_kind, est=True,
                       num_dmrs_symbols=c.num_dmrs_symbols, err=err,
                       subband_size=subband_size)
    out = a.out or os.path.join(
        common.ROOT,
        "out", "dmimo", "model", f"nn_pmi_mixer_{tag}.pt")
    meta = dict(
        tag=tag, arch="mixer",
        scenario=dict(num_trps=c.num_trps, rank=c.rank, num_ue_ant=c.num_ue_ant,
                      num_tx_ant=c.num_tx_ant, n_subcarriers=c.n_subcarriers,
                      qam_order=c.qam_order, code_rate=c.code_rate,
                      channel_kind=c.channel_kind, pathloss=c.pathloss,
                      tau_ns=list(c.tau_ns), cal_amp_error=c.cal_amp_error,
                      cal_pha_error=c.cal_pha_error, granularity=c.granularity,
                      use_channel_estimation=c.use_channel_estimation,
                      num_dmrs_symbols=c.num_dmrs_symbols,
                      subband_size=subband_size),
        train_loss=hist["train_loss"][-1], val_loss=hist["val_loss"][-1],
        train_rate=hist["train_rate"][-1], val_rate=hist["val_rate"][-1],
        val_type1_rate=hist["val_type1_rate"][-1],
        val_cjt_rate=hist["val_cjt_rate"][-1],
        gain_over_type1=hist["gain_over_type1"][-1],
        gap_to_cjt=hist["gap_to_cjt"][-1],
        epochs=len(hist["train_loss"]), steps_per_epoch=a.steps_per_epoch,
        batch_size=a.batch_size, lr=a.lr, snr_db=a.snr_db,
        early_stop_patience=a.early_stop, blocks=MIXER_BLOCKS,
        hidden=MIXER_HIDDEN, cjt_every=a.cjt_every,
        time=time.strftime("%Y%m%d_%H%M%S"),
        history={k: [round(x, 6) for x in v] for k, v in hist.items()})
    # wrap the trained network in the Precoder-protocol wrapper so the checkpoint
    # can be loaded directly by the link-level example (load_model -> NNMixerPMI)
    wrapper = NNMixerPMI(rank=c.rank, subband_size=subband_size,
                         n_subcarriers=c.n_subcarriers, num_ant=c.num_tx_ant,
                         num_trps=c.num_trps, blocks=MIXER_BLOCKS,
                         hidden=MIXER_HIDDEN, device=dev)
    wrapper.net.load_state_dict(model.state_dict())
    save_model(wrapper, out, meta)
    print("Saved ->", out)
    # also save the training history as CSV (one row per epoch, opens in Excel)
    hist_out = os.path.splitext(out)[0] + "_history.csv"
    cols = ["epoch", "train_loss", "train_rate", "val_loss", "val_rate",
            "val_type1_rate", "val_cjt_rate", "gain_over_type1", "gap_to_cjt",
            "cjt_computed", "epoch_time"]
    with open(hist_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(len(hist["train_loss"])):
            w.writerow([i] + [round(hist[k][i], 6) if k != "cjt_computed"
                              else int(hist[k][i]) for k in cols[1:]])
    print("History ->", hist_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
