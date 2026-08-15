"""
Downlink precoding.

Each TRP computes its own precoder **independently from its own channel only**,
ignoring the other TRPs and the timing/calibration errors. This is exactly the
reference behaviour whose gain the errors then erode.

A :class:`Precoder` protocol is exposed so a *learnable* (e.g. neural) precoder
can be plugged in later to jointly optimise the TRP precoders given the channel
and the (estimated) error statistics.

Precoder convention: ``w : [B, K, N_t, N]`` with unit-norm vectors over ``N_t``
per (batch, TRP, subcarrier).
"""
from __future__ import annotations

from typing import Protocol

import math

import torch


class Precoder(Protocol):
    """Callable mapping a channel to per-TRP precoders.

    Optional attribute ``precodes_from_errors : bool`` (default ``False``):
    ``True`` means the precoder should be computed from the *error-corrupted*
    channel ``h_err`` (the BS observes/compensates timing+calibration errors,
    e.g. CJT), ``False`` means it uses the clean channel (limited feedback, e.g.
    Type I / NN-PMI whose PMI carries no error information).
    """

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """Return precoders ``[B, K, N_t, N]`` from the channel ``[B, K, N_t, N]``."""
        ...


def _normalize(w: torch.Tensor) -> torch.Tensor:
    return w / (w.abs().square().sum(dim=2, keepdim=True).sqrt() + 1e-12)


def _canonical_basis(V: torch.Tensor) -> torch.Tensor:
    """Per-subcarrier *canonical* orthonormal basis for the SVD precoding subspace.

    ``V : [B, M, r, N]`` (batch x antennas x layers x subcarriers). The SVD
    right singular vectors have an arbitrary per-subcarrier phase *and* an
    arbitrary unitary rotation within degenerate (near-equal singular value)
    subspaces; CPU LAPACK and GPU cusolver resolve this differently, so the raw
    basis is not smooth in frequency and sparse-pilot LS channel estimation of
    the effective channel fails (observed est-MSE ~80-100% on GPU).

    Rotating each subcarrier's basis to the canonical form ``V' = V W^{-1}`` with
    ``W = V[:r]`` (first ``r`` antennas block), followed by a positive-diagonal
    QR, gives a unique basis that is *independent* of the arbitrary SVD rotation
    and therefore smooth across subcarriers for a smooth channel.
    """
    B, M, r, N = V.shape
    eye = torch.eye(r, dtype=V.dtype, device=V.device).view(1, r, r)
    out = []
    for k in range(N):
        v = V[:, :, :, k]                            # [B, M, r] orthonormal
        W = v[:, :r, :]                              # [B, r, r] first-r-antennas block
        W_inv = torch.linalg.inv(W + 1e-9 * eye)
        vp = v @ W_inv                               # V'[:r] = I
        q, R = torch.linalg.qr(vp)
        d = R.diagonal(dim1=-2, dim2=-1)             # make diagonal real positive
        sgn = torch.where(d.abs() > 1e-6, d / (d.abs() + 1e-12),
                          torch.ones_like(d))
        out.append(q * sgn.conj().unsqueeze(-2))
    return torch.stack(out, dim=-1)


class IndependentMRT:
    """Maximum-ratio (matched-filter) per TRP from its own channel, rank ``r``.

    For a channel ``h : [B, K, D, Nt, N]`` with ``D`` UE antennas, the rank-``r``
    precoder matches the first ``r`` UE receive antennas: column ``l`` is the
    normalised ``h[b, t, l, :, :]^H`` (conjugate beamforming), one per layer.
    """

    def __init__(self, rank: int = 1):
        self.rank = int(rank)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        B, K, D, Nt, N = h.shape
        r = min(self.rank, D)
        w = h.conj().permute(0, 1, 3, 2, 4)   # [B, K, Nt, D, N]
        w = w[..., :r, :]                     # [B, K, Nt, r, N]
        return _normalize(w)                  # unit columns over Nt


class IndependentZF:
    """Zero-forcing per TRP from its own channel, rank ``r``.

    ``W = H^H (H H^H + alpha I)^-1`` per TRP per subcarrier, normalised columns.
    ``alpha=0`` gives pure ZF; a small regularisation avoids ill-conditioning.
    """

    def __init__(self, rank: int = 1, alpha: float = 1e-4):
        self.rank = int(rank)
        self.alpha = float(alpha)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        B, K, D, Nt, N = h.shape
        r = min(self.rank, D)
        # G_t[k] = H_t[k] H_t[k]^H + alpha I : [B,K,D,D,N]
        hh = torch.einsum("bkdan,bkdbn->bkabn", h, h.conj())       # [B,K,D,D,N]
        eye = torch.eye(D, dtype=h.real.dtype, device=h.device)
        inv = torch.linalg.inv(hh + self.alpha * eye.view(1, 1, D, D, 1))  # [B,K,D,D,N]
        # W = H^H (H H^H + aI)^-1  ->  [B,K,Nt,D,N]
        w = torch.einsum("bkman,bkmln->bkaln", h.conj(), inv)
        w = w[..., :r, :]
        return _normalize(w)


def make_mrt(rank: int = 1) -> IndependentMRT:
    return IndependentMRT(rank=rank)


class CJTPrecoder:
    """Coherent Joint Transmission (CJT): joint precoding across all TRPs.

    Concatenates the per-TRP channels into a joint channel ``H_joint : [B, D, K*Nt, N]``
    and applies *joint* eigen-beamforming (per-subcarrier SVD), taking the ``r``
    dominant right singular vectors as the joint precoder. The result is split
    back into per-TRP blocks ``w : [B, K, Nt, r, N]``. This is the DMIMO upper
    bound that exploits coherent combining across TRPs.

    ``precodes_from_errors = True``: CJT has full channel state at the BS and the
    timing/calibration errors are observable (reciprocity / timing advance), so
    it precodes from the *error-corrupted* channel and is robust to them.

    ``subband_size``: ``None`` -> per-subcarrier joint SVD (finest granularity);
    an integer ``S`` -> one joint precoder per ``S`` subcarriers (eigen-beamformer
    of the subband-averaged Gram), broadcast over the subband -- same granularity
    as the learned NN-PMI (``S=12`` RB-level by default; changeable, e.g. ``48``).

    Parameters
    ----------
    rank : int
        Number of layers.
    subband_size : int | None
        Subcarriers per subband (``None`` = per-subcarrier).
    """

    precodes_from_errors = True

    def __init__(self, rank: int = 1, subband_size=None):
        self.rank = int(rank)
        self.subband_size = None if subband_size is None else int(subband_size)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        B, K, D, Nt, N = h.shape
        M = K * Nt
        r = min(self.rank, D, M)
        # H_joint[b,n] = [H_1 ... H_K] : [D, M]
        Hj = h.permute(0, 2, 1, 3, 4).reshape(B, D, M, N)   # [B,D,M,N]

        if self.subband_size is None:
            # ---- per-subcarrier joint SVD (finest granularity) ----------------
            Hm = Hj.permute(0, 3, 1, 2)                          # [B,N,D,M]
            _, _, Vh = torch.linalg.svd(Hm, full_matrices=False)  # Vh [B,N,M,M]
            # Right singular vectors are the ROWS of Vh; take the ``r`` dominant ones
            # as precoder columns: Wj : [B, M, r, N].
            Wj = Vh[..., :r, :].conj().permute(0, 3, 2, 1).contiguous()
        else:
            # ---- subband-level joint eigen-beamforming ------------------------
            # One precoder per `subband_size` subcarriers, broadcast over the
            # subband -- same granularity as the learned NN-PMI (RB-level by
            # default). The right singular vectors of the subband-stacked channel
            # ``[D*sb, M]`` are the eigenvectors of ``sum_k H[k]^H H[k]`` (they
            # maximise the average projected power over the subband) but computed
            # via SVD, which is numerically robust (batched ``eigh`` on the Gram
            # fails for near-degenerate eigenvalues on GPU).
            sb = self.subband_size
            S = (N + sb - 1) // sb
            pad = S * sb - N
            Hp = Hj
            if pad:
                Hp = torch.nn.functional.pad(Hp, (0, pad))
            Hs = Hp.reshape(B, D, M, S, sb)                       # [B,D,M,S,sb]
            Hstack = Hs.permute(0, 3, 1, 4, 2).reshape(B, S, D * sb, M)  # [B,S,D*sb,M]
            _, _, Vh = torch.linalg.svd(Hstack,
                                        full_matrices=False)      # Vh [B,S,D*sb,M]
            Wj = Vh[..., :r, :].conj().permute(0, 3, 2, 1).contiguous()  # [B,M,r,S]
        # Rotation-invariant canonical basis so the effective channel is smooth
        # in frequency and pilot-based LS estimation works (device-independent:
        # GPU/CPU SVDs resolve the basis differently). REQUIRED for LS estimation
        # with sparse (comb) pilots; harmless with dense pilots.
        Wj = _canonical_basis(Wj)
        # broadcast subband precoders to subcarriers (identity if per-subcarrier)
        Wj = Wj[..., (torch.arange(N, device=h.device) // self.subband_size).long()] \
            if self.subband_size is not None else Wj
        Wj = Wj.reshape(B, K, Nt, r, N)                        # split per TRP
        # Per-TRP power normalisation (each TRP has its own power budget, like MRT).
        return Wj / (Wj.abs().square().sum(dim=2, keepdim=True).sqrt() + 1e-12)


def type1_wideband_selection(h: torch.Tensor, rank: int = 1, oversmpl: int = 4) -> dict:
    """Type I wideband beam selection + per-subcarrier dual-polarisation projections.

    ``h : [B, K, D, 2P, N]`` (dual-polarised TRPs). Returns a dict with::

        Vr    [B, K, r, P]      wideband DFT beams (r orthogonal: m, m+P, ...)
        a     [B, K, r, D, N]   projections of ``hp`` (pol group 0) on ``Vr``
        c     [B, K, r, D, N]   projections of ``hm`` (pol group 1) on ``Vr``
        hp, hm, P, r, B, K, N, D

    Shared by :class:`TypeICodebook` and the neural subband-PMI precoder
    (:mod:`dmimo.nn_pmi`) so the wideband PMI semantics stay identical.
    """
    B, K, D, total_ant, N = h.shape
    P = total_ant // 2
    M = P * oversmpl
    r = min(int(rank), total_ant, D)
    from sionna.phy.mimo import grid_of_beams_dft_ula

    V = grid_of_beams_dft_ula(num_ant=P, oversmpl=oversmpl).to(h.device)  # [M, P]
    hp, hm = h[..., :P, :], h[..., P:, :]          # [B,K,D,P,N] two pol groups

    # wideband beam selection: best beam per TRP from per-antenna power
    pp = torch.einsum("mp,bkdpn->bkdmn", V.conj(), hp)  # [B,K,D,M,N]
    pm = torch.einsum("mp,bkdpn->bkdmn", V.conj(), hm)
    beam_power = (pp.abs().square() + pm.abs().square()).sum(dim=(2, 4))  # [B,K,M]
    beam_base = beam_power.argmax(dim=-1)          # [B,K]

    # r orthogonal beams: indices m, m+P, m+2P, ... (mod M)
    l_off = (torch.arange(r, device=h.device) * P).view(1, 1, r)
    beam_idx = (beam_base.unsqueeze(-1) + l_off) % M                 # [B,K,r]
    Vr = V[beam_idx]                                # [B,K,r,P]

    # per-layer selected projections a,c : [B,K,r,D,N]
    a = torch.einsum("bklp,bkdpn->bkldn", Vr.conj(), hp)
    c = torch.einsum("bklp,bkdpn->bkldn", Vr.conj(), hm)
    return dict(Vr=Vr, a=a, c=c, hp=hp, hm=hm, P=P, r=r, B=B, K=K, N=N, D=D)


class TypeICodebook:
    """3GPP Type I codebook precoding (rank 1-4, dual-polarised TRPs).

    Each TRP has a dual-polarised ULA (``Nt = 2P``). For rank ``r`` it selects
    ``r`` mutually orthogonal DFT beams (wideband) from the grid of beams and
    applies a QPSK co-phasing ``phi`` per layer between the two polarisation
    groups. ``phi`` is either wideband (one for the band) or subband (one per
    group of subcarriers).

    Precoder (rank r):  ``w[k][:, l] = (1/sqrt(2)) [v_l ; phi_l[k] * v_l]``,
    ``w : [B, K, 2P, r, N]``.

    Parameters
    ----------
    rank : int
        Number of layers (1..4). Clamped to ``min(rank, 2P, D)`` at runtime.
    oversmpl : int
        DFT grid-of-beams oversampling factor ``O``.
    subband_size : int | None
        Subcarriers per subband. ``None`` -> ``ceil(N / num_subbands)``.
    num_subbands : int
        Number of subbands (used when ``subband_size`` is ``None``).
    """

    def __init__(self, rank: int = 1, oversmpl: int = 4,
                 subband_size=None, num_subbands: int = 4):
        self.rank = int(rank)
        self.oversmpl = int(oversmpl)
        self.subband_size = subband_size
        self.num_subbands = int(num_subbands)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """``h : [B, K, D, 2P, N]`` -> precoders ``w : [B, K, 2P, r, N]``."""
        B, K, D, total_ant, N = h.shape
        sel = type1_wideband_selection(h, self.rank, self.oversmpl)
        a, c, P, r, Vr = sel["a"], sel["c"], sel["P"], sel["r"], sel["Vr"]

        # QPSK co-phasing per layer per subband: argmax_phi sum_{d,k in sub} |a + phi c|^2
        qpsk = torch.tensor([1, 1j, -1, -1j], dtype=torch.complex64, device=h.device)
        comb = a.unsqueeze(-1) + c.unsqueeze(-1) * qpsk                 # [B,K,r,D,N,4]
        power = comb.abs().square().sum(dim=3)                          # [B,K,r,N,4]

        s = self.subband_size or (int(math.ceil(N / self.num_subbands)))
        S = int(math.ceil(N / s))
        pad = S * s - N
        if pad:
            power = torch.nn.functional.pad(power, (0, 0, 0, pad))
        pwr = power.reshape(B, K, r, S, s, 4).sum(dim=4)                # [B,K,r,S,4]
        phi_idx = pwr.argmax(dim=-1)                                    # [B,K,r,S]
        sub_id = (torch.arange(N, device=h.device) // s).long()
        phi = qpsk[phi_idx[..., sub_id]]                                # [B,K,r,N]

        # w[k][:, l] = 1/sqrt(2) [v_l ; phi_l v_l]
        v_r = Vr.unsqueeze(-1).expand(B, K, r, P, N)                    # [B,K,r,P,N]
        top = v_r.permute(0, 1, 3, 2, 4)                                # [B,K,P,r,N]
        bot = (phi.unsqueeze(3) * v_r).permute(0, 1, 3, 2, 4)           # [B,K,P,r,N]
        w = torch.cat([top, bot], dim=2)                                # [B,K,2P,r,N]
        return w / (2.0 ** 0.5)

