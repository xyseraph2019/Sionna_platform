"""
Link adaptation: MCS / CQI selection from measured link quality.

Given a target transport-block error rate (BLER) and an operating SNR, this
module searches the MCS index space and picks the highest-rate MCS whose
predicted (measured) BLER stays below the target. This is the classic
\"select the MCS that maximises throughput subject to a BLER constraint\"
procedure used by link-level simulators and schedulers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from .config import SimConfig
from .simulator import LinkSimulator
from .metrics import LinkMetrics


@dataclass
class MCSConfig:
    """Result of an MCS selection for one SNR point."""

    snr_db: float
    mcs_index: int
    bler: float
    throughput_bps: float
    info_bits_per_tb: int
    cqi: int  # 3GPP CQI 1..15 mapped approximation

    def as_dict(self) -> dict:
        return asdict(self)


def _snr_to_cqi(snr_db: float) -> int:
    """Approximate mapping of SNR (dB) to CQI (1-15) using the classic
    \"set partitioning in hierarchical trees\" (SPHAT)-style thresholds.

    Heuristic table-based approximation; meant as a useful estimate, not a
    spec-exact quantisation.
    """
    thresholds = [
        (-6.0, 1), (-4.0, 2), (-2.0, 3), (0.0, 4), (2.0, 5), (4.0, 6),
        (6.0, 7), (8.0, 8), (10.0, 9), (12.0, 10), (14.0, 11),
        (16.0, 12), (18.0, 13), (20.0, 14), (22.0, 15),
    ]
    cqi = 0
    for thr, val in thresholds:
        if snr_db >= thr:
            cqi = val
    return min(max(cqi, 1), 15)


class LinkAdaptation:
    """Run BLER-driven MCS selection across SNR points.

    Parameters
    ----------
    base_cfg : SimConfig
        Scenario used as template (carrier / channel / device).
    target_bler : float
        BLER constraint for MCS selection (e.g. 0.1).
    num_trials : int
        Monte-Carlo trials used *per MCS candidate per SNR*.
    mcs_candidates : list[int] | None
        MCS indices to search. Defaults to table-1 indices 0..27.
        Indices whose code rate is not supported (e.g. LDPC r < 1/5) are
        skipped automatically.
    """

    def __init__(
        self,
        base_cfg: SimConfig,
        target_bler: float = 0.1,
        num_trials: int = 100,
        mcs_candidates: Optional[List[int]] = None,
    ) -> None:
        self.base_cfg = base_cfg
        self.target_bler = target_bler
        self.num_trials = num_trials
        self.mcs_candidates = mcs_candidates or list(range(0, 28))

    def select_at_snr(self, snr_db: float) -> MCSConfig:
        """Choose the highest-rate MCS whose measured BLER <= target at ``snr_db``."""
        chosen_mcs = None
        chosen_bler = 1.0
        chosen_tp = 0.0
        chosen_bits = 0
        fallback_mcs = None  # lowest MCS that can actually be built

        for mcs in self.mcs_candidates:
            cfg = self._sms_config(self.base_cfg, mcs)
            try:
                sim = LinkSimulator(cfg)
            except Exception:
                # Unsupported code rate (e.g. LDPC r < 1/5): cannot build -> skip.
                continue
            if fallback_mcs is None:
                fallback_mcs = mcs
            m = sim.run_snr(snr_db, self.num_trials)
            if m.bler <= self.target_bler and m.throughput_bps >= chosen_tp:
                chosen_mcs = mcs
                chosen_bler = m.bler
                chosen_tp = m.throughput_bps
                chosen_bits = m.num_info_bits_per_tb

        if chosen_mcs is None:
            # No candidate met the BLER target at this SNR: use the lowest
            # buildable MCS and report its (likely high) BLER.
            chosen_mcs = fallback_mcs if fallback_mcs is not None else self.mcs_candidates[0]
            try:
                sim = LinkSimulator(self._sms_config(self.base_cfg, chosen_mcs))
                m = sim.run_snr(snr_db, self.num_trials)
                chosen_bler = m.bler
                chosen_tp = m.throughput_bps
                chosen_bits = m.num_info_bits_per_tb
            except Exception:
                pass

        return MCSConfig(
            snr_db=snr_db,
            mcs_index=chosen_mcs,
            bler=chosen_bler,
            throughput_bps=chosen_tp,
            info_bits_per_tb=chosen_bits,
            cqi=_snr_to_cqi(snr_db),
        )

    def select_curve(self, snr_db: List[float]) -> List[MCSConfig]:
        """Run MCS selection over a full SNR sweep."""
        return [self.select_at_snr(float(s)) for s in snr_db]

    @staticmethod
    def _sms_config(cfg: SimConfig, mcs_index: int) -> SimConfig:
        """Return a copy of ``cfg`` with a different ``mcs_index``."""
        import copy

        new_cfg = copy.deepcopy(cfg)
        new_cfg.tb.mcs_index = int(mcs_index)
        return new_cfg


def select_mcs_for_snr(
    base_cfg: SimConfig,
    snr_db: float,
    target_bler: float = 0.1,
    num_trials: int = 100,
) -> MCSConfig:
    """Convenience one-shot helper returning the best MCS at a given SNR."""
    return LinkAdaptation(base_cfg, target_bler=target_bler, num_trials=num_trials).select_at_snr(snr_db)
