"""
Physical propagation channel models.

Wraps Sionna's ``AWGN``, 3GPP TR38.901 ``TDL`` / ``CDL`` stochastic models and the
system-level ``UMa`` / ``UMi`` models so that they can be applied in the
frequency domain over the PUSCH resource grid through Sionna's ``OFDMChannel``
block.  All fading / geometric models (TDL, CDL, UMa, UMi) support multiple
transmit (UE) and receive (BS) antennas.

For the link-level use case the SNR->noise mapping assumes a unit-power channel,
so:

  * TDL / CDL (whose PDPs are normalised to unit energy by Sionna) use the
    configured ``normalize_channel`` flag (default ``False``);
  * UMa / UMi additionally have their geometric pathloss and shadow fading
    switched off and the frequency-domain channel normalised to unit energy so
    that the BLER-vs-SNR curves stay comparable across scenarios.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from sionna.phy import PI

from .config import ChannelConfig
from .transmitter import PUSCHTransmitterWrapper


def _awgn_variance(snr_db, num_layers=1):
    """Map an SNR in dB to an AWGN noise variance per complex dimension.

    Assumes the transmit resource grid has unit average energy per resource
    element (the Sionna NR transmitter guarantees this through constellation
    power normalisation and precoder scaling). The noise power is scaled by the
    number of layers so that the per-layer SNR is meaningful.
    """
    no = 1.0 / (2.0 * 10.0 ** (snr_db / 10.0))
    return torch.tensor(no, dtype=torch.float32)


class ChannelModelWrapper(nn.Module):
    """Apply a 5G propagation channel to a frequency-domain PUSCH signal.

    For ``awgn`` the channel is an identity (no fading) plus additive noise.
    For ``tdl`` / ``cdl`` / ``uma`` / ``umi`` a 3GPP TR38.901 model is used and
    applied per resource element through Sionna's ``OFDMChannel`` block.

    The forward pass maps an SNR in dB (or an explicit ``no``) to the received
    resource grid ``y`` of shape ``[batch, num_rx, num_ant, n_symbols, fft]``.
    """

    def __init__(self, channel: ChannelConfig, resource_grid, device: str = "cpu", num_tx_ant: int = 1, num_rx_ant: Optional[int] = None):
        super().__init__()
        self.channel_cfg = channel
        self._type = channel.channel_type.lower()
        self.device = device
        self._ofdm_channel = None
        self._sl_model = None  # system-level UMa/UMi channel model (topology-driven)
        self.num_tx_ant = int(num_tx_ant)
        self.num_rx_ant = int(num_rx_ant) if num_rx_ant else int(num_tx_ant)

        # Build the channel through the registry so new propagation models can be
        # added with a single ``register("channel", ...)`` + a config line.
        from . import registry

        if not registry.has("channel", self._type):
            raise ValueError(
                f"Unknown channel_type '{channel.channel_type}'. "
                f"Registered: {registry.names('channel')}"
            )
        self._ofdm_channel, self._sl_model = registry.build(
            "channel", self._type, channel, resource_grid, device,
            self.num_tx_ant, self.num_rx_ant,
        )

    @property
    def is_awgn(self) -> bool:
        return self._type == "awgn"

    def noise_variance(self, snr_db: float, num_layers: int = 1) -> torch.Tensor:
        """AWGN variance per complex dimension for a given SNR (dB)."""
        return _awgn_variance(snr_db, num_layers=num_layers)

    def _set_topology(self, batch_size: int) -> None:
        """(Re-)set the UMa/UMi network topology for the current batch.

        A single UE (uplink transmitter) and a single BS (uplink receiver) are
        placed on a fixed geometry; only the batch dimension is sized to the
        current forward call so independent channel realisations are drawn per
        trial. The topology is identical across scenarios (only the UT distance /
        height come from the config) to keep the BLER curves comparable.
        """
        ch = self.channel_cfg
        dev = self.device
        dtype = torch.float32
        dist = float(ch.ut_distance)
        ut_h = float(ch.ut_height)

        # BS at the origin; UT at (distance, 0, height).
        # NOTE: tensors must be contiguous (not expanded views) so the scenario
        # can copy_() them into its pre-allocated topology buffers.
        bs_loc = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=dev).repeat(batch_size, 1, 1)
        ut_loc = torch.tensor([[dist, 0.0, ut_h]], dtype=dtype, device=dev).repeat(batch_size, 1, 1)
        # Orient the BS array toward the UT (which sits on the +x axis).
        bs_orientation = torch.full((batch_size, 1, 3), 0.0, dtype=dtype, device=dev)
        bs_orientation[:, :, 0] = PI
        ut_orientation = torch.zeros(batch_size, 1, 3, dtype=dtype, device=dev)
        # Static UE (no Doppler) for a clean link-level evaluation.
        ut_velocity = torch.zeros(batch_size, 1, 3, dtype=dtype, device=dev)
        # Outdoor UT state — shape [batch, num_ut] (one UT).
        in_state = torch.zeros(batch_size, 1, dtype=torch.bool, device=dev)

        # NOTE: topology buffers freeze at the first call, so batch_size must be
        # constant across forwards (the driver guarantees this by making
        # num_trials a multiple of the mini-batch size).
        # los=False forces NLoS: the fixed probe geometry would otherwise be a
        # clean single-ray LoS link whose channel is rank-deficient, which makes
        # spatial multiplexing (MIMO) inherently fail. NLoS gives rich multipath
        # so both SISO and MIMO BLER curves are well-conditioned.
        self._sl_model.set_topology(
            ut_loc, bs_loc, ut_orientation, bs_orientation, ut_velocity, in_state,
            los=False,
        )

    def forward(
        self, x: torch.Tensor, no: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the propagation channel.

        Parameters
        ----------
        x : [batch, num_tx, num_ant, n_symbols, fft] complex
            Transmitted frequency-domain resource grid.
        no : float-like tensor
            AWGN variance per complex dimension.

        Returns
        -------
        y : received resource grid.
        h : perfect frequency-domain channel (only for TDL/CDL), else ``None``.
        """
        if self._type == "awgn":
            no = no.to(dtype=torch.float32, device=self.device)
            # Map transmit antenna ports to the configured number of receive
            # antennas (identity / no-fade profile) for the AWGN channel.
            if x.shape[2] != self.num_rx_ant:
                rep = self.num_rx_ant // x.shape[2]
                if rep > 1:
                    x = x.repeat_interleave(rep, dim=2)
            noise = torch.randn(
                *x.shape,
                dtype=x.real.dtype,
                device=self.device,
            ) + 1j * torch.randn(
                *x.shape,
                dtype=x.real.dtype,
                device=self.device,
            )
            noise = noise * torch.sqrt(no)  # total noise power = 2*no
            y = x + noise
            return y, None

        if self._type in ("uma", "umi"):
            # System-level models need a topology matching the current batch.
            self._set_topology(int(x.shape[0]))

        y, h = self._ofdm_channel(x, no)
        return y, h

def build_channel(channel: ChannelConfig, resource_grid, device: str = "cpu", num_tx_ant: int = 1, num_rx_ant: Optional[int] = None) -> ChannelModelWrapper:
    """Factory helper returning a :class:`ChannelModelWrapper`."""
    return ChannelModelWrapper(channel, resource_grid, device=device, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant)
