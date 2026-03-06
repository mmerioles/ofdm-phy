"""Channel impairments from notebook Part 2."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelArtifacts:
    tx_packet_with_idle: np.ndarray
    rx_i: np.ndarray
    rx_ii: np.ndarray
    rx_iii: np.ndarray
    rx_iv: np.ndarray
    stf_start: int
    stf_stop: int


def apply_channel(
    tx_packet: np.ndarray,
    stf_len: int,
    idle_len: int = 100,
    attenuation: float = 1e-5,
    phase_shift: float = -3 * np.pi / 4,
    freq_offset: float = 0.00017,
    noise_variance: float = 1e-14,
    noise_seed: int = 257,
) -> ChannelArtifacts:
    idle_period = np.zeros(idle_len, dtype=complex)
    tx_packet_with_idle = np.concatenate([idle_period, tx_packet])

    rx_i = attenuation * tx_packet_with_idle
    rx_ii = rx_i * np.exp(1j * phase_shift)

    k = np.arange(rx_ii.size)
    rx_iii = rx_ii * np.exp(-1j * 2 * np.pi * freq_offset * k)

    sigma = np.sqrt(noise_variance / 2.0)
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(0.0, sigma, rx_iii.shape) + 1j * rng.normal(0.0, sigma, rx_iii.shape)
    rx_iv = rx_iii + noise

    stf_start = idle_len
    stf_stop = stf_start + stf_len
    return ChannelArtifacts(
        tx_packet_with_idle=tx_packet_with_idle,
        rx_i=rx_i,
        rx_ii=rx_ii,
        rx_iii=rx_iii,
        rx_iv=rx_iv,
        stf_start=stf_start,
        stf_stop=stf_stop,
    )
