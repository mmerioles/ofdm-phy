"""Transmitter path: bits -> QPSK -> OFDM -> preamble packet."""

from dataclasses import dataclass

import numpy as np

from .constants import (
    CP_LEN,
    DATA_BINS,
    LTF_PATTERN,
    N_FFT,
    PACKET_LENGTH,
    PILOT_BINS,
    SC53,
    STF_PATTERN,
)


@dataclass
class TxArtifacts:
    packet_bits: np.ndarray
    qpsk_symbols: np.ndarray
    ofdm_symbols: np.ndarray
    n_syms: int
    tx_signal: np.ndarray
    stf_td: np.ndarray
    stf_time: np.ndarray
    ltf_time: np.ndarray
    tx_packet: np.ndarray


def generate_packet_bits(packet_length: int = PACKET_LENGTH, seed: int | None = None) -> np.ndarray:
    if seed is None:
        return np.random.randint(0, 2, packet_length)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, packet_length)


def bits_to_qpsk_symbols(bits: np.ndarray) -> np.ndarray:
    bit_pairs = bits.reshape(-1, 2)
    symbol_map = {
        (0, 0): 1 + 0j,
        (0, 1): 0 + 1j,
        (1, 0): -1 + 0j,
        (1, 1): 0 - 1j,
    }
    return np.array([symbol_map[tuple(pair)] for pair in bit_pairs])


def pack_ofdm_fd(qpsk: np.ndarray) -> tuple[np.ndarray, int]:
    block = 48
    pad = (-len(qpsk)) % block
    if pad:
        qpsk = np.pad(qpsk, (0, pad), constant_values=0)
    n_syms = len(qpsk) // block

    data_blocks = qpsk.reshape(n_syms, block)
    x_fd = np.zeros((n_syms, N_FFT), dtype=complex)
    x_fd[:, PILOT_BINS] = 1 + 0j
    x_fd[:, DATA_BINS] = data_blocks
    return x_fd, n_syms


def ofdm_modulate(ofdm_symbols: np.ndarray) -> np.ndarray:
    time_domain_symbols = np.fft.ifft(ofdm_symbols, n=N_FFT, axis=1)
    cyclic_prefix = time_domain_symbols[:, -CP_LEN:]
    time_domain_with_cp = np.hstack((cyclic_prefix, time_domain_symbols))
    return time_domain_with_cp.reshape(-1)


def build_preambles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stf_fd = np.zeros(N_FFT, dtype=complex)
    stf_fd[SC53 % N_FFT] = STF_PATTERN.astype(complex)
    stf_td = np.fft.ifft(stf_fd, n=N_FFT)
    stf_short = stf_td[:16]
    stf_time = np.tile(stf_short, 10)

    ltf_fd = np.zeros(N_FFT, dtype=complex)
    ltf_fd[SC53 % N_FFT] = LTF_PATTERN.astype(complex)
    ltf_td = np.fft.ifft(ltf_fd, n=N_FFT)
    ltf_cp = ltf_td[-32:]
    ltf_time = np.concatenate([ltf_cp, ltf_td, ltf_td])
    return stf_td, stf_time, ltf_time


def build_tx_chain(packet_seed: int | None = None) -> TxArtifacts:
    packet = generate_packet_bits(seed=packet_seed)
    qpsk_symbols = bits_to_qpsk_symbols(packet)
    ofdm_symbols, n_syms = pack_ofdm_fd(qpsk_symbols)
    tx_signal = ofdm_modulate(ofdm_symbols)
    stf_td, stf_time, ltf_time = build_preambles()
    tx_packet = np.concatenate([stf_time, ltf_time, tx_signal])
    return TxArtifacts(
        packet_bits=packet,
        qpsk_symbols=qpsk_symbols,
        ofdm_symbols=ofdm_symbols,
        n_syms=n_syms,
        tx_signal=tx_signal,
        stf_td=stf_td,
        stf_time=stf_time,
        ltf_time=ltf_time,
        tx_packet=tx_packet,
    )
