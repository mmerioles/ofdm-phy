"""CFO/channel estimation and bit decoding."""

from dataclasses import dataclass

import numpy as np

from .constants import CP_LEN, DATA_BINS, LTF_PATTERN, N_FFT, PACKET_LENGTH, PILOT_BINS


@dataclass
class ReceiverArtifacts:
    ltf1_start: int
    ltf2_start: int
    cfo_estimate: float
    r_corr: np.ndarray
    h_est: np.ndarray
    rx_bits: np.ndarray
    tx_bits: np.ndarray
    bit_errors: int
    ber: float


def estimate_and_correct_cfo(r: np.ndarray, stf_start_idx: int, stf_len: int = 160, ltf_cp: int = 32) -> tuple[np.ndarray, int, int, float]:
    ltf1_start = stf_start_idx + stf_len + ltf_cp
    ltf2_start = ltf1_start + 64

    r1 = r[ltf1_start : ltf1_start + 64]
    r2 = r[ltf2_start : ltf2_start + 64]
    j = np.vdot(r1, r2)
    phi = np.angle(j)
    eps = -phi / (2 * np.pi * 64)

    k = np.arange(len(r))
    r_corr = r * np.exp(+1j * 2 * np.pi * eps * k)
    return r_corr, ltf1_start, ltf2_start, float(eps)


def estimate_channel(ltf_pattern: np.ndarray, r_corr: np.ndarray, ltf1_start: int, ltf2_start: int) -> np.ndarray:
    sc53 = np.r_[np.arange(-26, 0), [0], np.arange(1, 27)] % N_FFT
    s_ref_full = np.zeros(N_FFT, dtype=complex)
    s_ref_full[sc53] = ltf_pattern.astype(complex)

    r1 = np.fft.fft(r_corr[ltf1_start : ltf1_start + N_FFT], N_FFT)
    r2 = np.fft.fft(r_corr[ltf2_start : ltf2_start + N_FFT], N_FFT)
    ravg = 0.5 * (r1 + r2)

    h_est = np.zeros(N_FFT, dtype=complex)
    mask = s_ref_full != 0
    h_est[mask] = ravg[mask] / s_ref_full[mask]
    return h_est


def qpsk_demapper(points: np.ndarray) -> np.ndarray:
    mapping = {1 + 0j: (0, 0), 0 + 1j: (0, 1), -1 + 0j: (1, 0), 0 - 1j: (1, 1)}
    pts = np.asarray(points).ravel()
    out = np.empty((pts.size, 2), dtype=int)
    keys = tuple(mapping.keys())
    for i, z in enumerate(pts):
        k = min(keys, key=lambda c: abs(z - c) ** 2)
        out[i] = mapping[k]
    return out.reshape(-1)


def decode_bits(r_corr: np.ndarray, h_est: np.ndarray, ltf2_start: int, packet_length: int = PACKET_LENGTH) -> tuple[np.ndarray, int]:
    data_start = ltf2_start + N_FFT
    sym_len = N_FFT + CP_LEN
    n_samps = len(r_corr) - data_start
    n_syms_rx = n_samps // sym_len

    rx_td = r_corr[data_start : data_start + n_syms_rx * sym_len].reshape(n_syms_rx, sym_len)
    y = np.fft.fft(rx_td[:, CP_LEN:], N_FFT, axis=1)

    eps = 1e-12
    z_eq = y / (h_est + eps)
    for i in range(n_syms_rx):
        theta = np.angle(np.mean(z_eq[i, PILOT_BINS]))
        z_eq[i] *= np.exp(-1j * theta)

    xhat = z_eq[:, DATA_BINS]
    dec_bits_all = qpsk_demapper(xhat.reshape(-1))
    rx_bits = dec_bits_all[:packet_length]
    return rx_bits, n_syms_rx


def run_receiver(rx: np.ndarray, stf_start_idx: int, tx_bits: np.ndarray) -> ReceiverArtifacts:
    r_corr, ltf1_start, ltf2_start, cfo_estimate = estimate_and_correct_cfo(rx, stf_start_idx)
    h_est = estimate_channel(LTF_PATTERN, r_corr, ltf1_start, ltf2_start)
    rx_bits, _ = decode_bits(r_corr, h_est, ltf2_start, packet_length=len(tx_bits))
    tx_bits_trunc = tx_bits[: len(rx_bits)]
    bit_errors = int(np.sum(rx_bits != tx_bits_trunc))
    ber = bit_errors / len(tx_bits_trunc)
    return ReceiverArtifacts(
        ltf1_start=ltf1_start,
        ltf2_start=ltf2_start,
        cfo_estimate=cfo_estimate,
        r_corr=r_corr,
        h_est=h_est,
        rx_bits=rx_bits,
        tx_bits=tx_bits_trunc,
        bit_errors=bit_errors,
        ber=ber,
    )
