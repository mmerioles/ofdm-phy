"""Plot helpers that mirror notebook figures and save images."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.signal.windows import hann as hanning

from .constants import N_FFT


def _save(fig: plt.Figure, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_all(
    out_dir: Path,
    stf_time: np.ndarray,
    tx_signal: np.ndarray,
    stf_tx: np.ndarray,
    stf_rx_i: np.ndarray,
    stf_rx_ii: np.ndarray,
    stf_rx_iii: np.ndarray,
    stf_rx_iv: np.ndarray,
    detection_idx: np.ndarray,
    detection_metric: np.ndarray,
    pkt_start_est: int | None,
    xc_mag: np.ndarray,
    stf_start_idx: int,
    peaks: np.ndarray,
) -> dict[str, str]:
    files: dict[str, str] = {}

    fig = plt.figure()
    plt.plot(np.abs(stf_time))
    plt.title("STF magnitude")
    plt.xlabel("Sample index")
    plt.ylabel("|x[n]|")
    files["stf_magnitude"] = _save(fig, out_dir, "stf_magnitude")

    win = hanning(N_FFT, sym=False)
    f, pxx = welch(
        tx_signal,
        fs=N_FFT,
        window=win,
        nperseg=N_FFT,
        noverlap=N_FFT // 2,
        detrend=False,
        return_onesided=False,
        scaling="density",
    )
    f_shift = np.fft.fftshift(f) - N_FFT / 2.0
    pxx_shift = np.fft.fftshift(pxx)
    fig = plt.figure()
    plt.semilogy(f_shift, pxx_shift)
    plt.title("Power Spectrum Density of OFDM data 64-pt")
    plt.xlabel("Normalized subcarrier index")
    plt.ylabel("PSD")
    files["tx_psd"] = _save(fig, out_dir, "tx_psd")

    fig = plt.figure()
    plt.plot(np.abs(stf_tx), label="Before channel (STF)")
    plt.plot(np.abs(stf_rx_i), label="After (i): attenuated STF")
    plt.title("STF magnitude: before vs after attenuation (10^-5)")
    plt.xlabel("Sample index")
    plt.ylabel("|x[n]|")
    plt.legend()
    files["channel_attenuation"] = _save(fig, out_dir, "channel_attenuation")

    fig = plt.figure()
    plt.plot(np.unwrap(np.angle(stf_rx_i)), label="phase after (i)")
    plt.plot(np.unwrap(np.angle(stf_rx_ii)), label="phase after (i)+(ii)")
    plt.title("STF phase before/after fixed rotation")
    plt.xlabel("Sample index")
    plt.ylabel("phase [rad]")
    plt.legend()
    files["channel_phase_shift"] = _save(fig, out_dir, "channel_phase_shift")

    fig = plt.figure()
    plt.plot(np.unwrap(np.angle(stf_rx_ii)), label="phase after (i)+(ii)")
    plt.plot(np.unwrap(np.angle(stf_rx_iii)), label="phase after (i)+(ii)+(iii)")
    plt.title("STF phase with frequency offset")
    plt.xlabel("Sample index")
    plt.ylabel("phase [rad]")
    plt.legend()
    files["channel_freq_offset"] = _save(fig, out_dir, "channel_freq_offset")

    fig = plt.figure()
    plt.plot(np.abs(stf_rx_iii), label="after (i)+(ii)+(iii)")
    plt.plot(np.abs(stf_rx_iv), label="after (i)+(ii)+(iii)+(iv) noise", alpha=0.8)
    plt.title("STF magnitude: before vs after gaussian channel noise")
    plt.xlabel("Sample index")
    plt.ylabel("|x[n]|")
    plt.legend()
    files["channel_noise"] = _save(fig, out_dir, "channel_noise")

    fig = plt.figure()
    plt.plot(detection_idx, detection_metric, label="|R|/E")
    if pkt_start_est is not None:
        plt.axvline(pkt_start_est, color="r", ls="--", label="metric > 0.85")
    plt.ylim(0, 1.05)
    plt.xlabel("Sample index m")
    plt.ylabel("Self-correlation")
    plt.title("STF self-correlation (M=16), simple threshold=0.85")
    plt.grid(alpha=0.3)
    plt.legend()
    files["packet_detection"] = _save(fig, out_dir, "packet_detection")

    m_idx = np.arange(xc_mag.size)
    pad = 32
    z0 = max(0, stf_start_idx - pad)
    z1 = min(xc_mag.size, stf_start_idx + 16 * 10 + pad)
    fig = plt.figure()
    plt.plot(m_idx[z0:z1], xc_mag[z0:z1])
    for p in peaks[(peaks >= z0) & (peaks < z1)]:
        plt.axvline(p, ls="--", lw=1)
    plt.axvline(stf_start_idx, ls="-", lw=2, label=f"STF start ~ {stf_start_idx}")
    plt.title("Cross-correlation (slide formula) - 10 peaks, 16 apart")
    plt.xlabel("Sample index m")
    plt.ylabel("|R[m]|")
    plt.grid(True, alpha=0.3)
    plt.legend()
    files["packet_sync"] = _save(fig, out_dir, "packet_sync")

    return files
