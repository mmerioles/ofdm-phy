"""End-to-end simulation pipeline."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .channel import apply_channel
from .constants import PACKET_LENGTH
from .plotting import plot_all
from .receiver import run_receiver
from .synchronization import detect_packet_self_correlation, synchronize_with_stf
from .transmitter import build_tx_chain


@dataclass
class SimulationSummary:
    packet_length: int
    n_ofdm_symbols: int
    tx_packet_length_samples: int
    packet_detection_estimate: int | None
    sync_stf_start_idx: int
    sync_peaks_first_five: list[int]
    cfo_estimate: float
    bit_errors: int
    ber: float
    tx_bits_sample_first_64: list[int]
    rx_bits_sample_first_64: list[int]
    detection_indices_first_20: list[int]
    plot_files: dict[str, str]


def run_simulation(plot_dir: str = "artifacts/plots", packet_seed: int | None = None) -> SimulationSummary:
    tx = build_tx_chain(packet_seed=packet_seed)
    ch = apply_channel(tx.tx_packet, stf_len=len(tx.stf_time))
    det = detect_packet_self_correlation(ch.rx_iv)
    sync = synchronize_with_stf(ch.rx_iv, tx.stf_td)
    rx = run_receiver(ch.rx_iv, sync.stf_start_idx, tx.packet_bits)

    plots = plot_all(
        out_dir=Path(plot_dir),
        stf_time=tx.stf_time,
        tx_signal=tx.tx_signal,
        stf_tx=ch.tx_packet_with_idle[ch.stf_start : ch.stf_stop],
        stf_rx_i=ch.rx_i[ch.stf_start : ch.stf_stop],
        stf_rx_ii=ch.rx_ii[ch.stf_start : ch.stf_stop],
        stf_rx_iii=ch.rx_iii[ch.stf_start : ch.stf_stop],
        stf_rx_iv=ch.rx_iv[ch.stf_start : ch.stf_stop],
        detection_idx=det.metric_indices,
        detection_metric=det.metric,
        pkt_start_est=det.pkt_start_est,
        xc_mag=sync.xc_mag,
        stf_start_idx=sync.stf_start_idx,
        peaks=sync.peaks,
    )

    return SimulationSummary(
        packet_length=PACKET_LENGTH,
        n_ofdm_symbols=tx.n_syms,
        tx_packet_length_samples=len(tx.tx_packet),
        packet_detection_estimate=det.pkt_start_est,
        sync_stf_start_idx=sync.stf_start_idx,
        sync_peaks_first_five=[int(x) for x in sync.peaks[:5]],
        cfo_estimate=rx.cfo_estimate,
        bit_errors=rx.bit_errors,
        ber=float(rx.ber),
        tx_bits_sample_first_64=rx.tx_bits[:64].astype(int).tolist(),
        rx_bits_sample_first_64=rx.rx_bits[:64].astype(int).tolist(),
        detection_indices_first_20=det.detected_indices[:20].astype(int).tolist(),
        plot_files=plots,
    )


def summary_to_dict(summary: SimulationSummary) -> dict:
    return asdict(summary)
