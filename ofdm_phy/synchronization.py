"""Packet detection and synchronization helpers."""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class DetectionArtifacts:
    metric: np.ndarray
    metric_indices: np.ndarray
    detected_indices: np.ndarray
    pkt_start_est: int | None


@dataclass
class SyncArtifacts:
    xc_mag: np.ndarray
    stf_start_idx: int
    peaks: np.ndarray


def detect_packet_self_correlation(r: np.ndarray, m: int = 16, threshold: float = 0.85) -> DetectionArtifacts:
    w = sliding_window_view(r, m)
    r_corr = np.sum(w[m:] * np.conj(w[:-m]), axis=1)
    e = np.sum(np.abs(w[m:]) ** 2, axis=1)
    metric = np.abs(r_corr) / (e + 1e-12)
    metric_indices = np.arange(m, m + metric.size)

    det_mask = metric > threshold
    detected_indices = metric_indices[det_mask]
    pkt_start_est = int(detected_indices[0]) if detected_indices.size else None
    return DetectionArtifacts(
        metric=metric,
        metric_indices=metric_indices,
        detected_indices=detected_indices,
        pkt_start_est=pkt_start_est,
    )


def synchronize_with_stf(
    r: np.ndarray, stf_td: np.ndarray, idle_len: int = 100, m: int = 16, threshold_scale: float = 0.7
) -> SyncArtifacts:
    s_ref = stf_td[:m]
    w = sliding_window_view(r, m)
    xc = w @ np.conj(s_ref)
    xc_mag = np.abs(xc)

    search_lo = max(0, idle_len - 32)
    search_hi = min(xc_mag.size, idle_len + 200)
    seg = xc_mag[search_lo:search_hi]
    thr = threshold_scale * seg.max()
    rel = np.where(seg >= thr)[0][0]
    stf_start_idx = search_lo + rel

    peaks = stf_start_idx + 16 * np.arange(10)
    return SyncArtifacts(xc_mag=xc_mag, stf_start_idx=int(stf_start_idx), peaks=peaks)
