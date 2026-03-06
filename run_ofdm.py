"""CLI runner for the OFDM notebook-equivalent pipeline."""

import argparse
import json
from pathlib import Path

from ofdm_phy.pipeline import run_simulation, summary_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OFDM PHY simulation.")
    parser.add_argument("--plot-dir", default="artifacts/plots", help="Directory for PNG figures.")
    parser.add_argument("--summary-path", default="artifacts/run_summary.json", help="Path to JSON summary output.")
    parser.add_argument(
        "--packet-seed",
        type=int,
        default=None,
        help="Optional packet seed (default: notebook-style unseeded packet bits).",
    )
    args = parser.parse_args()

    summary = run_simulation(plot_dir=args.plot_dir, packet_seed=args.packet_seed)
    payload = summary_to_dict(summary)

    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("OFDM simulation complete.")
    print(f"Packet length (bits): {payload['packet_length']}")
    print(f"Number of OFDM symbols: {payload['n_ofdm_symbols']}")
    print(f"TX packet length (samples): {payload['tx_packet_length_samples']}")
    print(f"Packet detection estimate: {payload['packet_detection_estimate']}")
    print(f"Synchronization STF start index: {payload['sync_stf_start_idx']}")
    print(f"First five sync peaks: {payload['sync_peaks_first_five']}")
    print(f"CFO estimate: {payload['cfo_estimate']:.10f}")
    print(f"Bit errors: {payload['bit_errors']}")
    print(f"BER: {payload['ber']:.6e}")
    print(f"Summary JSON: {args.summary_path}")
    print(f"Plot directory: {args.plot_dir}")


if __name__ == "__main__":
    main()
