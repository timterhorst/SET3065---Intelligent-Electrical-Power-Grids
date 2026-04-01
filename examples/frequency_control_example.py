#!/usr/bin/env python3
"""Simple frequency-control simulation for a single-area power grid."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    nominal_frequency_hz: float = 50.0
    inertia: float = 8.0
    damping: float = 1.2
    droop_mw_per_hz: float = 300.0
    governor_time_constant_s: float = 5.0
    initial_generation_mw: float = 1_000.0
    initial_load_mw: float = 1_000.0
    disturbance_time_s: float = 10.0
    disturbance_size_mw: float = 80.0
    duration_s: float = 120.0
    dt_s: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple single-area frequency control simulation "
            "with a step load disturbance."
        )
    )
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation time step in seconds.")
    parser.add_argument(
        "--disturbance-time",
        type=float,
        default=10.0,
        help="Time in seconds when the load disturbance is applied.",
    )
    parser.add_argument(
        "--disturbance-size",
        type=float,
        default=80.0,
        help="Disturbance size in MW added to load.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional output CSV path for time-series results.",
    )
    return parser.parse_args()


def simulate(cfg: ModelConfig) -> list[dict[str, float]]:
    if cfg.dt_s <= 0:
        raise ValueError("dt must be positive.")
    if cfg.duration_s <= 0:
        raise ValueError("duration must be positive.")

    steps = int(cfg.duration_s / cfg.dt_s) + 1
    frequency_deviation_hz = 0.0
    generation_mw = cfg.initial_generation_mw
    load_mw = cfg.initial_load_mw
    disturbance_applied = False
    trace: list[dict[str, float]] = []

    for step in range(steps):
        t = step * cfg.dt_s

        if (not disturbance_applied) and t >= cfg.disturbance_time_s:
            load_mw += cfg.disturbance_size_mw
            disturbance_applied = True

        target_generation_mw = cfg.initial_generation_mw - (
            cfg.droop_mw_per_hz * frequency_deviation_hz
        )
        governor_response = (target_generation_mw - generation_mw) / cfg.governor_time_constant_s
        generation_mw += governor_response * cfg.dt_s

        power_imbalance_mw = generation_mw - load_mw
        frequency_rate_hz_per_s = (
            power_imbalance_mw - (cfg.damping * frequency_deviation_hz)
        ) / cfg.inertia
        frequency_deviation_hz += frequency_rate_hz_per_s * cfg.dt_s

        trace.append(
            {
                "time_s": t,
                "generation_mw": generation_mw,
                "load_mw": load_mw,
                "frequency_hz": cfg.nominal_frequency_hz + frequency_deviation_hz,
                "frequency_deviation_hz": frequency_deviation_hz,
                "power_imbalance_mw": power_imbalance_mw,
            }
        )

    return trace


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, float]], nominal_frequency_hz: float) -> None:
    min_frequency = min(row["frequency_hz"] for row in rows)
    final_frequency = rows[-1]["frequency_hz"]
    max_imbalance = max(abs(row["power_imbalance_mw"]) for row in rows)

    print("Simulation complete")
    print(f"  Samples: {len(rows)}")
    print(f"  Nominal frequency: {nominal_frequency_hz:.2f} Hz")
    print(f"  Minimum frequency: {min_frequency:.4f} Hz")
    print(f"  Final frequency:   {final_frequency:.4f} Hz")
    print(f"  Max |imbalance|:   {max_imbalance:.2f} MW")


def main() -> None:
    args = parse_args()
    cfg = ModelConfig(
        duration_s=args.duration,
        dt_s=args.dt,
        disturbance_time_s=args.disturbance_time,
        disturbance_size_mw=args.disturbance_size,
    )
    rows = simulate(cfg)
    print_summary(rows, cfg.nominal_frequency_hz)

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"  CSV written to:    {args.csv}")


if __name__ == "__main__":
    main()
