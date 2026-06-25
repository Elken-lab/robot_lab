#!/usr/bin/env python3
"""Parse tensorboard events; recommend play checkpoint at best feet_air (smoothed peak)."""
import argparse
import glob
import os
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed", file=sys.stderr)
    sys.exit(2)


def load_feet_air(events_dir: str) -> list[tuple[int, float]]:
    event_files = glob.glob(os.path.join(events_dir, "events.out.tfevents.*"))
    if not event_files:
        return []
    event_files.sort(key=os.path.getmtime)
    ea = EventAccumulator(event_files[-1])
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    key = "Episode_Reward/feet_air_time"
    if key not in tags:
        return []
    return [(int(e.step), float(e.value)) for e in ea.Scalars(key)]


def smooth(values: list[float], window: int = 50) -> list[float]:
    if len(values) < window:
        return list(values)
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def analyze(points: list[tuple[int, float]], plateau_eps: float = 0.00015) -> dict:
    if not points:
        return {"status": "no_data", "n": 0}
    steps = [p[0] for p in points]
    raw = [p[1] for p in points]
    sm = smooth(raw, 50)
    peak_idx = max(range(len(sm)), key=lambda i: sm[i])
    peak_step = steps[peak_idx]
    peak_val = sm[peak_idx]
    last_step = steps[-1]
    last_sm = sm[-1]
    # decline: smoothed dropped >30% from peak after peak+100 steps
    post_peak = [(steps[i], sm[i]) for i in range(len(steps)) if steps[i] >= peak_step + 100]
    declined = False
    if post_peak and peak_val > 5e-5:
        min_post = min(v for _, v in post_peak)
        declined = min_post < peak_val * 0.7 and last_sm < peak_val * 0.8
    # plateau at 1e-3: last 300 steps smoothed range tiny and mean ~1e-3
    tail = sm[-300:] if len(sm) >= 300 else sm
    tail_mean = sum(tail) / len(tail)
    tail_range = max(tail) - min(tail) if tail else 0
    plateau = tail_mean < 0.0025 and tail_range < 0.001 and last_step >= 800
    # stuck below locomotion bar (~0.01): peak never broke 0.003, long run useless
    low_ceiling = peak_val < 0.003 and last_step >= 1200 and tail_mean < 0.0025
    return {
        "status": "ok",
        "n": len(points),
        "last_step": last_step,
        "last_raw": raw[-1],
        "last_smoothed": last_sm,
        "peak_step": peak_step,
        "peak_smoothed": peak_val,
        "declined": declined,
        "plateau": plateau,
        "low_ceiling": low_ceiling,
        "tail_mean": tail_mean,
        "tail_range": tail_range,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events_dir")
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()
    info = analyze(load_feet_air(args.events_dir))
    if info["status"] != "ok":
        print(info)
        sys.exit(0)
    ckpt_step = (info["peak_step"] // args.save_interval) * args.save_interval
    info["ckpt_step"] = ckpt_step
    info["play_ckpt"] = f"model_{ckpt_step}.pt"
    info["play_reason"] = "feet_air_smoothed_peak"
    for k, v in info.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
