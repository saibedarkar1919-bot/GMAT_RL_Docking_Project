"""
summarize_flight.py — compress a recorded flight JSON into something small
enough to paste into a chat message.

record_ai_flight.py and play_manual.py both save every single simulation
frame (up to 6000 for a full stage-4 episode), which is too large to paste
usefully. This pulls out the numbers that actually matter for diagnosing
a flight: closest approach, final state, and a downsampled trend of range/
velocity/lateral offset/attitude over the episode, plus the outcome.

USAGE:
    python summarize_flight.py static_viz/ai_flight_data.json
    python summarize_flight.py static_viz/manual_flight_1234567890.json --points 40

Prints JSON to stdout — copy it straight into chat.
"""
import json
import sys
import os
import glob
import argparse
import math


def _resolve_flight_path(flight_path):
    """
    --- FIX: unhelpful FileNotFoundError on a guessed filename ---
    record_ai_flight.py and play_manual.py both name files with a real
    millisecond timestamp (e.g. ai_flight_1758302214123.json) — there's
    no way to guess one correctly, and a bad guess used to just crash
    with a bare traceback. `latest` picks the most recently written
    flight file in static_viz/, and any other miss now lists what's
    actually there instead of dying silently useless.
    """
    search_dir = "static_viz"

    if flight_path == "latest":
        candidates = [f for f in glob.glob(os.path.join(search_dir, "*.json"))
                      if os.path.basename(f) != "mission_log.json"]
        if not candidates:
            raise SystemExit(f"No flight files found in '{search_dir}/'. "
                              f"Run record_ai_flight.py or play_manual.py first.")
        newest = max(candidates, key=os.path.getmtime)
        print(f"[using latest flight: {newest}]", file=sys.stderr)
        return newest

    if os.path.exists(flight_path):
        return flight_path

    # Not found as given — show what's actually on disk so the next guess
    # doesn't have to be a guess.
    candidates = sorted(
        (f for f in glob.glob(os.path.join(search_dir, "*.json"))
         if os.path.basename(f) != "mission_log.json"),
        key=os.path.getmtime, reverse=True
    )
    msg = [f"'{flight_path}' does not exist."]
    if candidates:
        msg.append(f"Flight files actually present in '{search_dir}/' (newest first):")
        for f in candidates[:15]:
            msg.append(f"  {f}")
        msg.append(f"\nTip: run with 'latest' instead of a filename to auto-pick the newest one:")
        msg.append(f"  python summarize_flight.py latest")
    else:
        msg.append(f"No flight files found in '{search_dir}/' at all. "
                    f"Run record_ai_flight.py or play_manual.py first.")
    raise SystemExit("\n".join(msg))


def summarize(flight_path, num_points=30, tail=0):
    flight_path = _resolve_flight_path(flight_path)
    with open(flight_path, "r") as f:
        payload = json.load(f)

    frames = payload.get("frames", [])
    if not frames:
        return {"error": "no frames in this flight file"}

    ranges = [math.hypot(f["x"], f["y"]) for f in frames]
    closest_idx = min(range(len(frames)), key=lambda i: ranges[i])
    closest = frames[closest_idx]
    final = frames[-1]

    # Downsample to num_points evenly-spaced frames so the trend is visible
    # without pasting every single step.
    n = len(frames)
    if n <= num_points:
        sample_idx = list(range(n))
    else:
        step = n / num_points
        sample_idx = sorted(set(int(i * step) for i in range(num_points)))

    trend = [
        {
            "step": frames[i]["step"],
            "range_m": round(ranges[i], 3),
            "vx": round(frames[i]["vx"], 4),
            "vy": round(frames[i]["vy"], 4),
            "theta_deg": round(frames[i]["theta_deg"], 2),
            "action": frames[i].get("action_taken"),
        }
        for i in sample_idx
    ]

    result = {
        "flight_file": flight_path,
        "generated_at": payload.get("generated_at"),
        "total_steps": n,
        "outcome": final.get("outcome"),
        "cumulative_reward": round(final.get("cumulative_reward", 0.0), 2),
        "fuel_remaining_kg": round(final.get("fuel_kg", 0.0), 1),
        "closest_approach": {
            "step": closest["step"],
            "range_m": round(ranges[closest_idx], 3),
            "vx": round(closest["vx"], 4),
            "vy": round(closest["vy"], 4),
            "theta_deg": round(closest["theta_deg"], 2),
        },
        "final_state": {
            "x": round(final["x"], 3),
            "y": round(final["y"], 3),
            "vx": round(final["vx"], 4),
            "vy": round(final["vy"], 4),
            "theta_deg": round(final["theta_deg"], 2),
            "omega_dps": round(final.get("omega_dps", 0.0), 3),
        },
        "trend_sampled_every_nth_frame": trend,
    }

    # --- NEW: full-resolution tail ---
    # The downsampled trend above can smear out exactly what happened in
    # the last second or two before docking/crashing — which is usually
    # the part you actually need to see to diagnose overshoot, late
    # braking, or attitude drift right at the capture box. --tail N gives
    # you every single frame (no downsampling) for the last N steps.
    if tail > 0:
        tail_frames = frames[-tail:]
        result["tail_full_resolution"] = [
            {
                "step": f["step"],
                "x": round(f["x"], 3), "y": round(f["y"], 3),
                "range_m": round(math.hypot(f["x"], f["y"]), 3),
                "vx": round(f["vx"], 4), "vy": round(f["vy"], 4),
                "theta_deg": round(f["theta_deg"], 2),
                "omega_dps": round(f.get("omega_dps", 0.0), 3),
                "action": f.get("action_taken"),
                "reward": round(f.get("reward", 0.0), 3),
                "fuel_kg": round(f.get("fuel_kg", 0.0), 2),
            }
            for f in tail_frames
        ]

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flight_json",
                         help="Path to a flight JSON file, or 'latest' to auto-pick the "
                              "most recently written flight in static_viz/")
    parser.add_argument("--points", type=int, default=30, help="Number of trend points to sample (default: 30)")
    parser.add_argument("--tail", type=int, default=20,
                         help="Full-resolution (non-downsampled) frames to include from the end of "
                              "the flight — the last few seconds before dock/crash/timeout. "
                              "Set to 0 to omit. (default: 20)")
    parser.add_argument("--out", default=None, help="Optional: write summary to this file instead of stdout")
    args = parser.parse_args()

    summary = summarize(args.flight_json, num_points=args.points, tail=args.tail)
    output = json.dumps(summary, indent=2)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Summary written to {args.out} ({len(output)} chars, {len(output)//4} tokens approx.)")
    else:
        print(output)