#!/usr/bin/env python3

import argparse
import math
import threading
import time
from collections import deque

import matplotlib.pyplot as plt

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def ylim_expanding(lo: float, hi: float) -> tuple[float, float]:
    """Padding around session min/max so the plotted y-range only grows as new extremes appear."""
    if not math.isfinite(lo):
        return -1.0, 1.0
    span = hi - lo
    pad = 2.0 if span < 1e-9 else max(span * 0.05, 0.5)
    return lo - pad, hi + pad


def make_panels(fig, axes):
    panels = []
    for ax, m in zip(axes, MOTORS):
        ax.set_ylabel(m)
        ax.grid(True, alpha=0.3)
        (ln,) = ax.plot([], [], color="C0", lw=1.2)
        ann = ax.text(0.99, 0.97, "", transform=ax.transAxes, ha="right", va="top", fontsize=8, family="monospace", color="0.2")
        panels.append((ax, ln, ann))
    axes[-1].set_xlabel("Time (s)")
    return panels


def sync_panels(panels, series, mins, maxs, t: float) -> None:
    xr = max(t, 0.05)
    for (ax, ln, ann), m in zip(panels, MOTORS):
        pts = series[m]
        if pts:
            ln.set_data(*zip(*pts))
        ax.set_xlim(0.0, xr)
        lo, hi = mins[m], maxs[m]
        ax.set_ylim(*ylim_expanding(lo, hi))
        ann.set_text(f"min {lo:.2f}  max {hi:.2f}  Δ {hi - lo:.2f}" if math.isfinite(lo) else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live plot of SO101 arm joint positions.")
    ap.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port of the SO101 arm.")
    ap.add_argument("--id", type=str, default="my_awesome_follower_arm", help="Robot id for the calibration file.")
    ap.add_argument("--hz", type=float, default=20.0, help="Sampling frequency in Hz.")
    ap.add_argument("--save", type=str, default="", help="PNG path; default so101_positions_<timestamp>.png in cwd.")
    ap.add_argument("--torque-off-on-exit", action="store_true", help="Disable torque on exit; default off (avoids overload when moving by hand).")
    args = ap.parse_args()

    robot = SO101Follower(
        SO101FollowerConfig(port=args.port, id=args.id, disable_torque_on_disconnect=args.torque_off_on_exit)
    )
    if not robot.calibration:
        raise FileNotFoundError(f"No calibration file found for id '{args.id}'. Expected at: {robot.calibration_fpath}")

    print(f"Loaded calibration file: {robot.calibration_fpath}")
    robot.connect(calibrate=False)
    # connect()+configure() re-enables torque so servos hold position; disable for passive hand-guiding.
    robot.bus.disable_torque()
    print("Torque disabled — move the arm by hand; position readout still works.")

    input("Press ENTER to start live plotting...\n")
    print("Plotting. Press ENTER here to stop, or close the plot window.")

    series = {m: deque() for m in MOTORS}
    mins, maxs = {m: math.inf for m in MOTORS}, {m: -math.inf for m in MOTORS}

    def tick(pos: dict[str, float], t: float) -> None:
        for m in MOTORS:
            k = f"{m}.pos"
            if k in pos:
                v = float(pos[k])
                series[m].append((t, v))
                mins[m], maxs[m] = min(mins[m], v), max(maxs[m], v)

    period = 1.0 / max(args.hz, 1e-3)
    plt.ion()
    fig, axes = plt.subplots(len(MOTORS), 1, sharex=True, figsize=(10, 9), layout="constrained")
    fig.suptitle("SO101 joints (full session)")
    panels = make_panels(fig, axes)

    t0 = time.perf_counter()
    last_t = 0.0
    stop = [False]

    def _wait_enter():
        input()
        stop[0] = True

    threading.Thread(target=_wait_enter, daemon=True).start()
    try:
        while plt.fignum_exists(fig.number) and not stop[0]:
            t_loop = time.perf_counter()
            obs = robot.get_observation()
            pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
            last_t = time.perf_counter() - t0
            tick(pos, last_t)
            sync_panels(panels, series, mins, maxs, last_t)
            fig.canvas.draw()
            plt.pause(max(period - (time.perf_counter() - t_loop), 0.001))
    finally:
        if stop[0]:
            print("\nStopped.")
        plt.ioff()
        out_path = args.save or f"so101_positions_{time.strftime('%Y%m%d_%H%M%S')}.png"
        has_data = any(series[m] for m in MOTORS)
        if has_data:
            if plt.fignum_exists(fig.number):
                sync_panels(panels, series, mins, maxs, last_t)
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
            else:
                fig2, axes2 = plt.subplots(len(MOTORS), 1, sharex=True, figsize=(10, 9), layout="constrained")
                fig2.suptitle("SO101 joints (full session)")
                panels2 = make_panels(fig2, axes2)
                sync_panels(panels2, series, mins, maxs, last_t)
                fig2.savefig(out_path, dpi=150)
                plt.close(fig2)
            print(f"Saved plot to {out_path}")
        elif plt.fignum_exists(fig.number):
            plt.close(fig)
        try:
            robot.disconnect()
        except RuntimeError as e:
            print(f"Warning: torque-off step failed ({e}); closing port without disabling torque.")
            if robot.is_connected:
                robot.bus.disconnect(disable_torque=False)
        print("Disconnected.\n\nSession range (min, max, Δ):")
        for m in MOTORS:
            lo, hi = mins[m], maxs[m]
            if math.isfinite(lo):
                print(f"  {m}: min={lo:.4f}  max={hi:.4f}  Δ={hi - lo:.4f}")
            else:
                print(f"  {m}: (no samples)")


if __name__ == "__main__":
    main()
