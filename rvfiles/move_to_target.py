#!/usr/bin/env python3

import argparse
import time

from lerobot.motors import MotorNormMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def joint_limits_user_units(robot: SO101Follower | SO101Leader, m: str) -> tuple[float, float]:
    """(low, high) in the same units as ``get_observation()`` ``*.pos`` (degrees or 0–100)."""
    cal = robot.calibration[m]
    spec = robot.bus.motors[m]
    min_, max_ = cal.range_min, cal.range_max
    drive = bool(robot.bus.apply_drive_mode and cal.drive_mode)

    if spec.norm_mode is MotorNormMode.DEGREES:
        mid = (min_ + max_) / 2
        max_res = robot.bus.model_resolution_table[spec.model] - 1
        lo = (min_ - mid) * 360 / max_res
        hi = (max_ - mid) * 360 / max_res
        return (lo, hi) if lo <= hi else (hi, lo)

    if spec.norm_mode is MotorNormMode.RANGE_0_100:
        if max_ == min_:
            return 0.0, 100.0

        def to_u(raw: int) -> float:
            b = min(max_, max(min_, raw))
            u = ((b - min_) / (max_ - min_)) * 100.0
            return 100.0 - u if drive else u

        u_lo, u_hi = to_u(min_), to_u(max_)
        return (u_lo, u_hi) if u_lo <= u_hi else (u_hi, u_lo)

    if spec.norm_mode is MotorNormMode.RANGE_M100_100:
        if max_ == min_:
            return -100.0, 100.0

        def to_u(raw: int) -> float:
            b = min(max_, max(min_, raw))
            u = (((b - min_) / (max_ - min_)) * 200.0) - 100.0
            u = -u if drive else u
            return u

        u_lo, u_hi = to_u(min_), to_u(max_)
        return (u_lo, u_hi) if u_lo <= u_hi else (u_hi, u_lo)

    raise NotImplementedError(spec.norm_mode)


def print_calibration_info(robot: SO101Follower | SO101Leader) -> None:
    print(f"Calibration file: {robot.calibration_fpath}")
    hdr = f"{'motor':<16} {'id':>3} {'model':<10} {'norm_mode':<16} {'homing_offset':>14} {'range_min':>10} {'range_max':>10} {'drive_mode':>4}"
    print(hdr)
    print("-" * len(hdr))
    for m in MOTORS:
        cal = robot.calibration[m]
        spec = robot.bus.motors[m]
        print(
            f"{m:<16} {cal.id:>3} {spec.model:<10} {spec.norm_mode.value:<16} "
            f"{cal.homing_offset:>14} {cal.range_min:>10} {cal.range_max:>10} {cal.drive_mode:>4}"
        )
    print()
    print("Joint limits (from calibration → same units as observation *.pos):")
    uhdr = f"{'motor':<16} {'unit':<10} {'limit_low':>12} {'limit_high':>12}  (raw ticks: min … max from table above)"
    print(uhdr)
    print("-" * len(uhdr))
    for m in MOTORS:
        lo, hi = joint_limits_user_units(robot, m)
        spec = robot.bus.motors[m]
        unit = "deg" if spec.norm_mode is MotorNormMode.DEGREES else "0–100" if spec.norm_mode is MotorNormMode.RANGE_0_100 else "±100"
        print(f"{m:<16} {unit:<10} {lo:>12.4f} {hi:>12.4f}")
    print()


def obs_to_waypoint(obs: dict) -> dict[str, float]:
    return {m: float(obs[f"{m}.pos"]) for m in MOTORS}


def read_raw_waypoint(robot: SO101Follower | SO101Leader) -> dict[str, int]:
    pos = robot.bus.sync_read("Present_Position", normalize=False)
    return {m: int(pos[m]) for m in MOTORS}


def norm_wp_to_raw(robot: SO101Follower | SO101Leader, wp: dict[str, float]) -> dict[str, int]:
    """Observation-space waypoint → raw ``Goal_Position`` ticks (matches ``_unnormalize``)."""
    ids_val = {robot.bus.motors[m].id: float(wp[m]) for m in MOTORS}
    raw_by_id = robot.bus._unnormalize(ids_val)
    return {m: int(raw_by_id[robot.bus.motors[m].id]) for m in MOTORS}


def interpolate_raw_segment(
    robot: SO101Follower | SO101Leader, start_raw: dict[str, int], end_raw: dict[str, int], 
    duration_s: float, hz: float
) -> None:
    """Linear ramp ``start_raw`` → ``end_raw`` (encoder ticks) at ``hz`` over ``duration_s``."""
    dt = 1.0 / max(hz, 1e-6)
    n = max(int(round(duration_s * hz)), 1)
    t0 = time.perf_counter()
    for k in range(1, n + 1):
        alpha = k / n
        if k == n:
            raw_goal = {m: end_raw[m] for m in MOTORS}
        else:
            raw_goal = {m: round(start_raw[m] + alpha * (end_raw[m] - start_raw[m])) for m in MOTORS}
        robot.bus.sync_write("Goal_Position", raw_goal, normalize=False)
        next_tick = t0 + k * dt
        time.sleep(max(next_tick - time.perf_counter(), 0.0))


def interpolate_move(robot: SO101Follower | SO101Leader, target_raw: dict[str, int], duration_s: float, hz: float) -> None:
    """Ramp from present pose to ``target_raw`` (raw ticks; avoids degree-space quantization stalls)."""
    robot.bus.enable_torque()
    time.sleep(0.2)
    p0 = read_raw_waypoint(robot)
    n = max(int(round(duration_s * max(hz, 1e-6))), 1)
    interpolate_raw_segment(robot, p0, target_raw, duration_s, hz)
    print(f"Sent {n} interpolated raw goals over {duration_s:.1f}s at {hz:.0f} Hz.")

"""
Follower arm:
  python move_to_target.py --arm follower --port /dev/ttyACM0 --id my_awesome_follower_arm
Leader arm:
  python move_to_target.py --arm leader --port /dev/ttyACM0 --id my_awesome_leader_arm
"""
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Teach a target pose with torque off, move away, then servo to that target with torque on."
    )
    ap.add_argument("--arm", type=str, choices=("follower", "leader"), default="follower", help="Which SO101 arm type to use.")
    ap.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port of the SO101 arm.")
    ap.add_argument("--id", type=str, default=None, help="Robot id for the calibration file.")
    ap.add_argument("--duration", type=float, default=3.0, help="Seconds for the interpolated move (joint-space ramp).")
    ap.add_argument("--hz", type=float, default=20.0, help="Position command rate during the move.")
    args = ap.parse_args()

    robot_id = args.id or ("my_awesome_leader_arm" if args.arm == "leader" else "my_awesome_follower_arm")
    if args.arm == "leader":
        robot = SO101Leader(SO101LeaderConfig(port=args.port, id=robot_id))
    else:
        robot = SO101Follower(
            SO101FollowerConfig(port=args.port, id=robot_id, disable_torque_on_disconnect=True, max_relative_target=None)
        )
    if not robot.calibration:
        raise FileNotFoundError(f"No calibration file found for id '{robot_id}'. Expected at: {robot.calibration_fpath}")

    print_calibration_info(robot)
    robot.connect(calibrate=False)
    robot.bus.disable_torque()
    print("Torque is OFF — you can move the arm by hand.")

    input("1) Move the arm to the TARGET pose, then press ENTER.\n")
    if args.arm == "leader":
        target = obs_to_waypoint(robot.get_action())
    else:
        target = obs_to_waypoint(robot.get_observation())
    target_raw = read_raw_waypoint(robot)
    print("Saved target joint positions.")

    input("2) Move the arm away (starting pose), then press ENTER.\n")
    if args.arm == "leader":
        start = obs_to_waypoint(robot.get_action())
    else:
        start = obs_to_waypoint(robot.get_observation())
    print("Starting interpolated move from current pose to target...")
    for m in MOTORS:
        print(f"  {m}: {start[m]:.4f} -> {target[m]:.4f}  (Δ {target[m] - start[m]:+.4f})")

    interpolate_move(robot, target_raw, duration_s=args.duration, hz=args.hz)

    # robot.bus.disable_torque()
    try:
        robot.disconnect()
    except RuntimeError as e:
        print(f"Warning: disconnect issue ({e}); closing port.")
        if robot.is_connected:
            robot.bus.disconnect(disable_torque=True)
    print("Disconnected.")


if __name__ == "__main__":
    main()
