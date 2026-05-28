import numpy as np
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import os
from dataclasses import dataclass

# ----------------------------
# Maze drawing (same as yours)
# ----------------------------

@dataclass(frozen=True)
class Cell:
    r: int
    c: int


def add_maze_to_swift(
    env,
    vertical_walls,
    horizontal_walls,
    rows,
    cols,
    cell_size,
    origin_xy=(0.0, 0.0),
    z_plane=0.25,
    wall_height=0.04,
    wall_thickness=0.002,
    wall_color=(0.2, 0.2, 0.2, 1.0),
    floor=True,
    floor_thickness=0.002,
    floor_color=(0.9, 0.9, 0.9, 1.0),
):
    ox, oy = origin_xy

    z0 = z_plane - floor_thickness / 2.0
    zw = z_plane + wall_height / 2.0

    W = cols * cell_size
    H = rows * cell_size

    if floor:
        floor_box = sg.Box(
            [W, H, floor_thickness],
            pose=sm.SE3(ox + W / 2.0, oy + H / 2.0, z0),
            color=floor_color,
        )
        env.add(floor_box)

    # Vertical walls: x = ox + c*cell_size
    for r in range(rows):
        y_center = oy + (r + 0.5) * cell_size
        for c in range(cols + 1):
            if not vertical_walls[r, c]:
                continue
            x = ox + c * cell_size
            wall = sg.Box(
                [wall_thickness, cell_size, wall_height],
                pose=sm.SE3(x, y_center, zw),
                color=wall_color,
            )
            env.add(wall)

    # Horizontal walls: y = oy + r*cell_size
    for r in range(rows + 1):
        y = oy + r * cell_size
        for c in range(cols):
            if not horizontal_walls[r, c]:
                continue
            x_center = ox + (c + 0.5) * cell_size
            wall = sg.Box(
                [cell_size, wall_thickness, wall_height],
                pose=sm.SE3(x_center, y, zw),
                color=wall_color,
            )
            env.add(wall)


# ----------------------------
# Replay helpers
# ----------------------------

def replay_joint_trajectory(env, robot, q_traj, dt=0.05, loop=False, hold_last=True):
    """
    Replays a joint-angle trajectory by directly setting robot.q each step.

    q_traj: (T, 7) or (T, robot.n) array
    """
    q_traj = np.asarray(q_traj)

    if q_traj.ndim != 2:
        raise ValueError(f"q_traj must be 2D (T,n), got shape {q_traj.shape}")

    n = robot.n
    if q_traj.shape[1] < n:
        raise ValueError(f"q_traj has {q_traj.shape[1]} joints but robot has {n}")

    q_use = q_traj[:, :n]

    while True:
        # start at first pose
        robot.q = q_use[0].copy()
        robot.qd = np.zeros(n)
        env.step(dt)

        for k in range(q_use.shape[0]):
            robot.q = q_use[k].copy()
            robot.qd = np.zeros(n)
            env.step(dt)

        if hold_last:
            # keep sim running at last pose
            while True:
                env.step(dt)

        if not loop:
            break


def load_demo_q(npz_path, demo_idx=0, key="q_list"):
    """
    Loads a single demo from your saved dataset.
    Expected dataset key (from your script): q_list with shape (num_demos, T, 7)
    """
    data = np.load(npz_path)
    return data["arr_0"]
    # print(data)
    # import pdb; pdb.set_trace()
    # if key not in data:
    #     # fall back to a few common names
    #     for k in ["q_list", "q", "qs", "q_traj", "traj_q"]:
    #         if k in data:
    #             key = k
    #             break
    #     else:
    #         raise KeyError(f"Couldn't find joint trajectory key. Available keys: {list(data.keys())}")

    # q_all = data[key]
    # if q_all.ndim == 2:
    #     # already (T,7)
    #     return q_all
    # if q_all.ndim == 3:
    #     return q_all[demo_idx]
    # raise ValueError(f"Unexpected shape for {key}: {q_all.shape}")


# ----------------------------
# Main: draw maze + replay q
# ----------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset containing q_list")
    ap.add_argument("--demo", type=int, default=0, help="Demo index to replay")
    ap.add_argument("--dt", type=float, default=0.05, help="Simulation timestep for replay")
    ap.add_argument("--loop", action="store_true", help="Loop trajectory forever (ignored if --hold-last)")
    ap.add_argument("--hold-last", action="store_true", help="Hold the final pose forever after replay")
    ap.add_argument("--maze", type=str, default=None,
                    help="Optional: path to maze .npz made by get_or_make_maze (with vertical_walls/horizontal_walls). "
                         "If omitted, uses maze stored in dataset if present, else errors.")
    args = ap.parse_args()

    # ---- Load maze from either separate maze file or dataset ----
    if args.maze is not None:
        m = np.load(args.maze)
        vertical_walls = m["vertical_walls"].astype(bool)
        horizontal_walls = m["horizontal_walls"].astype(bool)
        rows = int(m["rows"]) if "rows" in m else vertical_walls.shape[0]
        cols = int(m["cols"]) if "cols" in m else horizontal_walls.shape[1]
        # these may not exist in your maze-only .npz, so keep defaults if missing
        cell_size = float(m["cell_size"]) if "cell_size" in m else 0.025
        origin_xy = tuple(m["origin_xy"].astype(float)) if "origin_xy" in m else (0.35, -0.30)
        z_plane = float(m["z_plane"]) if "z_plane" in m else 0.25
    else:
        d = np.load(args.dataset)
        if "vertical_walls" not in d or "horizontal_walls" not in d:
            raise ValueError(
                "No --maze provided and dataset doesn't contain vertical_walls/horizontal_walls."
            )
        vertical_walls = d["vertical_walls"].astype(bool)
        horizontal_walls = d["horizontal_walls"].astype(bool)
        rows = int(d["rows"])
        cols = int(d["cols"])
        cell_size = float(d["cell_size"])
        origin_xy = tuple(d["origin_xy"].astype(float))
        z_plane = float(d["z_plane"])

    # ---- Load joint trajectory ----
    q_traj = load_demo_q(args.dataset, demo_idx=args.demo, key="q_list")

    # ---- Swift + Panda ----
    env = swift.Swift()
    env.launch(realtime=True)

    add_maze_to_swift(
        env,
        vertical_walls,
        horizontal_walls,
        rows, cols,
        cell_size,
        origin_xy=origin_xy,
        z_plane=z_plane,
        wall_height=0.04,
        wall_thickness=0.002,
    )

    panda = rtb.models.Panda()
    env.add(panda)

    # optional: start in first pose
    panda.q = q_traj[0, :panda.n].copy()
    panda.qd = np.zeros(panda.n)
    env.step(args.dt)

    # ---- Replay ----
    replay_joint_trajectory(
        env,
        panda,
        q_traj,
        dt=args.dt,
        loop=args.loop,
        hold_last=args.hold_last,
    )


if __name__ == "__main__":
    main()
