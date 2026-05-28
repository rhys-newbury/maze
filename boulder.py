#!/usr/bin/env python3
"""
viz_maze_dataset.py

Usage:
  python viz_maze_dataset.py --path maze_dataset.npz --idx 0
  python viz_maze_dataset.py --path maze_dataset.npz --idx 12 --save out.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def draw_maze(ax, vertical_walls, horizontal_walls, rows, cols, cell_size, origin_xy):
    """Draw maze walls on ax."""
    ox, oy = origin_xy
    # vertical_walls: (rows, cols+1)
    for r in range(rows):
        y0 = oy + r * cell_size
        y1 = y0 + cell_size
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                x = ox + c * cell_size
                ax.plot([x, x], [y0, y1])

    # horizontal_walls: (rows+1, cols)
    for r in range(rows + 1):
        y = oy + r * cell_size
        for c in range(cols):
            if horizontal_walls[r, c]:
                x0 = ox + c * cell_size
                x1 = x0 + cell_size
                ax.plot([x0, x1], [y, y])


def cell_center(cell_rc, cell_size, origin_xy):
    """Return (x,y) center of a (r,c) cell."""
    r, c = int(cell_rc[0]), int(cell_rc[1])
    ox, oy = origin_xy
    x = ox + (c + 0.5) * cell_size
    y = oy + (r + 0.5) * cell_size
    return x, y


def plot_demo(dataset_path, idx=0, show=True, save=None):
    data = np.load(dataset_path, allow_pickle=True)

    vertical_walls = data["vertical_walls"].astype(bool)
    horizontal_walls = data["horizontal_walls"].astype(bool)
    rows = int(data["rows"])
    cols = int(data["cols"])
    cell_size = float(data["cell_size"])
    origin_xy = tuple(data["origin_xy"].astype(float))

    np.savez_compressed(
        "mazes/maze.npz",
        vertical_walls = vertical_walls,
        horizontal_walls = horizontal_walls,
        rows = rows,
        cols = cols,
        cell_size = cell_size,
        origin_xy = origin_xy,
    )
    starts = data["starts"]
    goals = data["goals"]
    ee_xy = data["ee_xy"]            # (N, Tmax, 2)
    lengths = data["lengths"]        # (N,)
    q_list = data.get("q_list", None)

    if idx < 0 or idx >= ee_xy.shape[0]:
        raise IndexError(f"--idx {idx} out of range (0..{ee_xy.shape[0]-1})")

    T = int(lengths[idx])
    traj_xy = ee_xy[idx, :T, :]      # (T,2)
    s_xy = cell_center(starts[idx], cell_size, origin_xy)
    g_xy = cell_center(goals[idx], cell_size, origin_xy)

    # --- Figure 1: maze + path
    fig1 = plt.figure(figsize=(7, 7))
    ax1 = fig1.gca()
    ax1.set_aspect("equal", adjustable="box")

    draw_maze(ax1, vertical_walls, horizontal_walls, rows, cols, cell_size, origin_xy)

    ax1.plot(traj_xy[:, 0], traj_xy[:, 1], linewidth=2)
    ax1.scatter([s_xy[0]], [s_xy[1]], marker="o", s=60, label="start")
    ax1.scatter([g_xy[0]], [g_xy[1]], marker="x", s=60, label="goal")

    ox, oy = origin_xy
    W = cols * cell_size
    H = rows * cell_size
    ax1.set_xlim(ox - 0.05, ox + W + 0.05)
    ax1.set_ylim(oy - 0.05, oy + H + 0.05)
    ax1.set_title(f"Maze + EE path (demo {idx}, T={T})")
    ax1.legend(loc="best")

    # --- Figure 2: joint angles (3x2 for q1..q6, and an extra for q7)
    q = q_list[idx, :T, :]
    t = np.arange(T)

    fig2 = plt.figure(figsize=(12, 8))
    gs = fig2.add_gridspec(4, 3)

    # q1..q6 in first two rows
    for j in range(6):
        r = j // 3
        c = j % 3
        ax = fig2.add_subplot(gs[r, c])
        ax.plot(t, q[:, j])
        ax.set_title(f"q{j+1}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("rad")

    # q7 spans full bottom row
    ax7 = fig2.add_subplot(gs[3, :])
    ax7.plot(t, q[:, 6])
    ax7.set_title("q7")
    ax7.set_xlabel("timestep")
    ax7.set_ylabel("rad")

    fig2.suptitle(f"Joint angles (demo {idx})")
    fig2.tight_layout()

    if save:
        # save main maze figure, and joint figs with suffixes
        fig1.savefig(save, dpi=200, bbox_inches="tight")
        if q_list is not None and "fig2" in locals() and fig2 is not None:
            fig2.savefig(save.replace(".png", "_q1-6.png"), dpi=200, bbox_inches="tight")
            plt.figure(3)  # the q7 figure if created
            plt.savefig(save.replace(".png", "_q7.png"), dpi=200, bbox_inches="tight")
        print(f"Saved plots to: {save} (+ suffixes for joints if applicable)")

    if show:
        plt.show()
    else:
        plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to maze_dataset.npz")
    ap.add_argument("--idx", type=int, default=0, help="Demo index to visualize")
    ap.add_argument("--save", default=None, help="Save maze plot to this PNG filename")
    ap.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows")
    args = ap.parse_args()

    plot_demo(args.path, idx=args.idx, show=not args.no_show, save=args.save)


if __name__ == "__main__":
    main()
