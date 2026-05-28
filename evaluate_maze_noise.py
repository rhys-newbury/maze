"""
evaluate_maze_noise.py

Evaluates maze-solving success rate under noise A (goal drift on Tep) and
noise D (actuation noise on qd) across 100 trials with random start/end cells.

Results are saved to a .npz file and a summary CSV.

Usage:
    python evaluate_maze_noise.py
"""

import qpsolvers as qp
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import csv
import os
import heapq
from dataclasses import dataclass
from tqdm import tqdm
import uuid


# ---------------------------------------------------------------------------
# Noise model (A + D only)
# ---------------------------------------------------------------------------

class DynamicsNoise:
    """
    Noise A: random-walk drift applied once to the target translation (Tep)
             before each servo call.
    Noise D: i.i.d. Gaussian noise added to qd after every QP solve.

    Both are disabled when their std is 0 (the default), so the same object
    can represent the baseline with no changes to calling code.
    """

    def __init__(
        self,
        # A — goal drift
        goal_drift_std: float = 0.0,         # metres, random-walk step std per waypoint
        goal_drift_bias: np.ndarray = None,  # fixed xyz offset (metres)
        # D — actuation noise
        actuation_std: float = 0.0,          # rad/s added to every joint velocity
        rng_seed: int = 42,
    ):
        self.goal_drift_std = goal_drift_std
        self.goal_drift_bias = (
            np.zeros(3) if goal_drift_bias is None else np.asarray(goal_drift_bias, dtype=float)
        )
        self.actuation_std = actuation_std
        self._rng = np.random.default_rng(rng_seed)
        self._goal_drift_accum = np.zeros(3)   # accumulates across waypoints in one episode

    def reset_episode(self):
        """Call at the start of each new trial to reset the random-walk accumulator."""
        self._goal_drift_accum = np.zeros(3)

    def perturb_target(self, t_xyz: np.ndarray) -> np.ndarray:
        """
        Noise A: advance the random walk by one step and return the perturbed
        target translation.  The walk accumulates across waypoints so the drift
        grows as the episode progresses, making later segments harder.
        """
        if self.goal_drift_std > 0:
            self._goal_drift_accum += self._rng.normal(0.0, self.goal_drift_std, 3)
        return t_xyz + self._goal_drift_accum + self.goal_drift_bias

    def perturb_qd(self, qd: np.ndarray) -> np.ndarray:
        """
        Noise D: add zero-mean Gaussian noise to the joint velocity vector
        returned by the QP solver.
        """
        if self.actuation_std == 0:
            return qd
        return qd + self._rng.normal(0.0, self.actuation_std, qd.shape)


# ---------------------------------------------------------------------------
# Maze data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    r: int
    c: int


def generate_perfect_maze(rows, cols, seed=0):
    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)
    visited = np.zeros((rows, cols), dtype=bool)
    stack = [Cell(0, 0)]
    visited[0, 0] = True
    rng = np.random.default_rng(seed)

    def neighbors(cell):
        r, c = cell.r, cell.c
        opts = []
        if r > 0:            opts.append(Cell(r - 1, c))
        if r < rows - 1:     opts.append(Cell(r + 1, c))
        if c > 0:            opts.append(Cell(r, c - 1))
        if c < cols - 1:     opts.append(Cell(r, c + 1))
        rng.shuffle(opts)
        return opts

    while stack:
        current = stack[-1]
        unvisited = [n for n in neighbors(current) if not visited[n.r, n.c]]
        if not unvisited:
            stack.pop()
            continue
        nxt = unvisited[0]
        dr = nxt.r - current.r
        dc = nxt.c - current.c
        if dr == -1:   horizontal_walls[current.r,     current.c] = False
        elif dr == 1:  horizontal_walls[current.r + 1, current.c] = False
        elif dc == -1: vertical_walls[current.r,  current.c]     = False
        elif dc == 1:  vertical_walls[current.r,  current.c + 1] = False
        visited[nxt.r, nxt.c] = True
        stack.append(nxt)

    return vertical_walls, horizontal_walls


def astar_on_maze(rows, cols, vertical_walls, horizontal_walls, start: Cell, goal: Cell):
    def h(a, b):
        return abs(a.r - b.r) + abs(a.c - b.c)

    def can_move(a, b):
        dr, dc = b.r - a.r, b.c - a.c
        if dr == -1: return not horizontal_walls[a.r,     a.c]
        if dr ==  1: return not horizontal_walls[a.r + 1, a.c]
        if dc == -1: return not vertical_walls[a.r, a.c]
        if dc ==  1: return not vertical_walls[a.r, a.c + 1]
        return False

    def neighbors(cell):
        r, c = cell.r, cell.c
        cands = []
        if r > 0:        cands.append(Cell(r - 1, c))
        if r < rows - 1: cands.append(Cell(r + 1, c))
        if c > 0:        cands.append(Cell(r, c - 1))
        if c < cols - 1: cands.append(Cell(r, c + 1))
        return [nb for nb in cands if can_move(cell, nb)]

    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (h(start, goal), 0, counter, start))
    came_from = {start: None}
    gscore = {start: 0}

    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            cur = current
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path
        for nb in neighbors(current):
            tentative = g + 1
            if nb not in gscore or tentative < gscore[nb]:
                gscore[nb] = tentative
                came_from[nb] = current
                heapq.heappush(open_heap, (tentative + h(nb, goal), tentative, counter, nb))
                counter += 1

    raise RuntimeError("No path found.")


def cells_to_waypoints(cell_path, cell_size, origin_xy):
    ox, oy = origin_xy
    return [(ox + (c.c + 0.5) * cell_size, oy + (c.r + 0.5) * cell_size) for c in cell_path]


# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------

def servo_to_pose(
    env, panda, Tep,
    noise: DynamicsNoise = None,
    n=7, gain=2.0, dt=0.05,
    arrived_thresh=0.005, max_steps=600,
    slack_weight_scale=1.0, Y=0.01,
):
    """
    Drives the end-effector toward Tep using p_servo + QP with slack.

    Noise A is applied before this function is called (caller perturbs Tep).
    Noise D is applied here, after each QP solve, before env.step.

    Returns (ee_trace Nx3, q_trace Nx7, success bool).
    """
    ee_trace = []
    q_trace = []
    succ = False

    for _ in range(max_steps):
        Te = panda.fkine(panda.q)
        ee_trace.append(Te.t.copy())
        q_trace.append(panda.q[:n].copy())

        eTep = Te.inv() * Tep
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))
        v, _ = rtb.p_servo(Te, Tep, gain)

        if e < arrived_thresh:
            succ = True
            break

        Q_mat = np.eye(n + 6)
        Q_mat[:n, :n] *= Y
        Q_mat[n:, n:] = (slack_weight_scale / max(e, 1e-9)) * np.eye(6)

        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
        beq = v.reshape((6,))

        Ain  = np.zeros((n + 6, n + 6))
        bin_ = np.zeros(n + 6)
        ps, pi = 0.05, 0.9
        Ain[:n, :n], bin_[:n] = panda.joint_velocity_damper(ps, pi, n)

        c_vec = np.r_[-panda.jacobm().reshape((n,)), np.zeros(6)]
        lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
        ub  =  np.r_[panda.qdlim[:n], 10 * np.ones(6)]

        qd_sol = qp.solve_qp(Q_mat, c_vec, Ain, bin_, Aeq, beq,
                              lb=lb, ub=ub, solver="proxqp")
        if qd_sol is None:
            break

        # Noise D: perturb joint velocities after QP solve
        qd_apply = noise.perturb_qd(qd_sol[:n]) if noise is not None else qd_sol[:n]
        panda.qd[:n] = qd_apply
        env.step(dt)

    return np.array(ee_trace), np.array(q_trace), succ


def set_robot_to_cell(env, panda, R, cell, cell_size, origin_xy, z_plane, dt=0.05):
    x = origin_xy[0] + (cell.c + 0.5) * cell_size
    y = origin_xy[1] + (cell.r + 0.5) * cell_size
    Tep = sm.SE3.Rt(R, [x, y, z_plane])
    sol = panda.ikine_LM(Tep, q0=panda.q, ilimit=80, slimit=80)
    if not sol.success:
        return False
    panda.q  = sol.q
    panda.qd = np.zeros_like(panda.q)
    env.step(dt)
    return True


# ---------------------------------------------------------------------------
# Noise configurations to sweep
# ---------------------------------------------------------------------------

NOISE_CONFIGS = [
    # label,  goal_drift_std,  actuation_std
    ("AD_mild2",               0.002, 0.08),
    ("AD_mild3",               0.005, 0.1),
    ("AD_mild1",               0.002, 0.05),

    ("AD_moderate",           0.005, 0.15),
    ("AD_severe",             0.010, 0.30),
]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    num_trials: int = 100,
    min_manhattan_dist: int = 8,   # minimum A* path length to ensure non-trivial tasks
    maze_seed: int = 0,
    rng_seed: int = 1,
    save_dir: str = ".",
):
    # ---- Sim setup ----
    env = swift.Swift()
    env.launch(realtime=False, headless=True)
    panda = rtb.models.Panda()
    env.add(panda)

    Te0 = panda.fkine(panda.qr)
    R = np.array(Te0.R).reshape(3, 3)

    # ---- Fixed maze params (same as training script) ----
    rows, cols   = 22, 22
    z_plane      = 0.4
    cell_size    = 0.0225
    origin_xy    = (0.19111111111111106, -0.27)

    # Load or generate maze
    maze_path = os.path.join("mazes", f"maze{rows}x{cols}.npz")
    if os.path.exists(maze_path):
        data = np.load(maze_path)
        vertical_walls   = data["vertical_walls"].astype(bool)
        horizontal_walls = data["horizontal_walls"].astype(bool)
        print(f"Loaded maze from {maze_path}")
    else:
        vertical_walls, horizontal_walls = generate_perfect_maze(rows, cols, seed=maze_seed)
        os.makedirs("mazes", exist_ok=True)
        np.savez_compressed(
            maze_path,
            vertical_walls=vertical_walls.astype(np.uint8),
            horizontal_walls=horizontal_walls.astype(np.uint8),
        )
        print(f"Generated and saved maze to {maze_path}")

    # ---- Sample (start, goal) pairs once; reuse across all noise configs ----
    trial_rng = np.random.default_rng(rng_seed)

    trials = []   # list of (start_cell, goal_cell, cell_path)
    print(f"Sampling {num_trials} (start, goal) pairs with min Manhattan dist {min_manhattan_dist}...")
    attempts = 0
    while len(trials) < num_trials:
        attempts += 1
        if attempts > num_trials * 200:
            raise RuntimeError("Could not sample enough valid (start, goal) pairs.")
        sr = int(trial_rng.integers(rows))
        sc = int(trial_rng.integers(cols))
        gr = int(trial_rng.integers(rows))
        gc = int(trial_rng.integers(cols))
        start = Cell(sr, sc)
        goal  = Cell(gr, gc)
        if start == goal:
            continue
        path = astar_on_maze(rows, cols, vertical_walls, horizontal_walls, start, goal)
        if len(path) < min_manhattan_dist:
            continue
        trials.append((start, goal, path))

    print(f"Sampled {num_trials} trials after {attempts} attempts.")

    # ---- Results storage ----
    # results[config_label] = list of dicts with per-trial outcome
    all_results = {label: [] for label, *_ in NOISE_CONFIGS}

    # ---- Outer loop: noise configs ----
    for cfg_label, goal_drift_std, actuation_std in NOISE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {cfg_label}  "
              f"(goal_drift_std={goal_drift_std}, actuation_std={actuation_std})")
        print(f"{'='*60}")

        noise = DynamicsNoise(
            goal_drift_std=goal_drift_std,
            actuation_std=actuation_std,
            rng_seed=rng_seed + hash(cfg_label) % (2**31),
        )

        successes = 0

        for trial_idx, (start, goal, cell_path) in enumerate(
            tqdm(trials, desc=cfg_label, leave=True)
        ):
            noise.reset_episode()

            # Reset robot to ready pose, then teleport to start cell
            panda.q  = panda.qr.copy()
            panda.qd = np.zeros_like(panda.q)
            env.step(0.05)

            ok = set_robot_to_cell(env, panda, R, start, cell_size, origin_xy, z_plane)
            if not ok:
                all_results[cfg_label].append({
                    "trial": trial_idx,
                    "start_r": start.r, "start_c": start.c,
                    "goal_r":  goal.r,  "goal_c":  goal.c,
                    "path_len": len(cell_path),
                    "success": False,
                    "failure_reason": "ik_teleport_failed",
                    "waypoints_completed": 0,
                    "waypoints_total": len(cell_path),
                })
                continue

            waypoints_xy = cells_to_waypoints(cell_path, cell_size, origin_xy)
            waypoints_completed = 0
            trial_ok = True
            failure_reason = ""

            for wp_idx, (x, y) in enumerate(waypoints_xy):
                # Noise A: perturb the target translation before servo
                t_target = noise.perturb_target(np.array([x, y, z_plane]))
                Tep = sm.SE3.Rt(R, t_target)

                # IK feasibility pre-check (uses perturbed target)
                sol = panda.ikine_LM(Tep, q0=panda.q, ilimit=30, slimit=30)
                if not sol.success:
                    trial_ok = False
                    failure_reason = "ik_infeasible"
                    break

                _, _, success = servo_to_pose(
                    env, panda, Tep,
                    noise=noise,
                    n=7, gain=2.0, dt=0.05,
                    arrived_thresh=0.005, max_steps=600,
                )

                if not success:
                    trial_ok = False
                    failure_reason = "servo_timeout"
                    break

                waypoints_completed += 1

            if trial_ok:
                successes += 1

            all_results[cfg_label].append({
                "trial": trial_idx,
                "start_r": start.r, "start_c": start.c,
                "goal_r":  goal.r,  "goal_c":  goal.c,
                "path_len": len(cell_path),
                "success": trial_ok,
                "failure_reason": failure_reason if not trial_ok else "",
                "waypoints_completed": waypoints_completed,
                "waypoints_total": len(cell_path),
            })

        success_rate = successes / num_trials * 100
        print(f"  → success rate: {successes}/{num_trials} ({success_rate:.1f}%)")

    # ---- Save results ----
    run_id = str(uuid.uuid4())[:8]
    npz_path = os.path.join(save_dir, f"eval_results_{run_id}.npz")
    csv_path = os.path.join(save_dir, f"eval_summary_{run_id}.csv")

    # NPZ: full per-trial data
    save_dict = {}
    for cfg_label, rows_list in all_results.items():
        prefix = cfg_label
        save_dict[f"{prefix}__success"]            = np.array([r["success"]            for r in rows_list])
        save_dict[f"{prefix}__path_len"]           = np.array([r["path_len"]           for r in rows_list])
        save_dict[f"{prefix}__waypoints_completed"]= np.array([r["waypoints_completed"] for r in rows_list])
        save_dict[f"{prefix}__waypoints_total"]    = np.array([r["waypoints_total"]     for r in rows_list])

    save_dict["trial_starts"] = np.array([[t[0].r, t[0].c] for t in trials], dtype=np.int32)
    save_dict["trial_goals"]  = np.array([[t[1].r, t[1].c] for t in trials], dtype=np.int32)
    save_dict["trial_path_lens"] = np.array([len(t[2]) for t in trials], dtype=np.int32)

    np.savez_compressed(npz_path, **save_dict)
    print(f"\nSaved full results to {npz_path}")

    # CSV: summary table (one row per config)
    summary_rows = []
    for cfg_label, goal_drift_std, actuation_std in NOISE_CONFIGS:
        rows_list = all_results[cfg_label]
        n_success = sum(r["success"] for r in rows_list)
        n_timeout = sum(r["failure_reason"] == "servo_timeout" for r in rows_list)
        n_ik_fail = sum(r["failure_reason"] in ("ik_infeasible", "ik_teleport_failed") for r in rows_list)
        avg_wps_completed = np.mean([
            r["waypoints_completed"] / max(r["waypoints_total"], 1) for r in rows_list
        ])
        avg_path_len = np.mean([r["path_len"] for r in rows_list])
        summary_rows.append({
            "config":            cfg_label,
            "goal_drift_std":    goal_drift_std,
            "actuation_std":     actuation_std,
            "num_trials":        num_trials,
            "successes":         n_success,
            "success_rate_%":    round(n_success / num_trials * 100, 1),
            "servo_timeouts":    n_timeout,
            "ik_failures":       n_ik_fail,
            "avg_path_len":      round(float(avg_path_len), 2),
            "avg_frac_completed":round(float(avg_wps_completed), 4),
        })

    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary CSV to {csv_path}")
    print("\n--- Summary ---")
    print(f"{'Config':<20} {'SuccessRate':>12} {'Timeouts':>10} {'IK fails':>10} {'AvgFracDone':>13}")
    print("-" * 70)
    for row in summary_rows:
        print(
            f"{row['config']:<20} "
            f"{row['success_rate_%']:>11.1f}% "
            f"{row['servo_timeouts']:>10} "
            f"{row['ik_failures']:>10} "
            f"{row['avg_frac_completed']:>13.4f}"
        )

    return all_results, summary_rows


if __name__ == "__main__":
    run_evaluation(
        num_trials=100,
        min_manhattan_dist=8,
        maze_seed=0,
        rng_seed=1,
        save_dir=".",
    )
