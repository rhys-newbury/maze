import qpsolvers as qp
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import heapq
from dataclasses import dataclass
from tqdm import tqdm
import os
import spatialgeometry as sg
from itertools import product

# ----------------------------
# Maze generation (perfect maze) + A* path planning
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
    """
    Add maze walls as thin boxes into Swift.

    Coordinates match your waypoint mapping:
      x = ox + (c + 0.5)*cell_size
      y = oy + (r + 0.5)*cell_size

    Walls are placed on top of a floor at z_plane (you can tune).
    """
    ox, oy = origin_xy

    # Put the floor slightly below z_plane, walls sit on top.
    z0 = z_plane - floor_thickness / 2.0
    zw = z_plane + wall_height / 2.0

    W = cols * cell_size
    H = rows * cell_size

    # Optional floor
    if floor:
        floor_box = sg.Box(
            [W, H, floor_thickness],
            pose=sm.SE3(ox + W/2.0, oy + H/2.0, z0),
            color=floor_color,
        )
        env.add(floor_box)

    # --- Vertical walls: between (r,c-1) and (r,c), located at x = ox + c*cell_size
    # segment spans one cell in y.
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

    # --- Horizontal walls: between (r-1,c) and (r,c), located at y = oy + r*cell_size
    # segment spans one cell in x.
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


cache = {}  # (x_r, y_r) -> bool

def all_cells_feasible(
    panda, R, rows, cols, cell_size, origin_xy, z_plane,
    ilimit=30, slimit=30,
    joint_margin=0.15,           # radians
    min_manipulability=0.04,     # tune this
    max_joint_dist=2.5           # radians total motion from qr
):
    q_nom = panda.qr.copy()

    # ---- Memoized constants / repeated values ----
    qmin = panda.qlim[0]
    qmax = panda.qlim[1]
    qmin_margin = qmin + joint_margin
    qmax_margin = qmax - joint_margin
    max_joint_dist2 = max_joint_dist * max_joint_dist

    ox, oy = origin_xy

    # Helper: one-cell feasibility (memoized)
    def cell_feasible(x, y):
        key = (round(float(x), 12), round(float(y), 12), round(float(z_plane), 12))
        hit = cache.get(key, None)
        if hit is not None:
            # print(hit)
            return hit

        Tep = sm.SE3.Rt(R, [x, y, z_plane])

        sol = panda.ikine_LM(Tep, q0=q_nom, ilimit=ilimit, slimit=slimit)
        if not sol.success:
            cache[key] = False, None
            return False, None

        q_sol = sol.q

        # ---- Joint limit margin check ----
        if np.any(q_sol < qmin_margin) or np.any(q_sol > qmax_margin):
            cache[key] = False, None
            return False, None

        # ---- Joint distance check ----
        dq = q_sol - q_nom
        if float(dq @ dq) > max_joint_dist2:
            cache[key] = False, None
            return False, None

        # ---- Manipulability check ----
        J = panda.jacobe(q_sol)
        JJt = J @ J.T
        detJJt = np.linalg.det(JJt)
        manipulability = np.sqrt(detJJt) if detJJt > 0 else 0.0

        ok = manipulability >= min_manipulability
        cache[key] = (ok, manipulability)
        return ok, manipulability

    # ---- Main loop ----
    # l = np.zeros((rows, cols))
    manips = []
    for r, c in iter_spiral_inward(rows, cols):
        # print(r,c)
        # l[r,c]=1
        y = oy + (r + 0.5) * cell_size
        x = ox + (c + 0.5) * cell_size
        ok, manip = cell_feasible(x, y)
        manips.append(manip)
        if not ok:
            return False, None

    # import pdb; pdb.set_trace()


    return True, np.mean(manips)


def iter_spiral_inward(rows, cols):
    """
    Yield (r,c) in an inward spiral-ish order that checks the outer boundary first:
      top row -> right col -> bottom row -> left col, then repeat on the inner ring.
    Works for non-square grids too.
    """
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1

    while top <= bottom and left <= right:
        # top edge (left -> right)
        for c in range(left, right + 1):
            yield top, c
        top += 1

        # right edge (top -> bottom)
        for r in range(top, bottom + 1):
            yield r, right
        right -= 1

        if top <= bottom:
            # bottom edge (right -> left)
            for c in range(right, left - 1, -1):
                yield bottom, c
            bottom -= 1

        if left <= right:
            # left edge (bottom -> top)
            for r in range(bottom, top - 1, -1):
                yield r, left
            left += 1


def choose_origin_for_center(rows, cols, cell_size, center_xy):
    cx, cy = center_xy
    W = cols * cell_size
    H = rows * cell_size
    return (cx - W/2.0, cy - H/2.0)


# def auto_maximize_maze_strict(panda, R, 
#                               center_y=0.0,
#                               center_x_candidates=None,
#                               rows_candidates=None,
#                               cols_candidates=None,
#                               cell_size_candidates=None,
#                               ik_ilimit=30, ik_slimit=30):
#     """
#     Finds the biggest (rows=cols, cell_size) such that ALL cell centers are IK-feasible,
#     while allowing the maze center x-position to vary over positive values.
#     The maze center y-position is fixed to center_y (default 0.0).
#     """
#     if center_x_candidates is None:
#         # Any positive x is allowed; we search a reasonable band.
#         # You can widen this if you want.
#         center_x_candidates = np.linspace(0.35, 0.85, 10)  # 0.35..0.85 in steps of 0.02

#     if rows_candidates is None:
#         rows_candidates = list(range(12, 22, 2))  # 22,16,...64
#         cols_candidates = list(range(12, 22, 2))  # 22,16,...64

#     z_plane_l = np.linspace(0.2, 0.4, 5)
#     if cell_size_candidates is None:
#         cell_size_candidates = [0.0225, 0.025, 0.0275, 0.030, 0.0325]

#     best = None
#     # best = (score, rows, cols, cell_size, origin_xy, center_x)

#     total = (
#         len(cell_size_candidates)
#         * len(rows_candidates)
#         * len(cols_candidates)
#         * len(center_x_candidates)
#         * len(z_plane_l)
#     )

#     for cell_size, rows, cols, z, cx in tqdm(
#         product(cell_size_candidates,
#                 rows_candidates,
#                 cols_candidates,
#                 z_plane_l,
#                 center_x_candidates),
#         total=total,
#         desc="Searching maze configs"
#     ):
#         if cx <= 0:
#             continue

#         # rows, cols = n, n

#         center_xy = (float(cx), float(center_y))
#         origin_xy = choose_origin_for_center(rows, cols, cell_size, center_xy)

#         ok, manips = all_cells_feasible(
#             panda, R, rows, cols, cell_size, origin_xy, z,
#             ilimit=ik_ilimit, slimit=ik_slimit
#         )
#         if not ok:
#             continue

#         W = cols * cell_size
#         H = rows * cell_size
#         area = W * H
#         num_cells = rows * cols

#         # Primary objective: largest physical area.
#         # Secondary: more cells.
#         # Tertiary: larger cell_size (optional; remove if you prefer smaller cells).
#         score = (num_cells, manips, cell_size)

#         if best is None or score > best[0]:
#             num_cells, manips, cs = score
#             print(
#                 f"[NEW BEST] "
#                 f"manips={manips:.4f}, "
#                 f"cells={num_cells}, "
#                 f"cell_size={cs:.4f}, "
#                 f"rows={rows}, "
#                 f"cols={cols}, "
#                 f"center_x={cx:.3f}, "
#                 f"z={z:.3f}, "

#                 f"origin_xy={origin_xy}"
#             )            
#             best = (score, rows, cols, cell_size, origin_xy, z, cx)

#     if best is None:
#         raise RuntimeError("No candidate (center_x, rows, cell_size) makes ALL cells IK-feasible.")

#     _, rows, cols, cell_size, origin_xy, z, center_x, = best
#     print(f"[auto_maximize] chose center_x={center_x:.3f}, center_y={center_y:.3f}")
#     return rows, cols, cell_size, origin_xy, z


import numpy as np
from itertools import product
from tqdm import tqdm


def auto_maximize_maze_strict(
    panda,
    R,
    center_y=0.0,
    center_x_candidates=None,
    rows_candidates=None,
    cols_candidates=None,
    cell_size_candidates=None,
    ik_ilimit=30,
    ik_slimit=30,
    enforce_square=True,
    prefer_larger_cell_size=True,
):
    """
    Finds the biggest grid (by number of cells) such that ALL cell centers are IK-feasible,
    while allowing the maze center x-position to vary over positive values.
    The maze center y-position is fixed to center_y (default 0.0).

    Optimization priority (lexicographic):
      1) maximize num_cells = rows*cols
      2) within that, maximize manips (returned by all_cells_feasible)
      3) within that, optionally prefer larger cell_size (toggle prefer_larger_cell_size)

    Returns:
      (rows, cols, cell_size, origin_xy, z, cx, manips)
    """
    if center_x_candidates is None:
        # Any positive x is allowed; we search a reasonable band.
        center_x_candidates = np.linspace(0.35, 0.85, 10)

    if rows_candidates is None:
        rows_candidates = list(range(12, 30, 2))
    if cols_candidates is None:
        cols_candidates = list(range(12, 30, 2))

    z_plane_l = np.linspace(0.2, 0.4, 5)

    if cell_size_candidates is None:
        cell_size_candidates = [0.0225, 0.025, 0.0275, 0.030, 0.0325]

    # ---- sort candidates to encourage "good" configs earlier (does NOT change correctness) ----
    rows_sorted = sorted(rows_candidates, reverse=True)
    cols_sorted = sorted(cols_candidates, reverse=True)
    cx_sorted = sorted([float(cx) for cx in center_x_candidates if float(cx) > 0.0])
    z_sorted = sorted([float(z) for z in z_plane_l])  # doesn't matter much; keep increasing

    cs_sorted = sorted(cell_size_candidates, reverse=True) if prefer_larger_cell_size else list(cell_size_candidates)

    # Build grid options, ordered by descending num_cells (primary objective).
    grid_options = []
    for r in rows_sorted:
        for c in cols_sorted:
            if enforce_square and r != c:
                continue
            grid_options.append((r, c))
    grid_options.sort(key=lambda rc: rc[0] * rc[1], reverse=True)

    if not grid_options:
        raise ValueError("No grid options available (check rows/cols candidates and enforce_square).")

    # We'll search size-by-size. For each size, we keep the best by (manips, cell_size).
    for rows, cols in grid_options:
        num_cells = rows * cols

        best_for_size = None
        # best_for_size = (inner_score, rows, cols, cell_size, origin_xy, z, cx, manips)
        # inner_score = (manips, cell_size) or (manips, -cell_size) depending on preference

        total_inner = len(cs_sorted) * len(cx_sorted) * len(z_sorted)
        desc = f"Searching {rows}x{cols} ({num_cells} cells)"

        for cell_size, cx, z in tqdm(
            product(cs_sorted, cx_sorted, z_sorted),
            total=total_inner,
            desc=desc,
            leave=False,
        ):
            center_xy = (float(cx), float(center_y))
            origin_xy = choose_origin_for_center(rows, cols, cell_size, center_xy)

            ok, manips = all_cells_feasible(
                panda,
                R,
                rows,
                cols,
                cell_size,
                origin_xy,
                float(z),
                ilimit=ik_ilimit,
                slimit=ik_slimit,
            )

            if not ok:
                continue

            manips = float(manips)
            cs = float(cell_size)

            # Secondary objective: maximize manips.
            # Tertiary: (optionally) maximize cell_size.
            inner_score = (manips, cs) if prefer_larger_cell_size else (manips,)

            if best_for_size is None or inner_score > best_for_size[0]:
                print(
                    f"[BEST @ {rows}x{cols}] "
                    f"manips={manips:.4f}, "
                    f"cell_size={cs:.4f}, "
                    f"center_x={cx:.3f}, "
                    f"z={float(z):.3f}, "
                    f"origin_xy={origin_xy}"
                )
                best_for_size = (inner_score, rows, cols, cs, origin_xy, float(z), float(cx), manips)

        # If we found ANY feasible config at this cell count, it's globally optimal by the primary objective.
        if best_for_size is not None:
            _, rows, cols, cs, origin_xy, z, cx, manips = best_for_size
            # return rows, cols, cs, origin_xy, z, cx, manips
            return rows, cols, cs, origin_xy, z

    raise RuntimeError("No candidate (center_x, rows/cols, cell_size, z) makes ALL cells IK-feasible.")


def generate_perfect_maze(rows, cols, seed=0):
    """
    Returns walls representation for a perfect maze using DFS backtracking.
    We represent walls between cells with booleans:
      vertical_walls[r][c] is wall between (r,c-1) and (r,c) for c in [0..cols]
      horizontal_walls[r][c] is wall between (r-1,c) and (r,c) for r in [0..rows]
    Outer border walls are included.
    """
    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)

    visited = np.zeros((rows, cols), dtype=bool)
    stack = [Cell(0, 0)]
    visited[0, 0] = True

    # NOTE: use numpy RNG for shuffle
    rng = np.random.default_rng(seed)

    def neighbors(cell):
        r, c = cell.r, cell.c
        opts = []
        if r > 0: opts.append(Cell(r - 1, c))
        if r < rows - 1: opts.append(Cell(r + 1, c))
        if c > 0: opts.append(Cell(r, c - 1))
        if c < cols - 1: opts.append(Cell(r, c + 1))
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
        if dr == -1:
            horizontal_walls[current.r, current.c] = False
        elif dr == 1:
            horizontal_walls[current.r + 1, current.c] = False
        elif dc == -1:
            vertical_walls[current.r, current.c] = False
        elif dc == 1:
            vertical_walls[current.r, current.c + 1] = False

        visited[nxt.r, nxt.c] = True
        stack.append(nxt)

    return vertical_walls, horizontal_walls

def astar_on_maze(rows, cols, vertical_walls, horizontal_walls, start: Cell, goal: Cell):
    """
    A* search over maze cells using the walls to determine adjacency.
    """
    def h(a: Cell, b: Cell):
        return abs(a.r - b.r) + abs(a.c - b.c)

    def can_move(a: Cell, b: Cell):
        dr = b.r - a.r
        dc = b.c - a.c
        if dr == -1:
            return not horizontal_walls[a.r, a.c]
        if dr == 1:
            return not horizontal_walls[a.r + 1, a.c]
        if dc == -1:
            return not vertical_walls[a.r, a.c]
        if dc == 1:
            return not vertical_walls[a.r, a.c + 1]
        return False

    def neighbors(cell: Cell):
        r, c = cell.r, cell.c
        nbrs = []
        if r > 0:
            up = Cell(r - 1, c)
            if can_move(cell, up): nbrs.append(up)
        if r < rows - 1:
            dn = Cell(r + 1, c)
            if can_move(cell, dn): nbrs.append(dn)
        if c > 0:
            lf = Cell(r, c - 1)
            if can_move(cell, lf): nbrs.append(lf)
        if c < cols - 1:
            rt = Cell(r, c + 1)
            if can_move(cell, rt): nbrs.append(rt)
        return nbrs

    open_heap = []
    counter = 0  # tie-breaker so heap never compares Cell
    heapq.heappush(open_heap, (h(start, goal), 0, counter, start))
    counter += 1

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

        for nbr in neighbors(current):
            tentative = g + 1
            if nbr not in gscore or tentative < gscore[nbr]:
                gscore[nbr] = tentative
                came_from[nbr] = current
                f = tentative + h(nbr, goal)
                heapq.heappush(open_heap, (f, tentative, counter, nbr))
                counter += 1

    raise RuntimeError("No path found (should not happen in a perfect maze).")

def cells_to_waypoints(cell_path, cell_size, origin_xy=(0.0, 0.0)):
    ox, oy = origin_xy
    waypoints = []
    for cell in cell_path:
        x = ox + (cell.c + 0.5) * cell_size
        y = oy + (cell.r + 0.5) * cell_size
        waypoints.append((x, y))
    return waypoints

def compress_waypoints(waypoints, tol=1e-9):
    if len(waypoints) <= 2:
        return waypoints[:]
    out = [waypoints[0]]
    prev = waypoints[0]
    cur = waypoints[1]
    prev_dir = (cur[0] - prev[0], cur[1] - prev[1])

    def norm_dir(d):
        dx, dy = d
        if abs(dx) > abs(dy):
            return (np.sign(dx), 0.0)
        else:
            return (0.0, np.sign(dy))

    prev_dir = norm_dir(prev_dir)

    for nxt in waypoints[2:]:
        cur_dir = norm_dir((nxt[0] - cur[0], nxt[1] - cur[1]))
        if (abs(cur_dir[0] - prev_dir[0]) > tol) or (abs(cur_dir[1] - prev_dir[1]) > tol):
            out.append(cur)
            prev_dir = cur_dir
        cur = nxt

    out.append(waypoints[-1])
    return out

# ----------------------------
# Robot control helpers (unchanged servo_to_pose)
# ----------------------------

def servo_to_pose(env, panda, Tep, n=7, gain=5.0, dt=0.05,
                  arrived_thresh=0.05, max_steps=400,
                  slack_weight_scale=1.0, Y=0.01):
    """
    Drives the end-effector toward Tep using p_servo + QP w/ slack.
    Returns recorded end-effector positions (Nx3) and q (Nx7).
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

        # print(e)

        if e < arrived_thresh:
            succ = True
            break

        Q = np.eye(n + 6)
        Q[:n, :n] *= Y
        Q[n:, n:] = (slack_weight_scale / max(e, 1e-9)) * np.eye(6)

        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
        beq = v.reshape((6,))

        Ain = np.zeros((n + 6, n + 6))
        bin_ = np.zeros(n + 6)

        ps = 0.05
        pi = 0.9
        Ain[:n, :n], bin_[:n] = panda.joint_velocity_damper(ps, pi, n)

        c = np.r_[-panda.jacobm().reshape((n,)), np.zeros(6)]
        # import pdb; pdb.set_trace()
        lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
        ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

        qd = qp.solve_qp(Q, c, Ain, bin_, Aeq, beq, lb=lb, ub=ub, solver="proxqp")
        if qd is None:
            print("QP solver failed; stopping.")
            break

        panda.qd[:n] = qd[:n]
        env.step(dt)

    return np.array(ee_trace), np.array(q_trace), succ

def get_or_make_maze(rows: int, cols: int,
                     cell_size: float,
                     origin_xy,
                     z_plane: float,
                     maze_dir: str = ".",
                     prefix: str = "maze"):

    # os.makedirs(maze_dir, exist_ok=True)
    maze_path = os.path.join(maze_dir, f"{prefix}{rows}x{cols}.npz")

    if os.path.exists(maze_path):
        print("loaded maze!!", maze_path)
        data = np.load(maze_path)

        vertical_walls = data["vertical_walls"].astype(bool)
        horizontal_walls = data["horizontal_walls"].astype(bool)

        return vertical_walls, horizontal_walls, cell_size, origin_xy, maze_path

    vertical_walls, horizontal_walls = generate_perfect_maze(rows, cols)

    np.savez_compressed(
        maze_path,
        rows=np.int32(rows),
        cols=np.int32(cols),
        cell_size=np.float32(cell_size),
        origin_xy=np.array(origin_xy, dtype=np.float32),
        z_plane=np.float32(z_plane),
        vertical_walls=vertical_walls.astype(np.uint8),
        horizontal_walls=horizontal_walls.astype(np.uint8),
    )

    print("saved maze!!", maze_path)
    return vertical_walls, horizontal_walls, cell_size, origin_xy, maze_path

def sample_goal_far(rows, cols, cur: Cell, min_dist: int):
    while True:
        g = Cell(int(np.random.randint(rows)), int(np.random.randint(cols)))
        if g != cur: # and (abs(g.r - cur.r) + abs(g.c - cur.c)) >= min_dist:
            return g

def set_robot_to_cell(env, panda, R, cell: Cell, cell_size, origin_xy, z_plane,
                      dt=0.05, ik_ilimit=80, ik_slimit=80):
    """
    IK the robot to the center of `cell` and set panda.q directly (no recorded trajectory).
    Returns True if successful.
    """
    x = origin_xy[0] + (cell.c + 0.5) * cell_size
    y = origin_xy[1] + (cell.r + 0.5) * cell_size
    Tep = sm.SE3.Rt(R, [x, y, z_plane])

    sol = panda.ikine_LM(Tep, q0=panda.q, ilimit=ik_ilimit, slimit=ik_slimit)
    if not sol.success:
        return False

    # set state directly (skip "getting to the first point")
    panda.q = sol.q

    print(sol.q)
    panda.qd = np.zeros_like(panda.q)
    env.step(dt)
    return True

def main():
    # -------------------------
    # Dataset params
    # -------------------------
    import sys
    num_demos = 100
    T = 10_000  # fixed length per trajectory (timesteps)
    import uuid
    save_path = f"/mnt/slow2/maze_dataset_{uuid.uuid4()}.npz"
    save_png_examples = True
    num_png = 16

    # -------------------------
    # Maze params (fixed maze)
    # -------------------------
    # rows, cols = 12, 12
    # cell_size = 0.025

    # origin_xy = (0.35, -0.30)
    # z_plane = 0.25

    env = swift.Swift()
    env.launch(realtime=False, headless=True)
    panda = rtb.models.Panda()

    env.add(panda)

    # Fixed EE orientation from ready pose
    Te0 = panda.fkine(panda.qr)
    R = Te0.R
    if hasattr(R, "A"):
        R = R.A
    R = np.array(R).reshape(3, 3)

    # print("Using maze:", maze_path)
    # rows, cols, cell_size, origin_xy, z, center_x
    # rows, cols, cell_size, origin_xy, z_plane = auto_maximize_maze_strict(
    #     panda, R, 
    #     cell_size_candidates=[0.0225, 0.025, 0.0275, 0.030, 0.0325],
    #     ik_ilimit=30,
    #     ik_slimit=30
    # )

    rows, cols = 22,22
    z_plane=0.4
    cell_size=0.0225
    origin_xy=(0.19111111111111106, -0.27)
    # cell_size = 0.0275
    # origin_xy = (0.21499999999999997, -0.275)
    

    vertical_walls, horizontal_walls, cell_size, origin_xy, maze_path = get_or_make_maze(rows, cols, cell_size, origin_xy, z_plane, maze_dir="mazes")

    print(f"STRICT max maze: {rows}x{cols}, cell_size={cell_size:.4f}, origin_xy={origin_xy}")
    # -------------------------
    # Swift + Panda (single sim)
    # -------------------------


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


    # panda = rtb.models.UR5()
    # panda.qdlim = np.array([3.1415926535897931, 3.1415926535897931, 3.1415926535897931, 3.1415926535897931, 3.1415926535897931, 3.1415926535897931])
    # panda.q = np.array([0, -np.pi/2,  np.pi/2, -np.pi/2, -np.pi/2, 0])
    # panda.qr = np.array([0, -np.pi/2,  np.pi/2, -np.pi/2, -np.pi/2, 0])


    min_dist = (rows + cols) // 5

    # -------------------------
    # Collect demos (fixed-length; resample goal when reached)
    # -------------------------
    ee_xy_arr = np.zeros((num_demos, T, 2), dtype=np.float32)
    q_arr = np.zeros((num_demos, T, 7), dtype=np.float32)

    starts = np.zeros((num_demos, 2), dtype=np.int32)
    first_goals = np.zeros((num_demos, 2), dtype=np.int32)

    # optional: save images of first goal segment only
    demo_idx = 0
    pbar = tqdm(total=num_demos * T)

    while demo_idx < num_demos:
        s = Cell(int(np.random.randint(rows)), int(np.random.randint(cols)))
        s = Cell(15, 19)
        cur = s

        # reset robot
        panda.q = panda.qr
        panda.qd = np.zeros_like(panda.q)
        env.step(0.05)

        g = sample_goal_far(rows, cols, cur, min_dist)
        first_goal_cell = s

        set_robot_to_cell(env, panda, R, s, cell_size, origin_xy, z_plane)

        # buffers
        t = 0


        # For optional PNG: store the XY trace for this trajectory (downsampled to T anyway)
        full_xy_for_plot = []

        # roll until we have T timesteps
        ok = True
        safety_segments = 0
        max_segments = 200  # prevents infinite loops if something keeps failing

        while t < T and safety_segments < max_segments:

            
            # print
            safety_segments += 1

            g = sample_goal_far(rows, cols, cur, min_dist)

            # plan
            # try:
            cell_path = astar_on_maze(rows, cols, vertical_walls, horizontal_walls, cur, g)
            # except Exception as e:
            #     print(e)
            #     continue

            subgoals_xy = cells_to_waypoints(cell_path, cell_size, origin_xy=origin_xy)
            # subgoals_xy = compress_waypoints(dense_waypoints)

            # execute subgoals in order
            for (x, y) in subgoals_xy:
                if t >= T:
                    break

                Tep = sm.SE3.Rt(R, [x, y, z_plane])

                # feasibility filter
                sol = panda.ikine_LM(Tep, q0=panda.q, ilimit=30, slimit=30)
                if not sol.success:
                    ok = False
                    break

                ee_trace, q_trace, success = servo_to_pose(
                    env, panda, Tep,
                    n=7, gain=2.0, dt=0.05,
                    arrived_thresh=0.005, max_steps=600,
                    slack_weight_scale=1.0, Y=0.01
                )

                if not success:
                    ok = False
                    break

                if ee_trace.size == 0:
                    ok = False
                    break

                ee_xy = ee_trace[:, :2].astype(np.float32)
                q7 = q_trace[:, :7].astype(np.float32)

                # write into fixed buffers with truncation
                k = min(ee_xy.shape[0], T - t)
                ee_xy_arr[demo_idx, t:t+k, :] = ee_xy[:k]
                q_arr[demo_idx, t:t+k, :] = q7[:k]
                full_xy_for_plot.append(ee_xy[:k])

                t += k
                pbar.n = demo_idx * T + t
                pbar.refresh()

                # print(t)

                if t >= T:
                    break

            if not ok:
                break


            # we reached the maze goal (end of subgoals) -> sample a NEW goal next loop
            cur = g

        if not ok or t < T:
            # discard this demo and try again
            print("not ok!!")
            continue
        import pdb; pdb.set_trace()

        starts[demo_idx] = np.array([s.r, s.c], dtype=np.int32)
        first_goals[demo_idx] = np.array([first_goal_cell.r, first_goal_cell.c], dtype=np.int32)

        # Optional PNG for first few demos: plot the entire 2000-step XY trace
        if save_png_examples and demo_idx < num_png:
            xy_plot = np.vstack(full_xy_for_plot) if len(full_xy_for_plot) else ee_xy_arr[demo_idx]
            fig = plt.figure()
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")

            ox, oy = origin_xy
            W = cols * cell_size
            H = rows * cell_size

            for r0 in range(rows):
                y0 = oy + r0 * cell_size
                y1 = y0 + cell_size
                for c0 in range(cols + 1):
                    if vertical_walls[r0, c0]:
                        x0 = ox + c0 * cell_size
                        ax.plot([x0, x0], [y0, y1])

            for r0 in range(rows + 1):
                y0 = oy + r0 * cell_size
                for c0 in range(cols):
                    if horizontal_walls[r0, c0]:
                        x0 = ox + c0 * cell_size
                        x1 = x0 + cell_size
                        ax.plot([x0, x1], [y0, y0])

            ax.plot(xy_plot[:, 0], xy_plot[:, 1], linewidth=2)
            ax.set_xlim(ox - 0.05, ox + W + 0.05)
            ax.set_ylim(oy - 0.05, oy + H + 0.05)
            ax.set_title(f"demo {demo_idx}: start={s.r,s.c} first_goal={first_goal_cell.r,first_goal_cell.c}")
            plt.savefig(f"demo_{demo_idx:04d}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        demo_idx += 1
        # pbar.update(1)

    pbar.close()

    # -------------------------
    # Save dataset
    # -------------------------
    np.savez_compressed(
        save_path,
        vertical_walls=vertical_walls.astype(np.uint8),
        horizontal_walls=horizontal_walls.astype(np.uint8),
        rows=np.int32(rows),
        cols=np.int32(cols),
        cell_size=np.float32(cell_size),
        origin_xy=np.array(origin_xy, dtype=np.float32),
        z_plane=np.float32(z_plane),

        starts=starts,
        goals=first_goals,   # first goal only (trajectory contains many internal goals)

        ee_xy=ee_xy_arr,
        q_list=q_arr,
        lengths=np.full((num_demos,), T, dtype=np.int32),
    )

    print(f"Saved dataset: {save_path}")
    print("ee_xy shape:", ee_xy_arr.shape)
    print("q_list shape:", q_arr.shape)

if __name__ == "__main__":
    main()
