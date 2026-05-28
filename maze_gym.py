# pip install qpsolvers[proxqp]

# git clone https://github.com/petercorke/robotics-toolbox-python.git
# cd robotics-toolbox-python
# pip3 install -e .

# git clone https://github.com/jhavl/swift.git
# cd swift
# pip3 install -e .

import os
import heapq
import numpy as np
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

import swift
import roboticstoolbox as rtb
import spatialmath as sm

import qpsolvers as qp


@dataclass(frozen=True)
class Cell:
    r: int
    c: int


def cells_to_waypoints(cell_path, cell_size, origin_xy=(0.0, 0.0)):
    ox, oy = origin_xy
    waypoints = []
    for cell in cell_path:
        x = ox + (cell.c + 0.5) * cell_size
        y = oy + (cell.r + 0.5) * cell_size
        waypoints.append((x, y))
    return waypoints


def servo_to_pose(env, panda, Tep, n=7, gain=5.0, dt=0.05,
                  arrived_thresh=0.05, max_steps=400,
                  slack_weight_scale=1.0, Y=0.01):
    """
    Your QP servo loop (unchanged in spirit).
    Drives end-effector toward Tep.
    """
    ee_trace = []
    q_trace = []

    for _ in range(max_steps):
        Te = panda.fkine(panda.q)
        ee_trace.append(Te.t.copy())
        q_trace.append(panda.q[:n].copy())

        eTep = Te.inv() * Tep
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

        v, _arrived = rtb.p_servo(Te, Tep, gain)

        if e < arrived_thresh:
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

        lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
        ub =  np.r_[panda.qdlim[:n], 10 * np.ones(6)]

        qd = qp.solve_qp(Q, c, Ain, bin_, Aeq, beq, lb=lb, ub=ub, solver="proxqp")
        if qd is None:
            # Fail safe
            break

        panda.qd[:n] = qd[:n]
        env.step(dt)

    return np.array(ee_trace), np.array(q_trace)


def load_maze_npz(maze_path: str):
    data = np.load(maze_path)
    vertical_walls = data["vertical_walls"].astype(bool)
    horizontal_walls = data["horizontal_walls"].astype(bool)

    # if rows/cols stored, prefer that; else infer
    rows = int(data["rows"]) if "rows" in data else int(vertical_walls.shape[0])
    cols = int(data["cols"]) if "cols" in data else int(horizontal_walls.shape[1])
    return rows, cols, vertical_walls, horizontal_walls


# ----------------------------
# Gymnasium Env
# ----------------------------

class MazePandaVelEnv(gym.Env):
    """
    Action: 7 joint velocities (rad/s), clipped to [-qdlim, +qdlim], optionally scaled.
    Reward: 1 if EE (x,y) within goal_thresh of goal waypoint, else 0.
    Termination: reached goal.
    Truncation: max_steps.

    Reset: samples start+goal cells, then uses your servo_to_pose controller to move to start.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        maze_path="mazes/maze_12x12.npz",
        rows=12,
        cols=12,
        cell_size=0.025,
        origin_xy=(0.35, -0.30),
        z_plane=0.25,
        dt=0.05,
        max_steps=300,
        min_dist=None,
        goal_thresh=0.01,
        realtime=False,
        seed=None,
        obs_mode="q"
    ):
        super().__init__()
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.goal_thresh = float(goal_thresh)
        self.realtime = bool(realtime)

        self.rng = np.random.default_rng(seed)

        # Maze
        if os.path.exists(maze_path):
            self.rows, self.cols, self.vertical_walls, self.horizontal_walls = load_maze_npz(maze_path)
        else:
            # allow passing rows/cols/cell_size without maze file (but you asked to load it)
            self.rows, self.cols = int(rows), int(cols)
            self.vertical_walls = None
            self.horizontal_walls = None

        self.cell_size = float(cell_size)
        self.origin_xy = (float(origin_xy[0]), float(origin_xy[1]))
        self.z_plane = float(z_plane)
        self.min_dist = min_dist if min_dist is not None else (self.rows + self.cols) // 3

        # Simulator + robot
        self.env = swift.Swift()
        self.env.launch(realtime=self.realtime)

        self.panda = rtb.models.Panda()
        self.panda.q = self.panda.qr
        self.panda.qd = np.zeros_like(self.panda.q)
        self.env.add(self.panda)

        # Fixed EE orientation from ready pose
        Te0 = self.panda.fkine(self.panda.q)
        R = Te0.R
        if hasattr(R, "A"):
            R = R.A
        self.R_fixed = np.array(R).reshape(3, 3)

        # Spaces
        self.n = 7
        qdlim = self.panda.qdlim[:self.n].astype(np.float32)

        # Action is normalized? Here: raw velocities in rad/s (bounded by qdlim)
        self.action_space = spaces.Box(low=-qdlim, high=qdlim, dtype=np.float32)

        self.obs_mode = obs_mode  # "ee" or "q"

        if self.obs_mode == "ee":
            low = -np.inf * np.ones(2, dtype=np.float32)
            high = np.inf * np.ones(2, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        elif self.obs_mode == "q":
            low = -np.inf * np.ones(7, dtype=np.float32)
            high = np.inf * np.ones(7, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        else:
            raise ValueError(f"Unknown obs_mode={self.obs_mode}. Use 'ee' or 'q'.")

        # Episode state
        self._step_count = 0
        self.start_cell = None
        self.goal_cell = None
        self.goal_xy = None

    def _cell_center_xy(self, cell: Cell):
        ox, oy = self.origin_xy
        x = ox + (cell.c + 0.5) * self.cell_size
        y = oy + (cell.r + 0.5) * self.cell_size
        return np.array([x, y], dtype=np.float32)

    def _sample_start_goal(self):
        while True:
            s = Cell(int(self.rng.integers(self.rows)), int(self.rng.integers(self.cols)))
            g = Cell(int(self.rng.integers(self.rows)), int(self.rng.integers(self.cols)))
            if s != g and (abs(s.r - g.r) + abs(s.c - g.c) >= self.min_dist):
                return s, g

    def _move_to_cell_with_controller(self, cell: Cell):
        """Use your controller to place EE at the start cell center before the episode begins."""
        xy = self._cell_center_xy(cell)
        Tep = sm.SE3.Rt(self.R_fixed, [float(xy[0]), float(xy[1]), self.z_plane])

        # Quick feasibility check (optional but helps avoid weird starts)
        sol = self.panda.ikine_LM(Tep, q0=self.panda.q, ilimit=30, slimit=30)
        if not sol.success:
            return False

        servo_to_pose(
            self.env, self.panda, Tep,
            n=7, gain=3.0, dt=self.dt,
            arrived_thresh=0.005, max_steps=600,
            slack_weight_scale=1.0, Y=0.01
        )
        return True

    def _get_obs(self):
        ee_xy = self.panda.fkine(self.panda.q).t[:2].astype(np.float32)
        # goal_xy = self.goal_xy.astype(np.float32)

        if self.obs_mode == "ee":
            return ee_xy.astype(np.float32)

        elif self.obs_mode == "q":
            q = self.panda.q[:7].astype(np.float32)
            return q.astype(np.float32)

        raise ValueError(f"Unknown obs_mode={self.obs_mode}")

    def _at_goal(self):
        ee = self.panda.fkine(self.panda.q).t[:2]
        return float(np.linalg.norm(ee - self.goal_xy)) <= self.goal_thresh

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0

        # Reset robot state
        self.panda.q = self.panda.qr
        self.panda.qd = np.zeros_like(self.panda.q)
        self.env.step(self.dt)

        # Sample start/goal; ensure we can reach start pose
        for _ in range(50):
            s, g = self._sample_start_goal()
            if self._move_to_cell_with_controller(s):
                self.start_cell, self.goal_cell = s, g
                self.goal_xy = self._cell_center_xy(g)
                break
        else:
            raise RuntimeError("Failed to initialize a valid start pose after many attempts.")

        obs = self._get_obs()
        info = {
            "start_cell": (self.start_cell.r, self.start_cell.c),
            "goal_cell": (self.goal_cell.r, self.goal_cell.c),
            "goal_xy": self.goal_xy.copy(),
        }
        self.curr_goal = self.goal_xy
        return obs, info

    def step(self, action):
        self._step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(self.n)

        # Clip to joint velocity limits
        qdlim = self.panda.qdlim[:self.n].astype(np.float32)
        qd = np.clip(action, -qdlim, qdlim)

        # Apply velocities
        self.panda.qd[:self.n] = qd
        self.env.step(self.dt)

        terminated = self._at_goal()
        truncated = self._step_count >= self.max_steps

        reward = 1.0 if terminated else 0.0

        obs = self._get_obs()
        info = {
            "ee_xy": self.panda.fkine(self.panda.q).t[:2].copy(),
            "goal_xy": self.goal_xy.copy(),
            "dist_to_goal": float(np.linalg.norm(self.panda.fkine(self.panda.q).t[:2] - self.goal_xy)),
            "step": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        # Swift is already rendering if launched with realtime=True.
        # For "human", do nothing.
        return None

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ----------------------------
# Quick smoke test
# ----------------------------
if __name__ == "__main__":
    env = MazePandaVelEnv(
        maze_path="mazes/maze_12x12.npz",
        realtime=True,
        obs_mode="q",
        goal_thresh=0.01,
        max_steps=400,
        dt=0.05,
    )

    obs, info = env.reset()
    print("reset info:", info)

    done = False
    while not done:
        # random policy
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        if r > 0:
            print("Reached goal!")
    env.close()
