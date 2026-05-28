import os
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm


def load_maze_npz(maze_path: str):
    data = np.load(maze_path)
    vertical_walls = data["vertical_walls"].astype(bool)
    horizontal_walls = data["horizontal_walls"].astype(bool)

    rows = int(data["rows"]) if "rows" in data else int(vertical_walls.shape[0])
    cols = int(data["cols"]) if "cols" in data else int(horizontal_walls.shape[1])
    return rows, cols, vertical_walls, horizontal_walls


def cell_center_xy(r, c, cell_size, origin_xy):
    ox, oy = origin_xy
    x = ox + (c + 0.5) * cell_size
    y = oy + (r + 0.5) * cell_size
    return np.array([x, y], dtype=np.float64)


def main(
    maze_path="mazes/maze_12x12.npz",
    cell_size=0.025,
    origin_xy=(0.35, -0.30),
    z_plane=0.25,
    out_path="precomputed/cell_joint_map.npz",
):
    # ----------------------------
    # Load maze
    # ----------------------------
    rows, cols, _, _ = load_maze_npz(maze_path)
    print(f"Loaded maze: {rows} x {cols}")

    # ----------------------------
    # Robot model
    # ----------------------------
    panda = rtb.models.Panda()
    panda.q = panda.qr.copy()

    # Fixed EE orientation from ready pose
    Te0 = panda.fkine(panda.q)
    R = Te0.R.A if hasattr(Te0.R, "A") else np.array(Te0.R)

    # ----------------------------
    # Storage
    # ----------------------------
    joint_map = np.zeros((rows, cols, 7), dtype=np.float32)
    valid_mask = np.zeros((rows, cols), dtype=bool)
    ee_xyz_map = np.zeros((rows, cols, 3), dtype=np.float32)

    # ----------------------------
    # IK for each cell
    # ----------------------------
    n_success = 0
    for r in range(rows):
        for c in range(cols):
            xy = cell_center_xy(r, c, cell_size, origin_xy)
            Tep = sm.SE3.Rt(R, [float(xy[0]), float(xy[1]), float(z_plane)])

            sol = panda.ikine_LM(
                Tep,
                q0=panda.qr,
                ilimit=50,
                slimit=50,
                tol=1e-6,
            )

            if sol.success:
                q = sol.q[:7]
                joint_map[r, c, :] = q.astype(np.float32)
                valid_mask[r, c] = True

                Te = panda.fkine(q)
                ee_xyz_map[r, c, :] = Te.t.astype(np.float32)

                n_success += 1
            else:
                valid_mask[r, c] = False

    # ----------------------------
    # Save
    # ----------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez(
        out_path,
        joint_map=joint_map,        # (rows, cols, 7)
        valid_mask=valid_mask,      # (rows, cols) bool
        ee_xyz_map=ee_xyz_map,      # (rows, cols, 3)
        rows=rows,
        cols=cols,
        cell_size=cell_size,
        origin_xy=np.array(origin_xy, dtype=np.float32),
        z_plane=float(z_plane),
        R_fixed=R.astype(np.float32),
    )

    print(f"Saved: {out_path}")
    print(f"Valid IK solutions: {n_success} / {rows * cols}")


if __name__ == "__main__":
    main()
