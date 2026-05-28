import numpy as np
from pathlib import Path

def _pad_to(arr, target_T):
    """
    Pad a time-major array from (N, T, D) -> (N, target_T, D) with zeros.
    If already target_T, returns arr.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (N,T,D), got shape {arr.shape}")
    N, T, D = arr.shape
    if T == target_T:
        return arr
    out = np.zeros((N, target_T, D), dtype=arr.dtype)
    out[:, :T, :] = arr
    return out

def combine_npz(in_pattern, out_path, ids=range(10), float_rtol=1e-6, float_atol=1e-6):
    paths = [Path(in_pattern.format(i=i)) for i in ids]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    data0 = np.load(paths[0], allow_pickle=False)

    # --- reference maze fields from file 0
    v0 = data0["vertical_walls"]
    h0 = data0["horizontal_walls"]
    rows0 = int(data0["rows"])
    cols0 = int(data0["cols"])
    cell0 = float(data0["cell_size"])
    origin0 = np.array(data0["origin_xy"], dtype=np.float32)
    z0 = float(data0["z_plane"])

    # --- collect episode fields
    starts_all, goals_all, lengths_all = [], [], []
    ee_all, q_all = [], []
    maxT = 0

    def check_same_maze(d, path):
        # Walls should match bitwise
        if d["vertical_walls"].shape != v0.shape or not np.array_equal(d["vertical_walls"], v0):
            raise ValueError(f"Maze mismatch in vertical_walls: {path}")
        if d["horizontal_walls"].shape != h0.shape or not np.array_equal(d["horizontal_walls"], h0):
            raise ValueError(f"Maze mismatch in horizontal_walls: {path}")

        # Scalars/metadata
        if int(d["rows"]) != rows0 or int(d["cols"]) != cols0:
            raise ValueError(f"Maze mismatch in rows/cols: {path}")
        if not np.isclose(float(d["cell_size"]), cell0, rtol=float_rtol, atol=float_atol):
            raise ValueError(f"Maze mismatch in cell_size: {path}")
        if not np.allclose(np.array(d["origin_xy"], dtype=np.float32), origin0, rtol=float_rtol, atol=float_atol):
            raise ValueError(f"Maze mismatch in origin_xy: {path}")
        if not np.isclose(float(d["z_plane"]), z0, rtol=float_rtol, atol=float_atol):
            raise ValueError(f"Maze mismatch in z_plane: {path}")

    # First pass: validate & find global max time length
    for p in paths:
        d = np.load(p, allow_pickle=False)
        check_same_maze(d, p)

        ee = d["ee_xy"]
        q = d["q_list"]
        if ee.ndim != 3 or q.ndim != 3:
            raise ValueError(f"Expected ee_xy and q_list to be 3D (N,T,D). Got {p}: ee {ee.shape}, q {q.shape}")
        if ee.shape[0] != q.shape[0] or ee.shape[1] != q.shape[1]:
            raise ValueError(f"N/T mismatch between ee_xy and q_list in {p}: ee {ee.shape}, q {q.shape}")

        maxT = max(maxT, ee.shape[1])

    # Second pass: gather and pad to maxT
    for p in paths:
        d = np.load(p, allow_pickle=False)

        starts = d["starts"]
        goals = d["goals"]
        lengths = d["lengths"]

        ee = d["ee_xy"].astype(np.float32, copy=False)
        q = d["q_list"].astype(np.float32, copy=False)

        # Basic consistency checks per-file
        N = ee.shape[0]
        if starts.shape[0] != N or goals.shape[0] != N or lengths.shape[0] != N:
            raise ValueError(
                f"Episode count mismatch in {p}: "
                f"N={N}, starts={starts.shape}, goals={goals.shape}, lengths={lengths.shape}"
            )

        ee = _pad_to(ee, maxT)
        q = _pad_to(q, maxT)

        starts_all.append(starts)
        goals_all.append(goals)
        lengths_all.append(lengths)
        ee_all.append(ee)
        q_all.append(q)

    # Concatenate along episode dimension
    starts_cat = np.concatenate(starts_all, axis=0)
    goals_cat = np.concatenate(goals_all, axis=0)
    lengths_cat = np.concatenate(lengths_all, axis=0)
    ee_cat = np.concatenate(ee_all, axis=0)
    q_cat = np.concatenate(q_all, axis=0)

    # Save combined (store maze once, plus concatenated episodes)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        vertical_walls=v0.astype(np.uint8),
        horizontal_walls=h0.astype(np.uint8),
        rows=np.int32(rows0),
        cols=np.int32(cols0),
        cell_size=np.float32(cell0),
        origin_xy=origin0.astype(np.float32),
        z_plane=np.float32(z0),
        starts=starts_cat,
        goals=goals_cat,
        ee_xy=ee_cat.astype(np.float32),
        q_list=q_cat.astype(np.float32),
        lengths=lengths_cat,
    )

    print(f"Saved: {out_path}")
    print(f"Combined episodes: {ee_cat.shape[0]}")
    print(f"Time length (padded): {ee_cat.shape[1]}")
    print(f"ee_xy shape: {ee_cat.shape}, q_list shape: {q_cat.shape}")

if __name__ == "__main__":
    # Example: files named "maze_0.npz" ... "maze_9.npz"
    combine_npz(
        in_pattern="maze_dataset_{i}.npz",
        out_path="maze.npz",
        ids=range(10),
    )
