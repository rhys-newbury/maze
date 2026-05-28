import numpy as np
import hashlib
from collections import defaultdict

def hash_trajectory(traj):
    """
    traj: (T, D) float32/float64 array
    Returns a stable SHA256 hash of its raw bytes.
    """
    traj = np.ascontiguousarray(traj)
    return hashlib.sha256(traj.tobytes()).hexdigest()

def check_unique_q_list(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    q_list = data["q_list"]

    if q_list.ndim != 3:
        raise ValueError(f"Expected q_list to be (N,T,D), got {q_list.shape}")

    seen = {}
    duplicates = defaultdict(list)

    for i in range(q_list.shape[0]):
        h = hash_trajectory(q_list[i])
        if h in seen:
            duplicates[h].append(i)
        else:
            seen[h] = i

    if duplicates:
        print("❌ Duplicate trajectories found!")
        for h, idxs in duplicates.items():
            print(f"Hash {h[:10]}... duplicates: {seen[h]} and {idxs}")
    else:
        print("✅ All trajectories are unique!")

    print(f"Total trajectories: {q_list.shape[0]}")
    print(f"Unique hashes: {len(seen)}")

check_unique_q_list("maze.npz")
