import os
import glob
import numpy as np

def combine_npz_datasets(
    src_dir="/mnt/slow",
    pattern="maze_dataset_*.npz",
    k=20,
    out_path=None,
):
    files = sorted(glob.glob(os.path.join(src_dir, pattern)))
    if len(files) < k:
        raise RuntimeError(f"Found only {len(files)} files in {src_dir} matching {pattern}, need {k}.")

    files = files[:k]
    print("Combining:")
    for f in files:
        print("  ", f)

    # Load all
    datas = [np.load(f, allow_pickle=False) for f in files]

    # ---- Sanity checks: require maze meta to match (walls/grid/geometry) ----
    meta_keys = ["rows", "cols", "cell_size", "origin_xy", "z_plane"]
    wall_keys = ["vertical_walls", "horizontal_walls"]

    ref = datas[0]
    for i, d in enumerate(datas[1:], start=1):
        for mk in meta_keys:
            if not np.allclose(d[mk], ref[mk]):
                raise ValueError(f"Mismatch in {mk} between file0 and file{i}")
        for wk in wall_keys:
            if d[wk].shape != ref[wk].shape or not np.array_equal(d[wk], ref[wk]):
                raise ValueError(f"Mismatch in {wk} between file0 and file{i}")

    # ---- Concatenate dataset fields ----
    cat_keys = ["starts", "goals", "ee_xy", "q_list", "lengths"]
    combined = {}
    for ck in cat_keys:
        combined[ck] = np.concatenate([d[ck] for d in datas], axis=0)

    # Carry over maze definition from first file
    combined["vertical_walls"] = ref["vertical_walls"]
    combined["horizontal_walls"] = ref["horizontal_walls"]
    combined["rows"] = ref["rows"]
    combined["cols"] = ref["cols"]
    combined["cell_size"] = ref["cell_size"]
    combined["origin_xy"] = ref["origin_xy"]
    combined["z_plane"] = ref["z_plane"]

    if out_path is None:
        out_path = os.path.join(src_dir, f"maze_dataset_combined_{k}.npz")

    np.savez_compressed(out_path, **combined)

    # Close NpzFile handles
    for d in datas:
        d.close()

    print(f"\nSaved combined dataset: {out_path}")
    print("Combined shapes:")
    print("  starts :", combined["starts"].shape)
    print("  goals  :", combined["goals"].shape)
    print("  ee_xy  :", combined["ee_xy"].shape)
    print("  q_list :", combined["q_list"].shape)
    print("  lengths:", combined["lengths"].shape)

    return out_path

if __name__ == "__main__":
    combine_npz_datasets()