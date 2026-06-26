# delete_simulations.py
#
# Find and delete MPS files and/or correlator CSVs + images for simulations
# listed in a simulations_input.txt file (same format as run_simulations.py).
#
# Usage:
#   python delete_simulations.py [--simulations_file FILE] [--out_dir DIR]
#
# The script will:
#   1. Locate every MPS file matching the listed simulations.
#   2. Find associated correlator CSVs and images.
#   3. Ask whether to delete EVERYTHING or only correlators/images.
#   4. Ask for final confirmation before touching any file.

from __future__ import annotations
from pathlib import Path
import argparse

from run_simulations import (
    _parse_simulations_file,
    _find_existing_output,
    _get_model_entry,
    MODEL_OUTPUT_SUBDIR_CANDIDATES,
    OUTPUT_ROOT_DIR,
)

DELETE_INPUT_FILE = "delete_input.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_mps_dir(model_key: str, out_root: Path) -> Path | None:
    """Return the mps_files sub-directory if it exists, without creating it."""
    for subdir in MODEL_OUTPUT_SUBDIR_CANDIDATES.get(model_key, []):
        candidate = out_root / subdir / "mps_files"
        if candidate.is_dir():
            return candidate
    return None


def _correlator_dirs(mps_path: Path) -> tuple[Path, Path]:
    """Mirror of _dirs_for_mps_file in correlators_extractor.py."""
    parent = mps_path.parent
    dataset_root = parent.parent if parent.name == "mps_files" else parent
    return dataset_root / "correlators_csv", dataset_root / "correlators_plots"


def collect_files(sim: dict, out_root: Path) -> dict:
    """
    Locate all files associated with one simulation row.
    Returns {'mps': Path|None, 'csvs': [Path], 'images': [Path]}.
    """
    model_key = sim["model"]
    entry = _get_model_entry(model_key)
    mps_dir = _resolve_mps_dir(model_key, out_root)
    if mps_dir is None:
        return {"mps": None, "csvs": [], "images": []}

    mps_path = _find_existing_output(
        sim_out_dir=mps_dir,
        geom_tag=entry["geom_tag"],
        alpha=sim["alpha"],
        r=sim["r"],
        U=sim["U"],
        mu=sim["mu"],
        t=sim["t"],
        conserve=sim["conserve"],
        L=sim["L"],
        N_uc=sim["N_in_unit_cell"],
        n_max=sim["n_max"],
    )
    if mps_path is None:
        return {"mps": None, "csvs": [], "images": []}

    tag = mps_path.stem
    csv_dir, img_dir = _correlator_dirs(mps_path)

    csvs = sorted(csv_dir.glob(f"{tag}_*.csv")) if csv_dir.is_dir() else []
    images = sorted(img_dir.glob(f"{tag}_*.png")) if img_dir.is_dir() else []

    return {"mps": mps_path, "csvs": csvs, "images": images}


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def _ask_deletion_mode() -> str:
    """Return '1' (everything), '2' (correlators only), or 'q' (quit)."""
    print()
    print("What do you want to delete?")
    print("  [1] Everything      — MPS file + correlator CSVs + images")
    print("  [2] Correlators only — CSVs + images (MPS files are kept)")
    print("  [q] Quit without deleting anything")
    while True:
        choice = input("Your choice [1/2/q]: ").strip().lower()
        if choice in ("1", "2", "q"):
            return choice
        print("  Invalid input. Please enter 1, 2, or q.")


def _confirm(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please enter y or n.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Delete MPS files and/or correlator CSVs/images for simulations "
            "listed in a simulations_input.txt file."
        )
    )
    parser.add_argument(
        "--simulations_file",
        default=DELETE_INPUT_FILE,
        help=f"Simulations input file (default: {DELETE_INPUT_FILE}).",
    )
    parser.add_argument(
        "--out_dir",
        default=OUTPUT_ROOT_DIR,
        help=f"Root output directory (default: {OUTPUT_ROOT_DIR}).",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    sim_rows = _parse_simulations_file(args.simulations_file)

    # --- Collect files for each simulation ---
    results = [(sim, collect_files(sim, out_root)) for sim in sim_rows]
    found     = [(s, f) for s, f in results if f["mps"] is not None]
    not_found = [(s, f) for s, f in results if f["mps"] is None]

    # --- Summary ---
    print(f"\nScanned {len(sim_rows)} simulation(s) from '{args.simulations_file}'.")
    print(f"  Located  : {len(found)}")
    print(f"  Not found: {len(not_found)}")

    if not_found:
        print("\n  Simulations with no matching MPS file:")
        for sim, _ in not_found:
            print(
                f"    model={sim['model']}  alpha={sim['alpha']}  r={sim['r']}"
                f"  U={sim['U']}  mu={sim['mu']}  t={sim['t']}"
                f"  L={sim['L']}  N_uc={sim['N_in_unit_cell']}  cons={sim['conserve']}"
            )

    if not found:
        print("\nNothing to delete.")
        return

    # --- Detailed file list ---
    total_csvs = sum(len(f["csvs"])   for _, f in found)
    total_imgs = sum(len(f["images"]) for _, f in found)

    print("\n--- Located files ---")
    for sim, files in found:
        print(f"\n  MPS   : {files['mps'].name}")
        if files["csvs"]:
            print(f"  CSVs  : {len(files['csvs'])} file(s)  "
                  f"[{files['csvs'][0].parent}]")
        else:
            print("  CSVs  : none")
        if files["images"]:
            print(f"  Images: {len(files['images'])} file(s)  "
                  f"[{files['images'][0].parent}]")
        else:
            print("  Images: none")

    print(
        f"\nTotal: {len(found)} MPS file(s), "
        f"{total_csvs} CSV file(s), {total_imgs} image file(s)."
    )

    # --- Ask what to delete ---
    mode = _ask_deletion_mode()
    if mode == "q":
        print("Aborted. Nothing deleted.")
        return

    delete_mps = (mode == "1")
    label = "MPS + CSVs + images" if delete_mps else "CSVs + images (MPS kept)"

    if not _confirm(f"\nConfirm: permanently delete [{label}] for {len(found)} simulation(s)?"):
        print("Aborted. Nothing deleted.")
        return

    # --- Delete ---
    deleted_mps  = 0
    deleted_csvs = 0
    deleted_imgs = 0
    errors       = 0

    for sim, files in found:
        try:
            if delete_mps:
                files["mps"].unlink()
                deleted_mps += 1
            for p in files["csvs"]:
                p.unlink()
                deleted_csvs += 1
            for p in files["images"]:
                p.unlink()
                deleted_imgs += 1
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            errors += 1

    # --- Report ---
    print("\nDone.")
    if delete_mps:
        print(f"  Deleted MPS    : {deleted_mps}")
    print(f"  Deleted CSVs   : {deleted_csvs}")
    print(f"  Deleted images : {deleted_imgs}")
    if errors:
        print(f"  Errors         : {errors}")


if __name__ == "__main__":
    main()
