"""
rename_files.py
===============
One-shot migration script.  Run this ONCE from the project root:

    python rename_files.py

What it does
------------
1. Creates the new directory tree:
       mps_files_infinite / {mps_files, correlators_csv, correlators_plots}
       mps_files_finite   / {mps_files, correlators_csv, correlators_plots}

2. Renames every file by replacing the old "model tag" with the new
   alpha/r tag, then moves it into the appropriate new folder:

   Source folder                       Old tag (or no tag)   New tag
   ---------------------------------   ------------------    -----------
   mps_files_with_damping              (no tag)              alpha3_r3
   mps_files_without_damping           (no tag)              alpha0_r3
   mps_files_legacy_infinte            legacy                alpha0_r1
   mps_files_with_damping_finite       damped                alpha3_r3

3. Renames files inside mps_dipole_gap in-place:
       _damped_     ->  _alpha3_r3_
       _notDamped_  ->  _alpha0_r3_
       _legacy_     ->  _alpha0_r1_

4. Prints a summary and does NOT delete the old folders (you can inspect
   and remove them manually once you are happy).

Safe to re-run: files already present at the destination are skipped with
a warning rather than overwritten.
"""

from __future__ import annotations
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()

# (source_folder, new_model_tag, destination_folder)
FOLDER_MAP = [
    ("mps_files_with_damping",        "alpha3_r3", "mps_files_infinite"),
    ("mps_files_without_damping",     "alpha0_r3", "mps_files_infinite"),
    ("mps_files_legacy_infinte",      "alpha0_r1", "mps_files_infinite"),
    ("mps_files_with_damping_finite", "alpha3_r3", "mps_files_finite"),
]

# Sub-folder pairs: (source subfolder name, dest subfolder name)
SUBFOLDER_PAIRS = [
    ("mps_files",         "mps_files"),
    ("correlators_csv",   "correlators_csv"),
    ("correlators_plots", "correlators_plots"),
]

# In-place renaming inside mps_dipole_gap
DIPOLE_GAP_DIR = ROOT / "mps_dipole_gap"
DIPOLE_GAP_RENAMES = {
    "_damped_":    "_alpha3_r3_",
    "_notDamped_": "_alpha0_r3_",
    "_legacy_":    "_alpha0_r1_",
}

# Known old tags that appear directly after the geom prefix in filenames
# where no explicit tag was written (only needed for the two "no-tag" folders)
_OLD_TAGS = {"damped_", "notDamped_", "legacy_"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_new_name(filename: str, new_model_tag: str) -> str:
    """
    Insert / replace the model tag in a filename.

    Handles:
      iMPS_U1_...          (no old tag)   -> iMPS_alpha3_r3_U1_...
      iMPS_legacy_U1_...   (legacy tag)   -> iMPS_alpha0_r1_U1_...
      fMPS_damped_U1_...   (damped tag)   -> fMPS_alpha3_r3_U1_...
      dipole_gaps_fMPS_damped_...         -> dipole_gaps_fMPS_alpha3_r3_...
    """
    for geom in ("iMPS", "fMPS"):
        # Case 1: file starts with "dipole_gaps_{geom}_"
        dp = f"dipole_gaps_{geom}_"
        if filename.startswith(dp):
            rest = filename[len(dp):]
            for old in _OLD_TAGS:
                if rest.startswith(old):
                    return dp + new_model_tag + "_" + rest[len(old):]
            return dp + new_model_tag + "_" + rest

        # Case 2: file starts with "{geom}_"
        gp = geom + "_"
        if filename.startswith(gp):
            rest = filename[len(gp):]
            for old in _OLD_TAGS:
                if rest.startswith(old):
                    return gp + new_model_tag + "_" + rest[len(old):]
            return gp + new_model_tag + "_" + rest

    return filename  # fallback: unchanged


def _move_file(src: Path, dst: Path, counters: dict) -> None:
    if dst.exists():
        print(f"  [SKIP] destination exists: {dst.name}")
        counters["skipped"] += 1
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    counters["moved"] += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def migrate_main_folders(counters: dict) -> None:
    for src_name, new_tag, dst_name in FOLDER_MAP:
        src_root = ROOT / src_name
        dst_root = ROOT / dst_name
        if not src_root.is_dir():
            print(f"[WARN] Source folder not found, skipping: {src_name}")
            continue

        print(f"\n>>> Migrating  {src_name}  ->  {dst_name}  (tag={new_tag})")

        for src_sub, dst_sub in SUBFOLDER_PAIRS:
            src_dir = src_root / src_sub
            dst_dir = dst_root / dst_sub
            if not src_dir.is_dir():
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(src_dir.iterdir())
            for f in files:
                if not f.is_file():
                    continue
                new_name = _compute_new_name(f.name, new_tag)
                dst_path = dst_dir / new_name
                _move_file(f, dst_path, counters)


def rename_dipole_gap(counters: dict) -> None:
    if not DIPOLE_GAP_DIR.is_dir():
        print(f"\n[WARN] mps_dipole_gap not found at {DIPOLE_GAP_DIR}")
        return

    print(f"\n>>> Renaming files in-place in  mps_dipole_gap/")
    for f in sorted(DIPOLE_GAP_DIR.iterdir()):
        if not f.is_file():
            continue
        new_name = f.name
        for old_str, new_str in DIPOLE_GAP_RENAMES.items():
            if old_str in new_name:
                new_name = new_name.replace(old_str, new_str, 1)
                break
        if new_name != f.name:
            dst = f.parent / new_name
            if dst.exists():
                print(f"  [SKIP] destination exists: {dst.name}")
                counters["skipped"] += 1
            else:
                f.rename(dst)
                counters["moved"] += 1
                print(f"  {f.name}  ->  {new_name}")


def main() -> None:
    counters = {"moved": 0, "skipped": 0}
    migrate_main_folders(counters)
    rename_dipole_gap(counters)

    print("\n" + "=" * 60)
    print(f"Done.  Moved/renamed: {counters['moved']}   Skipped: {counters['skipped']}")
    print(
        "Old source folders are left intact so you can verify before deleting.\n"
        "When satisfied, remove:\n"
        "  mps_files_with_damping/\n"
        "  mps_files_without_damping/\n"
        "  mps_files_legacy_infinte/\n"
        "  mps_files_with_damping_finite/"
    )


if __name__ == "__main__":
    main()
