# correlators_extractor.py
#
# Recursively scan for .mps files, compute correlators/density, and write CSV + plots.
# Existing CSV files are reused (no recomputation).

from pathlib import Path
import pickle

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tenpy.networks.mps import MPS


# Root folder scanned recursively for .mps files.
SCAN_ROOT = "."

# Global controls
RANGE = 100
I0 = 1           # starting site for infinite MPS correlators
BOUNDARY_AVOID = 50  # for finite MPS: skip this many sites at each boundary
PROGRESS_EVERY = 10


def _log_site_progress(kind: str, i: int, r: int, j: int, total: int):
    if total <= 0:
        return
    if r == 0 or r == total - 1 or ((r + 1) % PROGRESS_EVERY == 0):
        print(f"    [{kind}] i={i}, j={j} (r={r + 1}/{total})", flush=True)


def _is_infinite_mps(psi: MPS) -> bool:
    return getattr(psi, "bc", None) == "infinite"


def _r_max_for(kind: str, psi: MPS, i: int) -> int:
    """
    Safe number of r-points for each correlator kind.
    For infinite MPS, RANGE is used.
    For finite MPS, j is capped so it stays >= BOUNDARY_AVOID from the far end
    (i.e. the last j satisfies j <= L - 1 - BOUNDARY_AVOID, adjusted for each
    operator's look-ahead).
    """
    if _is_infinite_mps(psi):
        return RANGE

    L = int(psi.L)
    end = BOUNDARY_AVOID  # sites to leave free at the far boundary
    if kind == "current":
        # Uses j+2 and i+2; need j+2 <= L-1-end → j <= L-3-end
        max_points = L - i - 2 - end
    elif kind in ("dipole", "density"):
        # Uses j+1 and i+1; need j+1 <= L-1-end → j <= L-2-end
        max_points = L - i - 1 - end
    elif kind in ("single_particle", "density_density"):
        # Uses j; need j <= L-1-end
        max_points = L - i - end
    else:
        max_points = RANGE

    return max(0, min(RANGE, max_points))


def dipole_correlator(psi: MPS, i: int, r_max: int | None = None):
    # C_d(r) = < d^dag_i d_{i+r} >, with d^dag_j = b^dag_j b_{j+1}
    if r_max is None:
        r_max = RANGE
    rs, vals = [], []
    for r in range(r_max):
        j = i + r
        _log_site_progress("dipole_correlator", i, r, j, r_max)
        op = [("Bd", i), ("B", i + 1), ("B", j), ("Bd", j + 1)]
        vals.append(psi.expectation_value_term(op))
        rs.append(r)
    return np.array(rs), np.array(vals)


def dipole_density(psi: MPS, j: int):
    # n_tilde_j = b^dag_j b_{j+1} b^dag_{j+1} b_j = N_j (1 + N_{j+1})
    # Expanded to ascending site order to avoid TeNPy reordering (which
    # accesses unit_cell_width, absent in old pickled MPS objects).
    nj = psi.expectation_value_term([("N", j)])
    nj_njp1 = psi.expectation_value_term([("N", j), ("N", j + 1)])
    return nj + nj_njp1


def density_correlator(psi: MPS, i: int, r_max: int | None = None):
    # Connected dipole-density correlator
    if r_max is None:
        r_max = RANGE
    ni = dipole_density(psi, i)
    rs, vals = [], []
    for r in range(r_max):
        j = i + r
        _log_site_progress("density_correlator", i, r, j, r_max)
        nj = dipole_density(psi, j)
        ninj = psi.expectation_value_term([
            ("Bd", i), ("B", i + 1), ("Bd", i + 1), ("B", i),
            ("Bd", j), ("B", j + 1), ("Bd", j + 1), ("B", j),
        ])
        vals.append(ninj - ni * nj)
        rs.append(r)
    return np.array(rs), np.array(vals)


def current_correlator(psi: MPS, i: int, r_max: int | None = None):
    if r_max is None:
        r_max = RANGE
    rs, vals = [], []
    for r in range(r_max):
        j = i + r
        _log_site_progress("current_correlator", i, r, j, r_max)

        op_AA = [
            ("Bd", i), ("B", i + 1), ("Bd", i + 2), ("B", i + 1),
            ("Bd", j), ("B", j + 1), ("Bd", j + 2), ("B", j + 1),
        ]
        op_AB = [
            ("Bd", i), ("B", i + 1), ("Bd", i + 2), ("B", i + 1),
            ("Bd", j + 1), ("B", j + 2), ("Bd", j + 1), ("B", j),
        ]
        op_BA = [
            ("Bd", i + 1), ("B", i + 2), ("Bd", i + 1), ("B", i),
            ("Bd", j), ("B", j + 1), ("Bd", j + 2), ("B", j + 1),
        ]
        op_BB = [
            ("Bd", i + 1), ("B", i + 2), ("Bd", i + 1), ("B", i),
            ("Bd", j + 1), ("B", j + 2), ("Bd", j + 1), ("B", j),
        ]

        AA = psi.expectation_value_term(op_AA)
        AB = psi.expectation_value_term(op_AB)
        BA = psi.expectation_value_term(op_BA)
        BB = psi.expectation_value_term(op_BB)

        vals.append(-AA + AB + BA - BB)
        rs.append(r)

    return np.array(rs), np.array(vals)


def single_particle_correlator(psi: MPS, i: int, r_max: int | None = None):
    if r_max is None:
        r_max = RANGE
    rs, vals = [], []
    for r in range(r_max):
        j = i + r
        _log_site_progress("single_particle_correlator", i, r, j, r_max)
        vals.append(psi.expectation_value_term([("Bd", i), ("B", j)]))
        rs.append(r)
    return np.array(rs), np.array(vals)


def density_density_correlator(psi: MPS, i: int, r_max: int | None = None):
    # C_nn(r) = <n_i n_{i+r}> - <n_i><n_{i+r}>
    if r_max is None:
        r_max = RANGE
    rs, vals = [], []
    ni = psi.expectation_value_term([("N", i)])
    for r in range(r_max):
        j = i + r
        _log_site_progress("density_density_correlator", i, r, j, r_max)
        ninj = psi.expectation_value_term([("Bd", i), ("B", i), ("Bd", j), ("B", j)])
        nj = psi.expectation_value_term([("N", j)])
        vals.append(ninj - ni * nj)
        rs.append(r)
    return np.array(rs), np.array(vals)


def fft_corr(c):
    return np.real_if_close(np.fft.rfft(c, norm="forward"))


def plot_curve(x, y, title, ylabel, out_prefix: str):
    if len(x) == 0:
        return
    fig, ax = plt.subplots()
    ax.loglog(x, np.abs(y), "o-")
    ax.set_xlabel("r")
    ax.set_ylabel(f"|{ylabel}|")
    ax.set_title(f"{title} (abs log-log)")
    ax.grid(True, which="both")
    fig.savefig(f"{out_prefix}_abs_loglog.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fft(spectrum, title, out_prefix: str):
    if len(spectrum) == 0:
        return
    k = np.arange(len(spectrum))
    fig, ax = plt.subplots()
    ax.plot(k, spectrum, ".-")
    ax.set_xlabel("momentum k")
    ax.set_ylabel("S(k)")
    ax.set_title(f"{title} - Fourier spectrum")
    ax.grid(True)
    fig.savefig(f"{out_prefix}_FFT.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dipole_density_from_array(sites: np.ndarray, nt_values: np.ndarray, out_prefix: str):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.scatter(sites, nt_values, s=20)
    ax.set_xlabel("Site index")
    ax.set_ylabel(r"$\langle\tilde{n}_i\rangle$")
    ax.set_title("Dipole density profile")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.ticklabel_format(useOffset=False)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_dipole_density_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_density_from_array(sites: np.ndarray, n_values: np.ndarray, out_prefix: str, subtract: float | None = None):
    if subtract is None:
        y = n_values
        ylabel = "<n_i>"
    else:
        y = n_values - subtract
        ylabel = f"<n_i> - {subtract}"

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.scatter(sites, y, s=20)
    ax.set_xlabel("Site index")
    ax.set_ylabel(ylabel)
    ax.set_title("On-site density")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.ticklabel_format(useOffset=False)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_density.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dipole_density_profile(psi: MPS, out_prefix: str, num_sites: int | None = None):
    L = int(psi.L)
    is_infinite = _is_infinite_mps(psi)
    if num_sites is None:
        num_sites = (10 * L) if is_infinite else (L - 1)

    if is_infinite:
        nt_values = []
        for i in range(num_sites):
            j = i % L
            _log_site_progress("dipole_density_profile", 0, i, j, num_sites)
            nt_values.append(dipole_density(psi, j))
        nt_values = np.array(nt_values)
    else:
        num_sites = min(num_sites, L - 1)
        nt_values = []
        for i in range(num_sites):
            _log_site_progress("dipole_density_profile", 0, i, i, num_sites)
            nt_values.append(dipole_density(psi, i))
        nt_values = np.array(nt_values)

    sites = np.arange(len(nt_values))
    _plot_dipole_density_from_array(sites, nt_values, out_prefix)
    return nt_values


def plot_density(psi: MPS, out_prefix: str, num_sites: int | None = None, subtract: float | None = None):
    L = int(psi.L)
    is_infinite = _is_infinite_mps(psi)
    if num_sites is None:
        num_sites = (10 * L) if is_infinite else L

    if is_infinite:
        n_values = []
        for i in range(num_sites):
            _log_site_progress("density_profile", 0, i, i % L, num_sites)
            n_values.append(psi.expectation_value_term([("N", i % L)]))
        n_values = np.array(n_values)
    else:
        num_sites = min(num_sites, L)
        n_values = []
        for i in range(num_sites):
            _log_site_progress("density_profile", 0, i, i, num_sites)
            n_values.append(psi.expectation_value_term([("N", i)]))
        n_values = np.array(n_values)

    sites = np.arange(num_sites)
    _plot_density_from_array(sites, n_values, out_prefix, subtract=subtract)
    return n_values


def save_two_column_csv(
    x: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    base_tag: str,
    correlator_suffix: str,
    x_label: str = "r",
    y_label: str = "value",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{base_tag}_{correlator_suffix}.csv"
    path = out_dir / fname
    data = np.column_stack([x, y])
    header = f"{x_label},{y_label}"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def load_two_column_csv(path: Path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1]


def process_file(path: Path, csv_dir: Path, img_dir: Path):
    tag = path.stem
    print(f"[FILE] {tag}", flush=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    img_prefix_str = str(img_dir / tag)

    density_profile_csv = csv_dir / f"{tag}_density_profile.csv"
    dipole_density_profile_csv = csv_dir / f"{tag}_dipole_density_profile.csv"
    current_csv = csv_dir / f"{tag}_current_correlator.csv"
    dipole_csv = csv_dir / f"{tag}_dipole_correlator.csv"
    density_corr_csv = csv_dir / f"{tag}_density_correlator.csv"
    single_part_csv = csv_dir / f"{tag}_single_particle_correlator.csv"
    density_density_csv = csv_dir / f"{tag}_density_density_correlator.csv"

    need_psi = not (
        density_profile_csv.exists()
        and dipole_density_profile_csv.exists()
        and current_csv.exists()
        and dipole_csv.exists()
        and density_corr_csv.exists()
        and single_part_csv.exists()
        and density_density_csv.exists()
    )

    psi = None
    i_start = I0  # for finite MPS this is overridden below
    if need_psi:
        print("  Loading MPS from disk...", flush=True)
        with open(path, "rb") as f:
            psi = pickle.load(f)
        # Newer TeNPy added unit_cell_width to MPS; patch old pickled objects so
        # that expectation_value_term works on them without AttributeError.
        if not hasattr(psi, 'unit_cell_width'):
            psi.unit_cell_width = psi.L
        geom = "infinite" if _is_infinite_mps(psi) else "finite"
        print(f"  Geometry detected from MPS: {geom}", flush=True)
        if not _is_infinite_mps(psi):
            i_start = BOUNDARY_AVOID
            print(f"  Finite MPS: correlators will start at site {i_start} "
                  f"and end {BOUNDARY_AVOID} sites before the boundary.", flush=True)
    else:
        print("  All correlator CSVs already present. Skipping recomputation.", flush=True)

    # density profile
    if density_profile_csv.exists():
        sites, n_values = load_two_column_csv(density_profile_csv)
        _plot_density_from_array(sites, n_values, img_prefix_str)
    else:
        n_values = plot_density(psi, out_prefix=img_prefix_str)
        sites = np.arange(len(n_values))
        save_two_column_csv(sites, n_values, csv_dir, tag, "density_profile", "site", "n")

    # dipole density profile
    if dipole_density_profile_csv.exists():
        sites_nt, nt_values = load_two_column_csv(dipole_density_profile_csv)
        _plot_dipole_density_from_array(sites_nt, nt_values, img_prefix_str)
    else:
        nt_values = plot_dipole_density_profile(psi, out_prefix=img_prefix_str)
        sites_nt = np.arange(len(nt_values))
        save_two_column_csv(sites_nt, nt_values, csv_dir, tag, "dipole_density_profile", "site", "n_tilde")

    # current correlator
    if current_csv.exists():
        x_j, C_j = load_two_column_csv(current_csv)
    else:
        x_j, C_j = current_correlator(psi, i_start, r_max=_r_max_for("current", psi, i_start))
        save_two_column_csv(x_j, C_j, csv_dir, tag, "current_correlator", "r", "C_j")
    plot_curve(x_j, C_j, f"<j_d(i) j_d(j)> (i={i_start})", "<j_d j_d>", f"{img_prefix_str}_current")
    plot_fft(fft_corr(C_j), "j_d j_d correlator", f"{img_prefix_str}_current")

    # dipole correlator
    if dipole_csv.exists():
        x_d, C_d = load_two_column_csv(dipole_csv)
    else:
        x_d, C_d = dipole_correlator(psi, i_start, r_max=_r_max_for("dipole", psi, i_start))
        save_two_column_csv(x_d, C_d, csv_dir, tag, "dipole_correlator", "r", "C_d")
    plot_curve(x_d, C_d, f"<d^dag_i d_j> (i={i_start})", "<d^dag_i d_j>", f"{img_prefix_str}_dipole")
    plot_fft(fft_corr(C_d), "d^dag d correlator", f"{img_prefix_str}_dipole")

    # density correlator
    if density_corr_csv.exists():
        x_n, C_n = load_two_column_csv(density_corr_csv)
    else:
        x_n, C_n = density_correlator(psi, i_start, r_max=_r_max_for("density", psi, i_start))
        save_two_column_csv(x_n, C_n, csv_dir, tag, "density_correlator", "r", "C_n")
    plot_curve(x_n, C_n, f"<n_tilde_i n_tilde_j>_c (i={i_start})", "<n_tilde_i n_tilde_j>_c", f"{img_prefix_str}_density")
    plot_fft(fft_corr(C_n), "n_tilde n_tilde correlator", f"{img_prefix_str}_density")

    # single-particle correlator
    if single_part_csv.exists():
        x_s, C_s = load_two_column_csv(single_part_csv)
    else:
        x_s, C_s = single_particle_correlator(psi, i_start, r_max=_r_max_for("single_particle", psi, i_start))
        save_two_column_csv(x_s, C_s, csv_dir, tag, "single_particle_correlator", "r", "C_sp")
    plot_curve(x_s, C_s, f"<b^dag_i b_j> (i={i_start})", "<b^dag_i b_j>", f"{img_prefix_str}_single_particle")
    plot_fft(fft_corr(C_s), "b^dag b correlator", f"{img_prefix_str}_single_particle")

    # density-density correlator
    if density_density_csv.exists():
        x_nn, C_nn = load_two_column_csv(density_density_csv)
    else:
        x_nn, C_nn = density_density_correlator(psi, i_start, r_max=_r_max_for("density_density", psi, i_start))
        save_two_column_csv(x_nn, C_nn, csv_dir, tag, "density_density_correlator", "r", "C_nn")
    plot_curve(x_nn, C_nn, f"<n_i n_j> (i={i_start})", "<n_i n_j>", f"{img_prefix_str}_density_density")
    plot_fft(fft_corr(C_nn), "n n correlator", f"{img_prefix_str}_density_density")


def _dirs_for_mps_file(mps_path: Path) -> tuple[Path, Path]:
    """
    If MPS is in */mps_files/*.mps, use sibling folders:
      */correlators_csv and */correlators_plots
    Otherwise use the MPS parent folder.
    """
    parent = mps_path.parent
    dataset_root = parent.parent if parent.name == "mps_files" else parent
    return dataset_root / "correlators_csv", dataset_root / "correlators_plots"


def main():
    root = Path(SCAN_ROOT)
    if not root.is_dir():
        raise NotADirectoryError(f"SCAN_ROOT not found or not a directory: {SCAN_ROOT}")

    files = sorted(p for p in root.rglob("*.mps") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .mps files found under {root.resolve()}")

    print(f"Found {len(files)} .mps file(s) under {root.resolve()}.")

    processed = 0
    failed = 0
    for p in files:
        try:
            csv_dir, img_dir = _dirs_for_mps_file(p)
            csv_dir.mkdir(parents=True, exist_ok=True)
            img_dir.mkdir(parents=True, exist_ok=True)

            print(f"-> {p}", flush=True)
            process_file(p, csv_dir, img_dir)
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] Skipping {p.name} due to error: {exc}", flush=True)

    print("Done.")
    print(f"Processed files: {processed}")
    print(f"Failed files:    {failed}")


if __name__ == "__main__":
    main()
