# run_simulations.py
from __future__ import annotations
from pathlib import Path
import argparse
import pickle
import csv

from tenpy.algorithms import dmrg
from tenpy.algorithms.mps_common import SubspaceExpansion
from tenpy.networks.mps import MPS
import model_extended

# ---------------------------------------------------------------------
# GLOBAL INPUT/OUTPUT
# ---------------------------------------------------------------------
OUTPUT_ROOT_DIR = "."
SIMULATIONS_INPUT_FILE = "simulations_input.txt"

# ---------------------------------------------------------------------
# GLOBAL DMRG SETTINGS
# ---------------------------------------------------------------------
MAX_SIMULATION_HOURS = 6

DMRG_MAX_SWEEPS = 500
DMRG_MIN_SWEEPS = 20
DMRG_MAX_E_ERR = 1e-7
DMRG_MAX_S_ERR = 1e-6
DMRG_N_SWEEPS_CHECK = 10

DMRG_USE_MIXER = True
DMRG_MIXER_AMPLITUDE = 1e-5
DMRG_MIXER_DECAY = 2.0
DMRG_MIXER_DISABLE_AFTER = 30

USE_CHI_MAX_INCREMENT = True
DMRG_CHI_RAMP_SCHEDULE = [
    (0, 0.25),
    (10, 0.50),
    (20, 0.75),
    (30, 1.00),
]

# ---------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------
MODEL_REGISTRY = {
    "infinite": {
        "class": model_extended.CorrelatedHoppingBoseModelInfinite,
        "geom_tag": "iMPS",
        "bc_mps": "infinite",
    },
    "finite": {
        "class": model_extended.CorrelatedHoppingBoseModelFinite,
        "geom_tag": "fMPS",
        "bc_mps": "finite",
    },
}

MODEL_OUTPUT_SUBDIR_CANDIDATES = {
    "infinite": ["mps_files_infinite"],
    "finite":   ["mps_files_finite"],
}


def _sanitize_conserve(c: str | None):
    if c is None:
        return None
    if isinstance(c, str) and c.strip().lower() in ("none", "null", "false", ""):
        return None
    return c


def _conserve_tag(conserve: str | None) -> str:
    if conserve is None:
        return "None"
    if isinstance(conserve, str):
        c = conserve.strip()
        if c.lower() in ("none", "null", "false", ""):
            return "None"
        if c.lower() == "dipole":
            return "dipole"
        if c.upper() == "N":
            return "N"
        return c
    return str(conserve)


def _tag(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _bounded_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(value))))


def _get_model_entry(model_key: str):
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError:
        raise KeyError(
            f"Unknown model key '{model_key}'. "
            f"Valid options are: {', '.join(MODEL_REGISTRY.keys())}"
        )


def _resolve_output_dir_for_model(model_key: str, out_root: str | Path) -> Path:
    out_root = Path(out_root)
    candidates = MODEL_OUTPUT_SUBDIR_CANDIDATES.get(model_key)
    if not candidates:
        raise KeyError(f"Missing output folder mapping for model '{model_key}'.")

    for subdir in candidates:
        candidate_path = out_root / subdir
        if candidate_path.exists() and candidate_path.is_dir():
            mps_subdir = candidate_path / "mps_files"
            mps_subdir.mkdir(parents=True, exist_ok=True)
            return mps_subdir

    preferred = out_root / candidates[0]
    mps_subdir = preferred / "mps_files"
    mps_subdir.mkdir(parents=True, exist_ok=True)
    return mps_subdir


def _build_chi_list(chi_max: int):
    if not USE_CHI_MAX_INCREMENT:
        return None
    chi_list = {}
    for sweep, fraction in DMRG_CHI_RAMP_SCHEDULE:
        if sweep < 0:
            raise ValueError("DMRG_CHI_RAMP_SCHEDULE sweeps must be non-negative.")
        if fraction <= 0:
            raise ValueError("DMRG_CHI_RAMP_SCHEDULE fractions must be positive.")
        chi_list[int(sweep)] = _bounded_int(float(chi_max) * float(fraction), 1, int(chi_max))
    return chi_list or None


def _build_dmrg_params(*, chi_max: int, svd_min: float, bc_mps: str):
    chi_list = _build_chi_list(chi_max)
    min_sweeps = int(DMRG_MIN_SWEEPS)
    if chi_list:
        min_sweeps = max(min_sweeps, max(chi_list.keys()))

    n_sweeps_check = int(DMRG_N_SWEEPS_CHECK) if bc_mps == "infinite" else 1

    params = {
        "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
        "max_E_err": float(DMRG_MAX_E_ERR),
        "max_S_err": float(DMRG_MAX_S_ERR),
        "max_sweeps": int(DMRG_MAX_SWEEPS),
        "min_sweeps": int(min_sweeps),
        "max_hours": float(MAX_SIMULATION_HOURS),
        "N_sweeps_check": int(n_sweeps_check),
        "verbose": 1,
    }

    if chi_list:
        params["chi_list"] = chi_list

    if DMRG_USE_MIXER:
        params["mixer"] = SubspaceExpansion
        params["mixer_params"] = {
            "amplitude": float(DMRG_MIXER_AMPLITUDE),
            "decay": float(DMRG_MIXER_DECAY),
            "disable_after": int(DMRG_MIXER_DISABLE_AFTER),
        }
    else:
        params["mixer"] = False

    return params


def _extract_converged_chi(info, psi) -> int:
    sweep_stats = info.get("sweep_statistics", {}) if isinstance(info, dict) else {}
    max_chi_history = sweep_stats.get("max_chi", [])
    if max_chi_history:
        return int(max_chi_history[-1])
    return int(max(psi.chi))


def _extract_n_sweeps(info) -> int:
    sweep_stats = info.get("sweep_statistics", {}) if isinstance(info, dict) else {}
    E_history = sweep_stats.get("E", [])
    return len(E_history)


def _convergence_status(info, psi) -> str:
    """
    Returns one of:
      "converged"          – DMRG met the energy/entropy criteria before max_sweeps
      "max_sweeps_reached" – loop exhausted max_sweeps without satisfying criteria
      "timed_out"          – wall-clock budget was hit mid-run
    """
    shelved = bool(info.get("shelve", False)) if isinstance(info, dict) else False
    if not shelved:
        return "converged"
    n_sweeps = _extract_n_sweeps(info)
    if n_sweeps >= DMRG_MAX_SWEEPS:
        return "max_sweeps_reached"
    return "timed_out"


def _model_tag(alpha: int, r: int) -> str:
    return f"alpha{alpha}_r{r}"


def _build_output_filename(
    *,
    geom_tag: str,
    alpha: int,
    r: int,
    U: float,
    mu: float,
    t: float,
    conserve: str | None,
    converged_chi_max: int,
    L: int,
    N_uc: int,
    n_max: int,
) -> str:
    cons_tag = _conserve_tag(conserve)
    tag = _model_tag(alpha, r)
    return (
        f"{geom_tag}_{tag}_U{U:g}_mu{_tag(mu)}_t{_tag(t)}_cons{cons_tag}"
        f"_chi{int(converged_chi_max)}_L{L}_Nuc{N_uc}_nmax{n_max}.mps"
    )


def _find_existing_output(
    *,
    sim_out_dir: Path,
    geom_tag: str,
    alpha: int,
    r: int,
    U: float,
    mu: float,
    t: float,
    conserve: str | None,
    L: int,
    N_uc: int,
    n_max: int,
) -> Path | None:
    cons_tag = _conserve_tag(conserve)
    tag = _model_tag(alpha, r)
    pattern = (
        f"{geom_tag}_{tag}_U{U:g}_mu{_tag(mu)}_t{_tag(t)}_cons{cons_tag}"
        f"_chi*_L{L}_Nuc{N_uc}_nmax{n_max}.mps"
    )
    matches = sorted(sim_out_dir.glob(pattern))
    return matches[0] if matches else None


def simulate_wavefunction(
    *,
    U: float,
    t: float,
    mu: float,
    L: int,
    N_uc: int,
    n_max: int,
    conserve: str | None,
    chi_max: int,
    svd_min: float = 1e-8,
    model_key: str,
    alpha: int,
    r: int,
):
    if N_uc < 0:
        raise ValueError("N_in_unit_cell must be non-negative.")
    if n_max * L < N_uc:
        raise ValueError(f"n_max*L = {n_max*L} must be >= N_in_unit_cell = {N_uc}.")

    q, rem = divmod(N_uc, L)
    if q > n_max or (rem > 0 and q + 1 > n_max):
        raise ValueError(
            f"Incompatible N_in_unit_cell={N_uc}, L={L}, n_max={n_max}."
        )

    occupations = [q] * L
    for k in range(rem):
        occupations[(k * L) // rem] += 1

    product_state = [str(n) for n in occupations]

    model_params = dict(
        t=t, U=U, mu=mu, n_max=n_max, conserve=conserve,
        L=L, filling=N_uc / L,
        alpha=alpha, r=r,
    )

    entry = _get_model_entry(model_key)
    ModelClass = entry["class"]
    bc_mps = entry["bc_mps"]

    model = ModelClass(model_params)
    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=bc_mps)
    dmrg_params = _build_dmrg_params(chi_max=chi_max, svd_min=svd_min, bc_mps=bc_mps)
    info = dmrg.run(psi, model, dmrg_params)
    return {
        "psi": psi,
        "info": info,
        "convergence": _convergence_status(info, psi),
        "converged_chi_max": _extract_converged_chi(info, psi),
        "n_sweeps": _extract_n_sweeps(info),
    }


def _parse_simulations_file(simulations_file: str):
    """
    Parse the simulations input file.  One simulation per line (comma-separated):

      geometry, L, n_max, N_IN_UNIT_CELL, U, mu_value, Conserve,
      t_value, chi_max, svd_min, alpha, r

    Lines starting with '#' and empty lines are ignored.
    A header line (first token == "geometry" or "model") is skipped.
    """
    rows = []
    path = Path(simulations_file)
    if not path.exists():
        raise FileNotFoundError(f"Simulations file not found: '{simulations_file}'")

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.lstrip("\ufeff").strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 12:
                raise ValueError(
                    f"Invalid format at {simulations_file}:{line_no}. "
                    f"Expected 12 comma-separated values, got {len(parts)}.\n"
                    f"Format: geometry,L,n_max,N_IN_UNIT_CELL,U,mu,Conserve,"
                    f"t,chi_max,svd_min,alpha,r"
                )

            if parts[0].lower() in ("geometry", "model"):
                continue

            model_key = parts[0].lower()
            if model_key not in MODEL_REGISTRY:
                raise ValueError(
                    f"Invalid geometry '{parts[0]}' at {simulations_file}:{line_no}. "
                    f"Valid values: {', '.join(MODEL_REGISTRY.keys())}"
                )

            rows.append(
                dict(
                    model=model_key,
                    L=int(parts[1]),
                    n_max=int(parts[2]),
                    N_in_unit_cell=int(parts[3]),
                    U=float(parts[4]),
                    mu=float(parts[5]),
                    conserve=_sanitize_conserve(parts[6]),
                    t=float(parts[7]),
                    chi_max=int(parts[8]),
                    svd_min=float(parts[9]),
                    alpha=int(parts[10]),
                    r=int(parts[11]),
                )
            )

    return rows


def run_from_simulations_file(*, simulations_file: str, out_root: str):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    sim_rows = _parse_simulations_file(simulations_file)
    manifest_rows = []
    counts = {"converged": 0, "max_sweeps_reached": 0, "timed_out": 0, "skipped": 0}

    for sim in sim_rows:
        model_key = sim["model"]
        L = sim["L"]
        n_max = sim["n_max"]
        N_uc = sim["N_in_unit_cell"]
        U = sim["U"]
        mu = sim["mu"]
        conserve = sim["conserve"]
        t = sim["t"]
        chi_max = sim["chi_max"]
        svd_min = sim["svd_min"]
        alpha = sim["alpha"]
        r = sim["r"]

        entry = _get_model_entry(model_key)
        geom_tag = entry["geom_tag"]
        sim_out_dir = _resolve_output_dir_for_model(model_key, out_root)

        existing_fpath = _find_existing_output(
            sim_out_dir=sim_out_dir,
            geom_tag=geom_tag,
            alpha=alpha,
            r=r,
            U=U,
            mu=mu,
            t=t,
            conserve=conserve,
            L=L,
            N_uc=N_uc,
            n_max=n_max,
        )

        if existing_fpath is not None:
            print(f"File already exists, skipping: '{existing_fpath}'")
            counts["skipped"] += 1
            continue

        result = simulate_wavefunction(
            U=U, t=t, mu=mu, L=L, N_uc=N_uc, n_max=n_max,
            conserve=conserve, chi_max=chi_max, svd_min=svd_min,
            model_key=model_key, alpha=alpha, r=r,
        )

        convergence = result["convergence"]
        converged_chi_max = result["converged_chi_max"]
        n_sweeps = result["n_sweeps"]
        counts[convergence] += 1

        # Save MPS only if DMRG actually converged.
        if convergence == "converged":
            fname = _build_output_filename(
                geom_tag=geom_tag, alpha=alpha, r=r,
                U=U, mu=mu, t=t, conserve=conserve,
                converged_chi_max=converged_chi_max,
                L=L, N_uc=N_uc, n_max=n_max,
            )
            fpath = sim_out_dir / fname
            with open(fpath, "wb") as f:
                pickle.dump(result["psi"], f)
            saved_file = str(fpath)
            print(f"[OK] Converged in {n_sweeps} sweeps (chi={converged_chi_max}): {fname}")
        else:
            saved_file = ""
            print(
                f"[WARN] Not saving ({convergence}) — "
                f"model='{model_key}', alpha={alpha}, r={r}, "
                f"U={U:g}, mu={mu:g}, t={t:g}, L={L}, "
                f"sweeps={n_sweeps}, chi={converged_chi_max}."
            )

        manifest_rows.append(
            dict(
                file=saved_file,
                convergence=convergence,
                n_sweeps=n_sweeps,
                model=model_key,
                alpha=alpha,
                r=r,
                L=L,
                N_in_unit_cell=N_uc,
                U=U,
                mu=mu,
                t=t,
                n_max=n_max,
                conserve=_conserve_tag(conserve),
                chi_max_requested=chi_max,
                chi_max_converged=converged_chi_max,
                svd_min=svd_min,
            )
        )

    mpath = out_root / "manifest.csv"
    with open(mpath, "w", newline="") as csvfile:
        fieldnames = [
            "convergence", "n_sweeps", "file",
            "model", "alpha", "r", "L", "N_in_unit_cell",
            "U", "mu", "t", "n_max", "conserve",
            "chi_max_requested", "chi_max_converged", "svd_min",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    return mpath, counts


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DMRG simulations listed in a text file.  One simulation per line.\n"
            "Format: geometry,L,n_max,N_IN_UNIT_CELL,U,mu,Conserve,t,chi_max,svd_min,alpha,r"
        )
    )
    parser.add_argument(
        "--simulations_file",
        type=str,
        default=SIMULATIONS_INPUT_FILE,
        help="Path to the simulations input file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=OUTPUT_ROOT_DIR,
        help=(
            "Root directory.  Each simulation is routed into "
            "mps_files_infinite/ or mps_files_finite/ automatically."
        ),
    )

    args = parser.parse_args()
    manifest_path, counts = run_from_simulations_file(
        simulations_file=args.simulations_file,
        out_root=args.out_dir,
    )
    total = sum(counts.values())
    print(
        f"\nFinished.  {total} simulations processed:\n"
        f"  converged          : {counts['converged']}\n"
        f"  max_sweeps_reached : {counts['max_sweeps_reached']}\n"
        f"  timed_out          : {counts['timed_out']}\n"
        f"  skipped (existing) : {counts['skipped']}\n"
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
