# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.algorithms.mps_common import SubspaceExpansion
from tenpy.networks.mps import MPS

import model_extended

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
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

# ---------------------------------------------------------------------
# DEFAULTS  (DMRG settings only — physics points come from the input file)
# ---------------------------------------------------------------------
# These are aligned with run_simulations.py so each sector gets the same
# robust treatment that GS-only runs do (chi-ramp + gentle mixer that
# decays + is disabled after sweep 30).
DEFAULT_MPS_DIR             = "mps_dipole_gap"
DEFAULT_DIPOLE_GAP_DEF      = "simple"

DEFAULT_MAX_SIMULATION_HOURS = 6.0

DEFAULT_MAX_E_ERR           = 1e-7
DEFAULT_MAX_S_ERR           = 1e-6
DEFAULT_MAX_SWEEPS          = 500
DEFAULT_MIN_SWEEPS          = 20
DEFAULT_N_SWEEPS_CHECK      = 10

DEFAULT_USE_MIXER           = True
DEFAULT_MIXER_AMP           = 1e-5
DEFAULT_MIXER_DECAY         = 2.0
DEFAULT_MIXER_DISABLE_AFTER = 30

DEFAULT_USE_CHI_RAMP        = True
DEFAULT_CHI_RAMP_SCHEDULE   = [
    (0,  0.25),
    (10, 0.50),
    (20, 0.75),
    (30, 1.00),
]

DEFAULT_SVD_MIN             = 1e-8
DEFAULT_INPUT_FILE          = "gaps_input.txt"

# ---------------------------------------------------------------------
# SMALL UTILS
# ---------------------------------------------------------------------
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
        if c.lower() == "parity":
            return "parity"
        return c
    return str(conserve)


def _tag(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _model_tag(alpha: int, r: int) -> str:
    return f"alpha{alpha}_r{r}"


def _get_model_entry(model_key: str) -> Dict[str, Any]:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as e:
        raise KeyError(
            f"Unknown model key '{model_key}'. "
            f"Valid options are: {', '.join(MODEL_REGISTRY.keys())}"
        ) from e


# ---------------------------------------------------------------------
# CHI-RAMP + CONVERGENCE HELPERS  (ported from run_simulations.py)
# ---------------------------------------------------------------------
def _bounded_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(value))))


def _build_chi_list(chi_max: int, schedule, use_ramp: bool):
    if not use_ramp:
        return None
    chi_list: Dict[int, int] = {}
    for sweep, fraction in schedule:
        if sweep < 0:
            raise ValueError("chi-ramp sweeps must be non-negative.")
        if fraction <= 0:
            raise ValueError("chi-ramp fractions must be positive.")
        chi_list[int(sweep)] = _bounded_int(float(chi_max) * float(fraction), 1, int(chi_max))
    return chi_list or None


def _extract_n_sweeps(info) -> int:
    sweep_stats = info.get("sweep_statistics", {}) if isinstance(info, dict) else {}
    E_history = sweep_stats.get("E", [])
    return len(E_history)


def _extract_converged_chi(info, psi) -> int:
    sweep_stats = info.get("sweep_statistics", {}) if isinstance(info, dict) else {}
    max_chi_history = sweep_stats.get("max_chi", [])
    if max_chi_history:
        return int(max_chi_history[-1])
    try:
        return int(max(psi.chi))
    except Exception:
        return -1


def _convergence_status(info, max_sweeps: int) -> str:
    """
    'converged'          – DMRG met the energy/entropy criteria
    'max_sweeps_reached' – ran out of sweeps without converging
    'timed_out'          – hit max_hours mid-run
    """
    shelved = bool(info.get("shelve", False)) if isinstance(info, dict) else False
    if not shelved:
        return "converged"
    if _extract_n_sweeps(info) >= max_sweeps:
        return "max_sweeps_reached"
    return "timed_out"


# ---------------------------------------------------------------------
# OCCUPATION / SECTOR CONSTRUCTION
# ---------------------------------------------------------------------
def build_balanced_occupations(L: int, N_tot: int, n_max: int) -> list[int]:
    if N_tot < 0:
        raise ValueError("Total particle number must be non-negative.")
    if n_max * L < N_tot:
        raise ValueError(f"n_max * L = {n_max * L} < N_tot = {N_tot}.")

    q, r = divmod(N_tot, L)
    if q > n_max or (r > 0 and q + 1 > n_max):
        raise ValueError(
            f"Incompatible N_tot={N_tot}, L={L}, n_max={n_max} "
            f"(would require occupancies {q} or {q + 1} > n_max)."
        )

    occ = [q] * L
    if r > 0:
        for k in range(r):
            idx = (k * L) // r
            occ[idx] += 1
    return occ


def _add_one_leftmost(base_occ: list[int], n_max: int) -> list[int]:
    for i, n in enumerate(base_occ):
        if n + 1 <= n_max:
            occ = base_occ.copy()
            occ[i] += 1
            return occ
    raise ValueError(f"Cannot add particle without violating n_max={n_max}: {base_occ}")


def _remove_one_leftmost(base_occ: list[int]) -> list[int]:
    for i, n in enumerate(base_occ):
        if n > 0:
            occ = base_occ.copy()
            occ[i] -= 1
            return occ
    raise ValueError(f"Cannot remove particle from empty configuration: {base_occ}")


def _shift_one_right_leftmost(base_occ: list[int], n_max: int) -> list[int]:
    L = len(base_occ)
    for i in range(L - 1):
        if base_occ[i] <= 0:
            continue
        if base_occ[i + 1] + 1 <= n_max:
            occ = base_occ.copy()
            occ[i] -= 1
            occ[i + 1] += 1
            return occ
    raise ValueError(
        f"Cannot construct P+1 without violating n_max={n_max}: {base_occ}"
    )


def _shift_one_left_leftmost(base_occ: list[int], n_max: int) -> list[int]:
    L = len(base_occ)
    for i in range(L - 1):
        if base_occ[i + 1] <= 0:
            continue
        if base_occ[i] + 1 <= n_max:
            occ = base_occ.copy()
            occ[i + 1] -= 1
            occ[i] += 1
            return occ
    raise ValueError(
        f"Cannot construct P-1 without violating n_max={n_max}: {base_occ}"
    )


def build_sector_occupations(
    sector: str,
    L: int,
    N_base: int,
    n_max: int,
) -> Tuple[list[int], int]:
    sector_u = sector.upper()
    base_occ = build_balanced_occupations(L, N_base, n_max)

    if sector_u == "GS":
        return base_occ, N_base
    if sector_u == "N+1":
        return _add_one_leftmost(base_occ, n_max), N_base + 1
    if sector_u == "N-1":
        if N_base <= 0:
            raise ValueError("Cannot build N-1 sector if N_base <= 0.")
        return _remove_one_leftmost(base_occ), N_base - 1
    if sector_u == "P+1":
        return _shift_one_right_leftmost(base_occ, n_max), N_base
    if sector_u == "P-1":
        return _shift_one_left_leftmost(base_occ, n_max), N_base

    raise ValueError(f"Unknown sector '{sector}'. Expected GS, N+1, N-1, P+1, P-1.")


# ---------------------------------------------------------------------
# DIPOLE MOMENT + DIAGNOSTICS
# ---------------------------------------------------------------------
def get_dipole_moment_in_cell(psi, L: int):
    densities = psi.expectation_value("N")
    positions = np.arange(L)
    if psi.finite:
        return int(round(np.sum(densities * positions))), densities
    return int(round(np.sum(densities * positions))) % L, densities


def print_state_info(psi, model, energy=None, label="State"):
    L = model.lat.N_sites
    charge = psi.get_total_charge(only_physical_legs=True) if psi.finite else psi.get_total_charge()
    dipole_val, dens_profile = get_dipole_moment_in_cell(psi, L)

    print(f"\n--- {label} ---")
    print(f"Charge Vector: {charge}")
    print(f"Dipole (unit cell): {dipole_val}")
    if energy is not None:
        print(f"Energy        : {energy:.12f}")
        if psi.finite:
            print(f"Energy / site : {energy / L:.12f}")
    avg_filling = float(np.mean(dens_profile))
    print("density profile:")
    print(dens_profile)
    print(f"N tot: {int(round(sum(dens_profile)))} | avg filling: {avg_filling:.4f}")


# ---------------------------------------------------------------------
# MODEL + INITIAL MPS
# ---------------------------------------------------------------------
def sector_mps_path(
    base_dir: Path,
    *,
    U: float,
    t: float,
    mu: float,
    L: int,
    n_max: int,
    N_tot: int,
    conserve: str | None,
    chi_max: int,
    model_key: str,
    alpha: int,
    r: int,
    sector: str,
) -> Path:
    entry = _get_model_entry(model_key)
    geom_tag = entry["geom_tag"]
    tag = _model_tag(alpha, r)

    fname = (
        f"{geom_tag}_{tag}_U{U:g}_mu{_tag(float(mu))}_t{_tag(float(t))}"
        f"_cons{_conserve_tag(conserve)}"
        f"_chi{chi_max}_L{L}_N{N_tot}_nmax{n_max}_sector{sector.upper()}.mps"
    )
    return base_dir / fname


def build_model_and_initial_mps(
    *,
    occupations: list[int],
    U: float,
    t: float,
    mu: float,
    L: int,
    n_max: int,
    N_tot: int,
    conserve: str | None,
    model_key: str,
    alpha: int,
    r: int,
):
    entry = _get_model_entry(model_key)
    ModelClass = entry["class"]
    bc_mps = entry["bc_mps"]

    model_params = dict(
        t=float(t),
        U=float(U),
        mu=float(mu),
        L=int(L),
        n_max=int(n_max),
        filling=float(N_tot) / float(L),
        conserve=conserve,
        bc_MPS=bc_mps,
        alpha=int(alpha),
        r=int(r),
    )

    model = ModelClass(model_params)
    product_state = list(map(int, occupations))

    if bc_mps == "infinite":
        psi0 = MPS.from_product_state(
            model.lat.mps_sites(),
            product_state,
            bc="infinite",
            unit_cell_width=L,
        )
    else:
        psi0 = MPS.from_product_state(
            model.lat.mps_sites(),
            product_state,
            bc="finite",
        )

    return model, psi0


# ---------------------------------------------------------------------
# DMRG EXECUTION + CACHING
# ---------------------------------------------------------------------
def _dmrg_params(
    *,
    bc_mps: str,
    chi_max: int,
    svd_min: float,
    use_mixer: bool,
    mixer_amplitude: float,
    mixer_decay: float,
    mixer_disable_after: int,
    max_E_err: float,
    max_S_err: float,
    max_sweeps: int,
    min_sweeps: int,
    N_sweeps_check: int,
    max_hours: float,
    use_chi_ramp: bool,
    chi_ramp_schedule,
) -> Dict[str, Any]:
    chi_list = _build_chi_list(chi_max, chi_ramp_schedule, use_chi_ramp)

    # The ramp's last entry sets the earliest sweep at which we run at full chi;
    # convergence is checked only after we've actually reached full chi.
    effective_min_sweeps = int(min_sweeps)
    if chi_list:
        effective_min_sweeps = max(effective_min_sweeps, max(chi_list.keys()))

    # Same convention as run_simulations.py: infinite checks every
    # N_sweeps_check sweeps; finite checks after every sweep.
    n_sweeps_check_eff = int(N_sweeps_check) if bc_mps == "infinite" else 1

    params: Dict[str, Any] = {
        "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
        "max_E_err": float(max_E_err),
        "max_S_err": float(max_S_err),
        "max_sweeps": int(max_sweeps),
        "min_sweeps": int(effective_min_sweeps),
        "max_hours": float(max_hours),
        "N_sweeps_check": int(n_sweeps_check_eff),
        "verbose": 1,
    }

    if chi_list:
        params["chi_list"] = chi_list

    if use_mixer:
        params["mixer"] = SubspaceExpansion
        params["mixer_params"] = {
            "amplitude": float(mixer_amplitude),
            "decay": float(mixer_decay),
            "disable_after": int(mixer_disable_after),
        }
    else:
        params["mixer"] = False

    return params


def get_or_compute_sector(
    *,
    sector: str,
    U: float,
    t: float,
    mu: float,
    L: int,
    n_max: int,
    N_base: int,
    conserve: str | None,
    chi_max: int,
    svd_min: float,
    use_mixer: bool,
    mixer_amplitude: float,
    mixer_decay: float,
    mixer_disable_after: int,
    max_E_err: float,
    max_S_err: float,
    max_sweeps: int,
    min_sweeps: int,
    N_sweeps_check: int,
    max_hours: float,
    use_chi_ramp: bool,
    chi_ramp_schedule,
    model_key: str,
    alpha: int,
    r: int,
    mps_base_dir: Path,
    print_info: bool = False,
) -> Dict[str, Any]:
    """
    Returns a dict:
      { "E": float, "psi": ..., "mps_path": Path,
        "convergence": "converged" | "max_sweeps_reached" | "timed_out" | "loaded",
        "n_sweeps": int, "converged_chi_max": int }
    """
    occ, N_tot = build_sector_occupations(sector, L, N_base, n_max)
    mps_path = sector_mps_path(
        mps_base_dir,
        U=U, t=t, mu=mu, L=L, n_max=n_max, N_tot=N_tot,
        conserve=conserve, chi_max=chi_max,
        model_key=model_key, alpha=alpha, r=r, sector=sector,
    )

    if mps_path.exists():
        logger.info(f"[{sector}] Loading existing MPS file: {mps_path}")
        with open(mps_path, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "E" in payload and "psi" in payload:
            E = float(payload["E"])
            psi = payload["psi"]
            convergence = payload.get("convergence", "loaded")
            n_sweeps = int(payload.get("n_sweeps", -1))
            converged_chi_max = int(payload.get("converged_chi_max", -1))
        else:
            psi = payload
            model, _ = build_model_and_initial_mps(
                occupations=occ, U=U, t=t, mu=mu, L=L, n_max=n_max,
                N_tot=N_tot, conserve=conserve,
                model_key=model_key, alpha=alpha, r=r,
            )
            E = float(model.H_MPO.expectation_value(psi))
            convergence = "loaded"
            n_sweeps = -1
            try:
                converged_chi_max = int(max(psi.chi))
            except Exception:
                converged_chi_max = -1
            with open(mps_path, "wb") as g:
                pickle.dump({
                    "E": E, "psi": psi,
                    "convergence": convergence,
                    "n_sweeps": n_sweeps,
                    "converged_chi_max": converged_chi_max,
                }, g)

        if print_info:
            model, _ = build_model_and_initial_mps(
                occupations=occ, U=U, t=t, mu=mu, L=L, n_max=n_max,
                N_tot=N_tot, conserve=conserve,
                model_key=model_key, alpha=alpha, r=r,
            )
            print_state_info(psi, model, energy=E, label=f"{sector.upper()} loaded")

        return {
            "E": E, "psi": psi, "mps_path": mps_path,
            "convergence": convergence, "n_sweeps": n_sweeps,
            "converged_chi_max": converged_chi_max,
        }

    logger.info(f"[{sector}] Running new DMRG for {mps_path}")
    model, psi0 = build_model_and_initial_mps(
        occupations=occ, U=U, t=t, mu=mu, L=L, n_max=n_max,
        N_tot=N_tot, conserve=conserve,
        model_key=model_key, alpha=alpha, r=r,
    )

    bc_mps = _get_model_entry(model_key)["bc_mps"]
    params = _dmrg_params(
        bc_mps=bc_mps,
        chi_max=chi_max, svd_min=svd_min,
        use_mixer=use_mixer,
        mixer_amplitude=mixer_amplitude,
        mixer_decay=mixer_decay,
        mixer_disable_after=mixer_disable_after,
        max_E_err=max_E_err, max_S_err=max_S_err,
        max_sweeps=max_sweeps, min_sweeps=min_sweeps,
        N_sweeps_check=N_sweeps_check,
        max_hours=max_hours,
        use_chi_ramp=use_chi_ramp,
        chi_ramp_schedule=chi_ramp_schedule,
    )

    # Use dmrg.run for both finite and infinite — same path as
    # run_simulations.py.  dmrg.run picks the right engine and respects
    # chi_list / mixer / max_hours uniformly.
    info = dmrg.run(psi0, model, params)
    E = float(info["E"])
    psi = psi0
    convergence = _convergence_status(info, max_sweeps)
    n_sweeps = _extract_n_sweeps(info)
    converged_chi_max = _extract_converged_chi(info, psi)

    payload = {
        "E": E,
        "psi": psi,
        "convergence": convergence,
        "n_sweeps": n_sweeps,
        "converged_chi_max": converged_chi_max,
        "meta": {
            "sector": sector.upper(),
            "U": float(U), "t": float(t), "mu": float(mu),
            "L": int(L), "n_max": int(n_max), "N_tot": int(N_tot),
            "conserve": conserve, "chi_max": int(chi_max),
            "svd_min": float(svd_min),
            "model_key": model_key, "alpha": int(alpha), "r": int(r),
        },
    }
    with open(mps_path, "wb") as f:
        pickle.dump(payload, f)

    if print_info:
        print_state_info(psi, model, energy=E, label=f"{sector.upper()} final")

    return {
        "E": E, "psi": psi, "mps_path": mps_path,
        "convergence": convergence, "n_sweeps": n_sweeps,
        "converged_chi_max": converged_chi_max,
    }


# ---------------------------------------------------------------------
# INPUT-FILE PARSER
# ---------------------------------------------------------------------
def _parse_gaps_input_file(input_file: str) -> list[Dict[str, Any]]:
    """
    One point per line (comma-separated):

      geometry, L, n_max, N_IN_UNIT_CELL, U, mu, Conserve,
      t, chi_max, svd_min, alpha, r

    Lines starting with '#' and empty lines are ignored.
    A header line whose first token is "geometry" or "model" is skipped.
    """
    rows: list[Dict[str, Any]] = []
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Gaps input file not found: '{input_file}'")

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.lstrip("﻿").strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 12:
                raise ValueError(
                    f"Invalid format at {input_file}:{line_no}. "
                    f"Expected 12 comma-separated values, got {len(parts)}.\n"
                    f"Format: geometry,L,n_max,N_IN_UNIT_CELL,U,mu,Conserve,"
                    f"t,chi_max,svd_min,alpha,r"
                )

            if parts[0].lower() in ("geometry", "model"):
                continue

            model_key = parts[0].lower()
            if model_key not in MODEL_REGISTRY:
                raise ValueError(
                    f"Invalid geometry '{parts[0]}' at {input_file}:{line_no}. "
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


# ---------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------
def run_gaps_from_input_file(
    *,
    input_file: str,
    results_path: Path,
    mps_dir: str,
    use_mixer: bool,
    mixer_amplitude: float,
    mixer_decay: float,
    mixer_disable_after: int,
    max_E_err: float,
    max_S_err: float,
    max_sweeps: int,
    min_sweeps: int,
    N_sweeps_check: int,
    max_hours: float,
    use_chi_ramp: bool,
    chi_ramp_schedule,
    dipole_gap_def: str,
) -> None:
    dipole_gap_def = dipole_gap_def.lower().strip()
    if dipole_gap_def not in ("paper", "simple", "symmetric"):
        raise ValueError("--dipole_gap_def must be one of: 'paper', 'simple', 'symmetric'.")

    rows = _parse_gaps_input_file(input_file)
    if not rows:
        print(f"[WARN] No data rows in '{input_file}'.")
        return

    mps_base_dir = Path(mps_dir)
    mps_base_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_key", "alpha", "r", "L", "n_max", "N_in_unit_cell",
        "U", "mu", "t", "conserve", "chi_max", "svd_min", "dipole_gap_def",
        "E_GS", "E_Nplus1", "E_Nminus1", "E_Pplus1", "E_Pminus1",
        "mu_d_plus", "mu_d_minus", "Delta_c", "Delta_d",
        # convergence diagnostics per sector
        "conv_GS", "conv_Nplus1", "conv_Nminus1", "conv_Pplus1", "conv_Pminus1",
        "nsweeps_GS", "nsweeps_Nplus1", "nsweeps_Nminus1",
        "nsweeps_Pplus1", "nsweeps_Pminus1",
        "chi_GS", "chi_Nplus1", "chi_Nminus1", "chi_Pplus1", "chi_Pminus1",
        "all_converged",
    ]

    # "w" = overwrite each run so the CSV always contains exactly this run
    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            model_key = row["model"]
            L         = row["L"]
            n_max     = row["n_max"]
            N_uc      = row["N_in_unit_cell"]
            U         = float(row["U"])
            mu_val    = float(row["mu"])
            conserve  = row["conserve"]
            t_val     = float(row["t"])
            chi_max   = int(row["chi_max"])
            row_svd   = float(row["svd_min"])
            alpha     = int(row["alpha"])
            r         = int(row["r"])

            if n_max * L < N_uc:
                raise ValueError(
                    f"Row {idx}: n_max*L = {n_max*L} must be >= N_in_unit_cell = {N_uc}."
                )

            sep = "=" * 70
            print(f"\n{sep}")
            print(
                f"  [{idx}/{len(rows)}]  model={model_key}  alpha={alpha}  r={r}"
                f"  U={U:g}  mu={mu_val:g}  t={t_val:g}"
                f"  L={L}  N_uc={N_uc}  n_max={n_max}"
                f"  chi={chi_max}  cons={_conserve_tag(conserve)}"
            )
            print(sep)

            common = dict(
                U=U, t=t_val, mu=mu_val, L=L, n_max=n_max,
                N_base=N_uc, conserve=conserve,
                chi_max=chi_max, svd_min=row_svd,
                use_mixer=use_mixer,
                mixer_amplitude=mixer_amplitude,
                mixer_decay=mixer_decay,
                mixer_disable_after=mixer_disable_after,
                max_E_err=max_E_err, max_S_err=max_S_err,
                max_sweeps=max_sweeps, min_sweeps=min_sweeps,
                N_sweeps_check=N_sweeps_check,
                max_hours=max_hours,
                use_chi_ramp=use_chi_ramp,
                chi_ramp_schedule=chi_ramp_schedule,
                model_key=model_key, alpha=alpha, r=r,
                mps_base_dir=mps_base_dir, print_info=False,
            )

            res_GS = get_or_compute_sector(sector="GS",  **common)
            res_Np = get_or_compute_sector(sector="N+1", **common)
            res_Nm = get_or_compute_sector(sector="N-1", **common)
            res_Pp = get_or_compute_sector(sector="P+1", **common)

            if dipole_gap_def == "paper":
                res_Pm = get_or_compute_sector(sector="P-1", **common)
            else:
                res_Pm = None

            E_GS = res_GS["E"]
            E_Np = res_Np["E"]
            E_Nm = res_Nm["E"]
            E_Pp = res_Pp["E"]
            E_Pm = res_Pm["E"] if res_Pm is not None else float("nan")

            Delta_c    = float(E_Np + E_Nm - 2.0 * E_GS)
            mu_d_plus  = float(E_Pp - E_GS)
            mu_d_minus = float(E_GS - E_Pm) if dipole_gap_def == "paper" else float("nan")

            if dipole_gap_def == "simple":
                Delta_d = mu_d_plus
            elif dipole_gap_def == "symmetric":
                Delta_d = 2.0 * mu_d_plus
            else:
                Delta_d = float(mu_d_plus - mu_d_minus)

            # convergence summary
            sector_results = {
                "GS":      res_GS,
                "N+1":     res_Np,
                "N-1":     res_Nm,
                "P+1":     res_Pp,
            }
            if res_Pm is not None:
                sector_results["P-1"] = res_Pm

            all_converged = all(
                s["convergence"] in ("converged", "loaded")
                for s in sector_results.values()
            )

            print(f"\n  E(GS)   = {E_GS:.12e}   [{res_GS['convergence']}, "
                  f"n_sweeps={res_GS['n_sweeps']}, chi={res_GS['converged_chi_max']}]")
            print(f"  E(N+1)  = {E_Np:.12e}   [{res_Np['convergence']}, "
                  f"n_sweeps={res_Np['n_sweeps']}, chi={res_Np['converged_chi_max']}]")
            print(f"  E(N-1)  = {E_Nm:.12e}   [{res_Nm['convergence']}, "
                  f"n_sweeps={res_Nm['n_sweeps']}, chi={res_Nm['converged_chi_max']}]")
            print(f"  E(P+1)  = {E_Pp:.12e}   [{res_Pp['convergence']}, "
                  f"n_sweeps={res_Pp['n_sweeps']}, chi={res_Pp['converged_chi_max']}]")
            if res_Pm is not None:
                print(f"  E(P-1)  = {E_Pm:.12e}   [{res_Pm['convergence']}, "
                      f"n_sweeps={res_Pm['n_sweeps']}, chi={res_Pm['converged_chi_max']}]")
                print(f"  mu_d^+  = {mu_d_plus:.12e}")
                print(f"  mu_d^-  = {mu_d_minus:.12e}")
            print(f"\n  >>> Charge gap  Delta_c = {Delta_c:.8e}")
            print(f"  >>> Dipole gap  Delta_d = {Delta_d:.8e}  (def={dipole_gap_def})")
            if not all_converged:
                bad = [k for k, s in sector_results.items()
                       if s["convergence"] not in ("converged", "loaded")]
                print(f"  ⚠ NON-CONVERGED sectors: {', '.join(bad)}")

            writer.writerow(dict(
                model_key=model_key, alpha=alpha, r=r,
                L=L, n_max=n_max, N_in_unit_cell=N_uc,
                U=U, mu=mu_val, t=t_val,
                conserve=_conserve_tag(conserve),
                chi_max=chi_max, svd_min=row_svd,
                dipole_gap_def=dipole_gap_def,
                E_GS=E_GS, E_Nplus1=E_Np, E_Nminus1=E_Nm,
                E_Pplus1=E_Pp, E_Pminus1=E_Pm,
                mu_d_plus=mu_d_plus, mu_d_minus=mu_d_minus,
                Delta_c=Delta_c, Delta_d=Delta_d,
                conv_GS=res_GS["convergence"],
                conv_Nplus1=res_Np["convergence"],
                conv_Nminus1=res_Nm["convergence"],
                conv_Pplus1=res_Pp["convergence"],
                conv_Pminus1=res_Pm["convergence"] if res_Pm is not None else "skipped",
                nsweeps_GS=res_GS["n_sweeps"],
                nsweeps_Nplus1=res_Np["n_sweeps"],
                nsweeps_Nminus1=res_Nm["n_sweeps"],
                nsweeps_Pplus1=res_Pp["n_sweeps"],
                nsweeps_Pminus1=res_Pm["n_sweeps"] if res_Pm is not None else -1,
                chi_GS=res_GS["converged_chi_max"],
                chi_Nplus1=res_Np["converged_chi_max"],
                chi_Nminus1=res_Nm["converged_chi_max"],
                chi_Pplus1=res_Pp["converged_chi_max"],
                chi_Pminus1=res_Pm["converged_chi_max"] if res_Pm is not None else -1,
                all_converged=all_converged,
            ))
            # flush after each row so partial results are readable mid-run
            csvfile.flush()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute charge and dipole gaps via DMRG.\n"
            "Physics points are read from an input text file "
            "(one comma-separated row per point).\n"
            "Format: geometry,L,n_max,N_IN_UNIT_CELL,U,mu,Conserve,"
            "t,chi_max,svd_min,alpha,r"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- required: input file ---
    parser.add_argument(
        "--input_file", type=str, default=DEFAULT_INPUT_FILE,
        help=(
            f"Text file with one point per line "
            f"(default: {DEFAULT_INPUT_FILE}).  "
            f"Format: geometry,L,n_max,N_IN_UNIT_CELL,U,mu,Conserve,"
            f"t,chi_max,svd_min,alpha,r"
        ),
    )

    # --- output ---
    parser.add_argument(
        "--mps_dir", type=str, default=DEFAULT_MPS_DIR,
        help="Directory where sector MPS+energy files are stored/cached.",
    )
    parser.add_argument(
        "--results", type=str, default="",
        help="CSV output path (default: auto-generated inside --mps_dir).",
    )

    # --- gap definition ---
    parser.add_argument(
        "--dipole_gap_def", type=str, default=DEFAULT_DIPOLE_GAP_DEF,
        choices=["paper", "simple", "symmetric"],
        help=(
            "Dipole gap definition. "
            "'paper' = [E(P+1)-E(GS)] - [E(GS)-E(P-1)]; "
            "'simple' = E(P+1)-E(GS); "
            "'symmetric' = 2*(E(P+1)-E(GS))."
        ),
    )

    # --- DMRG settings (svd_min is per-row from the input file) ---
    parser.add_argument("--use_mixer",           type=lambda s: s.lower() not in ("0", "false", "no"),
                        default=DEFAULT_USE_MIXER,
                        help="Use SubspaceExpansion mixer (default: True).")
    parser.add_argument("--mixer_amplitude",     type=float, default=DEFAULT_MIXER_AMP)
    parser.add_argument("--mixer_decay",         type=float, default=DEFAULT_MIXER_DECAY)
    parser.add_argument("--mixer_disable_after", type=int,   default=DEFAULT_MIXER_DISABLE_AFTER,
                        help="Sweep at which mixer noise is turned off (default: 30).")
    parser.add_argument("--max_E_err",           type=float, default=DEFAULT_MAX_E_ERR)
    parser.add_argument("--max_S_err",           type=float, default=DEFAULT_MAX_S_ERR)
    parser.add_argument("--max_sweeps",          type=int,   default=DEFAULT_MAX_SWEEPS)
    parser.add_argument("--min_sweeps",          type=int,   default=DEFAULT_MIN_SWEEPS)
    parser.add_argument("--N_sweeps_check",      type=int,   default=DEFAULT_N_SWEEPS_CHECK)
    parser.add_argument("--max_hours",           type=float, default=DEFAULT_MAX_SIMULATION_HOURS,
                        help="Wall-clock budget (hours) per DMRG sector run.")
    parser.add_argument("--use_chi_ramp",        type=lambda s: s.lower() not in ("0", "false", "no"),
                        default=DEFAULT_USE_CHI_RAMP,
                        help="Ramp chi up over the first sweeps (default: True).")

    args = parser.parse_args()

    if args.results:
        results_path = Path(args.results)
    else:
        stem = Path(args.input_file).stem
        results_path = Path(args.mps_dir) / f"gaps_{stem}_Dd{args.dipole_gap_def}.csv"

    print(f"Input file   : {args.input_file}")
    print(f"MPS directory: {args.mps_dir}")
    print(f"Results (CSV): {results_path}")
    print(f"Gap def      : {args.dipole_gap_def}")
    print(f"Chi ramp     : {args.use_chi_ramp}  schedule={DEFAULT_CHI_RAMP_SCHEDULE}")
    print(f"Mixer        : use={args.use_mixer}, amp={args.mixer_amplitude:g}, "
          f"decay={args.mixer_decay:g}, disable_after={args.mixer_disable_after}")
    print(f"Sweeps       : min={args.min_sweeps}, max={args.max_sweeps}, "
          f"check_every={args.N_sweeps_check} (infinite only)")
    print(f"Tolerances   : max_E_err={args.max_E_err:g}, max_S_err={args.max_S_err:g}")
    print(f"Time budget  : {args.max_hours:g} h per sector")

    run_gaps_from_input_file(
        input_file=args.input_file,
        results_path=results_path,
        mps_dir=args.mps_dir,
        use_mixer=args.use_mixer,
        mixer_amplitude=args.mixer_amplitude,
        mixer_decay=args.mixer_decay,
        mixer_disable_after=args.mixer_disable_after,
        max_E_err=args.max_E_err,
        max_S_err=args.max_S_err,
        max_sweeps=args.max_sweeps,
        min_sweeps=args.min_sweeps,
        N_sweeps_check=args.N_sweeps_check,
        max_hours=args.max_hours,
        use_chi_ramp=args.use_chi_ramp,
        chi_ramp_schedule=DEFAULT_CHI_RAMP_SCHEDULE,
        dipole_gap_def=args.dipole_gap_def,
    )

    print(f"\n{'=' * 70}")
    print("Finished.")
    print(f"Results (CSV): {results_path}")
    print(f"MPS directory: {args.mps_dir}")


if __name__ == "__main__":
    main()
