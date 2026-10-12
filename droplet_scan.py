# -*- coding: utf-8 -*-
"""
droplet_scan.py
===============

Search for self-bound objects (quantum droplets / solitons) in the extended
correlated-hopping Bose-Hubbard chain

    H = - sum_{d=1}^{r} t/d^alpha sum_i ( b+_{i-d} b_i b_i b+_{i+d} + h.c. )
        + U/2 sum_i n_i (n_i - 1)

with FINITE DMRG at fixed particle number N in a box of L sites that is much
larger than the object (open boundaries, mu = 0, N and dipole moment conserved).

Physics in two lines (details in the Obsidian notes "Droplets and solitons"):
  * an isolated boson cannot move (pair hopping needs company), so a dilute gas
    has energy exactly 0;  particles that cluster gain correlated-hopping
    energy ~ -n^2 t  but pay U n^2/2  ->  a cluster with E(N) < 0 is self-bound.
  * droplet  = flat-top, density saturates at an equilibrium value n0,
               size grows ~ N / n0          (dR/dN > 0)
    soliton  = bell-shaped, peak density keeps growing, size shrinks with N
               (dR/dN < 0)

What this script does, for every (t, N, n_max, init):
  1. builds the initial product state
       compact : the N bosons in a block in the middle at density n_init
       spread  : the N bosons spread uniformly over the box (what
                 run_simulations.py does) -- usually a frozen local minimum
                 when N < L because single bosons cannot hop
  2. runs finite DMRG (dipole + N conserved) and stores
       energy, density profile, max P(n_i = n_max), chi, truncation error
  3. writes a summary row to <out>/droplet_summary.csv and the profile to
     <out>/profiles/<tag>.csv
Analyse with  analyze_droplets.py.

Example (from Restricted_model_extended):
    python droplet_scan.py --alpha 3 --r 3 --U 1 --t 0.10 0.15 0.20 ^
        --L 60 --N 6 10 14 20 28 36 --n_max 4 6 8 --chi 120 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def _set_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


FIELDS = ["alpha", "r", "U", "t", "L", "N", "n_max", "chi_max", "init", "n_init",
          "E", "E_per_N", "n_peak", "p_nmax", "S_mid", "trunc_err",
          "chi_reached", "converged", "wall_s", "profile_file"]


def tag_of(j: dict) -> str:
    f = lambda x: f"{x:g}".replace(".", "p").replace("-", "m")
    return (f"alpha{j['alpha']}_r{j['r']}_U{f(j['U'])}_t{f(j['t'])}_L{j['L']}"
            f"_N{j['N']}_nmax{j['n_max']}_chi{j['chi_max']}_{j['init']}{j['n_init']}")


def initial_occupations(L: int, N: int, n_max: int, init: str, n_init: int) -> list[int]:
    occ = [0] * L
    if init == "spread":
        q, rem = divmod(N, L)
        occ = [q] * L
        for k in range(rem):
            occ[(k * L) // rem] += 1
        return occ
    # compact block of density n_init centred in the box (symmetric -> same
    # dipole sector as the symmetric spread state when possible)
    n_init = max(1, min(n_init, n_max))
    width = -(-N // n_init)                       # ceil(N / n_init)
    if width > L:
        raise ValueError(f"N={N} does not fit in L={L} at density {n_init}")
    start = (L - width) // 2
    left = N
    for i in range(start, start + width):
        occ[i] = min(n_init, left)
        left -= occ[i]
    # put the remainder (last, partially filled site) in the middle instead of
    # at the edge to keep the block symmetric
    if N % n_init:
        occ[start + width - 1] = 0
        occ[start + width // 2] += N % n_init
        if occ[start + width // 2] > n_max:
            raise ValueError("n_init too large for n_max")
    return occ


def run_point(job: dict) -> dict:
    import numpy as np
    from tenpy.algorithms import dmrg
    from tenpy.algorithms.mps_common import SubspaceExpansion
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPS
    import model_extended

    L, N, n_max, chi = job["L"], job["N"], job["n_max"], job["chi_max"]
    t0 = time.time()
    model = model_extended.CorrelatedHoppingBoseModelFinite(dict(
        t=job["t"], U=job["U"], mu=0.0, L=L, n_max=n_max,
        conserve="dipole", filling=N / L, bc_MPS="finite",
        alpha=job["alpha"], r=job["r"],
    ))
    occ = initial_occupations(L, N, n_max, job["init"], job["n_init"])
    psi = MPS.from_product_state(model.lat.mps_sites(), occ, bc="finite")
    info = dmrg.run(psi, model, {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-10},
        "chi_list": {0: max(16, chi // 4), 4: max(16, chi // 2), 8: chi},
        "max_E_err": 1e-10, "max_S_err": 1e-3,
        # don't abort when chi is too small: record trunc_err and flag it
        "max_trunc_err": 1.0,
        "min_sweeps": 10, "max_sweeps": job["max_sweeps"],
        "max_hours": job["max_hours"],
        "mixer": SubspaceExpansion,     # essential: the pair hop acts on 3 sites
        "mixer_params": {"amplitude": 1e-3, "decay": 1.5, "disable_after": 20},
    })
    E = float(info["E"])
    dens = np.asarray(psi.expectation_value("N"), float)
    P_top = np.zeros((n_max + 1, n_max + 1)); P_top[n_max, n_max] = 1.0
    ops = [npc.Array.from_ndarray(P_top, [s.leg, s.leg.conj()], labels=["p", "p*"])
           for s in psi.sites]
    p_nmax = float(np.max(psi.expectation_value(ops)))
    stats = info.get("sweep_statistics", {})

    prof_dir = Path(job["out"]) / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    pf = prof_dir / (tag_of(job) + ".csv")
    with open(pf, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["site", "n", "n_initial"])
        for i, (x, o) in enumerate(zip(dens, occ)):
            w.writerow([i, f"{x:.10f}", o])

    return dict(
        alpha=job["alpha"], r=job["r"], U=job["U"], t=job["t"], L=L, N=N,
        n_max=n_max, chi_max=chi, init=job["init"], n_init=job["n_init"],
        E=E, E_per_N=E / N, n_peak=float(dens.max()), p_nmax=p_nmax,
        S_mid=float(psi.entanglement_entropy()[L // 2 - 1]),
        trunc_err=float(stats["max_trunc_err"][-1]) if stats.get("max_trunc_err") else float("nan"),
        chi_reached=int(max(psi.chi)),
        converged=not bool(info.get("shelve", False)),
        wall_s=round(time.time() - t0, 1),
        profile_file=str(pf.relative_to(job["out"])),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alpha", type=int, required=True)
    p.add_argument("--r", type=int, required=True)
    p.add_argument("--U", type=float, default=1.0)
    p.add_argument("--t", type=float, nargs="+", required=True)
    p.add_argument("--L", type=int, default=60)
    p.add_argument("--N", type=int, nargs="+", required=True)
    p.add_argument("--n_max", type=int, nargs="+", default=[4, 6],
                   help="several values = local-Hilbert-space convergence check")
    p.add_argument("--chi", type=int, default=120)
    p.add_argument("--init", choices=["compact", "spread", "both"], default="both")
    p.add_argument("--n_init", type=int, default=1,
                   help="density of the compact initial block")
    p.add_argument("--max_sweeps", type=int, default=60)
    p.add_argument("--max_hours", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--threads_per_worker", type=int, default=1)
    p.add_argument("--out", default="droplet_data")
    a = p.parse_args()
    _set_threads(a.threads_per_worker)

    S = sum(d ** (-a.alpha) for d in range(1, a.r + 1))
    ts = a.U / (4 * S)
    print(f"alpha={a.alpha}, r={a.r}: mean-field bunching t*/U = {ts / a.U:.6f}")
    for t in a.t:
        if t > ts:
            print(f"  [WARN] t={t} > t*: collapse regime, any self-bound object "
                  f"is only stabilised by n_max -> check the n_max dependence!")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summ = out / "droplet_summary.csv"
    done = set()
    if summ.exists():
        with open(summ, newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != FIELDS:
                raise SystemExit(f"{summ} has different columns; use another --out")
            done = {r["profile_file"] for r in rd}

    inits = ["compact", "spread"] if a.init == "both" else [a.init]
    jobs = []
    for t in a.t:
        for N in a.N:
            for nm in a.n_max:
                for ini in inits:
                    j = dict(alpha=a.alpha, r=a.r, U=a.U, t=t, L=a.L, N=N, n_max=nm,
                             chi_max=a.chi, init=ini, n_init=a.n_init,
                             max_sweeps=a.max_sweeps, max_hours=a.max_hours,
                             out=str(out))
                    if str(Path("profiles") / (tag_of(j) + ".csv")) not in done:
                        jobs.append(j)
    jobs.sort(key=lambda j: (j["N"], j["n_max"], j["t"]))
    print(f"{len(jobs)} runs to do ({len(done)} done) -> {summ}")
    if not jobs:
        return

    new = not summ.exists()
    with open(summ, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()

        def _write(res):
            w.writerow(res); f.flush()
            flag = "" if res["converged"] else "  [NOT CONVERGED]"
            if res["p_nmax"] > 1e-3:
                flag += f"  [p(n_max)={res['p_nmax']:.1e}: raise n_max]"
            if res["trunc_err"] > 1e-8:
                flag += f"  [trunc_err={res['trunc_err']:.1e}]"
            print(f"t={res['t']:g} N={res['N']:3d} nmax={res['n_max']} "
                  f"{res['init']:7s} E/N={res['E_per_N']:+.8f} "
                  f"n_peak={res['n_peak']:.4f} chi={res['chi_reached']} "
                  f"{res['wall_s']:.0f}s{flag}", flush=True)

        if a.workers == 1:
            for j in jobs:
                _write(run_point(j))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(run_point, j): j for j in jobs}
                for fu in as_completed(futs):
                    try:
                        _write(fu.result())
                    except Exception as exc:
                        j = futs[fu]
                        print(f"[ERROR] {tag_of(j)}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
