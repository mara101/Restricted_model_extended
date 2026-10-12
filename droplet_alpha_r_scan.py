# -*- coding: utf-8 -*-
"""
droplet_alpha_r_scan.py
=======================

Scan of the self-bound objects (droplets / soliton-like / collapse) over the
hopping range r and the damping exponent alpha of

    H = - sum_{d=1}^{r} t/d^alpha sum_i ( b+_{i-d} b_i b_i b+_{i+d} + h.c. )
        + U/2 sum_i n_i (n_i - 1)

Every (alpha, r) has its own mean-field bunching point
    t*(alpha, r) = U / (4 S),   S = sum_{d=1}^{r} d^-alpha,
so t is scanned in units of t*:  t = f * t*(alpha, r).  This makes diagrams of
different models directly comparable.  r = 1 is the Zechmann et al. model
(alpha is irrelevant there) and is the control case.

For every (alpha, r, f) the script runs finite DMRG (droplet_scan.run_point)
  * N in --N                       (droplet size scaling)
  * two compact initial states     (block at density 1 and at density 2:
                                    bistability test a la Schubert 2026)
  * n_max = --n_max_low for all N, and additionally --n_max_high for the
    largest N                      (local-Hilbert-space convergence test)
All jobs go into ONE process pool (good load balancing); results are appended
to <out>/droplet_summary.csv (same format as droplet_scan.py) and the scan can
be stopped and restarted at any time.

Then run   python droplet_phase_diagram.py --out <out>

Default grid: alpha in {0,1,2,3}, r in {1,2,3,4}, f in {0.3,...,1.3},
N in {8,16,24}  ->  13 models x 7 t x 8 runs = 728 DMRG runs.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alphas", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--rs", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--f", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3],
                   help="t in units of t*(alpha, r)")
    p.add_argument("--U", type=float, default=1.0)
    p.add_argument("--N", type=int, nargs="+", default=[8, 16, 24])
    p.add_argument("--n_init", type=int, nargs="+", default=[1, 2])
    p.add_argument("--n_max_low", type=int, default=6)
    p.add_argument("--n_max_high", type=int, default=8)
    p.add_argument("--chi", type=int, default=120)
    p.add_argument("--L_pad", type=int, default=12,
                   help="box length L = max(L_min, 2N + L_pad)")
    p.add_argument("--L_min", type=int, default=40)
    p.add_argument("--max_sweeps", type=int, default=60)
    p.add_argument("--max_hours", type=float, default=0.5, help="per run")
    p.add_argument("--workers", type=int, default=None,
                   help="default: number of PHYSICAL cores - 1 (hyper-threads don't help DMRG)")
    p.add_argument("--threads_per_worker", type=int, default=1)
    p.add_argument("--out", default="droplet_alpha_r_data")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    if a.workers is None:
        try:
            import psutil
            phys = psutil.cpu_count(logical=False) or 2
        except Exception:
            phys = max(1, (os.cpu_count() or 2) // 2)
        a.workers = max(1, phys - 1)

    # thread limits BEFORE numpy/tenpy are imported anywhere
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(a.threads_per_worker)

    import droplet_scan as ds          # run_point, tag_of, FIELDS

    # ---- pre-flight: TeNPy must support dipole conservation (TeNPy >= 1.0) ----
    import tenpy
    print(f"TeNPy {tenpy.__version__} from {tenpy.__file__}", flush=True)
    try:
        from tenpy.networks.site import BosonSite
        BosonSite(Nmax=2, conserve="dipole")
    except Exception as exc:
        raise SystemExit(f"[FATAL] this TeNPy cannot conserve the dipole moment ({exc}). "
                         "Install TeNPy >= 1.0 (pip install -U physics-tenpy).")

    models = []
    for r in a.rs:
        for al in (a.alphas if r > 1 else a.alphas[:1]):   # alpha irrelevant at r=1
            models.append((al, r))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summ = out / "droplet_summary.csv"
    done = set()
    if summ.exists():
        with open(summ, newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != ds.FIELDS:
                raise SystemExit(f"{summ} has different columns; use another --out")
            done = {r["profile_file"] for r in rd}

    Nmax_ = max(a.N)
    jobs = []
    for al, r in models:
        S = sum(d ** (-al) for d in range(1, r + 1))
        tstar = a.U / (4.0 * S)
        for f in a.f:
            t = round(f * tstar, 8)
            for N in a.N:
                L = max(a.L_min, 2 * N + a.L_pad)
                nmaxes = [a.n_max_low] + ([a.n_max_high] if N == Nmax_ else [])
                for nm in nmaxes:
                    for ni in a.n_init:
                        if ni > nm:
                            continue
                        j = dict(alpha=al, r=r, U=a.U, t=t, L=L, N=N, n_max=nm,
                                 chi_max=a.chi, init="compact", n_init=ni,
                                 max_sweeps=a.max_sweeps, max_hours=a.max_hours,
                                 out=str(out))
                        if str(Path("profiles") / (ds.tag_of(j) + ".csv")) not in done:
                            jobs.append(j)
    # cheap jobs first: small N, small t/t*
    jobs.sort(key=lambda j: (j["N"], j["t"] / (a.U / (4.0 * sum(d ** (-j["alpha"]) for d in range(1, j["r"] + 1)))), j["n_max"]))
    print(f"{len(models)} models: {models}")
    print(f"{len(jobs)} runs to do ({len(done)} done) -> {summ}; "
          f"{a.workers} workers x {a.threads_per_worker} thread(s)", flush=True)
    if a.dry_run or not jobs:
        return

    # ---- pre-flight 2: run the cheapest job in this process; abort on error ----
    test = dict(jobs[0]); test["out"] = str(out / "_preflight")
    try:
        r0 = ds.run_point(test)
        print(f"[preflight ok] E/N={r0['E_per_N']:+.6f} in {r0['wall_s']:.0f}s", flush=True)
    except Exception as exc:
        raise SystemExit(f"[FATAL] preflight run failed: {exc!r}")

    new = not summ.exists()
    t_start = time.time()
    with open(summ, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ds.FIELDS)
        if new:
            w.writeheader()
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(ds.run_point, j): j for j in jobs}
            n_ok = n_err = 0
            for k, fu in enumerate(as_completed(futs), 1):
                j = futs[fu]
                try:
                    res = fu.result()
                except Exception as exc:
                    print(f"[ERROR] {ds.tag_of(j)}: {exc}", file=sys.stderr, flush=True)
                    n_err += 1
                    if n_err >= 20 and n_ok == 0:
                        for other in futs:
                            other.cancel()
                        raise SystemExit("[FATAL] 20 failures and no success: stopping.")
                    continue
                n_ok += 1
                w.writerow(res); fh.flush()
                el = (time.time() - t_start) / 3600
                flag = "" if res["converged"] else " [NOT CONV]"
                if res["p_nmax"] > 1e-3:
                    flag += f" [p(nmax)={res['p_nmax']:.0e}]"
                print(f"[{k}/{len(jobs)} {el:.2f}h] a={res['alpha']} r={res['r']} "
                      f"t={res['t']:.4f} N={res['N']} nmax={res['n_max']} "
                      f"init={res['n_init']} E/N={res['E_per_N']:+.6f} "
                      f"peak={res['n_peak']:.3f} {res['wall_s']:.0f}s{flag}", flush=True)


if __name__ == "__main__":
    main()
