# -*- coding: utf-8 -*-
"""
bkt_from_dipole_correlators.py
==============================

Estimate the d-Mott -> d-Luttinger (BKT) point at integer filling from the
dipole correlators <d^dag_r d_0> that correlators_extractor.py already saved
(mps_files_infinite/correlators_csv/*_dipole_correlator.csv).

Criterion (paper, Sec. III and Fig. 5a): in the d-Luttinger liquid
        <d^dag_r d_0> ~ r^(-1/(2 K_d)),
and the BKT transition happens at K_d* = 2, i.e. a power-law exponent 1/4.
For each t we fit the log-log slope on a window [r_lo, r_hi], define the
effective K_d = -1 / (2 * slope), and interpolate where K_d crosses 2.
In the Mott phase the decay is exponential, so the "effective K_d" drops
quickly below 2 -- we also report which of the two fits (power law vs.
exponential) has the smaller residual.

Caveats: with finite chi the correlations always decay exponentially at large
r (xi ~ chi^kappa), so keep r_hi well below the correlation length and check
the result for two bond dimensions.

Usage
-----
    python bkt_from_dipole_correlators.py --alpha 3 --r 3 --n 1
    python bkt_from_dipole_correlators.py --alpha 3 --r 3 --n 2 --r_lo 8 --r_hi 40
Writes/updates  phase_diagram_data/bkt_points.csv  (columns n,t_bkt,alpha,r)
and a K_d(t) plot.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PAT = re.compile(
    r"iMPS_alpha(?P<alpha>\d+)_r(?P<r>\d+)_U(?P<U>[0-9p]+)_mu(?P<mu>m?[0-9p]+)"
    r"_t(?P<t>[0-9p]+)_cons(?P<cons>\w+?)_chi(?P<chi>\d+)_L(?P<L>\d+)"
    r"_Nuc(?P<N>\d+)_nmax(?P<nmax>\d+)_dipole_correlator\.csv$")


def _num(s: str) -> float:
    return float(s.replace("m", "-").replace("p", "."))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="mps_files_infinite/correlators_csv")
    p.add_argument("--alpha", type=int, required=True)
    p.add_argument("--r", type=int, required=True)
    p.add_argument("--n", type=int, required=True, help="integer filling")
    p.add_argument("--cons", default="dipole",
                   help="conservation tag of the runs to use (don't mix canonical "
                        "and grand-canonical runs); 'any' to disable the filter")
    p.add_argument("--r_lo", type=int, default=8)
    p.add_argument("--r_hi", type=int, default=50)
    p.add_argument("--out", default="phase_diagram_data/bkt_points.csv")
    a = p.parse_args()

    rows = []
    for f in Path(a.dir).glob("*_dipole_correlator.csv"):
        m = PAT.match(f.name)
        if not m:
            continue
        g = m.groupdict()
        if int(g["alpha"]) != a.alpha or int(g["r"]) != a.r:
            continue
        L, N = int(g["L"]), int(g["N"])
        if N != a.n * L or (a.cons != "any" and g["cons"] != a.cons):
            continue
        rows.append(dict(t=_num(g["t"]), chi=int(g["chi"]), nmax=int(g["nmax"]),
                         cons=g["cons"], file=f))
    if not rows:
        raise SystemExit("no matching correlator files")
    df = (pd.DataFrame(rows).sort_values(["t", "nmax", "chi"])
            .groupby("t", as_index=False).last())

    res = []
    for _, row in df.iterrows():
        c = pd.read_csv(row.file)
        x, y = c.iloc[:, 0].to_numpy(float), np.abs(c.iloc[:, 1].to_numpy(float))
        w = (x >= a.r_lo) & (x <= a.r_hi) & (y > 1e-14)
        if w.sum() < 4:
            continue
        lx, ly = np.log(x[w]), np.log(y[w])
        (s_pl, b_pl), r_pl = np.polyfit(lx, ly, 1, full=True)[:2]
        (s_ex, b_ex), r_ex = np.polyfit(x[w], ly, 1, full=True)[:2]
        res.append(dict(t=row.t, chi=row.chi, cons=row.cons,
                        exponent=-s_pl, K_d=-1.0 / (2.0 * s_pl) if s_pl < 0 else np.inf,
                        xi=-1.0 / s_ex if s_ex < 0 else np.inf,
                        better_fit="power" if (r_pl.size and r_ex.size and r_pl[0] < r_ex[0])
                        else "exp"))
    res = pd.DataFrame(res).sort_values("t")
    print(res.to_string(index=False))

    # first crossing K_d = 2 from below
    t_bkt = np.nan
    K = res.K_d.to_numpy(); T = res.t.to_numpy()
    B = res.better_fit.to_numpy()
    for i in range(len(K) - 1):
        if K[i] < 2.0 <= K[i + 1]:
            if B[i + 1] != "power":
                print(f"[skip] K_d crosses 2 at t~{T[i]:.4f}..{T[i+1]:.4f} but the decay "
                      "is still better fitted by an exponential -> not a BKT point")
                continue
            t_bkt = T[i] + (2.0 - K[i]) * (T[i + 1] - T[i]) / (K[i + 1] - K[i])
            break
    if np.isfinite(t_bkt):
        print(f"\nalpha={a.alpha}, r={a.r}, n={a.n}:  t_BKT/U ~ {t_bkt:.4f}")
    else:
        print(f"\nalpha={a.alpha}, r={a.r}, n={a.n}:  no BKT point found "
              "(dipole correlations exponential over the whole scan)")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["n", "t_bkt", "alpha", "r"])
    old = old[~((old.n == a.n) & (old.alpha == a.alpha) & (old.r == a.r))]
    if np.isfinite(t_bkt):
        new = pd.DataFrame([dict(n=a.n, t_bkt=t_bkt, alpha=a.alpha, r=a.r)])
        old = new if old.empty else pd.concat([old, new], ignore_index=True)
    old.to_csv(out, index=False)

    fig, ax = plt.subplots(figsize=(4, 2.8))
    ax.plot(res.t, np.clip(res.K_d, 0, 20), "k.-")
    ax.axhline(2, ls=":", color="0.4")
    if np.isfinite(t_bkt):
        ax.axvline(t_bkt, ls="--", color="firebrick")
    ax.set_xlabel("$t/U$"); ax.set_ylabel(r"effective $K_d$")
    ax.set_title(rf"$\alpha={a.alpha},\ r={a.r},\ n={a.n}$", fontsize=9)
    fig.tight_layout()
    fig.savefig(out.with_name(f"Kd_alpha{a.alpha}_r{a.r}_n{a.n}.png"), dpi=200)


if __name__ == "__main__":
    main()
