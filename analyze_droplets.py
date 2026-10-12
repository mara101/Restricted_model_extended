# -*- coding: utf-8 -*-
"""
analyze_droplets.py
===================

Reads <out>/droplet_summary.csv + profiles written by droplet_scan.py and

  1. keeps, for every (t, N, n_max), the LOWEST-energy run among the initial
     states (compact / spread): DMRG is variational, the lower energy wins;
  2. fits each density profile with the flat-top "symmetrised Fermi function"
     used by Morera et al. PRR 2, 022008 (2020), Eq. (2):

         n_i = n_M sinh(R/2a) / ( cosh(R/2a) + cosh((i - c)/a) )

     n_M = central density, R = size, a = edge (surface) thickness, c = centre;
     also computes the rms width  sigma = sqrt(<(i - c)^2>);
  3. decides, for every (t, N):
       - bound?    E(N) < 0  (a dilute gas of isolated bosons has E = 0 exactly,
                   because a single boson cannot hop in this model)
       - touches the box?  density at the boundaries > 1% of n_M  ->  "gas /
                   box-limited", not a self-bound object
       - n_max-converged?  peak density and E must not change (rel. tol.) between the
                   two largest n_max -- the local-Hilbert-space test
  4. classifies the sequence in N (at fixed t, largest n_max):
       droplet  : n_M saturates (flat top, R > ~3a) and sigma grows with N
       soliton  : no flat top (R <~ 2a), n_M grows and sigma shrinks with N
  5. plots: profiles, n_M(N), sigma(N), E/N(N), n_max dependence.

Usage:
    python analyze_droplets.py --out droplet_data --alpha 3 --r 3 --U 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fermi_profile(i, nM, R, a, c):
    a = np.abs(a) + 1e-9
    x = R / (2 * a)
    # sinh(x)/(cosh(x)+cosh(y)) written in an overflow-safe way
    y = (i - c) / a
    m = np.maximum(np.abs(x), np.abs(y))
    num = 0.5 * (np.exp(x - m) - np.exp(-x - m))
    den = 0.5 * (np.exp(x - m) + np.exp(-x - m) + np.exp(y - m) + np.exp(-y - m))
    return nM * num / den


def fit_profile(n: np.ndarray) -> dict:
    i = np.arange(len(n), dtype=float)
    N = n.sum()
    c0 = (i * n).sum() / N
    sigma = float(np.sqrt(((i - c0) ** 2 * n).sum() / N))
    p0 = [n.max(), max(1.0, N / max(n.max(), 1e-3)), 1.0, c0]
    try:
        popt, _ = curve_fit(fermi_profile, i, n, p0=p0, maxfev=20000,
                            bounds=([0, 0, 0.05, 0],
                                    [1.5 * n.max() + 1e-6, 10 * len(n), len(n), len(n)]))
        nM, R, a, c = popt
        resid = float(np.sqrt(np.mean((fermi_profile(i, *popt) - n) ** 2)))
    except Exception:
        nM, R, a, c, resid = n.max(), np.nan, np.nan, c0, np.nan
    edge = float(max(n[:2].max(), n[-2:].max()) / max(n.max(), 1e-12))
    return dict(n_M=float(nM), R=float(R), a=float(abs(a)), center=float(c),
                sigma=sigma, fit_rms=resid, edge_ratio=edge)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="droplet_data")
    p.add_argument("--alpha", type=int, required=True)
    p.add_argument("--r", type=int, required=True)
    p.add_argument("--U", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=0.02,
                   help="relative tolerance for n_max convergence")
    a = p.parse_args()

    out = Path(a.out)
    df = pd.read_csv(out / "droplet_summary.csv")
    df = df[(df.alpha == a.alpha) & (df.r == a.r) & np.isclose(df.U, a.U)]
    if df.empty:
        raise SystemExit("no matching runs")

    # 1. lowest energy among initial states
    best = (df.sort_values("E").groupby(["t", "N", "n_max"], as_index=False).first())
    # tell the user when the initial state mattered
    spread = df.pivot_table(index=["t", "N", "n_max"], columns="init", values="E")
    if {"compact", "spread"} <= set(spread.columns):
        d = (spread["spread"] - spread["compact"]).dropna()
        stuck = d[d.abs() > 1e-6]
        if len(stuck):
            print(f"[info] initial state mattered in {len(stuck)} cases "
                  f"(max |dE| = {stuck.abs().max():.3e}); lower energy kept.")

    # 2. fits
    fits = []
    for row in best.itertuples():
        prof = pd.read_csv(out / row.profile_file)
        fits.append(fit_profile(prof.n.to_numpy(float)))
    best = pd.concat([best.reset_index(drop=True), pd.DataFrame(fits)], axis=1)
    best["flat_top"] = best.R > 3 * best.a
    best["bound"] = best.E < -1e-8
    best["box_limited"] = best.edge_ratio > 0.01

    # 3. n_max convergence (compare the two largest n_max)
    conv = []
    for (t, N), g in best.groupby(["t", "N"]):
        g = g.sort_values("n_max")
        if len(g) < 2:
            conv.append(dict(t=t, N=N, nmax_converged=np.nan, dnM_rel=np.nan, dE_rel=np.nan))
            continue
        x, y = g.iloc[-2], g.iloc[-1]
        dn = abs(y.n_peak - x.n_peak) / max(abs(y.n_peak), 1e-12)
        dE = abs(y.E - x.E) / max(abs(y.E), 1e-12)
        conv.append(dict(t=t, N=N, nmax_converged=bool(dn < a.tol and dE < a.tol),
                         dnM_rel=dn, dE_rel=dE))
    conv = pd.DataFrame(conv)
    top = best.sort_values("n_max").groupby(["t", "N"], as_index=False).last()
    top = top.merge(conv, on=["t", "N"], how="left")

    # 4. classification along N at fixed t
    labels = []
    for t, g in top.groupby("t"):
        g = g.sort_values("N")
        Ns = g.N.to_numpy(float)
        dsig = np.gradient(g.sigma.to_numpy(), Ns) if len(g) > 1 else np.full(len(g), np.nan)
        # relative change of the peak density per added particle, times N
        # (= d ln n_peak / d ln N): ~0 for a saturated droplet, > 0 if growing
        dlnp = (np.gradient(np.log(g.n_peak.to_numpy()), np.log(Ns))
                if len(g) > 1 else np.full(len(g), np.nan))
        for (idx, row), ds, gp in zip(g.iterrows(), dsig, dlnp):
            top.loc[idx, "dln_npeak_dlnN"] = gp
            if not row.bound:
                lab = "unbound (gas)"
            elif row.box_limited:
                lab = "fills the box (gas / bulk phase)"
            elif ds < 0 and gp > 0:
                lab = "SOLITON-like (narrower and taller with N)"
            elif row.flat_top and ds > 0 and abs(gp) < 0.05:
                lab = "DROPLET (saturated flat top, grows with N)"
            elif ds > 0 and gp >= 0.05:
                lab = "growing peak: non-saturated droplet / soliton-like (need larger N)"
            else:
                lab = "crossover / unclear"
            if row.nmax_converged is False:
                lab += "  !! NOT converged in n_max"
            labels.append((idx, lab))
    for idx, lab in labels:
        top.loc[idx, "classification"] = lab

    cols = ["t", "N", "n_max", "E", "E_per_N", "n_peak", "n_M", "R", "a", "sigma",
            "edge_ratio", "dln_npeak_dlnN", "nmax_converged", "dnM_rel", "classification"]
    pd.set_option("display.width", 220)
    print(top[cols].sort_values(["t", "N"]).to_string(index=False,
          float_format=lambda v: f"{v:.4g}"))
    top.to_csv(out / f"droplet_analysis_alpha{a.alpha}_r{a.r}.csv", index=False)

    # 5. plots
    for t, g in top.groupby("t"):
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        for row in g.sort_values("N").itertuples():
            prof = pd.read_csv(out / row.profile_file)
            ax[0, 0].plot(prof.site, prof.n, label=f"N={row.N}")
        ax[0, 0].set_xlabel("site i"); ax[0, 0].set_ylabel(r"$\langle n_i\rangle$")
        ax[0, 0].legend(fontsize=7)
        g = g.sort_values("N")
        ax[0, 1].plot(g.N, g.n_peak, "o-"); ax[0, 1].set_xlabel("N"); ax[0, 1].set_ylabel("peak density")
        ax[1, 0].plot(g.N, g.sigma, "o-"); ax[1, 0].set_xlabel("N"); ax[1, 0].set_ylabel(r"rms width $\sigma$")
        ax[1, 1].plot(g.N, g.E_per_N, "o-"); ax[1, 1].axhline(0, color="0.5", lw=0.8)
        ax[1, 1].set_xlabel("N"); ax[1, 1].set_ylabel("E/N")
        # n_max dependence of the peak density
        for N, gg in best[best.t == t].groupby("N"):
            gg = gg.sort_values("n_max")
            if len(gg) > 1:
                ax[0, 1].plot([N] * len(gg), gg.n_peak, "x", color="firebrick", ms=4)
        fig.suptitle(rf"$\alpha={a.alpha}, r={a.r}, U={a.U:g}, t={t:g}$"
                     "   (red x: peak density for the other n_max)", fontsize=9)
        fig.tight_layout()
        f = out / f"droplets_alpha{a.alpha}_r{a.r}_t{t:g}.png"
        fig.savefig(f, dpi=180); plt.close(fig)
        print("saved", f)


if __name__ == "__main__":
    main()
