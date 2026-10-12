# -*- coding: utf-8 -*-
"""
droplet_phase_diagram.py
========================

Turns the output of droplet_alpha_r_scan.py into phase diagrams of the
self-bound objects as a function of (alpha, r, t).

For every (alpha, r, t) it uses the runs at the different N:
  * lowest energy among the initial states for each (N, n_max)
    (and flags "bistable?" when the two initial states converge to clearly
    different objects with close energies)
  * bound           : E(N) < 0 (a dilute gas has E = 0 exactly)
  * n_max converged : peak density and E at the largest N change by < tol
                      between n_max_low and n_max_high
  * saturation      : g = d ln(peak) / d ln N  between the two largest N
  * width trend     : d sigma / d N
  * flat top        : Fermi-function fit, R > 3a  (Morera 2020, Eq. 2)
and classifies:
  unbound | DROPLET (saturated) | growing (non-saturated / soliton-like)
  | SOLITON-like (narrower and taller) | collapse / cutoff-limited | unclear
For droplets: equilibrium density n0 (= peak at largest N) and bulk energy
per particle eps0 = dE/dN (from the two largest N).

Outputs in <out>/:
  droplet_phase_table.csv
  phase_map_f.png      x = t/t*,  y = r,  one panel per alpha
  phase_map_tU.png     x = t/U,   y = r,  one panel per alpha (t* marked)
  n0_map.png, eps0_map.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analyze_droplets import fit_profile

CLASSES = {
    "unbound": "#bdbdbd",
    "DROPLET": "#3b7dd8",
    "growing": "#f2a541",
    "SOLITON-like": "#d1495b",
    "collapse": "#5c5c5c",
    "unclear": "#e9e1c8",
}


def t_star(alpha, r, U=1.0):
    return U / (4.0 * sum(float(d) ** (-float(alpha)) for d in range(1, int(r) + 1)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="droplet_alpha_r_data")
    p.add_argument("--U", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--g_sat", type=float, default=0.05,
                   help="|d ln peak / d ln N| below this = saturated")
    a = p.parse_args()
    out = Path(a.out)
    df = pd.read_csv(out / "droplet_summary.csv")
    df = df[np.isclose(df.U, a.U)]
    if df.empty:
        raise SystemExit("droplet_summary.csv has no results yet - nothing to plot.")
    df["f"] = [t / t_star(al, r, a.U) for t, al, r in zip(df.t, df.alpha, df.r)]

    # fits for every run (cheap)
    # profile paths may have been written on Windows (backslashes)
    df["profile_file"] = df.profile_file.str.replace("\\", "/", regex=False)
    fits = [fit_profile(pd.read_csv(out / pf).n.to_numpy(float)) for pf in df.profile_file]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(fits)], axis=1)

    rows = []
    for (al, r, t), g in df.groupby(["alpha", "r", "t"]):
        nm_lo, nm_hi = g.n_max.min(), g.n_max.max()
        # best over initial states
        best = g.sort_values("E").groupby(["N", "n_max"], as_index=False).first()
        # bistability flag: two inits, both converged, different peaks, close E
        bist = False
        for (N, nm), gg in g.groupby(["N", "n_max"]):
            if len(gg) >= 2 and gg.converged.all():
                dE = gg.E.max() - gg.E.min()
                dp = gg.n_peak.max() - gg.n_peak.min()
                if dp > 0.1 and dE < 0.02 * abs(gg.E.min()):
                    bist = True
        lo = best[best.n_max == nm_lo].sort_values("N")
        Ns = lo.N.to_numpy(float)
        Nbig = Ns.max()
        big_lo = lo[lo.N == Nbig].iloc[0]
        big_hi_rows = best[(best.N == Nbig) & (best.n_max == nm_hi)]
        if nm_hi > nm_lo and len(big_hi_rows):
            bh = big_hi_rows.iloc[0]
            dpk = abs(bh.n_peak - big_lo.n_peak) / bh.n_peak
            dE = abs(bh.E - big_lo.E) / max(abs(bh.E), 1e-12)
            nconv = bool(dpk < a.tol and dE < a.tol)
            ref = bh
        else:
            dpk = dE = np.nan; nconv = None; ref = big_lo
        if len(lo) >= 2:
            l2 = lo.iloc[-2:]
            g_sat = float(np.log(l2.n_peak.iloc[1] / l2.n_peak.iloc[0])
                          / np.log(l2.N.iloc[1] / l2.N.iloc[0]))
            dsig = float((l2.sigma.iloc[1] - l2.sigma.iloc[0]) / (l2.N.iloc[1] - l2.N.iloc[0]))
            eps0 = float((l2.E.iloc[1] - l2.E.iloc[0]) / (l2.N.iloc[1] - l2.N.iloc[0]))
        else:
            g_sat = dsig = eps0 = np.nan
        bound = bool((lo.E < -1e-8).all())
        flat = bool(ref.R > 3 * ref.a)
        touches = bool(ref.edge_ratio > 0.01)

        if not bound:
            cls = "unbound"
        elif nconv is False:
            cls = "collapse"
        elif touches:
            cls = "unclear"
        elif abs(g_sat) < a.g_sat and flat and dsig > 0:
            cls = "DROPLET"
        elif g_sat >= a.g_sat and dsig < 0:
            cls = "SOLITON-like"
        elif g_sat >= a.g_sat:
            cls = "growing"
        else:
            cls = "unclear"
        rows.append(dict(alpha=al, r=r, t=t, f=t / t_star(al, r, a.U), cls=cls,
                         bistable=bist, n0=float(ref.n_peak) if cls == "DROPLET" else np.nan,
                         eps0=eps0 if cls == "DROPLET" else np.nan,
                         peak_bigN=float(ref.n_peak), g_sat=g_sat, dsigma_dN=dsig,
                         nmax_converged=nconv, d_peak_nmax=dpk, d_E_nmax=dE,
                         flat_top=flat, touches_box=touches,
                         all_converged=bool(g.converged.all()),
                         max_trunc_err=float(g.trunc_err.max()),
                         N_values=",".join(str(int(x)) for x in Ns)))
    tab = pd.DataFrame(rows).sort_values(["alpha", "r", "t"])
    tab.to_csv(out / "droplet_phase_table.csv", index=False)
    pd.set_option("display.width", 220)
    print(tab[["alpha", "r", "t", "f", "cls", "bistable", "n0", "eps0", "peak_bigN",
               "g_sat", "nmax_converged", "all_converged", "max_trunc_err"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    alphas = sorted(tab.alpha.unique())
    # r = 1 does not depend on alpha: show it in every panel
    r1 = tab[tab.r == 1]

    def panel_data(al):
        d = tab[(tab.alpha == al) & (tab.r > 1)]
        if len(r1):
            rr = r1.copy(); rr["alpha"] = al
            d = pd.concat([rr, d])
        return d

    for xkey, fname, xlabel in (("f", "phase_map_f.png", r"$t/t^*(\alpha,r)$"),
                                ("t", "phase_map_tU.png", r"$t/U$")):
        fig, axes = plt.subplots(1, len(alphas), figsize=(3.2 * len(alphas), 3.0),
                                 sharey=True, squeeze=False)
        for ax, al in zip(axes[0], alphas):
            d = panel_data(al)
            for row in d.itertuples():
                ax.scatter(getattr(row, xkey), row.r, s=150, marker="s",
                           color=CLASSES[row.cls], edgecolor="k", lw=0.4, zorder=3)
                if row.bistable:
                    ax.scatter(getattr(row, xkey), row.r, s=30, marker="x",
                               color="k", zorder=4)
            if xkey == "f":
                ax.axvline(1.0, color="k", ls="--", lw=0.8)
            else:
                rs = sorted(d.r.unique())
                ax.plot([t_star(al, r, a.U) for r in rs], rs, "k--", lw=0.8)
            ax.set_title(rf"$\alpha={al}$", fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_yticks(sorted(tab.r.unique()))
        axes[0, 0].set_ylabel("range $r$")
        handles = [Patch(color=c, label=k) for k, c in CLASSES.items()]
        handles.append(plt.Line2D([], [], marker="x", ls="", color="k", label="bistable?"))
        handles.append(plt.Line2D([], [], ls="--", color="k", label=r"$t^*$ (mean field)"))
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(out / fname, dpi=200); plt.close(fig)

    for key, fname, lab in (("n0", "n0_map.png", r"droplet density $n_0$"),
                            ("eps0", "eps0_map.png", r"bulk $\varepsilon_0=dE/dN$")):
        fig, axes = plt.subplots(1, len(alphas), figsize=(3.2 * len(alphas), 3.0),
                                 sharey=True, squeeze=False)
        vals = tab[key].dropna()
        vmin, vmax = (vals.min(), vals.max()) if len(vals) else (0, 1)
        sc = None
        for ax, al in zip(axes[0], alphas):
            d = panel_data(al).dropna(subset=[key])
            sc = ax.scatter(d.f, d.r, c=d[key], s=150, marker="s", cmap="viridis",
                            vmin=vmin, vmax=vmax, edgecolor="k", lw=0.4)
            ax.axvline(1.0, color="k", ls="--", lw=0.8)
            ax.set_title(rf"$\alpha={al}$", fontsize=9); ax.set_xlabel(r"$t/t^*$")
            ax.set_yticks(sorted(tab.r.unique()))
        axes[0, 0].set_ylabel("range $r$")
        if sc is not None:
            fig.colorbar(sc, ax=axes[0].tolist(), label=lab)
        fig.savefig(out / fname, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved:", ", ".join(str(out / f) for f in
          ("droplet_phase_table.csv", "phase_map_f.png", "phase_map_tU.png",
           "n0_map.png", "eps0_map.png")))


if __name__ == "__main__":
    main()
