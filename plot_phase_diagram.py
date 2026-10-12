# -*- coding: utf-8 -*-
"""
plot_phase_diagram.py
=====================

Fig.-1(b)-style (t/U, mu/U) phase diagram from the CSV of phase_diagram_scan.py.

For every (t, n):  mu_c^+ = E0(nL+1) - E0(nL),   mu_c^- = E0(nL) - E0(nL-1)
(Eq. (20) of Zechmann et al.).  The d-Mott lobe of filling n is the region
mu_c^- < mu < mu_c^+; its tip is where the two curves meet.  Everything else
below the bunching instability t* = U / (4 sum_d d^-alpha) has non-integer
filling (the d-supersolid region of the paper).

Usage
-----
    python plot_phase_diagram.py --alpha 3 --r 3
    python plot_phase_diagram.py --alpha 3 --r 3 --bkt phase_diagram_data/bkt_points.csv
    python plot_phase_diagram.py --alpha 3 --r 3 --compare 0,1 1,3
If the CSV holds several chain lengths L, the largest is drawn and the others
are overlaid as thin grey lines (finite-size check).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

C_MOTT, C_REST, C_BUNCH, C_LUTT = "#cfe3f3", "#e8f1e4", "#e3e3e3", "#f2c6c6"


def t_star(alpha: int, r: int) -> float:
    return 1.0 / (4.0 * sum(d ** (-alpha) for d in range(1, r + 1)))


def lobes_table(csv: str, alpha: int, r: int, U: float) -> pd.DataFrame:
    """One row per (L, n, t) with mu_minus, mu_plus in units of U."""
    df = pd.read_csv(csv)
    df = df[(df.alpha == alpha) & (df.r == r) & np.isclose(df.U, U)]
    if df.empty:
        raise SystemExit(f"no data for alpha={alpha}, r={r}, U={U} in {csv}")
    # keep the best (largest n_max, then chi) run of every point
    df = (df.sort_values(["n_max", "chi_max"])
            .groupby(["L", "n_fill", "t", "dN"], as_index=False).last())
    if "trunc_err" not in df:
        df["trunc_err"] = np.nan
    for name, cond in (("not converged", ~df.converged.astype(bool)),
                       ("n_max cutoff felt (p_nmax>1e-2)", df.p_nmax > 1e-2),
                       ("with truncation error > 1e-8", df.trunc_err > 1e-8)):
        if cond.any():
            bad = df[cond]
            print(f"[WARN] {len(bad)} runs {name}: "
                  + ", ".join(f"(t={x.t:g}, n={x.n_fill}, dN={x.dN:+d})"
                              for x in bad.itertuples())[:400])
    piv = df.pivot_table(index=["L", "n_fill", "t"], columns="dN", values="E0")
    piv = piv.dropna(subset=[-1, 0, 1]).reset_index()
    out = pd.DataFrame(dict(L=piv.L, n=piv.n_fill, t=piv.t / U,
                            mu_minus=(piv[0] - piv[-1]) / U,
                            mu_plus=(piv[1] - piv[0]) / U))
    out["gap"] = out.mu_plus - out.mu_minus           # charge gap Delta_c / U
    return out.sort_values(["L", "n", "t"]).reset_index(drop=True)


def lobe_polygon(g: pd.DataFrame):
    """t, lower, upper arrays of one lobe, closed at the tip (linear interp.)."""
    g = g.sort_values("t")
    t, lo, hi, gap = (g[c].to_numpy() for c in ("t", "mu_minus", "mu_plus", "gap"))
    open_ = gap > 0
    if not open_[0]:
        return None
    k = np.argmax(~open_) if (~open_).any() else len(t)   # first closed index
    tt, ll, hh = list(t[:k]), list(lo[:k]), list(hi[:k])
    if k < len(t):                                          # tip between k-1 and k
        f = gap[k - 1] / (gap[k - 1] - gap[k])
        t_tip = t[k - 1] + f * (t[k] - t[k - 1])
        mu_tip = 0.5 * ((lo[k - 1] + f * (lo[k] - lo[k - 1]))
                        + (hi[k - 1] + f * (hi[k] - hi[k - 1])))
        tt.append(t_tip); ll.append(mu_tip); hh.append(mu_tip)
    return np.array(tt), np.array(ll), np.array(hh)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="phase_diagram_data/lobes.csv")
    p.add_argument("--alpha", type=int, required=True)
    p.add_argument("--r", type=int, required=True)
    p.add_argument("--U", type=float, default=1.0)
    p.add_argument("--L", type=int, default=None, help="default: largest L")
    p.add_argument("--compare", nargs="*", default=[],
                   help="other models 'alpha,r' -> lobe boundaries as lines")
    p.add_argument("--bkt", default=None, help="CSV with columns n,t_bkt[,alpha,r]")
    p.add_argument("--label_rest", default="d-Supersolid?",
                   help="label of the non-integer-filling region")
    p.add_argument("--ymin", type=float, default=None,
                   help="default: just below the lowest lobe edge")
    p.add_argument("--ymax", type=float, default=2.5)
    p.add_argument("--xmax", type=float, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    tab = lobes_table(a.csv, a.alpha, a.r, a.U)
    Ls = sorted(tab.L.unique())
    L = a.L or Ls[-1]
    main_tab = tab[tab.L == L]
    ts = t_star(a.alpha, a.r)
    xmax = a.xmax or 1.2 * ts
    if a.ymin is None:
        a.ymin = min(-0.05, float(main_tab.mu_minus.min()) - 0.05)

    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    ax.axvspan(0, ts, color=C_REST, zorder=0, lw=0)
    ax.axvspan(ts, xmax, color=C_BUNCH, zorder=0, lw=0)
    ax.axvline(ts, color="k", lw=1.0, zorder=4)
    ax.text(0.5 * (ts + xmax), 0.5 * (a.ymin + a.ymax), "Bunching instability",
            rotation=90, ha="center", va="center", fontsize=8)
    ax.text(0.68 * ts, 0.93 * a.ymax, a.label_rest, fontsize=8, ha="center", va="top")

    lobe_poly = {}
    for n, g in main_tab.groupby("n"):
        poly = lobe_polygon(g)
        if poly is None:
            continue
        tt, lo, hi = poly
        lobe_poly[n] = poly
        ax.fill_between(tt, lo, hi, color=C_MOTT, lw=0, zorder=2)
        ax.plot(tt, lo, "k-", lw=1.1, zorder=5)
        ax.plot(tt, hi, "k-", lw=1.1, zorder=5)
        y = 0.5 * (lo[0] + hi[0]) - 0.1 * (hi[0] - lo[0])
        if a.ymin < y < a.ymax:
            ax.text(0.012 * xmax / 0.3 + 0.005, y, f"d-Mott\n$n={int(n)}$",
                    fontsize=8, va="center", zorder=6)

    # BKT: red cap of the lobe from t_BKT to the tip (d-Luttinger)
    if a.bkt and Path(a.bkt).exists():
        bk = pd.read_csv(a.bkt)
        if "alpha" in bk:
            bk = bk[(bk.alpha == a.alpha) & (bk.r == a.r)]
        for row in bk.itertuples():
            if row.n not in lobe_poly:
                continue
            tt, lo, hi = lobe_poly[row.n]
            m = tt >= row.t_bkt
            tc = np.r_[row.t_bkt, tt[m]]
            lc = np.r_[np.interp(row.t_bkt, tt, lo), lo[m]]
            hc = np.r_[np.interp(row.t_bkt, tt, hi), hi[m]]
            ax.fill_between(tc, lc, hc, color=C_LUTT, lw=0, zorder=3)
            ax.plot([tc[0], tc[0]], [lc[0], hc[0]], color="firebrick", lw=0.9, zorder=5)

    handles = []
    # finite-size check: other L as thin grey lines
    for L2 in Ls:
        if L2 == L:
            continue
        for n, g in tab[tab.L == L2].groupby("n"):
            g = g[g.gap > 0].sort_values("t")
            ax.plot(g.t, g.mu_minus, "-", color="0.55", lw=0.6, zorder=4)
            ax.plot(g.t, g.mu_plus, "-", color="0.55", lw=0.6, zorder=4)
        handles.append(Line2D([], [], color="0.55", lw=0.6, label=f"$L={L2}$"))

    # other models
    sty = [("--", "tab:orange"), (":", "tab:purple"), ("-.", "tab:green")]
    for k, spec in enumerate(a.compare):
        al, rr = (int(x) for x in spec.split(","))
        ls, col = sty[k % 3]
        try:
            t2 = lobes_table(a.csv, al, rr, a.U)
        except SystemExit as e:
            print(e); continue
        t2 = t2[t2.L == t2.L.max()]
        for n, g in t2.groupby("n"):
            poly = lobe_polygon(g)
            if poly is None:
                continue
            ax.plot(poly[0], poly[1], ls, color=col, lw=1, zorder=6)
            ax.plot(poly[0], poly[2], ls, color=col, lw=1, zorder=6)
        ax.axvline(t_star(al, rr), ls=ls, color=col, lw=0.8, zorder=4)
        handles.append(Line2D([], [], ls=ls, color=col, label=rf"$\alpha={al},\ r={rr}$"))
    if handles:
        handles.insert(0, Patch(facecolor=C_MOTT, edgecolor="k", lw=0.8,
                                label=rf"$\alpha={a.alpha},\ r={a.r}$, $L={L}$"))
        ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)

    ax.set_xlim(0, xmax)
    ax.set_ylim(a.ymin, a.ymax)
    ax.set_xlabel(r"Dipole hopping $t/U$")
    ax.set_ylabel(r"Chemical potential $\mu/U$")
    ax.set_title(rf"$\alpha={a.alpha},\ r={a.r}$", fontsize=9)
    fig.tight_layout()

    out = Path(a.out or Path(a.csv).with_name(f"phase_diagram_alpha{a.alpha}_r{a.r}.pdf"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    tab.to_csv(out.with_name(out.stem + "_lobes.csv"), index=False)
    print(f"saved {out}, {out.with_suffix('.png')}, {out.stem}_lobes.csv")


if __name__ == "__main__":
    main()
