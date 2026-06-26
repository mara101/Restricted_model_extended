# Correlated-Hopping Bose–Hubbard Model — DMRG Toolkit

A small set of Python tools I use, as an undergraduate research project in
theoretical physics, to study an **extended correlated-hopping (dipole-conserving)
Bose–Hubbard chain** with DMRG. The code is built on top of
[TeNPy](https://tenpy.readthedocs.io/) (Tensor Network Python).

The toolkit lets me:

1. Define the model (finite or infinite chain) with a tunable hopping range and
   power-law damping.
2. Run DMRG ground-state simulations from a plain-text list of parameter points.
3. Compute **charge** and **dipole** gaps by running DMRG in several particle /
   dipole sectors.
4. Extract a family of correlation functions (and density profiles) from the
   saved states, save them as CSV, and plot them.
5. Interactively browse the resulting plots in a Jupyter notebook.

---

## The model

The Hamiltonian is a Bose–Hubbard chain whose elementary move is a
**correlated (dipole-conserving) hop** — a particle hops left at one site while a
particle hops right a distance away, so that the total dipole moment is conserved:

```
H = Σ_{d=1}^{r}  -t/d^α  Σ_i ( b†_{i-d} b_i b_i b†_{i+d} + h.c. )
  + (U/2) Σ_i n_i (n_i - 1)
  -  μ Σ_i n_i
```

Parameters:

| Symbol  | Meaning                                                                 |
|---------|-------------------------------------------------------------------------|
| `t`     | correlated-hopping amplitude                                            |
| `r`     | maximum hopping range (`r=1` → nearest-neighbour only)                  |
| `α`     | power-law damping exponent (`α=0` → all ranges equal; `α=3` → `1/d³`)   |
| `U`     | on-site interaction strength                                            |
| `μ`     | chemical potential                                                      |
| `n_max` | maximum occupation per site                                             |
| `L`     | number of sites (length of the unit cell for the infinite chain)        |

The model is implemented for both geometries in
[model_extended.py](model_extended.py):

- `CorrelatedHoppingBoseModelInfinite` — `bc_MPS="infinite"` (iMPS / iDMRG)
- `CorrelatedHoppingBoseModelFinite`   — `bc_MPS="finite"` (finite DMRG)

A symmetry can be conserved via the `Conserve` field: `dipole` (dipole moment),
`N` (particle number), `parity`, or `None`.

---

## Repository layout

```
.
├── model_extended.py                  # TeNPy model definitions (finite + infinite)
├── run_simulations.py                 # ground-state DMRG runner (reads a text list)
├── charge_and_dipole_gaps_calculator_V3.py  # charge / dipole gap calculator
├── correlators_extractor.py           # compute & plot correlators from .mps files
├── browse_correlators_immages.ipynb   # interactive plot browser (ipywidgets)
├── delete_simulations.py              # helper to clean up runs + their outputs
├── rename_files.py                    # one-shot migration script (legacy → α/r naming)
│
├── simulations_input.txt              # input list for run_simulations.py
├── gaps_input.txt                     # input list for the gap calculator
├── delete_input.txt                   # input list for delete_simulations.py
│
├── mps_files_finite/                  # finite-chain outputs   (generated, git-ignored)
├── mps_files_infinite/                # infinite-chain outputs (generated, git-ignored)
└── mps_dipole_gap/                    # per-sector states for gap runs (generated, git-ignored)
```

Each output dataset folder is organised as:

```
mps_files_<geom>/
├── mps_files/           # saved DMRG ground states (.mps pickles)
├── correlators_csv/     # extracted correlators / profiles (.csv)
└── correlators_plots/   # plots of the above (.png)
```

> **Note on what is published.** The heavy generated artifacts — the `.mps` state
> files and the `.png` plots — are **not** committed (see `.gitignore`). They are
> reproducible from the input files and the scripts. Only the code and the small
> text input files are tracked.

---

## Requirements

- Python ≥ 3.9
- [TeNPy](https://tenpy.readthedocs.io/), NumPy, Matplotlib, pandas
- For the notebook: Jupyter and `ipywidgets`

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Usage

All scripts are driven by small comma-separated text files. Every input row uses
the **same 12-field format**:

```
geometry, L, n_max, N_IN_UNIT_CELL, U, mu, Conserve, t, chi_max, svd_min, alpha, r
```

| Field            | Description                                                   |
|------------------|--------------------------------------------------------------|
| `geometry`       | `finite` or `infinite`                                       |
| `L`              | number of sites (unit-cell length for infinite)             |
| `n_max`          | max occupation per site                                      |
| `N_IN_UNIT_CELL` | total particle number in the ground-state sector            |
| `U`              | on-site interaction                                          |
| `mu`             | chemical potential                                           |
| `Conserve`       | `dipole`, `N`, `parity`, or `None`                          |
| `t`              | hopping amplitude                                            |
| `chi_max`        | maximum MPS bond dimension                                   |
| `svd_min`        | SVD truncation threshold                                     |
| `alpha`          | damping exponent                                            |
| `r`              | maximum hopping range                                        |

Lines starting with `#` and blank lines are ignored; a header line whose first
field is `geometry` (or `model`) is skipped automatically.

### 1. Run ground-state DMRG

List the points in [simulations_input.txt](simulations_input.txt), then:

```bash
python run_simulations.py
# or
python run_simulations.py --simulations_file simulations_input.txt --out_dir .
```

For each row it runs DMRG (with a χ-ramp schedule, a subspace-expansion mixer,
a wall-clock budget, and convergence checks) and, **only if the run converged**,
saves the ground state into `mps_files_finite/` or `mps_files_infinite/`. A
[manifest.csv](manifest.csv) summarising convergence status, sweep counts and
the achieved bond dimension is written to the output root. Already-computed
points are detected and skipped.

### 2. Compute charge and dipole gaps

List the points in [gaps_input.txt](gaps_input.txt), then:

```bash
python charge_and_dipole_gaps_calculator_V3.py
```

For each point it runs DMRG in the relevant sectors (`GS`, `N±1`, and `P±1`,
i.e. one extra/missing particle or a shifted dipole) and computes:

- **Charge gap**  `Δ_c = E(N+1) + E(N−1) − 2 E(GS)`
- **Dipole gap**  `Δ_d`, with the definition selectable via `--dipole_gap_def`:
  - `simple`    → `E(P+1) − E(GS)` (default)
  - `symmetric` → `2·[E(P+1) − E(GS)]`
  - `paper`     → `[E(P+1) − E(GS)] − [E(GS) − E(P−1)]`

Sector states are cached in `mps_dipole_gap/` and reused on later runs. Results
(energies, gaps and per-sector convergence diagnostics) are written to a CSV
inside that folder. Useful flags: `--mps_dir`, `--results`, `--max_hours`,
`--chi`/`--mixer`/`--sweep` tuning options (see `--help`).

### 3. Extract and plot correlators

After ground states exist, scan recursively for `.mps` files and compute the
correlators:

```bash
python correlators_extractor.py
```

For every state it computes and saves (as CSV, then plots as PNG):

- on-site density profile `⟨n_i⟩` and dipole-density profile `⟨ñ_i⟩`
- dipole correlator `⟨d†_i d_{i+r}⟩`
- connected dipole-density correlator
- dipole-current correlator
- single-particle correlator `⟨b†_i b_{i+r}⟩`
- density–density correlator `⟨n_i n_{i+r}⟩_c`

plus the Fourier transform of each. CSVs are reused if already present (no
recomputation). For finite chains a boundary region is skipped to avoid edge
effects.

### 4. Browse the plots

Open [browse_correlators_immages.ipynb](browse_correlators_immages.ipynb) in
Jupyter. It indexes every plot under the `correlators_plots/` folders by parsing
the filenames and gives interactive dropdowns (`ipywidgets`) to filter by model,
geometry, parameters and correlator type.

### 5. Clean up runs

To remove the files associated with a set of runs, list them in
[delete_input.txt](delete_input.txt) and run:

```bash
python delete_simulations.py
```

It locates the matching `.mps` files, CSVs and images, then interactively asks
whether to delete everything or only the correlator outputs, with a final
confirmation before touching anything.

---

## Output filename convention

States and plots encode their parameters in the filename, e.g.

```
fMPS_alpha1_r3_U4_mu1_t0p25_consdipole_chi700_L160_Nuc100_nmax4.mps
```

- `fMPS`/`iMPS` — finite / infinite geometry
- `alpha1_r3`   — damping exponent and hopping range
- numeric tags use `p` for the decimal point and `m` for a minus sign
  (`t0p25` = `t = 0.25`, `mum1` = `μ = -1`)

---

## License

This is research code shared for transparency and reproducibility. Feel free to
read and adapt it; if you use it, a mention is appreciated.
