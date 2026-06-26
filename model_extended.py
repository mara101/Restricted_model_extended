from tenpy.models.lattice import Chain
from tenpy.networks.site import BosonSite
from tenpy.models.model import CouplingMPOModel


class CorrelatedHoppingBoseModelInfinite(CouplingMPOModel):
    r"""
    Infinite-chain correlated-hopping Bose-Hubbard model with tunable range r
    and power-law damping exponent alpha:

        H = sum_{d=1}^{r}  -t/d^alpha * sum_i ( b^dag_{i-d} b_i b_i b^dag_{i+d} + h.c. )
          + (U/2) sum_i n_i(n_i-1)  -  mu sum_i n_i

    model_params
    ------------
    alpha : int
        Damping exponent.  alpha=0 -> all couplings equal -t.
        alpha=3 with r=3 gives -t (d=1), -t/8 (d=2), -t/27 (d=3).
    r : int
        Maximum hopping range.  r=1 -> nearest-neighbour only.
        r=3 -> includes hops of distance 1, 2, 3.
    t, U, mu, n_max, conserve, L, filling : standard TeNPy parameters.
    """

    def init_sites(self, model_params):
        n_max = int(model_params.get("n_max", 4))
        conserve = model_params.get("conserve", "N")
        filling = float(model_params.get("filling", 0.0))
        return BosonSite(Nmax=n_max, conserve=conserve, filling=filling)

    def init_lattice(self, model_params):
        L = int(model_params.get("L", 4))
        bc_MPS = model_params.get("bc_MPS", "infinite")
        site = self.init_sites(model_params)
        return Chain(L, site, bc="periodic", bc_MPS=bc_MPS)

    def init_terms(self, model_params):
        t = float(model_params.get("t", 1.0))
        U = float(model_params.get("U", 0.0))
        mu = float(model_params.get("mu", 0.0))
        alpha = int(model_params.get("alpha", 0))
        r = int(model_params.get("r", 1))

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu - U / 2.0, u, "N")
            self.add_onsite(U / 2.0, u, "NN")

        conserve = model_params.get("conserve", "N")
        if not (isinstance(conserve, str) and conserve.lower() == "dipole"):
            self.add_coupling(-1e-5, 0, "Bd", 0, "B", 1, plus_hc=True)

        for d in range(1, r + 1):
            coupling = -t / float(d ** alpha) if alpha > 0 else -t
            ops_d = [
                ("Bd", [0], 0),
                ("B", [d], 0),
                ("B", [d], 0),
                ("Bd", [2 * d], 0),
            ]
            category = "CH" if d == 1 else f"SYM{d}"
            self.add_multi_coupling(
                coupling, ops_d, plus_hc=True, switchLR="middle_op", category=category
            )


class CorrelatedHoppingBoseModelFinite(CouplingMPOModel):
    r"""
    Finite-chain correlated-hopping Bose-Hubbard model with tunable range r
    and power-law damping exponent alpha.  Same Hamiltonian as the infinite
    version but with bc_MPS="finite".
    """

    def init_sites(self, model_params):
        n_max = model_params.get("n_max", 3, int)
        filling = model_params.get("filling", 0.5, "real")
        conserve = model_params.get("conserve", "N", str)
        if conserve == "best":
            conserve = "N"
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        return BosonSite(Nmax=n_max, conserve=conserve, filling=filling)

    def init_terms(self, model_params):
        t = model_params.get("t", 1.0, "real_or_array")
        U = model_params.get("U", 0.0, "real_or_array")
        mu = model_params.get("mu", 0.0, "real_or_array")
        alpha = model_params.get("alpha", 0, int)
        r = model_params.get("r", 1, int)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu - U / 2.0, u, "N")
            self.add_onsite(U / 2.0, u, "NN")

        conserve = model_params.get("conserve", "N", str)
        if not (isinstance(conserve, str) and conserve.lower() == "dipole"):
            self.add_coupling(-1e-5, 0, "Bd", 0, "B", 1, plus_hc=True)

        for d in range(1, r + 1):
            coupling = -t / float(d ** alpha) if alpha > 0 else -t
            ops_d = [
                ("Bd", [0], 0),
                ("B", [d], 0),
                ("B", [d], 0),
                ("Bd", [2 * d], 0),
            ]
            category = "CH" if d == 1 else f"SYM{d}"
            self.add_multi_coupling(
                coupling, ops_d, plus_hc=True, switchLR="middle_op", category=category
            )
