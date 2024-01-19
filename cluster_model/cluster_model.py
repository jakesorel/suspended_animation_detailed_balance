import equinox as eqx  # https://github.com/patrick-kidger/equinox
import numpy as np
import jax.numpy as jnp
from jax import jacrev
from scipy.integrate import solve_ivp
from jax import jit
from functools import partial


class OneDCluster(eqx.Module):
    n_clust: int
    i0: int

    """
    In this simplified form, D is treated as a single unit that does not influence A in any way. 
    
    We do get cluster size dependent polarity loss. 
    """

    @partial(jit, static_argnums=(3, 4))
    def f(self, t, y, n_clust, i0, params_start,params_end,tau):
        frac_start = jnp.exp(-t/tau)
        frac_end = 1- frac_start
        params = frac_start*params_start + frac_end*params_end
        D_A, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params

        _y = y.reshape(n_clust * 2 + 1, 2)
        p, b = _y[:n_clust], _y[n_clust:]

        k_AP_p = jnp.column_stack([jnp.zeros_like(p[:, 0]), jnp.ones_like(p[:, 0]) * _k_AP])
        k_AP = jnp.array([0, _k_AP])

        ##note that p is 1-indexed and b is 0-indexed

        ##Cluster dynamics

        i = np.arange(n_clust) + 1

        fw_rate_per = kbind_c * (i / i0) ** (1 / 3)
        rv_rate_per = kunbind_c

        p1 = p[0]
        fw_rate_cluster = p * jnp.expand_dims(fw_rate_per, 1) * p1
        fw_rate_monomers = p * kbind_m * p1 * jnp.expand_dims(i, 1)
        rv_rate_cluster = p * jnp.expand_dims(i, 1) * rv_rate_per
        rv_rate_monomers = p * jnp.expand_dims(i, 1) * kunbind_m
        rv_rate_active = p * jnp.expand_dims(i, 1) * k_AP_p

        fw_rate = jnp.concatenate((fw_rate_monomers[:i0 - 1], fw_rate_cluster[i0 - 1:-1]))
        rv_rate = jnp.concatenate((rv_rate_monomers[1:i0], rv_rate_cluster[i0:]))

        D_P = jnp.concatenate((jnp.array(((0.,),)), jnp.expand_dims((i <= i0) * D_A + (i > i0) * D_C, 1)))

        net_rate = fw_rate - rv_rate

        dtp = jnp.concatenate((jnp.array(((0., 0.),)), net_rate - rv_rate_active[1:])) - jnp.concatenate(
            (net_rate - rv_rate_active[1:], (jnp.array(((0., 0.),))))) \
              + D_P[1:] * (jnp.roll(p, -1, axis=1) - 2 * p + jnp.roll(p, 1, axis=1)) / (L / 2) ** 2

        ## loading and unloading
        A = (p.T * i).sum()
        # A_clust = (p[i0-1:]*i[i0-1:]).sum()
        B = (b.T).sum()

        b0 = b[0]
        b1 = b[1]
        _b = b[1:]
        _bm1 = jnp.concatenate((jnp.array(((0., 0.),)), _b[:-1]))
        _bp1 = jnp.concatenate((_b[1:], (jnp.array(((0., 0.),)))))
        pm1 = jnp.concatenate((jnp.array(((0., 0.),)), p[:-1]))

        A_cyto = A_tot - psi * A.mean()
        B_cyto = B_tot - psi * B.mean()

        # p1 = d_{1,0} + d_{1,1}
        # p1 = d_{1,0} + b1
        # if only d_{1,0} can unload, then the fraction that can unload is
        dtp1 = k_onA * A_cyto - k_offA * (p1 - b1) - net_rate.sum() - k_AP * p1
        dtp = dtp.at[0].add(dtp1)

        k_onB_f = k_onB_c * (k_rel * k_offB_f) / (k_seq * k_offB_c)  ###enforces detailed balance.

        b_load = (k_seq * b0 + k_onB_c * B_cyto) * (jnp.expand_dims(i, 1) * p - _b)
        b_unload = (k_rel + k_offB_c) * _b

        K_plus_i = jnp.expand_dims(kbind_c * (i / i0) ** (1 / 3) * (i >= i0) + kbind_m * (i < i0), 1) * jnp.ones((1, 2))
        K_plus_im1 = jnp.concatenate((jnp.array(((0., 0.),)), K_plus_i[:-1]))
        k_minus_i = jnp.expand_dims(k_offB_c * (i > i0) + k_offB_f * (i <= i0) * (i > 1), 1) * jnp.ones((1, 2))
        k_minus_i_inc_active = k_minus_i + k_AP_p
        k_minus_ip1_inc_active = jnp.concatenate((k_minus_i_inc_active[1:], (jnp.array(((0., 0.),)))))

        b_ad_load = K_plus_im1 * (_bm1 * p1 - pm1 * b1) - K_plus_i * p1 * _b
        b_ad_unload = jnp.expand_dims(i, 1) * (k_minus_ip1_inc_active * _bp1 - k_minus_i_inc_active * _b)

        dtb1_add = - (K_plus_i * b1 * p)[:-1].sum() + (k_minus_i * _b)[1:].sum()

        dtb0 = k_onB_f * B_cyto - k_offB_f * b0 + (k_rel * _b - k_seq * b0 * (jnp.expand_dims(i, 1) * p - _b)).sum()
        dt_b = b_ad_load + b_ad_unload + b_load - b_unload

        dtb = jnp.row_stack((dtb0, dt_b)) \
              + D_P * (jnp.roll(b, -1, axis=1) - 2 * b + jnp.roll(b, 1, axis=1)) / (L / 2) ** 2

        dtb = dtb.at[1].add(dtb1_add)

        ## compile
        dty = jnp.concatenate((dtp.ravel(), dtb.ravel()))

        return dty

    def __call__(self, t, y, params_start,params_end,tau):
        return self.f(t, y, self.n_clust, self.i0, params_start,params_end,tau)


class Simulate:
    def __init__(self, param_dict=None, anoxia_dict=None, t_eval_dict=None):
        self.initialise_param_dicts(param_dict,anoxia_dict)

        self.p_init_pre_polarisation, self.b_init_pre_polarisation, self.y_init_pre_polarisation = [], [], []

        self.model = OneDCluster(self.normoxia_param_dict["n_clust"], self.normoxia_param_dict["i0"])
        self.jac = jit(jacrev(self.model, argnums=[1, ]))

        if t_eval_dict is None:
            self.t_eval_dict = {'pre_polarisation': {"dt": 10, "tfin": 1e5},
                                'polarisation': {"dt": 10, "tfin": 1e3},
                                'anoxia': {"dt": 10, "tfin": 1e6}}
        else:
            assert "pre_polarisation" in t_eval_dict
            assert "polarisation" in t_eval_dict
            assert "anoxia" in t_eval_dict
            for key, val in t_eval_dict.items():
                assert "dt" in val
                assert "tfin" in val
            self.t_eval_dict = t_eval_dict

        self.t_evals = {}
        self.initialise_t_evals()
        self.y_polarisation, self.y_pre_polarisation, self.y_anoxia, self.y = [], [], [], []
        self.t_eval = []
        self.initialise_variables()

    def initialise_param_dicts(self,param_dict,anoxia_dict):

        self.param_names = "D_A,D_C,k_onA,k_offA,k_onB_c,k_offB_f,k_offB_c,kbind_c,kunbind_c,kbind_m,kunbind_m,k_seq,k_rel,A_tot,B_tot,psi,L,k_AP".split(
            ",")

        if param_dict is None:
            self.normoxia_param_dict = {'D_A': 1e-3,
                                        'D_C': 1e-4,  ##NOTE THAT FOR NOW, NO DIFFUSION OF B
                                        'k_onA': 1e-3,
                                        'k_offA': 5e-3,
                                        'k_onB_c': 1e-3,
                                        'k_offB_f': 5e-3,
                                        'k_offB_c': 5e-3,
                                        'kbind': 0.5,
                                        'kunbind': 0.002,
                                        'k_seq': 0.8,
                                        'k_rel': 0.01,
                                        'A_tot': 1.0,
                                        'B_tot': 1.0,
                                        'psi': 0.137,  ##check this!
                                        'L': 173.,
                                        'k_AP': 1e1,
                                        'n_clust': 128,
                                        'i0': 3,
                                        'advection_fraction': 0.99,
                                        "tau_pol":60,
                                        "tau_anox":600}
        else:
            self.normoxia_param_dict = param_dict

        self.normoxia_param_dict["kbind_c"] = self.normoxia_param_dict["kbind"] * self.normoxia_param_dict["i0"]
        self.normoxia_param_dict["kbind_m"] = self.normoxia_param_dict["kbind"]
        self.normoxia_param_dict["kunbind_c"] = self.normoxia_param_dict["kunbind"]
        self.normoxia_param_dict["kunbind_m"] = self.normoxia_param_dict["kunbind"]

        self.normoxia_prepolarisation_param_dict = self.normoxia_param_dict.copy()
        self.normoxia_prepolarisation_param_dict["k_AP"] = 0.

        if anoxia_dict is None:
            self.anoxia_dict = {"k_rel_multiplier": 2.0,
                                "kunbind_multiplier": 5.0,
                                "k_AP_multiplier": 0.0}
        else:
            self.anoxia_dict = anoxia_dict

        self.anoxia_param_dict = self.normoxia_param_dict.copy()
        self.anoxia_param_dict["k_AP"] *= self.anoxia_dict["k_AP_multiplier"]
        self.anoxia_param_dict["kunbind"] *= self.anoxia_dict["kunbind_multiplier"]
        self.anoxia_param_dict["kunbind_c"] *= self.anoxia_dict["kunbind_multiplier"]
        self.anoxia_param_dict["kunbind_m"] *= self.anoxia_dict["kunbind_multiplier"]
        self.anoxia_param_dict["k_rel"] *= self.anoxia_dict["k_rel_multiplier"]

        self.params_pre_polarisation = jnp.array([self.normoxia_prepolarisation_param_dict[nm] for nm in self.param_names])
        self.params_normoxia = jnp.array([self.normoxia_param_dict[nm] for nm in self.param_names])
        self.params_anoxia = jnp.array([self.anoxia_param_dict[nm] for nm in self.param_names])


    def initialise_variables(self):
        self.p_init_pre_polarisation = np.zeros((self.normoxia_param_dict["n_clust"], 2))
        self.b_init_pre_polarisation = np.zeros((self.normoxia_param_dict["n_clust"] + 1, 2))
        self.y_init_pre_polarisation = np.concatenate(
            (self.p_init_pre_polarisation.ravel(), self.b_init_pre_polarisation.ravel()))

    def initialise_t_evals(self):
        for key, val in self.t_eval_dict.items():
            self.t_evals[key] = np.arange(0, val["tfin"], val["dt"])
            self.t_eval_dict[key]["nt"] = len(self.t_evals[key])

    def solve(self, y0, t_eval, params_start,params_end,tau):
        t_span = [t_eval[0], t_eval[-1]]
        sol = solve_ivp(self.model, t_span, y0, method="LSODA", t_eval=t_eval, jac=self.jac,
                        args=(params_start,params_end,tau))
        return sol

    def simulate(self):
        self.y_pre_polarisation = self.solve(self.y_init_pre_polarisation, self.t_evals["pre_polarisation"],
                                             self.params_pre_polarisation,self.params_pre_polarisation,10).y
        _y_pre_polarisation = self.y_pre_polarisation.reshape(self.normoxia_param_dict["n_clust"] * 2 + 1, 2, -1)
        _y_post_advection = jnp.stack((_y_pre_polarisation[:, 0, -1] + _y_pre_polarisation[:, 1, -1] *
                                       self.normoxia_param_dict["advection_fraction"], _y_pre_polarisation[:, 1, -1] * (
                                               1 - self.normoxia_param_dict["advection_fraction"])), axis=1)
        self.y_post_advection = _y_post_advection.reshape(self.y_pre_polarisation.shape[:-1])

        self.y_polarisation = self.solve(self.y_post_advection, self.t_evals["polarisation"], self.params_pre_polarisation,self.params_normoxia,self.normoxia_param_dict["tau_pol"]).y
        self.y_anoxia = self.solve(self.y_polarisation[..., -1], self.t_evals["anoxia"], self.params_normoxia,self.params_anoxia,self.normoxia_param_dict["tau_anox"]).y

        self.y = jnp.column_stack((self.y_pre_polarisation, self.y_polarisation, self.y_anoxia))
        self.t_eval = jnp.concatenate((self.t_evals["pre_polarisation"],
                                       self.t_evals["polarisation"] + self.t_evals["pre_polarisation"][-1],
                                       self.t_evals["anoxia"] + self.t_evals["polarisation"][-1] +
                                       self.t_evals["pre_polarisation"][-1]))

    def polarity(self, X_t):
        return (X_t.max(axis=0) - X_t.min(axis=0)) / (X_t.max(axis=0) + X_t.min(axis=0))

    def extract_values(self, y):
        n_clust = self.normoxia_param_dict["n_clust"]
        _y = y.reshape(n_clust * 2 + 1, 2, -1)
        p_t, b_t = _y[:n_clust], _y[n_clust:]
        C_t = p_t * np.expand_dims(np.expand_dims((1 + np.arange(len(p_t))), axis=1), axis=1)
        C_t = C_t.sum(axis=0)
        B_t = b_t.sum(axis=0)
        m_average = C_t/p_t.sum(axis=0)
        b_frac = b_t[0]/B_t
        d_frac = b_t[1]/B_t
        F_t = b_t[0]
        A_membrane_frac = self.normoxia_param_dict["psi"]*C_t.mean(axis=0)/self.normoxia_param_dict["A_tot"]
        B_membrane_frac = self.normoxia_param_dict["psi"]*B_t.mean(axis=0)/self.normoxia_param_dict["B_tot"]

        sim_values = {"p_t": p_t,
                      "b_t": b_t,
                      "C_t": C_t,
                      "B_t": B_t,
                      "m_average":m_average,
                      "b_frac":b_frac,
                      "d_frac":d_frac,
                      "F_t":F_t,
                      "A_membrane_frac":A_membrane_frac,
                      "B_membrane_frac":B_membrane_frac}
        return sim_values

    def get_polarity(self, sim_values):
        B_pol = self.polarity(sim_values["B_t"])
        C_pol = self.polarity(sim_values["C_t"])
        return {"B_pol": B_pol, "C_pol": C_pol}
