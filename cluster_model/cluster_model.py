import equinox as eqx  # https://github.com/patrick-kidger/equinox
import numpy as np
import jax.numpy as jnp
from jax import jacrev
from scipy.integrate import solve_ivp
from jax import jit
from functools import partial
import jax
jax.config.update('jax_platform_name', 'cpu')



class OneDCluster(eqx.Module):
    n_clust: int
    i0: int

    """
    In this simplified form, D is treated as a single unit that does not influence A in any way. 
    
    We do get cluster size dependent polarity loss. 
    """

    @partial(jit, static_argnums=(3, 4))
    def f(self, t, y, n_clust, i0, params_start,params_end,tau,k_rel_passive):
        ###Define the ratio of parameters between params_start to params_end given an exponential temporal decay with timescale tau
        frac_start = jnp.exp(- t / tau)
        frac_end = 1 - frac_start
        params = frac_start * params_start + frac_end * params_end

        ##Seperate out the parameters
        D_A, D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params

        ##Enforce the single detailed balance constraint
        k_onB_f = k_onB_c * (k_rel_passive * k_offB_f) / (k_seq * k_offB_c)

        ##Unpack the variables
        _y = y.reshape(n_clust * 2 + 1, 2)
        p, b = _y[:n_clust], _y[n_clust:]
        _b = b[1:]
        s_p = jnp.row_stack([jnp.array([0., 0.]), p])  ##dummy variable given len(b)-len(p) = 1
        p1 = p[0]
        b0 = b[0]
        b1 = b[1]

        ##Specify removal of due to pPARs occurs only in the posterior.
        k_AP_p = jnp.column_stack([jnp.zeros_like(b[:, 0]), jnp.ones_like(b[:, 0]) * _k_AP])
        _k_AP_p = k_AP_p[1:]
        ##a k_AP reaction removes a monomer (or a dimer) into the cytoplasm. dimer 'choice' is indiscriminate.

        ##Dummy shifts in the parameters for calculations.
        p_m1 = jnp.row_stack([jnp.array([0., 0.]), p[:-1]])
        p_p1 = jnp.row_stack([p[1:], jnp.array([0., 0.])])
        b_m1 = jnp.row_stack([jnp.array([0., 0.]), b[:-1]])
        b_p1 = jnp.row_stack([b[1:], jnp.array([0., 0.])])
        s_p_m1 = jnp.row_stack([jnp.array([0., 0.]), s_p[:-1]])

        ##i is the cluster size.
        i = np.arange(n_clust + 1)
        _i = i[1:]

        ##Get the total masses across all oligomeric states for A (PAR3) and B
        A = (p.T * _i).sum()
        B = (b).sum()

        ##Calculate the cytoplasmic concentrations given the above.
        A_cyto = A_tot - psi * A.mean()
        B_cyto = B_tot - psi * B.mean()

        ##Specify the binding and unbinding rates
        Kp_i = (i >= 1) * (i < n_clust) * ((i * kbind_m) * (i <= i0) + (i / i0) ** (2 / 3) * kbind_c * (i > i0))
        Km_i = (i > 1) * ((i * kunbind_m) * (i <= i0) + i * kunbind_c * (i > i0))
        km_i = (i > 1) * ((kunbind_m) * (i <= i0) + kunbind_c * (i > i0))  ##Note that Km_i = km_i * i

        ##Dummy shifts of the above parameters to aid calculations.
        _Kp_i = Kp_i[1:]
        _Km_i = Km_i[1:]
        Kp_i_m1 = jnp.concatenate([jnp.array([0.]), Kp_i[:-1]])
        Km_i_p1 = jnp.concatenate([Km_i[1:], jnp.array([0.])])
        km_i_p1 = jnp.concatenate([km_i[1:], jnp.array([0.])])
        k_AP_p_p1 = jnp.row_stack([k_AP_p[1:], jnp.array([0., 0.])])
        _k_AP_p_p1 = k_AP_p_p1[1:]
        _Kp_i_m1 = Kp_i_m1[1:]
        _Km_i_p1 = Km_i_p1[1:]

        ##Calculate the rate of change of each of the oligomeric states, from the A-perspective
        dtpi = (jnp.expand_dims(_Kp_i_m1, 1) * jnp.expand_dims(p1, 0) * p_m1 - jnp.expand_dims(_Kp_i,
                                                                                               1) * jnp.expand_dims(p1,
                                                                                                                    0) * p) \
               + ((jnp.expand_dims(_Km_i_p1, 1) + _k_AP_p_p1 * jnp.expand_dims(i[1:], 1)) * p_p1 - (
                    jnp.expand_dims(_Km_i, 1) + _k_AP_p * jnp.expand_dims(i[:-1], 1)) * p)

        ##Include mass conservation terms for PAR3 monomers, and also binding and unbinding from cytoplasm
        ##Note there, k_AP is absent, as we assume that all pPAR phosphorylated protein goes directly to the cytoplasm.
        dtp1_add = (jnp.expand_dims(_Km_i, 1) * p - jnp.expand_dims(_Kp_i, 1) * jnp.expand_dims(p1, 0) * p).sum(axis=0) \
                   + k_onA * A_cyto - k_offA * (p1 - b1)

        ##Combine the additional terms.
        dtpi = dtpi.at[0].add(dtp1_add)

        ##Calculate the rate of change of B-mass of each oligomeric state
        dtbi = jnp.expand_dims(k_seq * b0 + k_onB_c * B_cyto, 0) * (
                jnp.expand_dims(i, 1) * s_p - jnp.expand_dims(i != 0, 1) * b) \
               - (k_rel + k_offB_c) * b * jnp.expand_dims(i != 0, 1) \
               + jnp.expand_dims(Kp_i_m1, 1) * (b_m1 * p1 + s_p_m1 * b1) - jnp.expand_dims(Kp_i, 1) * (p1 * b) \
               + jnp.expand_dims(i, 1) * (
                       (jnp.expand_dims(km_i_p1, 1) + k_AP_p_p1) * b_p1 - (jnp.expand_dims(km_i, 1) + k_AP_p) * b)

        ##Likewise, deal with mass conservation
        dtb0_add = (k_rel * b * jnp.expand_dims(i != 0, 1)
                    - jnp.expand_dims(k_seq * b0, 0) * (
                                jnp.expand_dims(i, 1) * s_p - jnp.expand_dims(i != 0, 1) * b)).sum(
            axis=0) \
                   + k_onB_f * B_cyto - k_offB_f * b0
        dtb1_add = (jnp.expand_dims(km_i, 1) * b - jnp.expand_dims(Kp_i, 1) * (b1 * s_p)).sum(axis=0)
        ##Note, as above, kAP directly leads to removal into cytoplasm.

        dtbi = dtbi.at[0].add(dtb0_add)
        dtbi = dtbi.at[1].add(dtb1_add)

        ##Calculate diffusion coefficient as a function of size.
        ##Here, len(D_P) = n_clust + 1, where D_P[0] = D_B for simplicity.
        D_P = jnp.concatenate(
            (jnp.expand_dims(jnp.array(((D_B),)), 1), jnp.expand_dims((_i <= i0) * D_A + (_i > i0) * D_C, 1)))

        ##Apply finite difference diffusion
        dtpi += D_P[1:] * (jnp.roll(p, -1, axis=1) - 2 * p + jnp.roll(p, 1, axis=1)) / (L / 2) ** 2
        dtbi += D_P * (jnp.roll(b, -1, axis=1) - 2 * b + jnp.roll(b, 1, axis=1)) / (L / 2) ** 2

        ##Assemble the variables.
        dty = jnp.concatenate((dtpi.ravel(), dtbi.ravel()))

        # frac_start = jnp.exp(-t/tau)
        # frac_end = 1- frac_start
        # params = frac_start*params_start + frac_end*params_end
        # D_A,D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params
        #
        # _y = y.reshape(n_clust * 2 + 1, 2)
        # p, b = _y[:n_clust], _y[n_clust:]
        #
        # k_AP_p = jnp.column_stack([jnp.zeros_like(p[:, 0]), jnp.ones_like(p[:, 0]) * _k_AP])
        # k_AP = jnp.array([0, _k_AP])
        #
        # ##note that p is 1-indexed and b is 0-indexed
        #
        # ##Cluster dynamics
        #
        # i = np.arange(n_clust) + 1
        #
        # fw_rate_per = kbind_c * (i / i0) ** (1 / 3)
        # rv_rate_per = kunbind_c
        #
        # p1 = p[0]
        # fw_rate_cluster = p * jnp.expand_dims(fw_rate_per, 1) * p1
        # fw_rate_monomers = p * kbind_m * p1 * jnp.expand_dims(i, 1)
        # rv_rate_cluster = p * jnp.expand_dims(i, 1) * rv_rate_per
        # rv_rate_monomers = p * jnp.expand_dims(i, 1) * kunbind_m
        # rv_rate_active = p * jnp.expand_dims(i, 1) * k_AP_p
        #
        # fw_rate = jnp.concatenate((fw_rate_monomers[:i0 - 1], fw_rate_cluster[i0 - 1:-1]))
        # rv_rate = jnp.concatenate((rv_rate_monomers[1:i0], rv_rate_cluster[i0:]))
        #
        # D_P = jnp.concatenate((jnp.expand_dims(jnp.array(((D_B),)),1), jnp.expand_dims((i <= i0) * D_A + (i > i0) * D_C, 1)))
        #
        # net_rate = fw_rate - rv_rate
        #
        # dtp = jnp.concatenate((jnp.array(((0., 0.),)), net_rate - rv_rate_active[1:])) - jnp.concatenate(
        #     (net_rate - rv_rate_active[1:], (jnp.array(((0., 0.),))))) \
        #       + D_P[1:] * (jnp.roll(p, -1, axis=1) - 2 * p + jnp.roll(p, 1, axis=1)) / (L / 2) ** 2
        #
        #     ## loading and unloading
        # A = (p.T * i).sum()
        # # A_clust = (p[i0-1:]*i[i0-1:]).sum()
        # B = (b.T).sum()
        #
        # b0 = b[0]
        # b1 = b[1]
        # _b = b[1:]
        # _bm1 = jnp.concatenate((jnp.array(((0., 0.),)), _b[:-1]))
        # _bp1 = jnp.concatenate((_b[1:], (jnp.array(((0., 0.),)))))
        # pm1 = jnp.concatenate((jnp.array(((0., 0.),)), p[:-1]))
        #
        # A_cyto = A_tot - psi * A.mean()
        # B_cyto = B_tot - psi * B.mean()
        #
        # # p1 = d_{1,0} + d_{1,1}
        # # p1 = d_{1,0} + b1
        # # if only d_{1,0} can unload, then the fraction that can unload is
        # dtp1 = k_onA * A_cyto - k_offA * (p1 - b1) - net_rate.sum(axis=0) - k_AP * p1
        # dtp = dtp.at[0].add(dtp1)
        #
        # k_onB_f = k_onB_c * (k_rel_passive * k_offB_f) / (k_seq * k_offB_c)  ###enforces detailed balance.
        #
        # b_load = (k_seq * b0 + k_onB_c * B_cyto) * (jnp.expand_dims(i, 1) * p - _b)
        # b_unload = (k_rel + k_offB_c) * _b
        #
        # K_plus_i = jnp.expand_dims(kbind_c * (i / i0) ** (1 / 3) * (i > i0) + i*kbind_m * (i <= i0), 1) * jnp.ones((1, 2))
        # K_plus_im1 = jnp.concatenate((jnp.array(((0., 0.),)), K_plus_i[:-1]))
        # k_minus_i = jnp.expand_dims(k_offB_c * (i > i0)*i + k_offB_f * (i <= i0) * (i > 1) * i, 1) * jnp.ones((1, 2))
        # k_minus_i_inc_active = k_minus_i + k_AP_p
        # k_minus_ip1_inc_active = jnp.concatenate((k_minus_i_inc_active[1:], (jnp.array(((0., 0.),)))))
        #
        # # b_ad_load = K_plus_im1 * (_bm1 * p1 - pm1 * b1) - K_plus_i * p1 * _b
        # b_ad_load = K_plus_im1 * (_bm1 * p1 + pm1 * b1) - K_plus_i * p1 * _b
        #
        # b_ad_unload = jnp.expand_dims(i, 1) * (k_minus_ip1_inc_active * _bp1 - k_minus_i_inc_active * _b)
        #
        # dtb1_add = - (K_plus_i * b1 * p)[:-1].sum() + (k_minus_i * _b)[1:].sum()
        #
        # dtb0 = k_onB_f * B_cyto - k_offB_f * b0 + (k_rel * _b - k_seq * b0 * (jnp.expand_dims(i, 1) * p - _b)).sum()
        # dt_b = b_ad_load + b_ad_unload + b_load - b_unload
        #
        # dtb = jnp.row_stack((dtb0, dt_b)) \
        #       + D_P * (jnp.roll(b, -1, axis=1) - 2 * b + jnp.roll(b, 1, axis=1)) / (L / 2) ** 2
        #
        # dtb = dtb.at[1].add(dtb1_add)
        #
        # ## compile
        # dty = jnp.concatenate((dtp.ravel(), dtb.ravel()))

        return dty

    def __call__(self, t, y, params_start,params_end,tau,k_rel_passive):
        return self.f(t, y, self.n_clust, self.i0, params_start,params_end,tau,k_rel_passive)


class Simulate:
    def __init__(self, param_dict=None, anoxia_dict=None, t_eval_dict=None):
        self.initialise_param_dicts(param_dict,anoxia_dict)

        self.p_init_pre_polarisation, self.b_init_pre_polarisation, self.y_init_pre_polarisation = [], [], []

        self.model = OneDCluster(self.normoxia_param_dict["n_clust"], self.normoxia_param_dict["i0"])
        self.jac = jit(jacrev(self.model, argnums=1))

        if t_eval_dict is None:
            self.t_eval_dict = {'pre_polarisation': {"dt": 10, "tfin": 1e5},
                                'polarisation': {"dt": 10, "tfin": 1e3},
                                'NEBD': {"dt": 10, "tfin": 1e3},
                                'anoxia': {"dt": 10, "tfin": 1e6}}
        else:
            assert "pre_polarisation" in t_eval_dict
            assert "polarisation" in t_eval_dict
            assert "NEBD" in t_eval_dict
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

        self.param_names = "D_A,D_B,D_C,k_onA,k_offA,k_onB_c,k_offB_f,k_offB_c,kbind_c,kunbind_c,kbind_m,kunbind_m,k_seq,k_rel,A_tot,B_tot,psi,L,k_AP".split(
            ",")
        assert param_dict is not None

        self.normoxia_param_dict = param_dict

        # self.normoxia_param_dict["kbind_c"] = self.normoxia_param_dict["kbind"] * self.normoxia_param_dict["i0"]
        # self.normoxia_param_dict["kbind_m"] = self.normoxia_param_dict["kbind"]
        self.normoxia_param_dict["kunbind_c"] = self.normoxia_param_dict["kunbind"]
        self.normoxia_param_dict["kunbind_m"] = self.normoxia_param_dict["kunbind"]

        self.postNEBD_param_dict = self.normoxia_param_dict.copy()
        self.postNEBD_param_dict["kunbind_c"] = self.postNEBD_param_dict["kunbind_postNEBD"]
        self.postNEBD_param_dict["kunbind_m"] = self.postNEBD_param_dict["kunbind_postNEBD"]

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
        self.anoxia_param_dict["kunbind"] = self.anoxia_dict["kunbind_anoxia"]
        self.anoxia_param_dict["kunbind_c"] = self.anoxia_dict["kunbind_anoxia"]
        self.anoxia_param_dict["kunbind_m"] = self.anoxia_dict["kunbind_anoxia"]
        self.anoxia_param_dict["k_rel"] *= self.anoxia_dict["k_rel_multiplier"]

        self.params_pre_polarisation = jnp.array([self.normoxia_prepolarisation_param_dict[nm] for nm in self.param_names])
        self.params_normoxia = jnp.array([self.normoxia_param_dict[nm] for nm in self.param_names])
        self.params_postNEBD = jnp.array([self.postNEBD_param_dict[nm] for nm in self.param_names])

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

    def solve(self, y0, t_eval, params_start,params_end,tau,method="LSODA"):
        t_span = [t_eval[0], t_eval[-1]]
        sol = solve_ivp(self.model, t_span, y0, method=method, t_eval=t_eval, jac=self.jac,
                        args=(params_start,params_end,tau,self.anoxia_param_dict["k_rel"]))
        return sol

    def simulate(self,method="LSODA",pre_polarisation_only=False):
        self.y_pre_polarisation = self.solve(self.y_init_pre_polarisation, self.t_evals["pre_polarisation"],
                                             self.params_pre_polarisation,self.params_pre_polarisation,10,method=method).y
        if not pre_polarisation_only:
            _y_pre_polarisation = self.y_pre_polarisation.reshape(self.normoxia_param_dict["n_clust"] * 2 + 1, 2, -1)
            _y_post_advection = jnp.stack((_y_pre_polarisation[:, 0, -1] + _y_pre_polarisation[:, 1, -1] *
                                           self.normoxia_param_dict["advection_fraction"], _y_pre_polarisation[:, 1, -1] * (
                                                   1 - self.normoxia_param_dict["advection_fraction"])), axis=1)
            self.y_post_advection = _y_post_advection.reshape(self.y_pre_polarisation.shape[:-1])

            self.y_polarisation = self.solve(self.y_post_advection, self.t_evals["polarisation"], self.params_pre_polarisation,self.params_normoxia,self.normoxia_param_dict["tau_pol"]).y
            self.y_postNEBD = None
            if self.t_evals["NEBD"].max()>0:
                self.y_postNEBD = self.solve(self.y_polarisation[...,-1], self.t_evals["NEBD"], self.params_normoxia,self.params_postNEBD,self.normoxia_param_dict["tau_NEBD"]).y
                self.y_anoxia = self.solve(self.y_postNEBD[..., -1], self.t_evals["anoxia"], self.params_postNEBD,self.params_anoxia,self.normoxia_param_dict["tau_anox"]).y
            else:
                self.y_postNEBD = np.zeros((0,self.y_polarisation.shape[1]))
                self.y_anoxia = self.solve(self.y_polarisation[..., -1], self.t_evals["anoxia"], self.params_normoxia,self.params_anoxia,self.normoxia_param_dict["tau_anox"]).y



            self.y = jnp.column_stack((self.y_pre_polarisation, self.y_polarisation, self.y_postNEBD,self.y_anoxia))
            self.t_eval = jnp.concatenate((self.t_evals["pre_polarisation"],
                                           self.t_evals["polarisation"] + self.t_evals["pre_polarisation"][-1],
                                           self.t_evals["NEBD"] + self.t_evals["polarisation"][-1] + self.t_evals["pre_polarisation"][-1],
                                           self.t_evals["anoxia"] +self.t_evals["NEBD"][-1] + self.t_evals["polarisation"][-1] + self.t_evals["pre_polarisation"][-1]))
        else:
            self.y = self.y_polarisation
            self.t_eval = self.t_evals["pre_polarisation"]

    def simulate_pre_and_post(self, method="LSODA"):
        self.y_pre_polarisation = self.solve(self.y_init_pre_polarisation, self.t_evals["pre_polarisation"],
                                             self.params_pre_polarisation, self.params_pre_polarisation, 10,
                                             method=method).y
        _y_pre_polarisation = self.y_pre_polarisation.reshape(self.normoxia_param_dict["n_clust"] * 2 + 1, 2, -1)
        _y_post_advection = jnp.stack((_y_pre_polarisation[:, 0, -1] + _y_pre_polarisation[:, 1, -1] *
                                       self.normoxia_param_dict["advection_fraction"],
                                       _y_pre_polarisation[:, 1, -1] * (
                                               1 - self.normoxia_param_dict["advection_fraction"])), axis=1)
        self.y_post_advection = _y_post_advection.reshape(self.y_pre_polarisation.shape[:-1])

        self.y_polarisation = self.solve(self.y_post_advection, self.t_evals["polarisation"],
                                         self.params_pre_polarisation, self.params_normoxia,
                                         self.normoxia_param_dict["tau_pol"]).y
        self.y_postNEBD = self.solve(self.y_polarisation[..., -1], self.t_evals["NEBD"], self.params_normoxia,
                                     self.params_postNEBD, self.normoxia_param_dict["tau_NEBD"]).y
        self.y_anoxia_postNEBD = self.solve(self.y_postNEBD[..., -1], self.t_evals["anoxia"], self.params_postNEBD,
                                   self.params_anoxia, self.normoxia_param_dict["tau_anox"]).y
        self.y_anoxia_preNEBD = self.solve(self.y_polarisation[..., -1], self.t_evals["anoxia"], self.params_normoxia,
                                   self.params_anoxia, self.normoxia_param_dict["tau_anox"]).y

        self.y_preNEBD_total = jnp.column_stack((self.y_pre_polarisation, self.y_polarisation, self.y_anoxia_preNEBD))
        self.y_postNEBD_total = jnp.column_stack((self.y_pre_polarisation, self.y_polarisation, self.y_postNEBD, self.y_anoxia_postNEBD))

        self.t_eval_preNEBD = jnp.concatenate((self.t_evals["pre_polarisation"],
                                       self.t_evals["polarisation"] + self.t_evals["pre_polarisation"][-1],
                                       self.t_evals["anoxia"] +
                                       self.t_evals["polarisation"][-1] + self.t_evals["pre_polarisation"][-1]))
        self.t_eval_postNEBD = jnp.concatenate((self.t_evals["pre_polarisation"],
                                       self.t_evals["polarisation"] + self.t_evals["pre_polarisation"][-1],
                                       self.t_evals["NEBD"] + self.t_evals["polarisation"][-1] +
                                       self.t_evals["pre_polarisation"][-1],
                                       self.t_evals["anoxia"] + self.t_evals["NEBD"][-1] +
                                       self.t_evals["polarisation"][-1] + self.t_evals["pre_polarisation"][-1]))


    def polarity(self, X_t):
        return (X_t.max(axis=0) - X_t.min(axis=0)) / (X_t.max(axis=0) + X_t.min(axis=0)+1e-17)

    def extract_values(self, y):
        n_clust = self.normoxia_param_dict["n_clust"]
        _y = y.reshape(n_clust * 2 + 1, 2, -1)
        p_t, b_t = _y[:n_clust], _y[n_clust:]
        C_t = p_t * np.expand_dims(np.expand_dims((1 + np.arange(len(p_t))), axis=1), axis=1)
        C_t = C_t.sum(axis=0)
        B_t = b_t.sum(axis=0)
        m_average = C_t/(p_t.sum(axis=0)+1e-17)
        b_frac = b_t[0]/(B_t + 1e-17)
        d_frac = b_t[1]/(B_t + 1e-17)
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
