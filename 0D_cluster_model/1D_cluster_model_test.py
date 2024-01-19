import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import grad as jgrad
from jax import hessian, jacrev, jacfwd
from jax import grad, vmap
from jaxopt import GradientDescent
from scipy.integrate import solve_ivp

import time
from jax import jit
import jax
import jax
from functools import partial
from jax.scipy import optimize
from scipy.optimize import minimize


class OneDCluster(eqx.Module):
    n_clust: int

    """
    In this simplified form, D is treated as a single unit that does not influence A in any way. 
    
    We do get cluster size dependent polarity loss. 
    """

    @partial(jit, static_argnums=(3,))
    def f(self, t, y, n_clust, params):
        D_A, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params

        _y = y.reshape(n_clust * 2 + 1, 2)
        p, b = _y[:n_clust], _y[n_clust:]

        k_AP_p = jnp.column_stack([jnp.zeros_like(p[:, 0]), jnp.ones_like(p[:, 0]) * _k_AP])
        k_AP = jnp.array([0, _k_AP])

        ##note that p is 1-indexed and b is 0-indexed

        ##Cluster dynamics

        i0 = 3
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

        #p1 = d_{1,0} + d_{1,1}
        #p1 = d_{1,0} + b1
        #if only d_{1,0} can unload, then the fraction that can unload is
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

    def __call__(self, t, y, params):
        return self.f(t, y, self.n_clust, params)


def solve(model, y0):
    term = diffrax.ODETerm(model)
    solver = diffrax.Tsit5()
    t0 = 0
    t1 = 10000
    dt0 = None
    max_steps = None
    controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    event = diffrax.SteadyStateEvent()
    adjoint = diffrax.ImplicitAdjoint()
    # This combination of event, t1, max_steps, adjoint is particularly
    # natural: we keep integration forever until we hit the event, with
    # no maximum time or number of steps. Backpropagation happens via
    # the implicit function theorem.
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        max_steps=max_steps,
        stepsize_controller=controller,
        discrete_terminating_event=event,
        adjoint=adjoint
    )
    (y1,) = sol.ys
    return y1


param_names = "D_A,D_C,k_onA,k_offA,k_onB_c,k_offB_f,k_offB_c,kbind_c,kunbind_c,kbind_m,kunbind_m,k_seq,k_rel,A_tot,B_tot,psi,L,k_AP".split(
    ",")
param_values_initialise = {'D_A': 1e-3,
                           'D_C': 0e-4,  ##NOTE THAT FOR NOW, NO DIFFUSION OF B
                           'k_onA': 1e-3,
                           'k_offA': 5e-3,
                           'k_onB_c': 1e-3,
                           'k_offB_f': 5e-3,
                           'k_offB_c': 5e-3,
                           'kbind_c': 0.5,
                           'kunbind_c': 1e-3,
                           'kbind_m': 0.05,
                           'kunbind_m': 0.002,
                           'k_seq': 0.8,
                           'k_rel': 0.01,
                           'A_tot': 1.0,
                           'B_tot': 0.0,
                           'psi': 0.137, ##check this!
                           'L': 173.,
                           'k_AP': 0e1,
                           'n_clust': 128}

params_initialise = [param_values_initialise[nm] for nm in param_names]

p0 = np.zeros((param_values_initialise["n_clust"], 2))


b0 = np.zeros((param_values_initialise["n_clust"] + 1, 2))
y0 = np.concatenate((p0.ravel(), b0.ravel()))

model = OneDCluster(param_values_initialise["n_clust"])
# y1 = solve(model,y0)
teval_init = np.arange(0, 1e6, 100)
tspan_init = [teval_init[0], teval_init[-1]]

jac = jit(jacrev(model, argnums=[1, ]))

sol_init = solve_ivp(model, tspan_init, y0, method="LSODA", t_eval=teval_init, jac=jac, args=(params_initialise,))
n_clust = param_values_initialise["n_clust"]
_y = sol_init.y.reshape(param_values_initialise["n_clust"] * 2 + 1, 2, -1)
p_t, b_t = _y[:n_clust], _y[n_clust:]

plt.plot(np.arange(1, n_clust + 1), p_t[:, 0, -1] / p_t[:, 0, -1].max())
plt.plot(np.arange(1, n_clust + 1), p_t[:, 1, -1] / p_t[:, 1, -1].max())

plt.show()

C_t = p_t * np.expand_dims(np.expand_dims((1 + np.arange(len(p_t))), axis=1), axis=1)
C_t = C_t.sum(axis=0)
B_t = b_t.sum(axis=0)

fig, ax = plt.subplots(1, 2)
ax[0].plot(teval_init, B_t.T)
ax[1].plot(teval_init, C_t.T)

# ax.plot(teval,p_t.T)

fig.show()

teval = np.arange(0, 1e5, 10)
tspan = [teval[0], teval[-1]]

param_values = param_values_initialise.copy()
param_values["k_AP"] = 0.0
param_values["kunbind_c"] = 2e-4


params = [param_values[nm] for nm in param_names]

sol1 = solve_ivp(model, tspan, sol_init.y[:, -1], method="LSODA", t_eval=teval, jac=jac, args=(params,))

_y = sol1.y.reshape(param_values_initialise["n_clust"] * 2 + 1, 2, -1)
p_t, b_t = _y[:n_clust], _y[n_clust:]
C_t = p_t * np.expand_dims(np.expand_dims((1 + np.arange(len(p_t))), axis=1), axis=1)
C_t = C_t.sum(axis=0)
B_t = b_t.sum(axis=0)


fig, ax = plt.subplots(1, 3)
ax[0].plot(teval, B_t.T)
ax[1].plot(teval, C_t.T)

# ax.plot(teval,p_t.T)

fig.show()


# plt.plot(np.arange(n_clust + 1), b_t[:, 0, -1] / b_t[:, 0, -1].max())
plt.plot(np.arange(1, n_clust + 1), p_t[:, 0, -1] / p_t[:, 0, -1].max())
plt.show()

def get_polarity(X_t):
    return (X_t.max(axis=0) - X_t.min(axis=0)) / (X_t.max(axis=0) + X_t.min(axis=0))


plt.plot(get_polarity(C_t))
plt.show()

plt.plot(C_t[0])
plt.show()

fig, ax = plt.subplots()
ax.plot(p_t[:, 0, 0])
fig.show()

fig, ax = plt.subplots()
ax.imshow()
