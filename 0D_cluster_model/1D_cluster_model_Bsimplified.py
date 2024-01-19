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
from jax import hessian, jacrev,jacfwd
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



    @partial(jit, static_argnums=(3,))
    def f(self, t, y,n_clust,params):
        D_A,D_C,k_onA,k_offA,k_onB_c,k_offB_f,k_offB_c,kbind_c,kunbind_c,kbind_m,kunbind_m,k_seq,k_rel,A_tot,B_tot,psi,L,_k_AP = params

        _y = y.reshape(n_clust+2,2)
        B, D, p = _y[0],_y[1],_y[2:]
        k_AP_p = jnp.column_stack([jnp.zeros_like(p[:,0]),jnp.ones_like(p[:,0])*_k_AP])
        k_AP = jnp.array([0,_k_AP])



        ##Cluster dynamics

        i0 = 3
        i = np.arange(n_clust)+1


        fw_rate_per = kbind_c*(i/i0)**(1/3)
        rv_rate_per = kunbind_c

        p1 = p[0]
        fw_rate_cluster = p*jnp.expand_dims(fw_rate_per,1)*p1
        fw_rate_monomers = p*kbind_m*p1*jnp.expand_dims(i,1)
        rv_rate_cluster = p*jnp.expand_dims(i,1)*rv_rate_per
        rv_rate_monomers = p*jnp.expand_dims(i,1)*kunbind_m
        rv_rate_active = p*jnp.expand_dims(i,1)*k_AP_p

        fw_rate = jnp.concatenate((fw_rate_monomers[:i0-1],fw_rate_cluster[i0-1:-1]))
        rv_rate = jnp.concatenate((rv_rate_monomers[1:i0],rv_rate_cluster[i0:]))

        D_P = jnp.expand_dims((i<=i0)*D_A + (i>i0)*D_C,1)

        net_rate = fw_rate - rv_rate

        dtp = jnp.concatenate((jnp.array(((0.,0.),)),net_rate-rv_rate_active[1:])) - jnp.concatenate((net_rate-rv_rate_active[1:],(jnp.array(((0.,0.),))))) \
              + D_P*(jnp.roll(p, -1,axis=1) - 2 * p + jnp.roll(p, 1,axis=1)) / (L/2) ** 2



        ## loading and unloading
        A = (p.T*i).sum()
        # A_clust = (p[i0-1:]*i[i0-1:]).sum()

        A_cyto = A_tot - psi*(A.mean())
        B_cyto = B_tot - psi*(B + D).mean()

        dtp1 = k_onA*A_cyto-k_offA*p1 -net_rate.sum() - k_AP*p1
        dtp = dtp.at[0].add(dtp1)

        k_onB_f = k_onB_c*(k_rel*k_offB_f)/(k_seq*k_offB_c) ###enforces detailed balance.

        dtB = k_onB_f*B_cyto - k_offB_f*B - k_seq*B*A + k_rel*D - k_AP*D
        dtD = k_onB_c*B_cyto*A - k_offB_c*D + k_seq*B*A - k_rel*D - k_AP*D

        ## compile
        dty = jnp.concatenate((dtB.ravel(),dtD.ravel(),dtp.ravel()))


        return dty

    def __call__(self, t, y,params):
        return self.f(t, y,self.n_clust,params)

def solve(model,y0):
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



param_names = "D_A,D_C,k_onA,k_offA,k_onB_c,k_offB_f,k_offB_c,kbind_c,kunbind_c,kbind_m,kunbind_m,k_seq,k_rel,A_tot,B_tot,psi,L,k_AP".split(",")
param_values_initialise = {'D_A':1e-3,
                'D_C':1e-4, ##NOTE THAT FOR NOW, NO DIFFUSION OF B
                'k_onA':1e-3,
                'k_offA':5e-3,
                'k_onB_c':1e-3,
                'k_offB_f':5e-3,
                'k_offB_c':5e-3,
                'kbind_c':0.5,
                'kunbind_c':1e-3,
                'kbind_m':0.05,
                'kunbind_m':0.002,
                'k_seq':0.3,
                'k_rel':0.01,
                'A_tot':1.0,
                'B_tot':10.0,
                'psi':0.137,
                'L':173.,
                'k_AP':1e1,
                'n_clust':300}

params_initialise = [param_values_initialise[nm] for nm in param_names]

B0 = np.array([0.2,1e-7])
D0 = np.array([0.02,1e-7])
p0 = np.zeros((param_values_initialise["n_clust"],2))
p0[0,0] = 1e-3
y0 = np.concatenate((B0.ravel(),D0.ravel(),p0.ravel()))

model = OneDCluster(param_values_initialise["n_clust"])
# y1 = solve(model,y0)
teval_init = np.arange(0,1e6,100)
tspan_init = [teval_init[0],teval_init[-1]]

jac = jit(jacrev(model,argnums=[1,]))

sol_init = solve_ivp(model,tspan_init,y0,method="LSODA",t_eval=teval_init,jac=jac,args=(params_initialise,))

_y = sol_init.y.reshape(param_values_initialise["n_clust"] + 2, 2,-1)
B_t, D_t, p_t = _y[0], _y[1], _y[2:]
C_t = p_t*np.expand_dims(np.expand_dims((1+np.arange(len(p_t))),axis=1),axis=1)
C_t = C_t.sum(axis=0)

fig, ax = plt.subplots(1,3)
ax[0].plot(teval_init,B_t.T)
ax[1].plot(teval_init,D_t.T)
ax[2].plot(teval_init,C_t.T)

# ax.plot(teval,p_t.T)

fig.show()

teval = np.arange(0,1e5,10)
tspan = [teval[0],teval[-1]]

param_values = param_values_initialise.copy()
param_values["k_AP"] = 0.0

params = [param_values[nm] for nm in param_names]

sol1 = solve_ivp(model,tspan,sol_init.y[:,-1],method="LSODA",t_eval=teval,jac=jac,args=(params,))



_y = sol1.y.reshape(param_values["n_clust"] + 2, 2,-1)
B_t, D_t, p_t = _y[0], _y[1], _y[2:]
C_t = p_t*np.expand_dims(np.expand_dims((1+np.arange(len(p_t))),axis=1),axis=1)
C_t = C_t.sum(axis=0)

fig, ax = plt.subplots(1,3)
ax[0].plot(teval,B_t.T)
ax[1].plot(teval,D_t.T)
ax[2].plot(teval,C_t.T)

# ax.plot(teval,p_t.T)

fig.show()

def get_polarity(X_t):
    return (X_t.max(axis=0)-X_t.min(axis=0))/(X_t.max(axis=0)+X_t.min(axis=0))

plt.plot(get_polarity(C_t))
plt.show()

plt.plot(C_t[0])
plt.show()

fig, ax = plt.subplots()
ax.plot(p_t[:,0,0])
fig.show()


fig, ax = plt.subplots()
ax.imshow()