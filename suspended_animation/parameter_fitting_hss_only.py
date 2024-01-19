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

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)

def fit_params(k_onF, k_offF, k_offB, inv_kappa_B, k_diss, k_unbind_late,k_bind,k_unbind_early,m0,k_unbind_anox,A_tot, B_tot,inv_kappa_B_anox, n, psi,rho0):
    y0 = jnp.array([0.1, 2., 10., 1e-3]) ##for unpolarised
    t_early = 15 #minutes
    t_late = 15 #minutes

    class Eqx_f(eqx.Module):
        log_fit_params: float

        @jit
        def f_anox_hss(self, t, y):
            fit_params = jnp.exp(self.log_fit_params)
            k_nuc,F0,k_onB = fit_params

            B, F, m, rho = y
            C = m * rho
            A = F + C

            A_cyto = A_tot - psi * A
            B_cyto = B_tot - psi * B
            phi = A / (A + inv_kappa_B_anox)

            nuc_rate = F ** n / (F0 ** n + F ** n)
            diss_rate = 1 / (1 + jnp.exp(m - m0))

            dtB = k_onB * B_cyto * C - k_offB * B * (1 - phi)
            dtF = k_onF * A_cyto - k_offF * F * phi \
                  - k_bind * (m ** (2 / 3)) * rho * F + k_unbind_anox * C \
                  - k_nuc * F * m * nuc_rate + k_diss * C * diss_rate
            dtm = k_bind * (m ** (2 / 3)) * F - k_unbind_anox * m
            dtrho = k_nuc * F * nuc_rate - k_diss * rho * diss_rate

            return jnp.array([dtB, dtF, dtm, dtrho])

        def __call__(self,t,y,args):
            return self.f_anox_hss(t, y)

    def solve(model):
        term = diffrax.ODETerm(model)
        solver = diffrax.Tsit5()
        t0 = 0
        t1 = jnp.inf
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
            adjoint=adjoint,
            args="f_anox_hss"
        )
        (y1,) = sol.ys
        return y1

    def loss(model):
        y1 = solve(model)
        fit_params = jnp.exp(model.log_fit_params)
        k_nuc, F0, k_onB = fit_params

        B, F, m, rho = y1
        C = m * rho
        A = F + C

        phi = A / (A + inv_kappa_B)

        nuc_rate = F ** n / (F0 ** n + F ** n)
        diss_rate = 1 / (1 + jnp.exp(m - m0))
        constraint1 = (k_onB * C * psi - k_offB * (1 - phi)) ** 2
        constraint2 = (k_onF * psi - k_offF * phi) ** 2
        constraint3 = (k_bind * m ** (2 / 3) * rho + k_nuc * C * nuc_rate - k_unbind_anox - k_diss * diss_rate) ** 2
        # print({1:constraint1,2:constraint2,3:constraint3})

        return constraint1 + constraint2 + constraint3


    @eqx.filter_jit
    def make_step(model,opt_state):
        grads = eqx.filter_grad(loss)(model)

        updates, opt_state = optim.update(grads, opt_state)

        model = eqx.apply_updates(model, updates)

        return model,opt_state

    n_repeat = 10
    log_fit_params_save = np.zeros((n_repeat,3))
    final_values_save = np.zeros((n_repeat,4))

    for repeat in range(n_repeat):
        #np.log((1e-2, 1e-2, 1e-2, 1e0,1e-3, 1e-1, 1e-4, 1e-4,1e-3, 1e0))
        log_fit_params0 = jnp.array(np.random.uniform(-4,0,3))
        model = Eqx_f(log_fit_params0)

        # optim = optax.sgd(1e-2, momentum=0.7, nesterov=True)
        optim = optax.adam(1e-3)

        opt_state = optim.init(model) ##I am assuming I need to specify only one of the models.


        total_loss = 1e9
        step = 0
        while (step<int(10000)):
            model, opt_state = make_step(model,opt_state)
            # if (step % 1000) ==0:
            #     total_loss = loss(model)
            #
            # if (step % 1000)==0:
            #     print("Repeat %d, "%repeat,loss(model),f"Step: {step}")
            step += 1
        print("Repeat %d, "%repeat,loss(model),f"Step: {step}")

        log_fit_params_save[repeat] = jnp.exp(model.log_fit_params)
        final_values_save[repeat] = solve(model)
    return log_fit_params_save,final_values_save

out,final_values = fit_params(k_onF=1e-2,
                 k_offF=1e-2,
                 k_offB=1e-2,
                 inv_kappa_B=40.,
                 k_diss=1e-2,
                 k_unbind_late=0.33,
                 k_bind=1,
                 k_unbind_early=0.33,
                 m0=3.,
                 k_unbind_anox=0.33,
                 A_tot=1,
                 B_tot=1,
                 inv_kappa_B_anox=40.,
                 n=4,
                 psi=0.174,
                 rho0=0.09)

for i in range(len(out)):
    plt.scatter(np.arange(len(out.T)),np.log10(out[i]))

plt.show()

for i in range(len(out)):
    plt.scatter(np.arange(len(final_values.T))+0.05*i,np.log10(final_values[i]))

plt.show()

F = final_values[:,1]
F0 = out[:,1]
knuc = out[:,0]

F**4/(F0**4+F**4) * F * knuc

for i in range(len(out)):
    plt.scatter(np.arange(len(out.T)),np.log10(out[i]))

plt.show()

fig, ax = plt.subplots()
ax.scatter(out[:,0],out[:,1])
fig.show()
