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

def fit_params(log2dm_early_late,k_offF_eff,k_offB_eff,MC_early,MC_late,phi_late_true,k_unbind_early,m0,k_unbind_anox,A_tot, B_tot,inv_kappa_B_anox, n, psi,rho0):
    y0 = jnp.array([0.1, 2., 10., 1e-3]) ##for unpolarised
    t_early = 15 #minutes
    t_late = 15 #minutes

    class Eqx_f(eqx.Module):
        log_fit_params: float

        @jit
        def f_anox_hss(self, t, y):
            fit_params = jnp.exp(self.log_fit_params)
            k_onF, k_offF, k_onB, k_offB, inv_kappa_B, k_bind, k_nuc, k_diss, F0, k_unbind_late = fit_params

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

        @jit
        def f_hss_normox(self, t, y):
            fit_params = jnp.exp(self.log_fit_params)
            k_onF, k_offF, k_onB, k_offB, inv_kappa_B, k_bind, k_nuc, k_diss, F0, k_unbind_late = fit_params

            B, F, m, rho = y
            C = m * rho
            A = F + C

            A_cyto = A_tot - psi * A
            B_cyto = B_tot - psi * B
            phi = A / (A + inv_kappa_B)

            nuc_rate = F ** n / (F0 ** n + F ** n)
            diss_rate = 1 / (1 + jnp.exp(m - m0))

            dtB = k_onB * B_cyto * C - k_offB * B * (1 - phi)
            dtF = k_onF * A_cyto - k_offF * F * phi \
                  - k_bind * (m ** (2 / 3)) * rho * F + k_unbind_early * C \
                  - k_nuc * F * m * nuc_rate + k_diss * C * diss_rate
            dtm = k_bind * (m ** (2 / 3)) * F - k_unbind_early * m
            dtrho = k_nuc * F * nuc_rate - k_diss * rho * diss_rate

            return jnp.array([dtB, dtF, dtm, dtrho])

        @jit
        def f_early_polarised_normox(self, t, y):
            fit_params = jnp.exp(self.log_fit_params)
            k_onF, k_offF, k_onB, k_offB, inv_kappa_B, k_bind, k_nuc, k_diss, F0, k_unbind_late = fit_params

            B, F, m, rho = y
            C = m * rho
            A = F + C

            A_cyto = A_tot - psi * A/2
            B_cyto = B_tot - psi * B/2
            phi = A / (A + inv_kappa_B)

            nuc_rate = F ** n / (F0 ** n + F ** n)
            diss_rate = 1 / (1 + jnp.exp(m - m0))

            dtB = k_onB * B_cyto * C - k_offB * B * (1 - phi)
            dtF = k_onF * A_cyto - k_offF * F * phi \
                  - k_bind * (m ** (2 / 3)) * rho * F + k_unbind_early * C \
                  - k_nuc * F * m * nuc_rate + k_diss * C * diss_rate
            dtm = k_bind * (m ** (2 / 3)) * F - k_unbind_early * m
            dtrho = k_nuc * F * nuc_rate - k_diss * rho * diss_rate

            return jnp.array([dtB, dtF, dtm, dtrho])

        @jit
        def f_late_polarised_normox(self, t, y):
            fit_params = jnp.exp(self.log_fit_params)
            k_onF, k_offF, k_onB, k_offB, inv_kappa_B, k_bind, k_nuc, k_diss, F0, k_unbind_late = fit_params

            B, F, m, rho = y
            C = m * rho
            A = F + C

            A_cyto = A_tot - psi * A / 2
            B_cyto = B_tot - psi * B / 2
            phi = A / (A + inv_kappa_B)

            nuc_rate = F ** n / (F0 ** n + F ** n)
            diss_rate = 1 / (1 + jnp.exp(m - m0))

            dtB = k_onB * B_cyto * C - k_offB * B * (1 - phi)
            dtF = k_onF * A_cyto - k_offF * F * phi \
                  - k_bind * (m ** (2 / 3)) * rho * F + k_unbind_late * C \
                  - k_nuc * F * m * nuc_rate + k_diss * C * diss_rate
            dtm = k_bind * (m ** (2 / 3)) * F - k_unbind_late * m
            dtrho = k_nuc * F * nuc_rate - k_diss * rho * diss_rate

            return jnp.array([dtB, dtF, dtm, dtrho])

        def __call__(self,t,y,args="f_anox_hss"):
            return (args=="f_anox_hss")*self.f_anox_hss(t, y)+ \
                (args=="f_hss_normox")*self.f_hss_normox(t, y)+ \
                (args=="f_early_polarised_normox")*self.f_early_polarised_normox(t, y)+ \
                (args=="f_late_polarised_normox")*self.f_late_polarised_normox(t, y)


    def loss_anox_hss(model):
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
        fit_params = jnp.exp(model.log_fit_params)
        k_onF, k_offF, k_onB, k_offB, inv_kappa_B, k_bind, k_nuc, k_diss, F0, k_unbind_late = fit_params

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


    def loss_pol(model):
        term_unpol = diffrax.ODETerm(model)
        solver_unpol = diffrax.Tsit5()
        t0_unpol = 0
        t1_unpol = 10*60.
        dt0_unpol = None
        max_steps_unpol = None
        controller_unpol = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        adjoint_unpol = diffrax.ImplicitAdjoint()
        sol_unpol = diffrax.diffeqsolve(
            term_unpol,
            solver_unpol,
            t0_unpol,
            t1_unpol,
            dt0_unpol,
            y0,
            max_steps=max_steps_unpol,
            stepsize_controller=controller_unpol,
            adjoint=adjoint_unpol,
            args="f_hss_normox"
        )
        (y1_unpol,) = sol_unpol.ys



        term_pol = diffrax.ODETerm(model)
        solver_pol = diffrax.Tsit5()
        t0_pol = 0
        t1_pol = t_early*60.
        dt0_pol = None
        y0_pol = jnp.array((y1_unpol[0]*2,y1_unpol[1]*2,y1_unpol[2],y1_unpol[3]*3))
        max_steps_pol = None
        controller_pol = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        adjoint_pol = diffrax.ImplicitAdjoint()
        sol_pol = diffrax.diffeqsolve(
            term_pol,
            solver_pol,
            t0_pol,
            t1_pol,
            dt0_pol,
            y0_pol,
            max_steps=max_steps_pol,
            stepsize_controller=controller_pol,
            adjoint=adjoint_pol,
            args="f_early_polarised_normox"
        )
        (y1_pol,) = sol_pol.ys

        term_late = diffrax.ODETerm(model)
        solver_late = diffrax.Tsit5()
        t0_late = 0
        t1_late = t_late * 60.
        dt0_late = None
        y0_late = y1_pol
        max_steps_late = None
        controller_late = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        adjoint_late = diffrax.ImplicitAdjoint()
        sol_late = diffrax.diffeqsolve(
            term_late,
            solver_late,
            t0_late,
            t1_late,
            dt0_late,
            y0_late,
            max_steps=max_steps_late,
            stepsize_controller=controller_late,
            adjoint=adjoint_late,
            args="f_late_polarised_normox"
        )
        (y1_late,) = sol_late.ys

        fit_params = jnp.exp(model.log_fit_params)
        k_onF, k_offF, k_onB, k_offB, inv_kappa_B, k_bind, k_nuc, k_diss, F0, k_unbind_late = fit_params

        B_unpol, F_unpol, m_unpol, rho_unpol = y1_unpol
        B_pol, F_pol, m_pol, rho_pol = y1_pol
        B_late, F_late, m_late, rho_late = y1_late

        A_pol = F_pol + m_pol*rho_pol
        A_late = F_late + m_late*rho_late
        phi_late = A_late / (A_late + inv_kappa_B)
        phi_early = A_pol / (A_pol + inv_kappa_B)

        constraint4 = (m0*rho_unpol - m0*rho0) ** 2
        constraint5 = (k_offF_eff*A_tot*psi - k_offF*phi_late*A_tot*psi)**2
        constraint6 = (k_offB_eff*B_tot*psi - k_offB*(1-phi_late)*B_tot*psi)**2
        constraint7 = (A_pol - A_tot*MC_early/psi)**2
        constraint8 = (A_late - A_tot*MC_late/psi)**2
        constraint9 = (phi_late*B_tot*psi - phi_late_true*B_tot*psi)**2
        constraint10 = (jnp.log2(m_late) - jnp.log2(m_pol) - log2dm_early_late)**2
        print({4:constraint4,
               5:constraint5,
               6:constraint6,
               7:constraint7,
               8:constraint8,
               9:constraint9,
               10:constraint10})


        return constraint4 + constraint5 + constraint6 + constraint7 + constraint8 + constraint9 + constraint10

    def loss(model):
        return 100*(loss_anox_hss(model)) + loss_pol(model)


    @eqx.filter_jit
    def make_step(model,opt_state):
        grads = eqx.filter_grad(loss)(model)

        updates, opt_state = optim.update(grads, opt_state)

        model = eqx.apply_updates(model, updates)

        return model,opt_state

    n_repeat = 5
    log_fit_params_save = np.zeros((n_repeat,10))
    for repeat in range(n_repeat):
        #np.log((1e-2, 1e-2, 1e-2, 1e0,1e-3, 1e-1, 1e-4, 1e-4,1e-3, 1e0))
        log_fit_params0 = jnp.array(np.random.uniform(-4,0,10))
        model = Eqx_f(log_fit_params0)

        # optim = optax.sgd(1e-2, momentum=0.7, nesterov=True)
        optim = optax.adam(1e-3)

        opt_state = optim.init(model) ##I am assuming I need to specify only one of the models.


        total_loss = 1e9
        step = 0
        while (step<int(50000)):
            model, opt_state = make_step(model,opt_state)
            if (step % 1000) ==0:
                total_loss = loss(model)

            if (step % 1000)==0:
                print("Repeat %d, "%repeat,loss(model),loss_pol(model),loss_anox_hss(model),f"Step: {step}")
            step += 1
        log_fit_params_save[repeat] = jnp.exp(model.log_fit_params)
    return log_fit_params_save

"""
log2dm_early_late=2.
k_offF_eff=5.4e-3
k_offB_eff=5.4e-3
MC_early=0.3
MC_late=0.15
phi_late_true=0.2
k_unbind_early=0.15
m0=3
k_unbind_anox=0.05
A_tot=1.
B_tot=1.
inv_kappa_B_anox=0.7
n=2
psi=0.174
rho0=0.0905
"""

out = fit_params(log2dm_early_late=2.,
           k_offF_eff=5.4e-3,
           k_offB_eff=5.4e-3,
           MC_early=0.3,
           MC_late=0.15,
           phi_late_true=0.2,
           k_unbind_early=0.15,
           m0=3,
           k_unbind_anox=0.05,
           A_tot=1.,
           B_tot=1.,
           inv_kappa_B_anox=0.7,
           n=2,
           psi=0.174,
           rho0=0.0905)

for i in range(5):
    plt.scatter(np.arange(10)+0.051*i,np.log10(out[i]))

plt.show()
"""
This approach doesn't give clear optima, although optima do cluster

One 

"""