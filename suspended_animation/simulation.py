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


class Simulation:
    def __init__(self):
        ...


@jit
def del_1d(x,dx):
    return (jnp.roll(x, -1) - 2 * x + jnp.roll(x, 1)) / dx ** 2


@jit
def f(t,y,params):
    D_B,D_F,D_C,k_onF,k_offF,k_onB,k_offB,inv_kappa_B,k_bind,k_unbind,k_nuc,k_diss,A_tot,B_tot,F0,m0,n,psi,dx = params
    B,F,m,rho = y[::4],y[1::4],y[2::4],y[3::4]
    C = m * rho
    A = F + C

    A_cyto = A_tot - psi*A.mean()
    B_cyto = B_tot - psi*B.mean()
    phi = A/(A+inv_kappa_B)

    nuc_rate = F**n/(F0**n + F**n)
    diss_rate = 1/(1+jnp.exp(m-m0))

    dtB = D_B*del_1d(B,dx) + k_onB*B_cyto*C - k_offB*B*(1-phi)
    dtF = D_F*del_1d(F,dx) + k_onF*A_cyto - k_offF*F*phi \
          - k_bind*(m**(2/3))*rho*F + k_unbind*C \
          - k_nuc*F*m*nuc_rate + k_diss*C*diss_rate
    dtm = k_bind*(m**(2/3))*F - k_unbind*m
    dtrho = D_C*del_1d(C,dx) +k_nuc*F*nuc_rate - k_diss*rho*diss_rate

    return jnp.column_stack([dtB,dtF,dtm,dtrho]).ravel()



t = 0

D_B = 1e-3
D_F = 1e-3
D_C = 1e-4
k_onF = 0.1
k_offF = 5.4e-3
k_onB = 0.1
k_offB = 5.4e-3
inv_kappa_B = 1.
k_bind = 0.5
k_unbind = 0.33
k_nuc = 1e-3
k_diss = 1e-3
A_tot = 1
B_tot = 1
F0 = 2
m0 = 3
n = 5
psi = 0.137
dx = 80

params = D_B, D_F,D_C, k_onF, k_offF, k_onB, k_offB, kappa_B, k_bind, k_unbind, k_nuc, k_diss, A_tot, B_tot, F0, m0, n, psi, dx


y0 = np.zeros(8) + 1e-17
y0[:4] = 0,1.3,0.005,10

f(t,y0,params)


f_jac = jit(jacfwd(f,argnums=1))
t_span = np.arange(0,1000,10)
t0 = time.time()
for i in range(100):
    res = solve_ivp(f,t_span=[t_span[0],t_span[-1]],t_eval=t_span,y0=y0,args=(params,),method="LSODA",jac=f_jac)
print((time.time()-t0)/100*1000)
labels = ["B","F","m","rho"]
fig, ax = plt.subplots()
for i in range(4):
    ax.plot(t_span,res.y[i::4][0],color=plt.cm.plasma(i/4),label=labels[i])
    ax.plot(t_span,res.y[i::4][1],color=plt.cm.plasma(i/4))
ax.legend()
fig.savefig("tests/1.pdf")


"""
This is remarkably fast

So let's try build this: 

1. Initial condition - M/C ratio initially for A and B
- then we have two unknowns: F/(F+C) and rho_init. 
- Fix rho_init and equilibrate. Use this as a penalty in the cost function
- Then simulate forward the transition from early to late maintenance 
--> k_unbind increases 

2. Anoxia
- Energy dependent terms: 
(a) k_unbind 
(b) k_rel or 1/Kappa (inv_kappa_B)
aka two additional parameters
- Simulate for 24hrs, score polarity 

3. Steady states
- run the simulation further (4 days) in anoxia and normoxia. Check that polarity in principle decays and decays faster in normoxic conditions 

4. Mutants: 
CDC42/Par6 KD: polarity decay 
Early vs late maintenance 

Costs: 

1. rho is stable (dtrho approaches 0) at normoxia
2. M/C in KO
3. M/C in normoxia (?)
4. M/C in anoxia (?)
5. Trajectory of polarity decay
6. Trajectories in the mutants 
7. PAR6-bound PAR3: Dickinson (20%) and dissociation constant 0.15-1s

"PAR3_cluster_turnover_anoxia_preNEBD": 1/30,  # (10-60)
"PAR3_cluster_turnover_anoxia_postNEBD": 1/30,  # (10-60)
"PAR3_growth_preNEBD":2.,  #Fig. 4A2 (assuming this is preNEBD)
"PAR3_growth_postNEBD":4.,

Optimisation: 

- known_parameters: 
(a) k_offA
(b) k_offB
(c) k_unbind_preNEBD ? postNEBD ? check notes
(d) A_tot (this is effective anyway) 
(e) psi
(f) diffusion constants
(g) rho_0 (N_clusters_init = 400)

- unknown_parameters: 
(a) k_onA
(b) k_onB
(c) k_bind
(d) k_unbind_anoxia_multiplier
(e) inv_kappa_B
(f) inv_kappa_B_anoxia_multiplier
(g) k_nuc
(h) k_diss
(i) F0
(j) m0
(k) n

Optimisation strategy 

Easiest solution is simplex with many random starting conditions


"""