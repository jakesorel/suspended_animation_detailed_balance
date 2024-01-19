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


class ZeroDCluster(eqx.Module):
    params: float


    @jit
    def f(self, t, y,params):
        K,Kr,kbind,kunbind = params
        i0 = 3
        i = np.arange(y.size)+1


        fw_rate_per = K*(i/i0)**(1/3)
        rv_rate_per = Kr

        p1 = y[0]
        fw_rate_cluster = y*fw_rate_per*p1
        fw_rate_monomers = y*kbind*p1*i
        rv_rate_cluster = y*i*rv_rate_per
        rv_rate_monomers = y*i*kunbind

        fw_rate = jnp.concatenate((fw_rate_monomers[:i0-1],fw_rate_cluster[i0-1:-1]))
        rv_rate = jnp.concatenate((rv_rate_monomers[1:i0],rv_rate_cluster[i0:]))
        net_rate = fw_rate - rv_rate

        dty = jnp.concatenate((jnp.array((0.,)),net_rate)) - jnp.concatenate((net_rate,(jnp.array((0.,)))))
        dty = dty.at[0].add(-net_rate.sum())

        return dty

    def __call__(self, t, y,params):
        return self.f(t, y,params)

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




y0 = np.zeros(200)
y0[0] = 1.0
params = jnp.array((600.,1.,0.05,0.01))
model1 = ZeroDCluster(params)
# y1 = solve(model,y0)
teval = np.arange(0,1e4,1)
tspan = [teval[0],teval[-1]]

jac1 = jit(jacrev(model1,argnums=[1,]))

sol1 = solve_ivp(model1,tspan,y0,method="LSODA",t_eval=teval,jac=jac1,args=(params,))
print("first done")
params = jnp.array((350.,1.,0.05,0.01))
model = ZeroDCluster(params)
jac = jit(jacrev(model,argnums=[1,]))

sol = solve_ivp(model,tspan,sol1.y[:,-1],method="LSODA",t_eval=teval,jac=jac,args=(params,))



fig, ax = plt.subplots()
n_plot = 100
for i in range(n_plot):
    t = int(sol.y.shape[1]/n_plot * i)

    y1 = sol.y[:,t]

    ax.plot(np.arange(y1.size)+1,y1/y1.max(),color=plt.cm.plasma(i/n_plot))

fig.show()


fig, ax = plt.subplots(10,1,figsize=(4,10))
ti_range = np.logspace(-1,np.log10(int(len(teval)-1)),500).astype(int)
# ti_range = np.linspace(0,200,10).astype(int)

fig2,ax2 = plt.subplots()

cg_values = np.zeros((len(ti_range),2))

for q, ti in enumerate(ti_range):

    y1 = sol.y[:,ti]



    i0 = 3
    j = np.arange(y0.size-i0)+1
    pi0 = y1[i0]
    K = float(params[0])/float(params[1])
    p0 = y1[0]

    ####THIS IS THE MAGIC DESCRIPTOR

    pp = pi0*(i0+1)/(K*p0)*(K*p0/i0**(1/3))**j * (i0/(i0+j))**(1/3) *np.cumprod((i0+j)**(-2/3))

    ##or

    pp2 = pi0*(i0+1)/(K*p0) * (i0/(i0+j))**(1/3) *np.exp(j*np.log(K*p0/i0**(1/3)) -2/3 * np.cumsum(np.log(i0+j)))





    ###IT DESCRIBES THE Pr of every cluster size, starting from i0
    ###Through numerics, I had to adjust based on the maths, so the proof needs some work. But this is indeed true.

    # ax[q].plot(y1[i0+1:]/y1[i0+1:].max())
    # ax[q].plot(pp2[1:]/pp2[1:].max())
    # ax[q].set(xlim=(1,20))

    # print(y1[i0+1:].sum()-pp2[1:].sum())

    plt.show()

    n = y1.size

    C= (y1*np.arange(1,n+1))
    Cleqi0 = C[:i0+1].sum()
    Ctot = C.sum()
    # Cgi0 = Ctot - Cleqi0

    Cgi0 = (y1[i0+1:]*np.arange(i0+2,n+1)).sum()
    Cgi0_approx = (pp[1:]*np.arange(i0+2,n+1)).sum()
    print(Cgi0-Cgi0_approx)

    # cg_values[q] = Cgi0,Cgi0_approx
    cg_values[q] = y1[i0 + 1:].sum(),pp2[1:].sum()

    # ax2.scatter(Cgi0,Cgi0_approx,color=plt.cm.plasma(q/len(ti_range)))
# ax.set(xscale="log",yscale="log")
fig.show()
# fig2.show()



fig, ax = plt.subplots()
ax.plot(teval[ti_range],(np.log(cg_values[:,1])-np.log(cg_values[:,0])))
# ax.set(xscale="log",xlim=(0.1,1e5))
fig.show()

fig, ax = plt.subplots()
ax.plot(teval[ti_range],cg_values[:,0])
ax.plot(teval[ti_range],cg_values[:,1])

ax.set(xscale="log",xlim=(0.1,1e5))
fig.show()


"""
Conclusion: 

It seems as though the equilibrium assumption of this model 
is violated 

Equalisation is achieved only when equilibrium is reached, which is dodgy. 

A huge shame given the result is pretty beautiful. 



Another observation: somehow the magnitude is way off, but the resultant normalised distribution is OK very fast.  


"""


@partial(jit,static_argnums=(3,4))
def get_pp(K,pi0,p0,i0,n):
    j = jnp.arange(n - i0) + 1

    pp = pi0 * (i0 + 1) / (K * p0) * (i0 / (i0 + j)) ** (1 / 3) \
         * jnp.exp(j * jnp.log(K * p0 / i0 ** (1 / 3)) - 2 / 3 * jnp.cumsum(jnp.log(i0 + j)))

    return pp

@jit
def _get_ptot(pp):
    return pp.sum()

@partial(jit,static_argnums=(2,3))
def _get_Ctot(pp,i0,n):
    j = jnp.arange(n - i0) + i0




t0 = time.time()
for i in range(int(1e5)):
    get_pp(K, pi0, p0, i0, 200)
t1 = time.time()
print((t1-t0)/1e5*1e6)




####AVERAGE MASS

j = np.arange(y0.size - i0) + 1
pi0 = y1[i0]
K = float(params[0])
p0 = y1[0]

pp = pi0 * (i0 + 1) / (K * p0) * (K * p0 / i0 ** (1 / 3)) ** j * (i0 / (i0 + j)) ** (1 / 3) * np.cumprod((i0 + j) ** (-2 / 3))

def cumprod_val(i0,j):
    val = 1
    for i in range(1,j+1):
        val *= (i0+j)**(-2/3)
    return val

def get_p_alt(j):
    pp_alt = pi0 * (i0+1)/(i0 + j + 1) *((K*p0)/(i0**(1/3)))**j * cumprod_val(i0,j)
    return pp_alt

p_alt = np.array([get_p_alt(j) for j in np.arange(y0.size - i0) + 1])


plt.plot(pp)
plt.plot(p_alt)
plt.show()



j = np.arange(y0.size - i0) + 1
pi0 = y1[i0-1]
K = float(params[0])
p0 = y1[0]
pp = pi0 * (K * p0 / i0 ** (1 / 3)) ** j * (i0 / (i0 + j)) ** (1 / 3) * np.cumprod(
    (i0 + j) ** (-2 / 3))


fig, ax = plt.subplots()
ax.plot(y1[i0:])
ax.plot(pp[:])
fig.show()

def get_rho(Kp0,pi0):
    j = np.arange(1000) + 1
    pp = pi0 *  (i0 / (i0 + j)) ** (1 / 3) * np.exp(j*np.log(Kp0 / i0 ** (1 / 3)) - 2/3 * np.cumsum(np.log((i0 + j))))
    return pp.sum()

Kp0_range = np.logspace(-3,2,500)
pp_range = np.array([get_rho(Kp0,1) for Kp0 in Kp0_range])
fig, ax = plt.subplots()
ax.plot(Kp0_range,pp_range)
ax.set(yscale="log",xscale="log")
fig.show()

def get_mrho(Kp0,pi0):
    j = np.arange(3000) + 1
    pp = pi0 *  (i0 / (i0 + j)) ** (1 / 3) * np.exp(j*np.log(Kp0 / i0 ** (1 / 3)) - 2/3 * np.cumsum(np.log((i0 + j))))
    mrho = ((i0+j)*pp).sum()
    return mrho


### We can approximate the m(Kp0) function with the below function
def approximation_of_m(Kp0,alpha,beta,n,m):
    return alpha + (Kp0**(n))/(Kp0+beta**m)

mrho_range = np.array([get_mrho(Kp0,1) for Kp0 in Kp0_range])
fig, ax = plt.subplots()
ax.plot(Kp0_range,mrho_range/pp_range)
# ax.set(xscale="log")
fig.show()

from scipy.optimize import curve_fit

res,__ = curve_fit(approximation_of_m,Kp0_range,(mrho_range/pp_range),[4.0,0.2,1.,1.],bounds=(1e-7,np.inf))

fig, ax = plt.subplots()
ax.plot(Kp0_range,mrho_range/pp_range)
ax.plot(Kp0_range,approximation_of_m(Kp0_range,*res))
ax.set(yscale="log",xscale="log")
fig.show()


