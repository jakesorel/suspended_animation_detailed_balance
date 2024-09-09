import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.integrate import odeint,solve_ivp
import multiprocessing
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
plt.rcParams.update({'pdf.fonttype': 42})
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', family='Helvetica Neue')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@jit(nopython=True)
def del_1d(x,dx):
    return (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / dx ** 2


def FP_fn(x,num_x,dx,l_cue):
    """Defines 'shape' of the cue. Defined as a Gaussian, maximal at the posterior.

    Adapted from Gross et al., 2018"""
    xx = (x >= num_x * dx / 2) * (-num_x * dx + x) + (
            x < num_x * dx / 2) * x  # accounts for periodic boundary
    return (np.exp(-(xx ** 2) / (l_cue ** 2)))

def fP(t,T_cue_on,T_cue_off,tau_Pon,tau_Poff):
    """Defines the temporal dynamics of the cue. Rate parameters described in __init__"""
    return 0.5 * (np.tanh((t - T_cue_on) / tau_Pon) - np.tanh((t - T_cue_off) / tau_Poff))

def c_P(t,kappa_cue,T_cue_on,T_cue_off,tau_Pon,tau_Poff,FP):
    """Defines cue strength in coarse_grained model"""
    return kappa_cue * fP(t,T_cue_on,T_cue_off,tau_Pon,tau_Poff)*FP


L = 134.6
psi = 0.174
A_tot, P_tot = 1.56, 1
k_onA, k_onP = 8.58 * 10 ** -3, 4.74 * 10 ** -2
k_offA, k_offP = 5.4 * 10 ** -3, 7.3 * 10 ** -3
k_AP = 0.19
k_PA = 2
D_A, D_P = 0.28, 0.15


kappa_cue = 0.951  # Pre-factor "strength" of cue
l_cue = 28.6 / L  # Length parameter for cue
tau_Pon = 1.62  # Timescale of cue turning on
tau_Poff = 1.62  # Timescale for cue turning off
T_cue_on = 100  # Time when cue turns on
T_cue_off = 1500  # Time when cue turns off

num_x = 256
_dx = 1/num_x
_x = np.arange(0,1,_dx)
FP = FP_fn(_x,num_x,_dx,l_cue)

dx = L/num_x
x = np.arange(0,L,dx)

dt = 0.1
tfin = 10000
t_span = np.arange(0,tfin,dt)


def f(y,t):
    a,p = y[::2],y[1::2]
    a_cyto = A_tot - psi*a.mean()
    p_cyto = P_tot - psi*p.mean()
    dta = D_A*del_1d(a,dx)+k_onA*a_cyto - k_offA*a - k_AP*a*p
    dtp =D_P*del_1d(p,dx)+k_onP*p_cyto - k_offP*p - k_PA*p*a**2 *(1-c_P(t,kappa_cue,T_cue_on,T_cue_off,tau_Pon,tau_Poff,FP))
    dty = np.zeros_like(y)
    dty[::2],dty[1::2] = dta, dtp
    return dty

"""
k_onA * (A_tot - psi*A) = k_offA*A
k_onA*A_tot/(k_offA + psi*k_onA)
"""

y0 = np.zeros(num_x*2)
y0[::2] = k_onA*A_tot/(k_offA + psi*k_onA)
y0[1::2] = k_onP*P_tot/(k_offP + psi*k_onP) * 0.01

sol = odeint(f,y0,t_span)
plt.imshow(sol[:,::2],aspect="auto")
plt.show()
print((sol[-1,::2]>(sol[-1,::2].max()/2)).mean())