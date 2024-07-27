import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.integrate import odeint
import multiprocessing
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

plt.rcParams.update({'pdf.fonttype': 42})

"""
NOTE: the solver has some issues. 

needs to make sure it doesn't terminate precociously. check the polarity afterwards etc. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', family='Helvetica Neue')

def format_ax(fig, ax):
    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.8, top=0.8)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(frameon=False)

"""
Have included a multiplier on a0 such that kPA is rescaled to approximately retain polarity in some range. 
"""


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
grey_magenta_cmap = make_colormap([c('magenta'), c('grey')])


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
grey_magenta_cmap = make_colormap([c('darkmagenta'), c('grey')])
grey_cmap = make_colormap([c('lightgrey'), c('black')])

"""
Consider the literature model

"""


def make_extent(x_range, y_range, xscale="linear", yscale="linear", center=True):
    if xscale == "log":
        x_range = np.log10(x_range)
    if yscale == "log":
        y_range = np.log10(y_range)
    if center is False:
        extent = [x_range[0], x_range[-1] + x_range[1] - x_range[0], y_range[0], y_range[-1] + y_range[1] - y_range[0]]
    else:
        extent = [x_range[0] - (x_range[1] - x_range[0]) / 2, x_range[-1] + (x_range[1] - x_range[0]) / 2,
                  y_range[0] - (y_range[1] - y_range[0]) / 2, y_range[-1] + (y_range[1] - y_range[0]) / 2]

    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    return extent, aspect


@jit(nopython=True)
def del_1d(x, dx):
    return (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / dx ** 2


def FP_fn(x, num_x, dx, l_cue):
    """Defines 'shape' of the cue. Defined as a Gaussian, maximal at the posterior.

    Adapted from Gross et al., 2018"""
    xx = (x >= num_x * dx / 2) * (-num_x * dx + x) + (
            x < num_x * dx / 2) * x  # accounts for periodic boundary
    return (np.exp(-(xx ** 2) / (l_cue ** 2)))


def fP(t, T_cue_on, T_cue_off, tau_Pon, tau_Poff):
    """Defines the temporal dynamics of the cue. Rate parameters described in __init__"""
    return 0.5 * (np.tanh((t - T_cue_on) / tau_Pon) - np.tanh((t - T_cue_off) / tau_Poff))


def c_P(t, kappa_cue, T_cue_on, T_cue_off, tau_Pon, tau_Poff, FP):
    """Defines cue strength in coarse_grained model"""
    return kappa_cue * fP(t, T_cue_on, T_cue_off, tau_Pon, tau_Poff) * FP


def get_aa_save_at_24hr(a_0_mult=2, k_conversion=5e-3):
    L = 134.6
    psi = 0.174
    A_tot, P_tot = 1.56, 1
    k_onA, k_onP = 8.58 * 10 ** -3, 4.74 * 10 ** -2
    k_offA, k_offP = 5.4 * 10 ** -3, 7.3 * 10 ** -3
    k_AP = 0.19
    k_PA = 2
    D_A, D_P = 0.28, 0.15

    a_0, p_0 = 0.32 * a_0_mult, 0.7
    d_a = D_A / k_offA / L ** 2
    d_p = D_P / k_offP / L ** 2
    kappa_AP = k_AP * P_tot / (psi * k_offA)
    kappa_PA = (1 / a_0_mult ** 2) * k_PA * A_tot ** 2 / (k_offP * psi ** 2)

    kappa_cue = 0.951  # Pre-factor "strength" of cue
    l_cue = 28.6 / L  # Length parameter for cue
    tau_Pon = 1.62  # Timescale of cue turning on
    tau_Poff = 1.62  # Timescale for cue turning off
    T_cue_on = 3  # Time when cue turns on
    T_cue_off = 11  # Time when cue turns off

    num_x = 64
    dx = 1 / num_x
    x = np.arange(0, 1, dx)
    FP = FP_fn(x, num_x, dx, l_cue)

    dt = 0.001
    tfin = 50
    t_span = np.arange(0, tfin, dt)

    def f_initialise(y, t):
        a, p = y[::2], y[1::2]
        dta = d_a * del_1d(a, dx) + (a_0 / (1 - a_0)) * (1 - a.mean()) - a - kappa_AP * a * p
        dtp = d_p * del_1d(p, dx) + (p_0 / (1 - p_0)) * (1 - p.mean()) - p - kappa_PA * p * a ** 2 * (
                    1 - c_P(t, kappa_cue, T_cue_on, T_cue_off, tau_Pon, tau_Poff, FP))
        dty = np.empty_like(y)
        dty[::2], dty[1::2] = dta, dtp
        return dty

    a_init = np.ones_like(num_x) * a_0
    p_init = np.ones_like(num_x) * p_0 * 0.01
    y_init = np.zeros(num_x * 2)
    y_init[0::2], y_init[1::2] = a_init, p_init

    sol = odeint(f_initialise, y_init, t_span)

    a_init_anox = sol[-1, ::2]
    aa_init_anox = np.zeros(a_init_anox.shape[0] * 2)
    aa_init_anox[::2] = a_init_anox
    ap_init_anox = np.zeros(a_init_anox.shape[0] * 2)
    ap_init_anox[1::2] = sol[-1, 1::2]
    # p_init_anox = sol[-1,1::2]
    # y_init_anox = np.zeros(num_x*2)
    # y_init_anox[0::2],y_init_anox[1::2] = a_init_anox,p_init_anox

    eps_a = (a_0 / (1 - a_0))

    def fa_anox_dimensional(aa, t, D_A, k_offA, eps_a, k_conversion=1e-3, d_immob_ratio=0, turnover_ratio=0,
                            k_reversal=0):
        """
        Here, eps = k_onA*A_tot/k_offA
        """
        a_mob, a_imob = aa[::2], aa[1::2]
        dta_mob = D_A * del_1d(a_mob, dx) / L ** 2 + eps_a * (
                    1 - (a_mob + a_imob).mean()) * k_offA - a_mob * k_offA - a_mob * k_conversion + a_imob * k_reversal
        dta_imob = D_A * del_1d(a_imob,
                                dx) * d_immob_ratio / L ** 2 - a_imob * turnover_ratio * k_offA + a_mob * k_conversion - a_imob * k_reversal
        dta = np.zeros_like(aa)
        dta[::2], dta[1::2] = dta_mob, dta_imob
        return dta

    t_span_dimensional = np.logspace(-1, np.log10(24 * 60 * 60), 500)

    # t_span_dimensional = np.arange(0,10.0**(4.5)*60,1)
    aa_save = odeint(fa_anox_dimensional, aa_init_anox, t_span_dimensional, args=(D_A, k_offA, eps_a, k_conversion))
    return aa_save


K_conv_range = np.logspace(-1, 1, 2)
a_0_mult_range = np.linspace(0.5, 2.5, 2)
k_conv_range = K_conv_range * 5.4 * 10 ** -3

KK, A0 = np.meshgrid(k_conv_range, a_0_mult_range, indexing="ij")
num_cores = multiprocessing.cpu_count()
aa_save_results = Parallel(n_jobs=num_cores)(
    delayed(get_aa_save_at_24hr)(a_0_mult, k_conv) for (a_0_mult, k_conv) in
    zip(A0.ravel(), KK.ravel()))



aa_save_results = np.array(aa_save_results).reshape(KK.shape + aa_save_results[0].shape)
aa_save_tot_results = aa_save_results[...,::2] + aa_save_results[...,1::2]

num_x = 64
dx = 1 / num_x
x = np.arange(0, 1, dx)

fig, ax = plt.subplots(2,2,sharex=True,sharey=True)
for i in range(2):
    for j in range(2):
        format_ax(fig,ax[i,j])
        ax[i,j].plot(x[:int(len(x) / 2)], aa_save_tot_results[i,j][0][int(len(x) / 2):],color="grey",linestyle="--")

        ax[i,j].plot(x[:int(len(x) / 2)], aa_save_tot_results[i,j][-1][int(len(x) / 2):],color="darkred")
        ax[1,j].set_xticks([0,0.25,0.5],labels=["0","L/4","L/2"])
        ax[i,j].set_title(r"$k_{imm}/k_{off}=%.3f a_0^{mult} = %.3f$"%(K_conv_range[i],a_0_mult_range[j]),fontsize=4)
fig.savefig("literature_model_Figs1-3/time_series.pdf",dpi=300)