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


def get_polarity_at_24hr(a_0_mult=2, k_conversion=5e-3):
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

    def polarity(a_save):
        mx, mn = np.max(a_save, 1), np.min(a_save, 1)
        return (mx - mn) / (mx + mn)

    # t_span_dimensional = np.arange(0,10.0**(4.5)*60,1)
    aa_save = odeint(fa_anox_dimensional, aa_init_anox, t_span_dimensional, args=(D_A, k_offA, eps_a, k_conversion))
    a_total_save = aa_save[:, ::2] + aa_save[:, 1::2]
    pol_save = polarity(a_total_save)
    if t_span_dimensional.max() < 24 * 60 * 60:
        pol = pol_save[-1]
    else:
        intrp = interp1d(t_span_dimensional, pol_save)
        pol = intrp(24 * 60 * 60)
        print("done")
    return pol, aa_init_anox[::2].mean()


K_conv_range = np.logspace(-3, 3, 20)
a_0_mult_range = np.linspace(0.01, 3, 20)
k_conv_range = K_conv_range * 5.4 * 10 ** -3

KK, A0 = np.meshgrid(k_conv_range, a_0_mult_range, indexing="ij")
num_cores = multiprocessing.cpu_count()
pol_after_24_hrs_results = Parallel(n_jobs=num_cores)(
    delayed(get_polarity_at_24hr)(a_0_mult, k_conv) for (a_0_mult, k_conv) in
    zip(A0.ravel(), KK.ravel()))

pol_after_24_hrs_results, MC_at_start = np.array(pol_after_24_hrs_results).T.reshape((2,) + KK.shape)

MC_range = MC_at_start[0]
intrp = RectBivariateSpline(np.log10(K_conv_range), MC_range, pol_after_24_hrs_results)

MC_range_dense = np.linspace(MC_range[0], MC_range[-1], 100)
log_K_conv_dense = np.linspace(np.log10(K_conv_range[0]), np.log10(K_conv_range[-1]), 100)

plt.imshow(intrp(MC_range_dense, log_K_conv_dense))
plt.show()

cmap = plt.cm.inferno

def format_ax(fig, ax):
    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.8, top=0.8)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(frameon=False)


levels = np.arange(-1, 8)
fig, ax = plt.subplots(figsize=(4, 4))
extent, aspect = make_extent(log_K_conv_dense, MC_range_dense, "linear", "linear")

im = intrp(MC_range_dense, log_K_conv_dense).copy()
ax.imshow(np.flip(im.T, axis=0), extent=extent, aspect=aspect, cmap=cmap, vmin=0, vmax=1)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=1, vmin=0))
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
cl.set_label("Polarity after 24h")
cl_ticks = [0, 0.5, 1]

ax.set(xlabel=r"$log_{10} \ k_{imm}/k_{off,A}$")
ax.set(ylabel="Membrane bound fraction \n (normoxia)")
fig.subplots_adjust(left=0.3, right=0.8, bottom=0.3, top=0.8, wspace=1)
fig.savefig("literature_model_Figs1-3/plots/membrane bound fraction vs K_imm.pdf", dpi=300)

import seaborn as sns
fig, ax = plt.subplots(figsize=(4,4))
format_ax(fig,ax)

for i, M in enumerate(np.linspace(MC_range[0], MC_range[-1], 6)):
    ax.plot(K_conv_range,intrp(np.log10(K_conv_range),M),color=sns.color_palette("mako_r",as_cmap=True)(i/5))
    ax.set(xscale="log")
sm = plt.cm.ScalarMappable(cmap=sns.color_palette("mako_r",as_cmap=True), norm=plt.Normalize(vmax=MC_range[-1], vmin=MC_range[0]))

cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
cl.set_label("Membrane bound fraction\n (normoxia)")

ax.set(xlabel=r"$k_{imm}/k_{off,A}$",ylabel="Polarity after 24hr")
fig.savefig("literature_model_Figs1-3/plots/membrane bound fraction vs K_imm lineplot.pdf", dpi=300)

fig.show()
fig, ax = plt.subplots(figsize=(4, 3))
for i, k_conv in enumerate(k_conv_space):
    k_conversion = k_conv
    CC = np.ones_like(KK) * k_conversion
    num_cores = multiprocessing.cpu_count()
    pol_after_24_hrs_results_no_conversion = Parallel(n_jobs=num_cores)(
        delayed(get_polarity_after_24_hrs)(D_A, k_offA, k_conv) for (D_A, k_offA, k_conv) in
        zip(DD.ravel(), KK.ravel(), CC.ravel()))
    pol_after_24_hrs_results_no_conversion = np.array(pol_after_24_hrs_results_no_conversion).reshape(DD.shape)
    ax.plot(k_off_space, pol_after_24_hrs_results_no_conversion[0], color=plt.cm.plasma(
        (np.log10(k_conversion) - np.min(np.log10(k_conv_space))) / (
                    np.log10(np.max(k_conv_space)) - np.min(np.log10(k_conv_space)))), alpha=0.5)

k_conversion = 1.5 / 60 / 60 / 24
CC = np.ones_like(KK) * k_conversion
num_cores = multiprocessing.cpu_count()
pol_after_24_hrs_results_no_conversion = Parallel(n_jobs=num_cores)(
    delayed(get_polarity_after_24_hrs)(D_A, k_offA, k_conv) for (D_A, k_offA, k_conv) in
    zip(DD.ravel(), KK.ravel(), CC.ravel()))
pol_after_24_hrs_results_no_conversion = np.array(pol_after_24_hrs_results_no_conversion).reshape(DD.shape)
ax.plot(k_off_space, pol_after_24_hrs_results_no_conversion[0], color=plt.cm.plasma(
    (np.log10(k_conversion) - np.min(np.log10(k_conv_space))) / (
                np.log10(np.max(k_conv_space)) - np.min(np.log10(k_conv_space)))))

# a_init_anox
fin = (a_init_anox + (1 - a_init_anox.mean()))
pol = (fin.max() - fin.min()) / (fin.max() + fin.min())
ax.plot(k_off_space, np.ones_like(k_off_space) * pol, linestyle="--", alpha=0.5, color="grey")
ax.set(xscale="log")
ax.set(ylabel="Polarity after 24h", xlabel=r"$k_{off} \ [ s^{-1} ]$")

# cont = ax.contour(np.log10(depol_time_mins),extent=extent,levels=levels,cmap=plt.cm.Greys)
# ax.clabel(cont, levels, fmt=format_contour_label,zorder=1000)
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                           norm=plt.Normalize(vmax=np.max(np.log10(k_conv_space)), vmin=np.min(np.log10(k_conv_space))))
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
cl.set_label(r"$k_{conv}$")
cl_ticks = np.arange(np.log10(k_conv_space.min()), np.log10(k_conv_space.max()) + 1, 2)
cl.set_ticks(cl_ticks)
cl.set_ticklabels([r"$10^{%d}$" % i for i in cl_ticks])
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8)

fig.savefig("exp_paper_April2022/plots/polarity after 24 hrs k_conv k_off.pdf", dpi=300)

####HEATMAPS

D_A_space = np.logspace(-4, 1, 25)
k_off_space = np.logspace(-6, 0, 25)
DD, KK = np.meshgrid(D_A_space, k_off_space, indexing="ij")

k_conversion = 0
CC = np.ones_like(KK) * k_conversion
num_cores = multiprocessing.cpu_count()
pol_after_24_hrs_results_no_conversion = Parallel(n_jobs=num_cores)(
    delayed(get_polarity_after_24_hrs)(D_A, k_offA, k_conv) for (D_A, k_offA, k_conv) in
    zip(DD.ravel(), KK.ravel(), CC.ravel()))
pol_after_24_hrs_results_no_conversion = np.array(pol_after_24_hrs_results_no_conversion).reshape(DD.shape)

k_conversion = 1.5 / 60 / 60
CC = np.ones_like(KK) * k_conversion
num_cores = multiprocessing.cpu_count()
pol_after_24_hrs_results = Parallel(n_jobs=num_cores)(
    delayed(get_polarity_after_24_hrs)(D_A, k_offA, k_conv) for (D_A, k_offA, k_conv) in
    zip(DD.ravel(), KK.ravel(), CC.ravel()))
pol_after_24_hrs_results = np.array(pol_after_24_hrs_results).reshape(DD.shape)

extent, aspect = make_extent(D_A_space, k_off_space, "log", "log")


def format_contour_label(x):
    return " " + r"$10^{%d}$" % x + "  "


cmap = plt.cm.inferno

levels = np.arange(-1, 8)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

im = pol_after_24_hrs_results_no_conversion.copy()
# im[im < 0.05] = np.nan
ax[0].imshow(np.flip(im.T, axis=0), extent=extent, aspect=aspect, cmap=cmap, vmin=0, vmax=1, interpolation="bicubic")
#
levels = (-1, 0.1, 1.1)
cs = ax[0].contour(np.log10(D_A_space), np.log10(k_off_space), pol_after_24_hrs_results_no_conversion.T, levels,
                   cmap="Greys")

im = pol_after_24_hrs_results.copy()
# im[im < 0.05] = np.nan
ax[1].imshow(np.flip(im.T, axis=0), extent=extent, aspect=aspect, cmap=cmap, vmin=0, vmax=1, interpolation="bicubic")
cs = ax[1].contour(np.log10(D_A_space), np.log10(k_off_space), pol_after_24_hrs_results.T, levels, cmap="Greys")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=0, vmin=1))
cl = plt.colorbar(sm, ax=ax[0], pad=0.05, fraction=0.073, aspect=12, orientation="vertical")

cl = plt.colorbar(sm, ax=ax[1], pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
cl.set_label("Polarity after 24h")
cl_ticks = [-1, 0, 1, 2, 3, 4]
cl.set_ticks(cl_ticks)
cl.set_ticklabels([r"$10^{%d}$" % i for i in cl_ticks])
for axx in ax:
    axx.set(ylabel=r"$k_{offA}$" + " " + r"$[s^{-1}]$")
    axx.set(xlabel=r"$D_A$" + " " + r"$[\mu m^2 s^{-1}]$")

for axx in ax:
    xtck = np.array([-3, -1, 1])
    ytck = np.array([-5, -3, -1])
    axx.xaxis.set_ticks(xtck)
    axx.xaxis.set_ticklabels([r"$10^{%d}$" % i for i in xtck])
    axx.yaxis.set_ticks(ytck)
    axx.yaxis.set_ticklabels([r"$10^{%d}$" % i for i in ytck])

    axx.set(ylim=(np.log10(k_off_space[0]), np.log10(k_off_space[-1])),
            xlim=(np.log10(D_A_space[0]), np.log10(D_A_space[-1])))
fig.subplots_adjust(left=0.3, right=0.8, bottom=0.3, top=0.8, wspace=1)
fig.show()

fig.savefig("exp_paper_April2022/plots/polarity with and without immobilisation.pdf", dpi=300)

fig, ax = plt.subplots(figsize=(4, 3))

for i, k_conv in enumerate(k_conv_space):
    ax.plot(*get_pol_save(1e-3, 5.4e-3, k_conv), color=plt.cm.plasma(
        (np.log10(k_conv) - np.min(np.log10(k_conv_space))) / (
                    np.log10(np.max(k_conv_space)) - np.min(np.log10(k_conv_space)))), zorder=-i, alpha=0.5)
ax.set(xscale="log", xlim=(10 ** 0, 24 * 60 * 60), ylabel="Polarity after 24h", xlabel="Time [s]")

k_conv = 1.5 / 60 / 60
ax.plot(*get_pol_save(1e-3, 5.4e-3, k_conv), color=plt.cm.plasma((np.log10(k_conv) - np.min(np.log10(k_conv_space))) / (
        np.log10(np.max(k_conv_space)) - np.min(np.log10(k_conv_space)))), zorder=-i, alpha=0.5)

# cont = ax.contour(np.log10(depol_time_mins),extent=extent,levels=levels,cmap=plt.cm.Greys)
# ax.clabel(cont, levels, fmt=format_contour_label,zorder=1000)
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                           norm=plt.Normalize(vmax=np.max(np.log10(k_conv_space)), vmin=np.min(np.log10(k_conv_space))))
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
cl.set_label(r"$k_{conv}$")
cl_ticks = np.arange(np.log10(k_conv_space.min()), np.log10(k_conv_space.max()) + 1, 2)
cl.set_ticks(cl_ticks)
cl.set_ticklabels([r"$10^{%d}$" % i for i in cl_ticks])
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8)
fig.savefig("exp_paper_April2022/plots/polarity after 24h for varying k_conv.pdf", dpi=300)

#
#
# grey_blue_cmap = make_colormap([c('royalblue'),c('grey')])
#
#
# fig, ax = plt.subplots(figsize=(5,3))
# ti = 100
# D_A = 10**-3
# k_off_plot = np.flip([10**-6,10**-5,10**-4,10**-3,10**-2])
# for i, k_offA in enumerate(k_off_plot):
#
#     t_span_dimensional,pol = get_pol_save(D_A,k_offA,k_conversion=0)
#
#     ax.plot(t_span_dimensional/60,pol,color=grey_blue_cmap(i/(len(k_off_plot)-1)),linewidth=2,linestyle="--",alpha=0.5)
# ax.set(xscale="log",xlim=(10**-1,10**5),ylim=(0.0,1),xlabel="Time into anoxia\n(minutes)",ylabel="Polarity")
#
# ytck = np.array([0,0.5,1])
# ax.yaxis.set_ticks(ytck)
# for i, k_offA in enumerate(k_off_plot):
#
#     t_span_dimensional,pol = get_pol_save(D_A,k_offA,k_conversion=1/24/60/60)
#
#     ax.plot(t_span_dimensional/60,pol,color=grey_blue_cmap(i/(len(k_off_plot)-1)),linewidth=2)
#
# fig.subplots_adjust(left=0.3,right=0.7,bottom=0.3,top=0.7)
# fig.show()
# fig.savefig("exp_paper_April2022/plots/depol time with immobilisation.pdf",dpi=300)
#
#
#
# fig, ax = plt.subplots()
# k_offA = 5.4 * 10.0**-3
# D_A = 0.28
# N = 6
# k_conversion_range = np.logspace(-3,-1,N)
# for i in range(N):
#
#     t_span_dimensional, pol = get_pol_save(D_A, k_offA, k_conversion=k_conversion_range[i],override=False)
#     if t_span_dimensional.max()< 10**3.5:
#         t_span_dimensional = np.append(t_span_dimensional,10**3.5)
#         pol = np.append(pol,pol[-1])
#
#     ax.plot(t_span_dimensional, pol, color=grey_magenta_cmap(i / (len(k_off_plot) - 1)), linewidth=2)
# # t_span_dimensional, pol = get_pol_save(D_A, k_offA, k_conversion=1.5 / 60 )
# #
# # ax.plot(t_span_dimensional, pol, color=grey_magenta_cmap(i / (len(k_off_plot) - 1)), linewidth=2)
#
# ax.set(xscale="log",xlim=(10**-3,10**5),ylim=(0.0,1),xlabel="Time into anoxia",ylabel="Polarity")
# fig.show()
#
#
# t_span_dimensional = np.arange(0,24*60*60,1)
#
# aa_save_L = odeint(fa_anox_dimensional, aa_init_anox, t_span_dimensional, args=(10**-3,10**-5,eps_a,0))
# aa_save_M = odeint(fa_anox_dimensional, aa_init_anox, t_span_dimensional, args=(10**-3,10**-5,eps_a,1/24/60/60))
#
# aa_save_R = odeint(fa_anox_dimensional, aa_init_anox, t_span_dimensional, args=(10**-3,10**-5,eps_a,1/10/60))
#
# N = 7
# fig, ax = plt.subplots(1,3,figsize=(4,1.2),sharey=True)
# for j, aa_save in enumerate([aa_save_L,aa_save_M,aa_save_R]):
#     i_range = np.linspace(0,len(aa_save)-1,N).astype(int)
#     for i in range(N):
#         ax[j].plot(x[:int(len(x)/2)],(aa_save[i_range[i],::2]+aa_save[i_range[i],1::2])[int(len(x)/2):],color=grey_cmap(i/N))
# fig.subplots_adjust(bottom=0.3)
# fig.savefig("exp_paper_April2022/plots/decay_modes immobilisation.pdf",dpi=300)
