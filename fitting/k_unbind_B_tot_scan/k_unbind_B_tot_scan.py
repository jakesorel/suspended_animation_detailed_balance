import sys
import os

sys.dont_write_bytecode = True

SCRIPT_DIR = "../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = "../../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = ".."
sys.path.append(os.path.dirname(SCRIPT_DIR))
from joblib import Parallel, delayed

from cluster_model.cluster_model import Simulate
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import griddata

import threading
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


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            print("Presuming path exists")


if __name__ == "__main__":

    param_dict = {'D_A': 0.2,
                  "D_B": 0.28,
                  'D_C': 0.001,
                  'k_onA': 1e-3,
                  'k_offA': 1.,
                  'k_onB_c': 1e-3,
                  'k_offB_f': 5.4e-3,
                  'k_offB_c': 5.4e-3,
                  'kbind_c': 0.051794745,
                  'kbind_m': 0.051794745,
                  'kunbind': 1 / 30,
                  "kunbind_postNEBD": 0.1,
                  'k_seq': 0.0071968567,
                  'k_rel': 0.01,
                  'k_seq_multiplier': 1.,
                  'A_tot': 1.0,
                  'B_tot': 4.666,
                  'psi': 0.174,
                  'L': 134.6,
                  'k_AP': 1e1,
                  'n_clust': 256,
                  'i0': 3,
                  'advection_fraction': 0.99,
                  "tau_pol": 60,
                  "tau_NEBD": 60,
                  "tau_anox": 600}

    anoxia_dict = {"k_rel_multiplier": 1.0,
                   "kunbind_anoxia": 0.0042,
                   "k_AP_multiplier": 0.0}

    t_eval_dict = {'pre_polarisation': {"dt": 10, "tfin": 3e4},
                   'polarisation': {"dt": 10, "tfin": 1e3},
                   'NEBD': {"dt": 10, "tfin": 1e3},
                   'anoxia': {"dt": 10, "tfin": 24*60*60}}

    sim = Simulate(param_dict, anoxia_dict, t_eval_dict)

    df = pd.read_csv("fitting/27072024_fitting/data/ASI_normalised.csv", index_col=0)

    t_span_data = np.arange(0, 62., 2.)
    t_span_data_used = np.arange(0, 60., 2.)
    asi_mat = df[t_span_data_used.astype(str)].values
    is_kd = df["KD"].values
    is_early = (df["Stage"] == "early maint.").values

    asi_norm = np.zeros((2, 2, len(t_span_data_used)))
    asi_sd = np.zeros((2, 2, len(t_span_data_used)))

    for i, kd in enumerate([False, True]):
        for j, early in enumerate([True, False]):
            mask = (is_kd == kd) * (is_early == early)
            asi_norm[i, j] = asi_mat[mask].mean(axis=0)
            asi_sd[i, j] = asi_mat[mask].std(axis=0)

    fit_param_names = ['k_onA', 'k_onB_c', 'kbind_c', 'kbind_m', 'k_rel', 'k_seq_multiplier', 'k_rel_multiplier',
                       "tau_anox"]

    log10_fit_params = pd.read_csv("fitting/27072024_fitting/fit_results/opt_param.csv")
    log10_fit_params = log10_fit_params[fit_param_names].values.ravel()

    _param_dict = param_dict.copy()
    _anoxia_dict = anoxia_dict.copy()
    for i, nm in enumerate(fit_param_names):
        assert (nm in _param_dict) or (nm in _anoxia_dict), "Names incorrect"
        if nm in _param_dict:
            _param_dict[nm] = 10.0 ** (log10_fit_params[i])
        else:
            _anoxia_dict[nm] = 10.0 ** (log10_fit_params[i])

    ##impose the constraint that k_onB_c > k_onB_f implicitly through this, given k_seq_multiplier > 1
    _param_dict["k_seq"] = _anoxia_dict["k_rel_multiplier"] * _param_dict["k_rel"] * _param_dict["k_seq_multiplier"]

    param_dict = _param_dict.copy()
    anoxia_dict = _anoxia_dict.copy()

    def get_polarity(B_tot,k_unbind_anoxia):
        _param_dict = param_dict.copy()
        _anoxia_dict = anoxia_dict.copy()
        _anoxia_dict["kunbind_anoxia"] = k_unbind_anoxia
        _param_dict["B_tot"] = B_tot

        ##Simulate WT, from pre and postNEBD
        sim.initialise_param_dicts(_param_dict, _anoxia_dict)
        sim.simulate_pre_and_post()
        sim_values_pre_polarisation = sim.extract_values(sim.y_pre_polarisation).copy()
        sim_values_polarisation = sim.extract_values(sim.y_polarisation).copy()
        sim_values_postNEBD = sim.extract_values(sim.y_postNEBD).copy()
        sim_values_anoxia_preNEBD = sim.extract_values(sim.y_anoxia_preNEBD)
        sim_values_anoxia_postNEBD = sim.extract_values(sim.y_anoxia_postNEBD)
        polarity_preNEBD = sim.get_polarity(sim_values_anoxia_preNEBD)
        polarity_postNEBD = sim.get_polarity(sim_values_anoxia_postNEBD)
        return sim_values_anoxia_preNEBD,sim_values_anoxia_postNEBD,polarity_preNEBD,polarity_postNEBD

    B_tot_range = np.logspace(-3,1,15)
    k_unbind_anoxia_range = np.logspace(-4,-1,15)
    BB,KK = np.meshgrid(B_tot_range,k_unbind_anoxia_range,indexing="ij")

    results = Parallel(n_jobs=-1)(delayed(get_polarity)(B_tot,k_unbind_anoxia) for (B_tot,k_unbind_anoxia) in zip(BB.ravel(),KK.ravel()))

    pol = np.zeros((BB.ravel().shape + (2,)))
    for i, res in enumerate(results):
        pol[i,0] = res[2]["C_pol"][-1]
        pol[i,1] = res[3]["C_pol"][-1]

    pol = pol.reshape(len(B_tot_range),len(k_unbind_anoxia_range),2)

    B_tot_range_fine = np.linspace(np.log10(B_tot_range[0]),np.log10(B_tot_range[-1]),100)
    k_unbind_anoxia_range_fine = np.linspace(np.log10(k_unbind_anoxia_range[0]),np.log10(k_unbind_anoxia_range[-1]),100)
    BBf,KKf = np.meshgrid(B_tot_range_fine,k_unbind_anoxia_range_fine,indexing="ij")

    z_0 = griddata((np.log10(BB).flatten(), np.log10(KK).flatten()), pol[...,0].flatten(), (BBf, KKf), method='cubic')
    z_1 = griddata((np.log10(BB).flatten(), np.log10(KK).flatten()), pol[...,1].flatten(), (BBf, KKf), method='cubic')


    def make_extent(x_range, y_range, xscale="linear", yscale="linear", center=True):
        if xscale == "log":
            x_range = np.log10(x_range)
        if yscale == "log":
            y_range = np.log10(y_range)
        if center is False:
            extent = [x_range[0], x_range[-1] + x_range[1] - x_range[0], y_range[0],
                      y_range[-1] + y_range[1] - y_range[0]]
        else:
            extent = [x_range[0] - (x_range[1] - x_range[0]) / 2, x_range[-1] + (x_range[1] - x_range[0]) / 2,
                      y_range[0] - (y_range[1] - y_range[0]) / 2, y_range[-1] + (y_range[1] - y_range[0]) / 2]

        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        return extent, aspect

    extent,aspect = make_extent(k_unbind_anoxia_range_fine,B_tot_range_fine)

    cmap = sns.color_palette("rocket",as_cmap=True)
    levels = [0.1,0.3,0.5,0.7]
    fig, ax = plt.subplots(1,2,sharex=True,sharey=True)
    for axx in ax:
        format_ax(fig, axx)
    ax[0].imshow(np.flip(z_0[:,],axis=0),vmin=0,vmax=1,extent=extent,aspect=aspect,interpolation="nearest",cmap=cmap)
    ax[0].contour(z_0,levels=levels,colors="white",extent=extent)

    ax[1].imshow(np.flip(z_1,axis=0),vmin=0,vmax=1,extent=extent,aspect=aspect,interpolation="nearest",cmap=cmap)
    ax[1].contour(z_1,levels=levels,colors="white",extent=extent)
    ax[0].set(ylabel=r"$log_{10} \ B_{tot}$",xlabel=r"$log_{10} \ k_{unbind}^{anoxia}$")
    ax[1].set(xlabel=r"$log_{10} \ k_{unbind}^{anoxia}$")
    for axx in ax:
        axx.plot(np.ones_like(B_tot_range_fine)*np.log10(0.0042),B_tot_range_fine,linestyle="--",color="grey")
    ax[0].set_title("Early Maint.")
    ax[1].set_title("Late Maint.")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=1,
                                                                       vmin=0))
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.02, aspect=12, orientation="vertical")
    cl.set_label("Polarity after\n24hr")

    fig.show()
