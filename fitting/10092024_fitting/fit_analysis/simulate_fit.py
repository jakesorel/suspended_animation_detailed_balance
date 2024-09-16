import sys
import os

sys.dont_write_bytecode = True

SCRIPT_DIR = "../../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = "../../../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = "../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))

from cluster_model.cluster_model import Simulate
import numpy as np
import pandas as pd
import seaborn as sns
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


try:
    import thread
except ImportError:
    import _thread as thread
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import pickle
import gzip
from scipy.interpolate import splrep, BSpline


# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

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
                   'anoxia': {"dt": 10, "tfin": 3720.}}

    sim = Simulate(param_dict, anoxia_dict, t_eval_dict)

    df = pd.read_csv("fitting/10092024_fitting/data/ASI_normalised.csv", index_col=0)

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

    log10_fit_params = pd.read_csv("fitting/10092024_fitting/fit_results/opt_param.csv")
    log10_fit_params = log10_fit_params[fit_param_names].values.ravel()

    _param_dict = param_dict.copy()
    _anoxia_dict = anoxia_dict.copy()
    for i, nm in enumerate(fit_param_names):
        assert (nm in _param_dict) or (nm in _anoxia_dict), "Names incorrect"
        if nm in _param_dict:
            _param_dict[nm] = 10.0 ** (log10_fit_params[i])
        else:
            _anoxia_dict[nm] = 10.0 ** (log10_fit_params[i])
    # _anoxia_dict["kunbind_anoxia"] = 1/30

    # _anoxia_dict["kunbind_anoxia"] = 1/30
    # _anoxia_dict["k_rel_multiplier"] = 1.

    ##impose the constraint that k_onB_c > k_onB_f implicitly through this, given k_seq_multiplier > 1
    _param_dict["k_seq"] = _anoxia_dict["k_rel_multiplier"] * _param_dict["k_rel"] * _param_dict["k_seq_multiplier"]

    _param_dict_KD = _param_dict.copy()
    _param_dict_KD["B_tot"] = 0

    _param_dict_CR1_mutant = _param_dict.copy()
    _param_dict_CR1_mutant["kbind_m"] = 0
    _param_dict_CR1_mutant["kbind_c"] = 0

    ##Simulate CR1 mutant
    sim.initialise_param_dicts(_param_dict_CR1_mutant, _anoxia_dict)
    sim.simulate(pre_polarisation_only=True)
    sim_values_CR1_pre_polarisation = sim.extract_values(sim.y_pre_polarisation).copy()

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

    ##Simulate KD, from pre and postNEBD
    sim.initialise_param_dicts(_param_dict_KD, _anoxia_dict)
    sim.simulate_pre_and_post()
    sim_values_polarisation_KD = sim.extract_values(sim.y_polarisation).copy()
    sim_values_postNEBD_KD = sim.extract_values(sim.y_postNEBD).copy()
    sim_values_anoxia_preNEBD_KD = sim.extract_values(sim.y_anoxia_preNEBD)
    sim_values_anoxia_postNEBD_KD = sim.extract_values(sim.y_anoxia_postNEBD)
    polarity_preNEBD_KD = sim.get_polarity(sim_values_anoxia_preNEBD_KD)
    polarity_postNEBD_KD = sim.get_polarity(sim_values_anoxia_postNEBD_KD)

    col_dict = {1:"#ba3579",0:"#9772a0"}
    col_dict_pastel = {1:"#B5839B",0:"#A69DAC"}

    fig, ax = plt.subplots(1,2, sharey=True,figsize=(8.5,4))
    for i, c_pol in enumerate([polarity_preNEBD["C_pol"],
                               polarity_postNEBD["C_pol"],
                               polarity_preNEBD_KD["C_pol"],
                               polarity_postNEBD_KD["C_pol"]]):
        ...
        ax[i//2].plot(sim.t_evals["anoxia"] / 60, c_pol,color=col_dict[i%2])
        ax[i//2].scatter(t_span_data_used, asi_norm.reshape(4, -1)[i],s=5,color=col_dict_pastel[i%2])
        ax[i//2].plot(t_span_data_used, asi_norm.reshape(4, -1)[i],linewidth=0.5,color=col_dict_pastel[i%2])

        ax[i//2].fill_between(t_span_data_used, asi_norm.reshape(4, -1)[i]-asi_sd.reshape(4, -1)[i],asi_norm.reshape(4, -1)[i]+asi_sd.reshape(4, -1)[i],alpha=0.5,linewidth=0,color=col_dict_pastel[i%2],zorder=-100)
        format_ax(fig, ax[i//2])
    for axx in ax:
        axx.set(xlim=(0, None),ylim=(0,None))
        axx.set_yticks([0,0.5,1],labels=["0","0.5","1"])
        axx.yaxis.grid(True)

    ax[0].set(ylabel="ASI")
    ax[0].set_title("ctrl (RNAi)")
    ax[1].set_title("aPAR (RNAi)")
    ax[1].set(xlabel="Time (min)")
    for axx in ax:
        axx.set(xlim=(0,58))
    fig.show()

    fig.savefig("fitting/10092024_fitting/plots/fit.pdf")

    ##Plot the cluster distribution

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



    max_val = 0
    fig, ax = plt.subplots(2,2)

    for axx,sim_i,nm in zip(ax.ravel(),[sim_values_anoxia_preNEBD,
                                     sim_values_anoxia_postNEBD,
                                     sim_values_anoxia_preNEBD_KD,
                                     sim_values_anoxia_postNEBD_KD],
                            ["preNEBD","postNEBD","preNEBD_KD","postNEBD_KD"]):
        im = sim_i["p_t"][:,:,:361]*np.expand_dims(np.arange(1,129),axis=(1,2))
        if np.percentile(im,98) > max_val:
            max_val = np.percentile(im,98)
    fig, ax = plt.subplots(2,2)
    for axx, sim_i, nm in zip(ax.ravel(), [sim_values_anoxia_preNEBD,
                                           sim_values_anoxia_postNEBD,
                                           sim_values_anoxia_preNEBD_KD,
                                           sim_values_anoxia_postNEBD_KD],
                              ["preNEBD", "postNEBD", "preNEBD_KD", "postNEBD_KD"]):
        im = sim_i["p_t"][:, 0, :361] * np.expand_dims(np.arange(1, 129), 1)
        extent,aspect = make_extent(np.arange(0,3610,10)/60,np.arange(1,120))
        axx.imshow(np.flip(im[:255],axis=0),vmin=0,vmax=max_val,extent=extent,aspect=aspect,cmap=sns.color_palette("Spectral",as_cmap=True))
        axx.set(ylabel=nm)
    fig.show()

    fig.savefig("fitting/10092024_fitting/plots/aPAR anterior protein distribution.pdf")

    fig, ax = plt.subplots(2,2)
    for axx, sim_i, nm in zip(ax.ravel(), [sim_values_anoxia_preNEBD,
                                           sim_values_anoxia_postNEBD,
                                           sim_values_anoxia_preNEBD_KD,
                                           sim_values_anoxia_postNEBD_KD],
                              ["preNEBD", "postNEBD", "preNEBD_KD", "postNEBD_KD"]):
        im = sim_i["p_t"][:, 1, :361] * np.expand_dims(np.arange(1, 129), 1)
        extent,aspect = make_extent(np.arange(0,3610,10)/60,np.arange(1,129))
        axx.imshow(np.flip(im[:100],axis=0),vmin=0,vmax=max_val,extent=extent,aspect=aspect,cmap=sns.color_palette("Spectral",as_cmap=True))
        axx.set(ylabel=nm)
    fig.show()
    fig.savefig("fitting/10092024_fitting/plots/aPAR posterior protein distribution.pdf")

    fig, ax = plt.subplots(figsize=(4,4))

    cax = ax.imshow(im, cmap=sns.color_palette("Spectral",as_cmap=True), vmin=0, vmax=max_val)

    cbar = fig.colorbar(cax, ax=ax)

    cbar.set_label('PAR3 mass concentration')

    fig.savefig("fitting/27072024_fitting/plots/colorbar protein distribution.pdf")

    ### plot other features

    fig, ax = plt.subplots(figsize=(7,4))
    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.6, top=0.8)
    for i, (sim_i,nm) in enumerate(zip([sim_values_anoxia_preNEBD,
                                           sim_values_anoxia_postNEBD,
                                           sim_values_anoxia_preNEBD_KD,
                                           sim_values_anoxia_postNEBD_KD],["preNEBD","postNEBD","preNEBD_KD","postNEBD_KD"])):

        linestyle = "-"
        if i//2 == 1:
            print("yes")
            linestyle = "--"
        ax.plot(sim.t_evals["anoxia"] / 60, sim_i["m_average"][0], color = col_dict[i % 2],label=nm,linestyle=linestyle)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    ax.set(xlim=(0, 20), xlabel="Time (min)", ylabel="Cluster size")

    fig.savefig("fitting/27072024_fitting/plots/cluster size.pdf")


    fig, ax = plt.subplots(figsize=(7,4))
    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.6, top=0.8)
    for i, (sim_i,nm) in enumerate(zip([sim_values_anoxia_preNEBD,
                                           sim_values_anoxia_postNEBD,
                                           sim_values_anoxia_preNEBD_KD,
                                           sim_values_anoxia_postNEBD_KD],["preNEBD","postNEBD","preNEBD_KD","postNEBD_KD"])):

        linestyle = "-"
        if i//2 == 1:
            print("yes")
            linestyle = "--"
        ax.plot(sim.t_evals["anoxia"] / 60, sim_i["C_t"][0], color = col_dict[i % 2],label=nm,linestyle=linestyle)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    ax.set(xlim=(0, 60), xlabel="Time (min)", ylabel=r"$[A]_{Anterior}$")

    fig.savefig("fitting/27072024_fitting/plots/conc anterior.pdf")



    fig, ax = plt.subplots(figsize=(7,4))
    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.6, top=0.8)
    for i, (sim_i,nm) in enumerate(zip([sim_values_anoxia_preNEBD,
                                           sim_values_anoxia_postNEBD,
                                           sim_values_anoxia_preNEBD_KD,
                                           sim_values_anoxia_postNEBD_KD],["preNEBD","postNEBD","preNEBD_KD","postNEBD_KD"])):

        linestyle = "-"
        if i//2 == 1:
            print("yes")
            linestyle = "--"
        ax.plot(sim.t_evals["anoxia"] / 60, sim_i["C_t"][1], color = col_dict[i % 2],label=nm,linestyle=linestyle)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    ax.set(xlim=(0, 60), xlabel="Time (min)", ylabel=r"$[A]_{Posterior}$")

    fig.savefig("fitting/27072024_fitting/plots/conc posterior.pdf")


    ##Without active feedback

    _param_dict = param_dict.copy()
    _anoxia_dict = anoxia_dict.copy()
    for i, nm in enumerate(fit_param_names):
        assert (nm in _param_dict) or (nm in _anoxia_dict), "Names incorrect"
        if nm in _param_dict:
            _param_dict[nm] = 10.0 ** (log10_fit_params[i])
        else:
            _anoxia_dict[nm] = 10.0 ** (log10_fit_params[i])
    # _anoxia_dict["kunbind_anoxia"] = 1/30
    _anoxia_dict["k_rel_multiplier"] = 1.

    _anoxia_dict_preNEBD = _anoxia_dict.copy()
    _anoxia_dict_postNEBD = _anoxia_dict.copy()
    _anoxia_dict_preNEBD["kunbind_anoxia"] = 1/30
    _anoxia_dict_postNEBD["kunbind_anoxia"] = 0.1


    ##impose the constraint that k_onB_c > k_onB_f implicitly through this, given k_seq_multiplier > 1
    _param_dict["k_seq"] = _anoxia_dict["k_rel_multiplier"] * _param_dict["k_rel"] * _param_dict["k_seq_multiplier"]

    _param_dict_KD = _param_dict.copy()
    _param_dict_KD["B_tot"] = 0

    _param_dict_CR1_mutant = _param_dict.copy()
    _param_dict_CR1_mutant["kbind"] = 0

    ##Simulate WT, from pre and postNEBD
    sim.initialise_param_dicts(_param_dict, _anoxia_dict_preNEBD)
    sim.simulate_pre_and_post()
    nof_polarity_preNEBD = sim.get_polarity( sim.extract_values(sim.y_anoxia_preNEBD))
    sim.initialise_param_dicts(_param_dict, _anoxia_dict_postNEBD)
    sim.simulate_pre_and_post()
    nof_polarity_postNEBD = sim.get_polarity( sim.extract_values(sim.y_anoxia_postNEBD))


    ##Simulate KD, from pre and postNEBD
    sim.initialise_param_dicts(_param_dict_KD, _anoxia_dict_preNEBD)
    sim.simulate_pre_and_post()
    nof_polarity_preNEBD_KD = sim.get_polarity( sim.extract_values(sim.y_anoxia_preNEBD))
    sim.initialise_param_dicts(_param_dict_KD, _anoxia_dict_postNEBD)
    sim.simulate_pre_and_post()
    nof_polarity_postNEBD_KD = sim.get_polarity( sim.extract_values(sim.y_anoxia_postNEBD))

    col_dict = {1:"#ba3579",0:"#9772a0"}
    col_dict_pastel = {1:"#B5839B",0:"#A69DAC"}

    fig, ax = plt.subplots(1,2, sharey=True,figsize=(8.5,4))
    for i, c_pol in enumerate([polarity_preNEBD["C_pol"],
                               polarity_postNEBD["C_pol"],
                               polarity_preNEBD_KD["C_pol"],
                               polarity_postNEBD_KD["C_pol"]]):
        ...
        ax[i//2].plot(sim.t_evals["anoxia"] / 60, c_pol,color=col_dict[i%2],alpha=0.4)
        format_ax(fig, ax[i//2])
    for i, c_pol in enumerate([nof_polarity_preNEBD["C_pol"],
                               nof_polarity_postNEBD["C_pol"],
                               nof_polarity_preNEBD_KD["C_pol"],
                               nof_polarity_postNEBD_KD["C_pol"]]):
        ...
        ax[i//2].plot(sim.t_evals["anoxia"] / 60, c_pol,color=col_dict[i%2])
        format_ax(fig, ax[i//2])
    for axx in ax:
        axx.set(xlim=(0, None),ylim=(0,None))
        axx.set_yticks([0,0.5,1],labels=["0","0.5","1"])
        axx.yaxis.grid(True)

    ax[0].set(ylabel="ASI")
    ax[0].set_title("ctrl (RNAi)")
    ax[1].set_title("aPAR (RNAi)")
    ax[1].set(xlabel="Time (min)")
    for axx in ax:
        axx.set(xlim=(0,58))
    fig.savefig("fitting/27072024_fitting/plots/fit_without_active_feedback.pdf")

    ###