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
                   'anoxia': {"dt": 1, "tfin": 24*60*60+1}}

    sim = Simulate(param_dict, anoxia_dict, t_eval_dict)

    df = pd.read_csv("ASI_normalised.csv", index_col=0)

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

    log10_fit_params = pd.read_csv("opt_param.csv")
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

    B_tot_range = np.logspace(-3,1,50)
    k_unbind_anoxia_range = np.logspace(-4,-1,50)
    BB,KK = np.meshgrid(B_tot_range,k_unbind_anoxia_range,indexing="ij")
    BB_f, KK_f = BB.ravel(),KK.ravel()

    df_out = pd.read_csv("fitting/k_unbind_B_tot_scan/t24_concat.txt",header=None)
    df_out.columns = ("index",'C_pol_pre', 'B_pol_pre', 'A_membrane_frac_pre', 'B_membrane_frac_pre',
                    'p0_t_A_pre', 'p0_t_P_pre', ' b0_t_A_pre', ' b0_t_P_pre', ' d1_t_A_pre',
                    ' d1_t_P_pre', ' C_t_A_pre', ' C_t_P_pre', ' B_t_A_pre', ' B_t_P_pre',
                    ' m_average_A_pre', ' m_average_P_pre', ' b_frac_A_pre',
                    ' b_frac_P_pre', ' d_frac_A_pre', ' d_frac_P_pre', 'C_pol_post',
                    'B_pol_post', 'A_membrane_frac_post', 'B_membrane_frac_post',
                    'p0_t_A_post', 'p0_t_P_post', ' b0_t_A_post', ' b0_t_P_post',
                    ' d1_t_A_post', ' d1_t_P_post', ' C_t_A_post', ' C_t_P_post',
                    ' B_t_A_post', ' B_t_P_post', ' m_average_A_post', ' m_average_P_post',
                    ' b_frac_A_post', ' b_frac_P_post', ' d_frac_A_post', ' d_frac_P_post',
                    't')
    df_out["idx"] = [int(nm.split("/")[1].strip(".txt")) for nm in df_out["index"]]
    df_out.index = df_out["idx"]
    df_out = df_out.sort_index()
    df_out["C_pol_pre"].values

    plt.imshow(df_out["C_pol_pre"].values.reshape(50, 50))
    plt.imshow(df_out["C_pol_post"].values.reshape(50, 50))
    plt.imshow(df_out["A_membrane_frac_post"].values.reshape(50, 50))
    plt.imshow(df_out["B_membrane_frac_post"].values.reshape(50, 50))
