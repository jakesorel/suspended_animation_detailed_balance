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
                  "D_B":0.28,
                   'D_C': 0.001,
                   'k_onA': 1e-3,
                   'k_offA': 1.,
                   'k_onB_c': 1e-3,
                   'k_offB_f': 5.4e-3,
                   'k_offB_c': 5.4e-3,
                   'kbind_c': 0.051794745,
                  'kbind_m': 0.051794745,
                   'kunbind': 1/30,
                    "kunbind_postNEBD": 0.1,
                   'k_seq': 0.0071968567,
                   'k_rel': 0.01,
                    'k_seq_multiplier':1.,
                   'A_tot': 1.0,
                   'B_tot': 4.666,
                   'psi': 0.174,
                   'L': 134.6,
                   'k_AP': 1e1,
                   'n_clust': 256,
                    'i0':3,
                    'advection_fraction':0.99,
                  "tau_pol":60,
                  "tau_NEBD":60,
                  "tau_anox":600}

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

    export_path = "results"
    mkdir(export_path)
    mkdir(export_path + "/all")
    mkdir(export_path + "/t24")

    def extract_synopsis(sim_values):
        polarity = sim.get_polarity(sim_values)
        C_pol = polarity["C_pol"]
        B_pol = polarity["B_pol"]
        p0_t = sim_values["p_t"][0,:,:]
        b0_t = sim_values["b_t"][0,:,:]
        d1_t = sim_values["b_t"][1,:,:]
        C_t, B_t, m_average,b_frac,d_frac,A_membrane_frac,B_membrane_frac = [sim_values[key] for key in ['C_t', 'B_t', 'm_average', 'b_frac', 'd_frac','A_membrane_frac', 'B_membrane_frac']]
        out_dct = dict(zip("C_pol,B_pol,A_membrane_frac,B_membrane_frac".split(","),[C_pol,B_pol,A_membrane_frac,B_membrane_frac]))
        for key, val in zip("p0_t, b0_t, d1_t, C_t, B_t, m_average, b_frac, d_frac".split(","),[p0_t, b0_t, d1_t, C_t, B_t, m_average, b_frac, d_frac]):
            out_dct[key+"_A"] = val[0]
            out_dct[key+"_P"] = val[1]
        return out_dct


    def get_result(index, B_tot,k_unbind_anoxia):
        _param_dict = param_dict.copy()
        _anoxia_dict = anoxia_dict.copy()
        _anoxia_dict["kunbind_anoxia"] = k_unbind_anoxia
        _param_dict["B_tot"] = B_tot

        ##Simulate WT, from pre and postNEBD
        sim.initialise_param_dicts(_param_dict, _anoxia_dict)
        sim.simulate_pre_and_post()
        sim_values_anoxia_preNEBD = sim.extract_values(sim.y_anoxia_preNEBD)
        sim_values_anoxia_postNEBD = sim.extract_values(sim.y_anoxia_postNEBD)

        synopsis_preNEBD = extract_synopsis(sim_values_anoxia_preNEBD)
        synopsis_postNEBD = extract_synopsis(sim_values_anoxia_postNEBD)
        out_dct = {}
        for key, val in synopsis_preNEBD.items():
            out_dct[key+"_pre"] = val
        for key, val in synopsis_postNEBD.items():
            out_dct[key+"_post"] = val

        df_out = pd.DataFrame(out_dct)
        df_out["t"] = sim.t_evals["anoxia"]
        for col in df_out.columns:
            df_out[col] = df_out[col].astype(np.float32)
        t_save = np.concatenate([np.arange(0,600,1),np.arange(600,86500,100)])
        df_out_reduced = df_out.loc[t_save]
        df_out_reduced.to_csv(export_path + "/all/" + str(index) + ".csv")
        df_out_final = df_out.iloc[[-1]]
        fl = open(export_path + "/t24/"+ str(index) + ".txt","w")
        fl.write(",".join(df_out_final.values.ravel().astype(np.float32).astype(str)))
        fl.close()
        return

    B_tot_range = np.logspace(-3,1,50)
    k_unbind_anoxia_range = np.logspace(-4,-1,50)
    BB,KK = np.meshgrid(B_tot_range,k_unbind_anoxia_range,indexing="ij")
    BB_f, KK_f = BB.ravel(),KK.ravel()
    index = int(sys.argv[1])
    get_result(index, BB_f[index], KK_f[index])


