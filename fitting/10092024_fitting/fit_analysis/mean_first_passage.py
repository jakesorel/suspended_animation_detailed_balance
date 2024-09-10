###BETA: This currently is not correct. I believe because we have a non-linear problem.


import sys
import os
from scipy import sparse

sys.dont_write_bytecode = True

SCRIPT_DIR = "../../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = "../../../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = "../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))
from joblib import Parallel, delayed

from cluster_model.cluster_model import Simulate
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import griddata
from scipy.integrate import solve_ivp
import threading
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.optimize import minimize
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

def f(t,p, n_clust, i0, terminal_state,b_frac,A_cyto):
    D_A,D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params

    i = np.arange(n_clust) + 1

    fw_rate_per = kbind_c * (i / i0) ** (1 / 3)
    rv_rate_per = kunbind_c

    p1 = p[0]
    fw_rate_cluster = p * fw_rate_per * p1
    fw_rate_monomers = p * kbind_m * p1 * i
    rv_rate_cluster = p * i * rv_rate_per
    rv_rate_monomers = p * i * kunbind_m

    fw_rate = np.concatenate((fw_rate_monomers[:i0 - 1], fw_rate_cluster[i0 - 1:-1]))
    rv_rate = np.concatenate((rv_rate_monomers[1:i0], rv_rate_cluster[i0:]))
    rv_rate[terminal_state] = 0. ##absorbing state

    net_rate = fw_rate - rv_rate

    dtp = np.concatenate((np.array([0]), net_rate )) - np.concatenate(
        (net_rate , (np.array([0]))))

    # p1 = d_{1,0} + d_{1,1}
    # p1 = d_{1,0} + b1
    # if only d_{1,0} can unload, then the fraction that can unload is
    dtp1 = k_onA * A_cyto - k_offA * p1*b_frac - net_rate.sum(axis=0)
    dtp[0] += dtp1

    return dtp




def _get_steady_state_distribution(X, params, n_clust, i0, b_frac,A_cyto):
    p1 = X[0]
    D_A, D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params

    i = np.arange(n_clust) + 1

    fw_rate_per = kbind_c * (i / i0) ** (1 / 3)
    rv_rate_per = kunbind_c

    fw_rate_cluster = fw_rate_per * p1
    fw_rate_monomers = kbind_m * p1 * i
    rv_rate_cluster = i * rv_rate_per
    rv_rate_monomers = i * kunbind_m

    fw_rate = np.concatenate((fw_rate_monomers[:i0 - 1], fw_rate_cluster[i0 - 1:-1]))
    rv_rate = np.concatenate((rv_rate_monomers[1:i0], rv_rate_cluster[i0:]))

    p_ss = np.concatenate([np.array([p1]),p1*np.cumproduct(fw_rate/rv_rate)])


    _fw_rate_cluster = p_ss*fw_rate_per*p1
    _fw_rate_monomers = p_ss*kbind_m  * i
    _rv_rate_cluster = p_ss*i * rv_rate_per
    _rv_rate_monomers = p_ss*i * kunbind_m

    fw_rate_per_monomer = np.concatenate((_fw_rate_monomers[:i0 - 1], _fw_rate_cluster[i0 - 1:-1]))
    rv_rate_per_monomer = np.concatenate((_rv_rate_monomers[1:i0], _rv_rate_cluster[i0:]))

    dtp1 = -(fw_rate_per_monomer - rv_rate_per_monomer).sum() +k_onA*A_cyto - k_offA*b_frac*p1 - fw_rate_per_monomer[0] + rv_rate_per_monomer[0]

    return np.abs(dtp1)

def get_steady_state_distribution(params, n_clust, i0, b_frac,A_cyto):

    p1_range = np.logspace(-2,0,5000)
    c = np.array([_get_steady_state_distribution([p1],*(params, n_clust, i0, b_frac,A_cyto)) for p1 in p1_range])
    p1_opt = p1_range[np.argmin(c)]
    for i in range(20):
        lims = [p1_range[np.argsort(c)[0]],p1_range[np.argsort(c)[1]]]
        lims = sorted(lims)
        p1_range = np.linspace(lims[0],lims[1],1000)
        c = np.array([_get_steady_state_distribution([p1],*(params, n_clust, i0, b_frac,A_cyto)) for p1 in p1_range])
        p1_opt =  p1_range[np.argmin(c)]
        print(c.min())
    p1 = p1_opt

    D_A, D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params

    i = np.arange(n_clust) + 1

    fw_rate_per = kbind_c * (i / i0) ** (1 / 3)
    rv_rate_per = kunbind_c

    fw_rate_cluster = fw_rate_per * p1
    fw_rate_monomers = kbind_m * p1 * i
    rv_rate_cluster = i * rv_rate_per
    rv_rate_monomers = i * kunbind_m

    fw_rate = np.concatenate((fw_rate_monomers[:i0 - 1], fw_rate_cluster[i0 - 1:-1]))
    rv_rate = np.concatenate((rv_rate_monomers[1:i0], rv_rate_cluster[i0:]))

    p_ss = np.concatenate([np.array([p1]),p1*np.cumproduct(fw_rate/rv_rate)])
    return p_ss

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
              'n_clust': 64,
              'i0': 3,
              'advection_fraction': 0.99,
              "tau_pol": 60,
              "tau_NEBD": 60,
              "tau_anox": 600}

anoxia_dict = {"k_rel_multiplier": 1.0,
               "kunbind_anoxia": 0.0042,
               "k_AP_multiplier": 0.0}

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


anoxia_param_dict = _param_dict
anoxia_param_dict["k_AP"] *= _anoxia_dict["k_AP_multiplier"]
anoxia_param_dict["kunbind"] = _anoxia_dict["kunbind_anoxia"]
anoxia_param_dict["kunbind_c"] = _anoxia_dict["kunbind_anoxia"]
anoxia_param_dict["kunbind_m"] = _anoxia_dict["kunbind_anoxia"]
anoxia_param_dict["k_rel"] *= _anoxia_dict["k_rel_multiplier"]

param_names = "D_A,D_B,D_C,k_onA,k_offA,k_onB_c,k_offB_f,k_offB_c,kbind_c,kunbind_c,kbind_m,kunbind_m,k_seq,k_rel,A_tot,B_tot,psi,L,k_AP".split(
    ",")

params = np.array([anoxia_param_dict[nm] for nm in param_names])

b_frac=0.3
A_cyto=0.3

state_j = _param_dict["i0"]+1
p_ss = get_steady_state_distribution(params, _param_dict["n_clust"], _param_dict["i0"], b_frac=b_frac,A_cyto=A_cyto)

sol = solve_ivp(f,[0,10],p_ss,args=(_param_dict["n_clust"], _param_dict["i0"], 55,b_frac,A_cyto),t_eval=np.linspace(0,10,100))



x = np.concatenate([np.array([A_cyto]),p_ss])

_T = generate_transition_matrix(p_ss,params,_param_dict["n_clust"], _param_dict["i0"], b_frac=b_frac,A_cyto=A_cyto)
dt = _T.max()/2

T = _T * dt + (1 - dt * _T.sum(axis=1)) * np.eye(len(_T))
mfpt = get_mean_first_passage_time(T,state_j,dt=dt)


Tp = T.copy()
Tp[state_j,:] = 0
Tp[state_j,state_j] = 1


n_iter = 1000
D_fin = np.zeros((n_iter,Tp.shape[0],Tp.shape[1]))
Tps = sparse.csr_matrix(Tp)
Tpsn = Tps
D_fin[0] = Tpsn.A
for i in range(1,n_iter):
    Tpsn = Tpsn*Tps
    D_fin[i] = Tpsn.A



_x_fin_j = np.tensordot(D_fin,np.eye(len(Tp)),axes=([2],[0]))
x_fin_j = _x_fin_j[...,state_j]

d_x_fin_j = (x_fin_j[1:] - x_fin_j[:-1])*dt

plt.plot(d_x_fin_j[:,1])

plt.show()
mfpt = (d_x_fin_j*np.expand_dims(np.arange(len(d_x_fin_j)),1)*dt).sum(axis=0)/dt

plt.plot(mfpt)
