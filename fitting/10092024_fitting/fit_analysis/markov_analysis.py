"""
This kind of works. I'm presuming that the extremely heavy tail you get from this model
leads to a massive deviation between the median and the mean.

May be reasonable to report the median to give a sense.

This should ultimately be run on real distributions from the simulations rather than the analytical result, which I think has its flaws.

The below builds a per-capita Markov model, considering the cluster distribution as static, and asks what is the stability of


"""

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
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

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


def generate_transition_matrix(pS,params,n_clust,i0,b_frac):
    D_A, D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params
    T = np.zeros((n_clust+1,n_clust+1))

    i = np.arange(n_clust) + 1


    fw_rate_per = kbind_c * (i / i0) ** (1 / 3)
    rv_rate_per = kunbind_c

    p1 = pS[0]
    fw_rate_cluster = fw_rate_per * p1
    fw_rate_monomers = kbind_m * p1 * i
    rv_rate_cluster = i * rv_rate_per * ((i-1)/i) ##per capita, remember
    rv_rate_monomers = i * kunbind_m * ((i-1)/i)

    fw_rate_per_cluster = np.concatenate((fw_rate_monomers[:i0 - 1], fw_rate_cluster[i0 - 1:-1]))
    rv_rate_per_cluster = np.concatenate((rv_rate_monomers[1:i0], rv_rate_cluster[i0:]))

    T[i[:-1],i[1:]] = fw_rate_per_cluster
    T[i[1:],i[:-1]] = rv_rate_per_cluster

    T[1,0] = k_offA*(1-b_frac)

    _fw_rate_cluster = pS*fw_rate_per
    _fw_rate_monomers = pS*kbind_m  * i
    _rv_rate_cluster = i * rv_rate_per * (1/i)
    _rv_rate_monomers = i * kunbind_m * (1/i)

    fw_rate_per_monomer = np.concatenate((_fw_rate_monomers[:i0 - 1], _fw_rate_cluster[i0 - 1:-1]))
    rv_rate_per_monomer = np.concatenate((_rv_rate_monomers[1:i0], _rv_rate_cluster[i0:]))

    T[1,i[1:]] += fw_rate_per_monomer
    T[i[1:],1] += rv_rate_per_monomer
    return T


def get_steady_state_distribution(params, n_clust, i0, b_frac,A_cyto):

    D_A, D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params
    p1 = k_onA*A_cyto/(k_offA*(1-b_frac))

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

def get_mean_first_passage_time(T,state_j,dt=0.01):
    i = np.arange(len(T))
    i = list(i)
    i.remove(state_j)
    i = np.array(i)
    Q = T[i][:,i]
    N = np.linalg.inv(np.eye(len(Q))-Q)
    N_h = np.zeros_like(T)
    N_h[:state_j,:state_j] = N[:state_j,:state_j]
    N_h[state_j+1:,state_j+1:] = N[state_j:,state_j:]
    N_h.sum()
    return N_h.sum(axis=1)*dt

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

b_frac=0
A_cyto=0.08
A_cyto_range = np.linspace(0,0.1,100)
out = np.zeros((100,2))


for i, A_cyto in enumerate(A_cyto_range):
    state_j = 0
    # p_ss[1:] = 0.
    p_ss = get_steady_state_distribution(params, _param_dict["n_clust"], _param_dict["i0"], b_frac=b_frac,
                                         A_cyto=A_cyto)
    p_ss /= (p_ss*np.arange(len(p_ss))).sum()
    p_ss *= (1-A_cyto)

    m_av = (p_ss*np.arange(len(p_ss))).sum()/p_ss.sum()

    x = np.concatenate([np.array([A_cyto]),p_ss])

    _T = generate_transition_matrix(p_ss,params,_param_dict["n_clust"], _param_dict["i0"], b_frac=b_frac)
    dt = 0.001

    T = _T * dt + (1 - dt * _T.sum(axis=1)) * np.eye(len(_T))
    mfpt = get_mean_first_passage_time(T,state_j,dt=dt)
    out[i] = m_av,mfpt[1]

plt.scatter(A_cyto_range,1/out[:,1])
plt.show()


print(1/mfpt[1])

Tp = T.copy()
# Tp[state_j,:] = 0
# Tp[state_j,state_j] = 1
A_cyto_range = np.linspace(0,0.1,10)
out = np.zeros(len(A_cyto_range))
for j, A_cyto in enumerate(A_cyto_range):
    state_j = 0
    p_ss = get_steady_state_distribution(params, _param_dict["n_clust"], _param_dict["i0"], b_frac=b_frac,
                                         A_cyto=A_cyto)

    m_av = (p_ss*np.arange(len(p_ss))).sum()/p_ss.sum()

    x = np.concatenate([np.array([A_cyto]),p_ss])

    _T = generate_transition_matrix(p_ss,params,_param_dict["n_clust"], _param_dict["i0"], b_frac=b_frac)
    dt = 0.1

    T = _T * dt + (1 - dt * _T.sum(axis=1)) * np.eye(len(_T))
    Tp = T.copy()
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

    med_residence_time = interp1d(x_fin_j[:,1],np.arange(0,n_iter*dt,dt))(0.5)
    out[j] = med_residence_time
    print(med_residence_time)
