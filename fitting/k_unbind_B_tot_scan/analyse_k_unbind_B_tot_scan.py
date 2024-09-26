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

    log10_fit_params = pd.read_csv("fitting/k_unbind_B_tot_scan/opt_param.csv")
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



    fig, ax = plt.subplots(1,2)

    ax[0].imshow(df_out["C_pol_pre"].values.reshape(50, 50))
    ax[1].imshow(df_out["C_pol_post"].values.reshape(50, 50))
    fig.show()


    def format_contour_label(x):
        return " " + r"$10^{%d}$" % x + "  "


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

    extent,aspect = make_extent(np.log10(B_tot_range),np.log10(k_unbind_anoxia_range))

    vmin = 0
    vmax = 1
    levels = np.arange(0,1.25,0.25)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.flip(df_out["C_pol_pre"].values.reshape(50, 50).T, axis=0), extent=extent, aspect=aspect, cmap=plt.cm.viridis, vmin=vmin,
              vmax=vmax,interpolation="bicubic")
    cont = ax.contour(df_out["C_pol_pre"].values.reshape(50, 50).T, extent=extent, levels=levels, cmap=plt.cm.Greys)
    # ax.clabel(cont, levels, zorder=1000)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=vmax, vmin=vmin))
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
    cl.set_label("Polarity after 24hr")
    cl_ticks = [0,0.5,1]
    cl.set_ticks(cl_ticks)
    ax.set(xlabel=r"$B_{tot}$")
    ax.set(ylabel=r"$k_{U}^{anoxia}$")
    ax.scatter(np.log10(param_dict["B_tot"]),np.log10(anoxia_dict["kunbind_anoxia"]),marker=(5, 2),s=60,color="white",zorder=1000)
    ax.annotate('', xy=(-3,np.log10(anoxia_dict["kunbind_anoxia"])), xytext=(np.log10(param_dict["B_tot"])-0.2,np.log10(anoxia_dict["kunbind_anoxia"])), arrowprops=dict(arrowstyle='->', lw=2,color="white",linestyle="--"))

    xtck = np.array([-3, -1, 1])
    ytck = np.array([-4, -3,-2, -1])
    ax.xaxis.set_ticks(xtck)
    ax.xaxis.set_ticklabels([r"$10^{%d}$" % i for i in xtck])
    ax.yaxis.set_ticks(ytck)
    ax.yaxis.set_ticklabels([r"$10^{%d}$" % i for i in ytck])

    ax.set(xlim=(np.log10(B_tot_range[0]), np.log10(B_tot_range[-1])),
           ylim=(np.log10(k_unbind_anoxia_range[0]), np.log10(k_unbind_anoxia_range[-1])))
    fig.subplots_adjust(left=0.3, right=0.8, bottom=0.3, top=0.8)
    fig.savefig("fitting/10092024_fitting/plots/preNEBD_param_scan.pdf")


    vmin = 0
    vmax = 1
    levels = np.arange(0,1.25,0.25)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.flip(df_out["C_pol_post"].values.reshape(50, 50).T, axis=0), extent=extent, aspect=aspect, cmap=plt.cm.viridis, vmin=vmin,
              vmax=vmax,interpolation="bicubic")
    cont = ax.contour(df_out["C_pol_post"].values.reshape(50, 50).T, extent=extent, levels=levels, cmap=plt.cm.Greys)
    # ax.clabel(cont, levels, zorder=1000)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmax=vmax, vmin=vmin))
    cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.073, aspect=12, orientation="vertical")
    cl.set_label("Polarity after 24hr")
    cl_ticks = [0,0.5,1]
    cl.set_ticks(cl_ticks)
    ax.set(xlabel=r"$B_{tot}$")
    ax.set(ylabel=r"$k_{U}^{anoxia}$")
    ax.scatter(np.log10(param_dict["B_tot"]),np.log10(anoxia_dict["kunbind_anoxia"]),marker=(5, 2),s=60,color="white",zorder=1000)
    ax.annotate('', xy=(-3,np.log10(anoxia_dict["kunbind_anoxia"])), xytext=(np.log10(param_dict["B_tot"])-0.2,np.log10(anoxia_dict["kunbind_anoxia"])), arrowprops=dict(arrowstyle='->', lw=2,color="white",linestyle="--"))

    xtck = np.array([-3, -1, 1])
    ytck = np.array([-4, -3,-2, -1])
    ax.xaxis.set_ticks(xtck)
    ax.xaxis.set_ticklabels([r"$10^{%d}$" % i for i in xtck])
    ax.yaxis.set_ticks(ytck)
    ax.yaxis.set_ticklabels([r"$10^{%d}$" % i for i in ytck])

    ax.set(xlim=(np.log10(B_tot_range[0]), np.log10(B_tot_range[-1])),
           ylim=(np.log10(k_unbind_anoxia_range[0]), np.log10(k_unbind_anoxia_range[-1])))
    fig.subplots_adjust(left=0.3, right=0.8, bottom=0.3, top=0.8)
    fig.savefig("fitting/10092024_fitting/plots/postNEBD_param_scan.pdf")



    ##Membrane bound
    def get_df_out(path):
        B_tot_range = np.logspace(-3,1,50)
        k_unbind_anoxia_range = np.logspace(-4,-1,50)
        BB,KK = np.meshgrid(B_tot_range,k_unbind_anoxia_range,indexing="ij")
        BB_f, KK_f = BB.ravel(),KK.ravel()
        df_out = pd.read_csv(path,header=None)
        df_out.columns = ("index","none",'C_pol_pre', 'B_pol_pre', 'A_membrane_frac_pre', 'B_membrane_frac_pre',
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

        df_out["idx"] = [int(nm.split("/")[1].strip(".csv")) for nm in df_out["index"]]
        df_out.index = df_out["idx"]
        df_out = df_out.sort_index()
        df_out["B_tot"] = [BB_f[i] for i in df_out["idx"]]
        df_out["k_unbind_anoxia"] = [KK_f[i] for i in df_out["idx"]]
        df_out["log_B_tot"] = np.log10(df_out["B_tot"])
        df_out["log_k_unbind_anoxia"] = np.log(df_out["k_unbind_anoxia"])

        df_out["F_t_A_pre"] = df_out["p0_t_A_pre"]-df_out[" d1_t_A_pre"]
        df_out["F_t_P_pre"] = df_out["p0_t_P_pre"]-df_out[" d1_t_P_pre"]
        df_out["F_t_A_post"] = df_out["p0_t_A_post"]-df_out[" d1_t_A_post"]
        df_out["F_t_P_post"] = df_out["p0_t_P_post"]-df_out[" d1_t_P_post"]
        df_out["F_frac_t_A_pre"] = df_out["F_t_A_pre"]/df_out[" C_t_A_pre"]
        df_out["F_frac_t_P_pre"] = df_out["F_t_P_pre"]/df_out[" C_t_P_pre"]
        df_out["F_frac_t_A_post"] = df_out["F_t_A_post"]/df_out[" C_t_A_post"]
        df_out["F_frac_t_P_post"] = df_out["F_t_P_post"]/df_out[" C_t_P_post"]
        return df_out


    df_outs = [get_df_out(path) for path in ["fitting/k_unbind_B_tot_scan/t%d_concat.csv"%i for i in (0,10,60,600)]]

    fig, ax = plt.subplots(2,2)
    vmax = np.max((df_out["F_t_A_pre"].max(),df_out["F_t_P_pre"].max(),df_out["F_t_A_post"].max(),df_out["F_t_P_post"].max()))
    ax[0,0].imshow(df_out["F_t_A_pre"].values.reshape(50, 50),vmax=vmax,vmin=0)
    ax[0,0].set_title("A")
    ax[0,1].set_title("P")
    ax[0,0].set(ylabel="PRE")
    ax[1,0].set(ylabel="POST")

    ax[0,1].imshow(df_out["F_t_P_pre"].values.reshape(50, 50),vmax=vmax,vmin=0)
    ax[1, 0].imshow(df_out["F_t_A_post"].values.reshape(50, 50),vmax=vmax,vmin=0)
    ax[1, 1].imshow(df_out["F_t_P_post"].values.reshape(50, 50),vmax=vmax,vmin=0)

    fig.show()



    fig, ax = plt.subplots(2,2)
    # vmax = np.log10(np.max((df_out["F_frac_t_A_pre"].max(),df_out["F_frac_t_P_pre"].max(),df_out["F_frac_t_A_post"].max(),df_out["F_frac_t_P_post"].max())))
    # vmin = np.log10(np.min((df_out["F_frac_t_A_pre"].min(),df_out["F_frac_t_P_pre"].min(),df_out["F_frac_t_A_post"].min(),df_out["F_frac_t_P_post"].min())))
    vmax = np.log10(np.max((df_out["F_frac_t_P_pre"].max(),df_out["F_frac_t_P_post"].max())))
    vmin = np.log10(np.min((df_out["F_frac_t_P_pre"].min(),df_out["F_frac_t_P_post"].min())))
    levels = np.linspace(vmin,vmax,4)

    ax[0,0].imshow(np.log10(np.flip(df_out["F_frac_t_A_pre"].values.reshape(50, 50).T,axis=0)),vmax=vmax,vmin=vmin)
    ax[0,0].contour(np.log10(np.flip(df_out["F_frac_t_A_pre"].values.reshape(50, 50).T,axis=0)),levels=levels,cmap=plt.cm.Reds)

    ax[0,0].set_title("A")
    ax[0,1].set_title("P")
    ax[0,0].set(ylabel="PRE")
    ax[1,0].set(ylabel="POST")

    ax[0,1].imshow(np.log10(np.flip(df_out["F_frac_t_P_pre"].values.reshape(50, 50).T,axis=0)),vmax=vmax,vmin=vmin)
    ax[0,1].contour(np.log10(np.flip(df_out["F_frac_t_P_pre"].values.reshape(50, 50).T,axis=0)),levels=levels,cmap=plt.cm.Reds)

    ax[1, 0].imshow(np.log10(np.flip(df_out["F_frac_t_A_post"].values.reshape(50, 50).T,axis=0)),vmax=vmax,vmin=vmin)
    ax[1, 0].contour(np.log10(np.flip(df_out["F_frac_t_A_post"].values.reshape(50, 50).T,axis=0)),levels=levels,cmap=plt.cm.Reds)

    ax[1, 1].imshow(np.log10(np.flip(df_out["F_frac_t_P_post"].values.reshape(50, 50).T,axis=0)),vmax=vmax,vmin=vmin)
    ax[1, 1].contour(np.log10(np.flip(df_out["F_frac_t_P_post"].values.reshape(50, 50).T,axis=0)),levels=levels,cmap=plt.cm.Reds)

    fig.show()

    fig, ax = plt.subplots()
    # ax.scatter(np.log10(1-df_out["A_membrane_frac_pre"].ravel()), df_out["C_pol_pre"].ravel(),c=np.log10(df_out["k_unbind_anoxia"].ravel()))
    # ax.scatter(np.log10(df_out["F_frac_t_P_pre"].ravel()), df_out["C_pol_pre"].ravel(),c="grey",alpha=0.05)
    ax.scatter(np.log10(df_outs[3]["F_t_P_post"].ravel()), df_out["C_pol_post"].ravel(),c=np.log10(df_out["B_tot"].ravel()))
    # ax.scatter(np.log10(df_out["F_t_P_post"].ravel()), df_out["C_pol_post"].ravel(),c=np.log10(df_out["B_tot"].ravel()))

    # ax.scatter(df_out["A_membrane_frac_post"].ravel(), df_out["C_pol_post"].ravel(),c=np.log10(df_out["k_unbind_anoxia"].ravel()))

    # ax.scatter(np.log10(df_out["F_frac_t_P_post"].ravel()), df_out["C_pol_post"].ravel(),c=np.log10(df_out["B_tot"].ravel()),alpha=0.3)

    fig.show()

    df_red = df_out[['C_pol_pre', 'B_pol_pre', 'A_membrane_frac_pre', 'B_membrane_frac_pre',
                    'p0_t_A_pre', 'p0_t_P_pre', ' b0_t_A_pre', ' b0_t_P_pre', ' d1_t_A_pre',
                    ' d1_t_P_pre', ' C_t_A_pre', ' C_t_P_pre', ' B_t_A_pre', ' B_t_P_pre',
                    ' m_average_A_pre', ' m_average_P_pre', ' b_frac_A_pre',
                    ' b_frac_P_pre', ' d_frac_A_pre', ' d_frac_P_pre', 'C_pol_post',
                    'B_pol_post', 'A_membrane_frac_post', 'B_membrane_frac_post',
                    'p0_t_A_post', 'p0_t_P_post', ' b0_t_A_post', ' b0_t_P_post',
                    ' d1_t_A_post', ' d1_t_P_post', ' C_t_A_post', ' C_t_P_post',
                    ' B_t_A_post', ' B_t_P_post', ' m_average_A_post', ' m_average_P_post',
                    ' b_frac_A_post', ' b_frac_P_post', ' d_frac_A_post', ' d_frac_P_post',"log_B_tot","log_k_unbind_anoxia",
                     "F_frac_t_A_pre","F_frac_t_A_post","F_frac_t_P_pre","F_frac_t_P_post",
                     "F_t_A_pre","F_t_A_post","F_t_P_pre","F_t_P_post"]]

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().set_output(transform="pandas")
    scaled_X_train = scaler.fit_transform(df_red)
    transformed = PCA(n_components=2).fit_transform(scaled_X_train.values)
    pca = PCA(n_components=2).fit(scaled_X_train.values)
    print(np.array(df_red.columns)[np.argsort(pca.components_[0])])

    fig, ax = plt.subplots()
    ax.scatter(*transformed.T,c=df_red["A_membrane_frac_pre"])
    fig.show()

    """
    Potential working hypothesis
    
    initial flux into posterior is quenched by B
    Starting condition is different depepnding on cell cycle state. 
    
    """

