"""
Edited Aug 2024 28th
"""
import numpy as np
import pandas as pd
import seaborn as sns
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

df_cost = pd.read_csv("fitting/10092024_fitting/fit_results/cost_concatenated.csv",header=None)
df_cost_dict = pd.read_csv("fitting/10092024_fitting/fit_results/cost_dict_concatenated.csv",header=None)
df_params = pd.read_csv("fitting/10092024_fitting/fit_results/opt_param_concatenated.csv",header=None)

df_cost.columns = ["index","cost"]
df_cost.index = [int(idx.split("/")[1].split(".txt")[0]) for idx in df_cost["index"]]
df_cost = df_cost.drop("index",axis=1)
df_cost = df_cost.sort_index()

fit_param_names = ['k_onA', 'k_onB_c', 'kbind_c', 'kbind_m', 'k_rel', 'k_seq_multiplier', 'k_rel_multiplier',
                   "tau_anox","kunbind_anoxia"]

cost_dict_columns = ['ASI', 'CR1_membrane_frac', 'B_bound_frac', 'preNEBD_cluster_size_fold_increase', 'postNEBD_cluster_size_fold_increase', 'preNEBD_membrane_frac', 'postNEBD_membrane_frac', 'polarisation_g4', 'postNEBD_g4', 'preNEBD_KD_minconc', 'postNEBD_KD_minconc', 'preNEBD_minconc', 'postNEBD_minconc', 'cluster_size_regularisation_preNEBD', 'cluster_size_regularisation_postNEBD', 'cluster_size_regularisation_preNEBD_KD', 'cluster_size_regularisation_postNEBD_KD']

df_cost_dict.columns = ["index"] + cost_dict_columns
df_cost_dict.index = [int(idx.split("/")[1].split(".txt")[0]) for idx in df_cost_dict["index"]]
df_cost_dict = df_cost_dict.drop("index",axis=1)
df_cost_dict = df_cost_dict.sort_index()

df_params.columns = ["index"] + fit_param_names
df_params.index = [int(idx.split("/")[1].split(".txt")[0]) for idx in df_params["index"]]
df_params = df_params.drop("index",axis=1)
df_params = df_params.sort_index()

indices = [set(list(dfi.index)) for dfi in [df_cost_dict,df_params,df_cost]]
shared_idx = list(indices[0].intersection(indices[1].intersection(indices[2])))

df_cost_dict = df_cost_dict.loc[shared_idx]
df_cost = df_cost.loc[shared_idx]
df_params = df_params.loc[shared_idx]

#####

df_params_opt = df_params.loc[df_cost.index[df_cost["cost"]<np.percentile(df_cost["cost"],5)]]
# df_params_opt["cost"] = df_cost.loc[df_cost.index[np.log10(df_cost["cost"])<0.5]]["cost"]


i = df_cost.index[df_cost["cost"]==df_cost["cost"].min()]

cost_dict = dict(df_cost_dict.iloc[i])

ground_truths \
    = {"CR1_membrane_frac": 0.05,
       "B_bound_frac": 0.2,
       "preNEBD_cluster_size_fold_increase": 2.,
       "postNEBD_cluster_size_fold_increase": 4.,
       "preNEBD_membrane_frac": 0.3,
       "postNEBD_membrane_frac": 0.15,
       "polarisation_g4": 8.6 / 100,
       "postNEBD_g4": 0.4 / 100,
       }

cost_weighting = {"ASI": 10,
                  "CR1_membrane_frac": 1,
                  "B_bound_frac": 1.,
                  "preNEBD_cluster_size_fold_increase": 1 / ground_truths["preNEBD_cluster_size_fold_increase"] ** 2,
                  "postNEBD_cluster_size_fold_increase": 1 / ground_truths["postNEBD_cluster_size_fold_increase"] ** 2,
                  "preNEBD_membrane_frac": 4.,
                  "postNEBD_membrane_frac": 4.,
                  "preNEBD_minconc": 0.1,
                  "postNEBD_minconc": 0.1,
                  "preNEBD_KD_minconc": 0.1,
                  "postNEBD_KD_minconc": 0.1,
                  "polarisation_g4": 4.,
                  "postNEBD_g4": 4.,
                  "cluster_size_regularisation_preNEBD": 4,
                  "cluster_size_regularisation_postNEBD": 4,
                  "cluster_size_regularisation_preNEBD_KD": 4,
                  "cluster_size_regularisation_postNEBD_KD": 4

                  }


def get_cost(i):
    cost_dict = dict(df_cost_dict.loc[i])
    cost_weighted = np.array([cost_weighting[key] * cost_dict[key] for key in cost_weighting.keys()])
    cost = cost_weighted.sum()
    return cost

costs_recalculated = []
for i in df_cost_dict.index:
    costs_recalculated += [get_cost(i)]



log10_fit_param_lims = {'k_onA':[-4,0],
                      'k_onB_c':[-3,2],
                      'kbind_c':[-np.infty,np.infty],
                      'kbind_m': [-np.infty, np.infty],
                      'k_rel':[-np.infty,np.infty],
                      'k_seq_multiplier':[0,2], ##to impose the k_onBf/konB_c constraint.
                      'k_rel_multiplier':[-3,0],
                        "tau_anox":[1,3],
                        "kunbind_anoxia":[-3,-2]}

log10_fit_param_lims_init = log10_fit_param_lims.copy()
for key, val in log10_fit_param_lims_init.items():
    mn, mx = val
    if np.isinf(mn):
        mn = -2
    if np.isinf(mx):
        mx = 2
    log10_fit_param_lims_init[key] = [mn, mx]

fig, ax = plt.subplots(1,df_params_opt.shape[1],figsize=(15,3))
for i, key in enumerate(df_params_opt.columns):
    axj = ax[i].twinx()
    kde1 = sns.kdeplot(x=df_params.loc[df_cost.index[df_cost["cost"]<=np.percentile(df_cost["cost"],80)]][key],color="grey",fill=True,ax=ax[i])
    kde2 = sns.kdeplot(x=df_params.loc[df_cost.index[df_cost["cost"]<=sorted(list(df_cost["cost"]))[20]]][key],color="magenta",fill=True,ax=axj)
    ylim1 = ax[i].get_ylim()
    ylim2 = axj.get_ylim()
    ax[i].plot(log10_fit_param_lims_init[key],(0,0),lw=3,color="black")
    ax[i].scatter(log10_fit_param_lims_init[key],(0,0),color="black",s=50)
    dx = np.diff(log10_fit_param_lims_init[key])
    lim = log10_fit_param_lims_init[key]
    ax[i].set(xlim=(lim[0]-dx,lim[1]+dx))
    axj.set(xlim=(lim[0]-dx,lim[1]+dx))

    ax[i].set(ylim=(-ylim1[1]*0.2,ylim1[1]))
    axj.set(ylim=(-ylim2[1]*0.2,ylim2[1]),ylabel="")
    format_ax(fig, ax[i])
    format_ax(fig, axj)
    axj.set_yticks([])
    ax[i].set_yticks([])
    ax[i].spines[['right', 'top',"left"]].set_visible(False)

    ax[i].set(ylabel="")
    ax[i].set_xlabel(xlabel=r"$log_{10}$"+" " + key,rotation=45,fontsize=5)
fig.savefig("fitting/27072024_fitting/plots/fit distribution top100 vs top50percent vs initial.pdf")

from sklearn.decomposition import PCA

data = df_params_opt.values
# Standardize the data
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data_standardized = (data - data_mean) / data_std

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_standardized)

print("Principal Components:\n", principal_components)

# Plot the data
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.title('PCA of 2D Matrix')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


sns.pairplot(df_params_opt)
plt.show()

df_params_best = df_params.loc[df_cost.index[df_cost["cost"]==sorted(df_cost["cost"].values)[0]]]
df_params_best.to_csv("fitting/10092024_fitting/fit_results/opt_param.csv")


cost_dict_best = df_cost_dict.loc[df_cost.index[df_cost["cost"]==sorted(df_cost["cost"].values)[0]]]



fig, ax = plt.subplots(1,df_params_opt.shape[1],figsize=(15,3))
for i, key in enumerate(df_params_opt.columns):
    axj = ax[i].twinx()
    kde1 = sns.kdeplot(x=df_params.loc[df_cost.index[df_cost["cost"]<=np.percentile(df_cost["cost"],90)]][key],color="grey",fill=True,ax=ax[i])
    kde2 = sns.kdeplot(x=df_params.loc[df_cost.index[df_cost["cost"]<=sorted(list(df_cost["cost"]))[20]]][key],color="magenta",fill=True,ax=axj)
    ylim1 = ax[i].get_ylim()
    ylim2 = axj.get_ylim()
    ax[i].plot(log10_fit_param_lims_init[key],(0,0),lw=3,color="black")
    ax[i].scatter(log10_fit_param_lims_init[key],(0,0),color="black",s=50)
    dx = np.diff(log10_fit_param_lims_init[key])
    lim = log10_fit_param_lims_init[key]
    ax[i].set(xlim=(lim[0]-dx,lim[1]+dx))
    axj.set(xlim=(lim[0]-dx,lim[1]+dx))

    ax[i].set(ylim=(-ylim1[1]*0.2,ylim1[1]))
    axj.set(ylim=(-ylim2[1]*0.2,ylim2[1]),ylabel="")
    format_ax(fig, ax[i])
    format_ax(fig, axj)
    axj.set_yticks([])
    ax[i].set_yticks([])
    ax[i].spines[['right', 'top',"left"]].set_visible(False)

    ax[i].set(ylabel="")
    ax[i].set_xlabel(xlabel=r"$log_{10}$"+" " + key,rotation=45,fontsize=5)

for i in range(len(log10_fit_params)):
    ax[i].scatter(log10_fit_params[i],0,color="red",s=50)

fig.show()


print(pd.DataFrame(dict(zip(df_params_opt.columns,log10_fit_params)),index=[0]).T)
fig, ax = plt.subplots(1,df_params_opt.shape[1],figsize=(15,3))
for i, key in enumerate(df_params_opt.columns):
    axj = ax[i].twinx()
    ax[i].plot(log10_fit_param_lims_init[key],(0,0),lw=3,color="black")
    ax[i].scatter(log10_fit_param_lims_init[key],(0,0),color="black",s=10)
    dx = np.diff(log10_fit_param_lims_init[key])
    lim = log10_fit_param_lims_init[key]
    ax[i].set(xlim=(lim[0]-dx,lim[1]+dx))
    axj.set(xlim=(lim[0]-dx,lim[1]+dx))

    ax[i].set(ylim=(-ylim1[1]*0.2,ylim1[1]))
    axj.set(ylim=(-ylim2[1]*0.2,ylim2[1]),ylabel="")
    format_ax(fig, ax[i])
    format_ax(fig, axj)
    axj.set_yticks([])
    ax[i].set_yticks([])
    ax[i].spines[['right', 'top',"left"]].set_visible(False)

    ax[i].set(ylabel="")
    ax[i].set_xlabel(xlabel=r"$log_{10}$"+" " + key,rotation=45,fontsize=5)

for i in range(len(log10_fit_params)):
    ax[i].scatter(log10_fit_params[i],0,color="red",s=10)

fig.show()