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

def mkdir(nm):
    if not os.path.exists(nm):
        os.mkdir(nm)


df_cost = pd.read_csv("paper_scripts/fit_comparison/out/df_cost.csv",index_col=0)

plot_dir = "paper_scripts/fit_comparison/plots/"
mkdir(plot_dir)

fit_param_names = ['k_onA', 'k_onB_c', 'kbind_c', 'kbind_m', 'k_rel', 'k_seq_multiplier', 'k_rel_multiplier',
                   "tau_anox"]

log_fit_param_names = ["log_%s"%nm for nm in fit_param_names]
"""
Plot histogram of costs
"""
fig, ax = plt.subplots(figsize=(4,3))
ax.hist(np.log10(df_cost["cost"]),bins=100,density=True)
format_ax(fig, ax)
ax.set(xlabel=r"$log_{10}$"" Cost",ylabel="Density")
fig.show()
fig.savefig(plot_dir+"cost_histogram.pdf")

"""
Subset on a threshold of cost
"""
thresh = 2.4
df_cost_opt = df_cost[df_cost["cost"]<thresh]

g = sns.pairplot(df_cost_opt[log_fit_param_names])
g.map_lower(sns.kdeplot, levels=4, color=".2")

df_cost[df_cost["cost"] == np.nanmin(df_cost["cost"])].to_csv("paper_scripts/fit_comparison/out/out_param.csv")

fig, ax = plt.subplots(figsize=(4,3))
nm = log_fit_param_names[7]
ax.hist(df_cost[nm],bins=100,density=True)
ax.hist(df_cost_opt[nm],bins=100,density=True)
ax.set(xlim=(0,4))
format_ax(fig, ax)
ax.set_title(nm)
ax.set(xlabel=r"$log_{10}$"" Cost",ylabel="Density")
fig.show()
fig.savefig(plot_dir+"cost_jointplot.pdf")
