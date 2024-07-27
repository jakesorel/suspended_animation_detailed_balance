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



dfs = []
for i in range(1,5001):
    try:
        cost = float(open("paper_scripts/current_best/cost/%d.txt"%i).readlines()[0].strip("\n"))
        df_cost_dict = pd.read_csv("paper_scripts/current_best/cost_dict/%d.txt"%i)
        opt_param = open("paper_scripts/current_best/opt_param/%d.txt"%i).readlines()[0].strip("\n").split(",")
        opt_param = np.array(opt_param).astype(float)
        fit_param_names = ['k_onA', 'k_onB_c', 'kbind_c','kbind_m', 'k_rel', 'k_seq_multiplier', 'k_rel_multiplier',"tau_anox"]
        dfi = df_cost_dict
        dfi["cost"] = cost
        dfi["index"] = i
        for nm,v in zip(fit_param_names,opt_param):
            dfi["log_"+nm] = v
        dfs += [dfi]
    except:
        print("%d not in file"%i)

df_cost = pd.concat(dfs)
df_cost.index = df_cost["index"]

mkdir("paper_scripts/fit_comparison/out")

df_cost.to_csv("paper_scripts/fit_comparison/out/df_cost.csv")


