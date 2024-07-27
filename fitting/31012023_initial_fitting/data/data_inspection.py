import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("fitting/31012023_initial_fitting/data/Intensities_ASI_Jake.csv")
df["TimeApprox"] = np.round(df["TimeMin"]*2,-1)/2
df["date"] = [int(nm.split("_")[0][-2:]) for nm in df["EmbryoID"]]
df = df[df["EmbryoID"]!=' 22Nov23_kk1216_cdc42RNAi_CH4_Stage4_postNEBD']

df = df[df["date"]==23]
dfnonzero = df[df["TimeMin"]>-1]
norm_by_e = {}
for e in dfnonzero["EmbryoID"].unique():
    dfj = dfnonzero[dfnonzero["EmbryoID"]==e]
    dfi = dfj[dfj["TimeMin"] == dfj["TimeMin"].min()]
    norm_by_e[e] = dfi["ASI"].values
df["norm"] = [norm_by_e[e][0] for e in df["EmbryoID"]]
df["ASINormtot1_new"] = df["ASI"]/df["norm"]



fig, ax = plt.subplots(1,3,sharey=True)
for i, rnai in enumerate(["ctrlRNAi","cdc42RNAi","P6RNAi"]):

    dfi = df[df["RNAi"] == rnai]
    sns.lineplot(ax=ax[i],data=dfi,x="TimeApprox",y="ASINormtot1_new",hue="StageSimple")
    ax[i].set_title(rnai)
    ax[i].set(xlim=(0,60),ylim=(0,1.4))
    # sns.lineplot(ax=ax[i],data=dfi,x="TimeMin",y="ASINormtot1",hue="CellCycle_RNAi")

# sns.lineplot(data=dfi,x="TimeMin",y="MeanMembPost",hue="EmbryoID")

# ax.legend("off")
fig.show()


fig, ax = plt.subplots(sharey=True)
rnai = "cdc42RNAi"
dfj = df[df["RNAi"] == rnai]
for embryo in dfj["EmbryoID"].unique():
    dfi = dfj[(dfj["EmbryoID"] == embryo)*(dfj["StageSimple"]=="late maint.")]
    if len(dfi)>0:
        if "22" in embryo:
            ax.plot(dfi["TimeMin"],dfi["ASINormtot1"],label=embryo)
ax.legend()
ax.set(xlabel="Time",ylabel="ASI norm to t0",xlim=(0,55))
fig.show()
# sns.lineplot(ax=ax[i],data=dfi,x="TimeApprox",y="ASINormtot1",hue="StageSimple")
    # ax[i].set_title(rnai)
    # ax[i].set(xlim=(0,60))
    # sns.lineplot(ax=ax[i],data=dfi,x="TimeMin",y="ASINormtot1",hue="CellCycle_RNAi")

# sns.lineplot(data=dfi,x="TimeMin",y="MeanMembPost",hue="EmbryoID")

