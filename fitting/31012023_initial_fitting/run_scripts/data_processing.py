import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/cornwaj/PycharmProjects/suspended_animation_detailed_balance/fitting/31012023_initial_fitting/data/Intensities_ASI.csv")
df["CellCycle_full"] = [nm.split("_")[-1] for nm in df["EmbryoID"]]
df["CellCycle"] = [nm if nm=="postNEBD" else "preNEBD" for nm in df["CellCycle_full"]]
df["CellCycle_RNAi"] = ["_".join((a,b)) for (a,b) in zip(df["CellCycle_full"],df["RNAi"])]
df["StageSimple_RNAi"] = ["_".join((a,b)) for (a,b) in zip(df["StageSimple"],df["RNAi"])]


MeanMembAntNorm,MeanMembPostNorm,ASI_norm = np.zeros(len(df)),np.zeros(len(df)),np.zeros(len(df))
for emb in df["EmbryoID"].unique():
    mask = df["EmbryoID"] == emb
    dfi = df[mask]
    MeanMembTot = (dfi["MeanMembAnt"] + dfi["MeanMembPost"])/2
    t0_mask = dfi["TimeMin"] < 5
    MeanMembTot = MeanMembTot[t0_mask].values.mean()
    MeanMembAntNorm[mask] = (dfi["MeanMembAnt"])/MeanMembTot
    MeanMembPostNorm[mask] = (dfi["MeanMembPost"])/MeanMembTot
    asi = (dfi["MeanMembAnt"]-dfi["MeanMembPost"])/(dfi["MeanMembAnt"]+dfi["MeanMembPost"])
    ASI_norm[mask] = asi/asi[t0_mask].mean()

df["MeanMembAntNorm"] = MeanMembAntNorm
df["MeanMembPostNorm"] = MeanMembPostNorm
df["ASI_new"] = ASI_norm
dfi = df[df["EmbryoID"] == " 11Jun22_kk1216_ctrlRNAi_CH4_Stage2_postNEBD"]



fig, ax = plt.subplots(2,len(df["RNAi"].unique()),figsize=(12,8))
for i, rnai in enumerate(df["RNAi"].unique()):
    dfi = df[df["RNAi"] == rnai]
    sns.lineplot(ax=ax[0,i],data=dfi,x="TimeMin",y="MeanMembAntNorm",hue="StageSimple")
    sns.lineplot(ax=ax[0,i],data=dfi,x="TimeMin",y="MeanMembAntNorm_model",hue="StageSimple")

    sns.lineplot(ax=ax[1,i],data=dfi,x="TimeMin",y="MeanMembPostNorm",hue="StageSimple")
    sns.lineplot(ax=ax[1,i],data=dfi,x="TimeMin",y="MeanMembPostNorm_model",hue="StageSimple")

# sns.lineplot(data=dfi,x="TimeMin",y="MeanMembPost",hue="EmbryoID")
    ax[0,i].set_title(rnai)
# ax.legend("off")
fig.show()



fig, ax = plt.subplots(1,len(df["RNAi"].unique()),figsize=(8,3),sharey=True)
for i, rnai in enumerate(df["RNAi"].unique()):
    dfi = df[df["RNAi"] == rnai]
    sns.lineplot(ax=ax[i],data=dfi,x="TimeMin",y="ASI_new_model",hue="StageSimple")
    sns.lineplot(ax=ax[i],data=dfi,x="TimeMin",y="ASI_new",hue="StageSimple")

    ax[i].set_title(rnai)

# sns.lineplot(data=dfi,x="TimeMin",y="MeanMembPost",hue="EmbryoID")

# ax.legend("off")
fig.show()


fig, ax = plt.subplots(1,2,sharey=True)
for i, rnai in enumerate(df["RNAi"].unique()[[0,3]]):
    dfi = df[df["RNAi"] == rnai]
    sns.lineplot(ax=ax[i],data=dfi,x="TimeMin",y="ASINormtot1",hue="CellCycle_RNAi")
    sns.lineplot(ax=ax[i],data=dfi,x="TimeMin",y="ASINormtot1",hue="CellCycle_RNAi")

# sns.lineplot(data=dfi,x="TimeMin",y="MeanMembPost",hue="EmbryoID")

# ax.legend("off")
fig.show()


rnai_vals = "ctrlRNAi","cdc42RNAi"
cellcycle_vals = "proMeet","postNEBD"
dt = 5
time_points = np.arange(0,60+dt,dt)
t_mid = (time_points[1:] + time_points[:-1])/2
ant_mem_mat = np.zeros((len(t_mid),len(rnai_vals),len(cellcycle_vals)))
post_mem_mat = np.zeros((len(t_mid),len(rnai_vals),len(cellcycle_vals)))

df_out = {"t":[],"RNAi":[],"CellCycle":[],"RNAi_CellCycle":[],"AntMemNorm":[],"PostMemNorm":[],"ASINormtot1":[],"StageSimple":[]}
for i, t in enumerate(t_mid):
    for j, rnai in enumerate(rnai_vals):
        for k, cc in enumerate(cellcycle_vals):
            dfi = df[(df["CellCycle_full"]==cc)*(df["RNAi"]==rnai)*(df["TimeMin"]>=t-dt/2)*(df["TimeMin"]<t+dt/2)]
            ant_mem_mat[i,j,k] = dfi["MeanMembAntNorm"].mean()
            post_mem_mat[i,j,k] = dfi["MeanMembPostNorm"].mean()
            df_out["t"].append(t)
            df_out["StageSimple"].append(dfi["StageSimple"].values[0])

            df_out["RNAi"].append(rnai)
            df_out["CellCycle"].append(cc)
            df_out["RNAi_CellCycle"].append(rnai + "_" + cc)
            df_out["AntMemNorm"].append(dfi["MeanMembAntNorm"].mean())
            df_out["PostMemNorm"].append(dfi["MeanMembPostNorm"].mean())
            df_out["ASINormtot1"].append(dfi["ASINormtot1"].mean())



df_out = pd.DataFrame(df_out)
t0ASI = {}
for state in df_out["RNAi_CellCycle"].unique():
    dfi = df_out[(df_out["RNAi_CellCycle"]==state)]
    dfit0 = dfi[dfi["t"] == np.min(dfi["t"])]
    t0ASI[state] = dfit0["ASINormtot1"].values[0]

ASINorm2tot1 = []
for i in range(len(df_out)):
    dfi = df_out.iloc[i]
    ASINorm2tot1.append(dfi["ASINormtot1"]/t0ASI[dfi["RNAi_CellCycle"]])

df_out["ASINorm2tot1"] = ASINorm2tot1

df_out.to_csv("../data/intensities_processed.csv")

fig, ax = plt.subplots(1,2,sharey=True)
for i, rnai in enumerate(df["RNAi"].unique()[[0,3]]):
    dfi = df_out[df_out["RNAi"] == rnai]
    sns.lineplot(ax=ax[i],data=dfi,x="t",y="ASINorm2tot1",hue="RNAi_CellCycle")
# sns.lineplot(data=dfi,x="TimeMin",y="MeanMembPost",hue="EmbryoID")

# ax.legend("off")
fig.show()



fig, ax = plt.subplots(2,2)
ax[0,0].plot(ant_mem_mat[:,0])
ax[0,1].plot(ant_mem_mat[:,1])
ax[1,0].plot(post_mem_mat[:,0])
ax[1,1].plot(post_mem_mat[:,1])
fig.show()


fig, ax = plt.subplots(1,len(df["RNAi"].unique()))
for i, rnai in enumerate(df["RNAi"].unique()):
    dfi = df[df["RNAi"] == rnai]
    sns.lineplot(ax=ax[i],data=dfi,x="TimeMin",y="MeanMembAntNorm",hue="CellCycle_RNAi")
    ax[i].plot(t_mid,ant_mem_mat[:, 0])
    ax[i].plot(t_mid,ant_mem_mat[:, 1])
# sns.lineplot(data=dfi,x="TimeMin",y="MeanMembPost",hue="EmbryoID")

# ax.legend("off")
fig.show()


df_TIRF = pd.read_csv("/Users/cornwaj/PycharmProjects/suspended_animation_detailed_balance/fitting/31012023_initial_fitting/data/TIRF_cluster_RNAi.csv")
df_TIRF["ASI"] = (1-df_TIRF["NumParticleFrameNormAreaPAratio"])/(1+df_TIRF["NumParticleFrameNormAreaPAratio"])

fig, ax = plt.subplots(1,len(df_TIRF["RNAiSimple"].unique()),sharey=True)
for i, rnai in enumerate(df_TIRF["RNAiSimple"].unique()):
    dfi = df_TIRF[df_TIRF["RNAiSimple"] == rnai]
    sns.lineplot(ax=ax[i],data=dfi,x="TimeMin",y="ASI",hue="StageSimple")
    ax[i].set_title(rnai)
    # ax[i].plot(t_mid,ant_mem_mat[:, 0])
    # ax[i].plot(t_mid,ant_mem_mat[:, 1])
# sns.lineplot(data=dfi,x="TimeMin",y="MeanMembPost",hue="EmbryoID")

# ax.legend("off")
fig.show()

#(A - P)/(A+P)
#(1 - P/A)/(1+P/A)