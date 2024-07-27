import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("fitting/31012023_initial_fitting/data/MC_ASI_aPAR_RNAi_AllData_Nate_NormSelect.csv")
df_meta = df[['EmbryoID', 'CellCycle', 'Stage', 'RNAi']]
df_ASI = df[['ASI_1', 'ASI_2', 'ASI_3',
       'ASI_4', 'ASI_5', 'ASI_6', 'ASI_7', 'ASI_8', 'ASI_9', 'ASI_10',
       'ASI_11', 'ASI_12', 'ASI_13', 'ASI_14', 'ASI_15', 'ASI_16', 'ASI_17',
       'ASI_18', 'ASI_19', 'ASI_20', 'ASI_21', 'ASI_22', 'ASI_23', 'ASI_24',
       'ASI_25', 'ASI_26', 'ASI_27', 'ASI_28', 'ASI_29', 'ASI_30', 'ASI_31']]
df_time = df[[ 'Time_1', 'Time_2', 'Time_3', 'Time_4', 'Time_5', 'Time_6', 'Time_7',
       'Time_8', 'Time_9', 'Time_10', 'Time_11', 'Time_12', 'Time_13',
       'Time_14', 'Time_15', 'Time_16', 'Time_17', 'Time_18', 'Time_19',
       'Time_20', 'Time_21', 'Time_22', 'Time_23', 'Time_24', 'Time_25',
       'Time_26', 'Time_27', 'Time_28', 'Time_29', 'Time_30', 'Time_31']]
df_ASI_Norm = df[['ASI_Norm_1', 'ASI_Norm_2', 'ASI_Norm_3', 'ASI_Norm_4', 'ASI_Norm_5',
       'ASI_Norm_6', 'ASI_Norm_7', 'ASI_Norm_8', 'ASI_Norm_9', 'ASI_Norm_10',
       'ASI_Norm_11', 'ASI_Norm_12', 'ASI_Norm_13', 'ASI_Norm_14',
       'ASI_Norm_15', 'ASI_Norm_16', 'ASI_Norm_17', 'ASI_Norm_18',
       'ASI_Norm_19', 'ASI_Norm_20', 'ASI_Norm_21', 'ASI_Norm_22',
       'ASI_Norm_23', 'ASI_Norm_24', 'ASI_Norm_25', 'ASI_Norm_26',
       'ASI_Norm_27', 'ASI_Norm_28', 'ASI_Norm_29', 'ASI_Norm_30',
       'ASI_Norm_31']]

t_span = np.arange(0,np.nanmax(df_time)+2,2)

def interpolate_ASI(t,asi,t_span):
    interp = interp1d(t,asi)
    return interp(t_span)

ASI_interp = np.zeros((len(df_ASI_Norm),len(t_span)))
for i in range(len(df_ASI_Norm)):
    ASI_interp[i] = interpolate_ASI(df_time.iloc[i], df_ASI_Norm.iloc[i], t_span)

df_combined = pd.DataFrame(ASI_interp)
df_combined.columns = [t for t in t_span]
for key in df_meta.columns:
    df_combined[key] = df_meta[key]

df_combined["KD"] = [nm != "ctrlRNAi" for nm in df_combined["RNAi"]]

df_combined.to_csv("fitting/27072024_fitting/data/ASI_normalised.csv")

df_combined_melt = df_combined.melt(id_vars=tuple(df_meta.columns)+("KD",))


fig, ax = plt.subplots(2,2,sharey=True)
for i, stage in enumerate(["early maint.","late maint."]):
    for j, kd in enumerate([False,True]):
        sns.lineplot(ax=ax[i,j],data=df_combined_melt[(df_combined_melt["Stage"]==stage)*(df_combined_melt["KD"]==kd)],x="variable",y="value",errorbar="sd")

fig.show()
