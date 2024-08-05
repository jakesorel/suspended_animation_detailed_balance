import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_cost = pd.read_csv("fitting/27072024_fitting/fit_results/cost_concatenated.csv",header=None)
df_cost_dict = pd.read_csv("fitting/27072024_fitting/fit_results/cost_dict_concatenated.csv",header=None)
df_params = pd.read_csv("fitting/27072024_fitting/fit_results/opt_param_concatenated.csv",header=None)

df_cost.columns = ["index","cost"]
df_cost.index = [int(idx.split("/")[1].split(".txt")[0]) for idx in df_cost["index"]]
df_cost = df_cost.drop("index",axis=1)
df_cost = df_cost.sort_index()

fit_param_names = ['k_onA', 'k_onB_c', 'kbind_c', 'kbind_m', 'k_rel', 'k_seq_multiplier', 'k_rel_multiplier',
                   "tau_anox"]

cost_dict_columns = "ASI,CR1_membrane_frac,B_bound_frac,preNEBD_cluster_size_fold_increase,postNEBD_cluster_size_fold_increase,preNEBD_membrane_frac,postNEBD_membrane_frac,N_clusters,preNEBD_KD_minconc,postNEBD_KD_minconc,preNEBD_minconc,postNEBD_minconc".split(",")

df_cost_dict.columns = ["index"] + cost_dict_columns
df_cost_dict.index = [int(idx.split("/")[1].split(".txt")[0]) for idx in df_cost_dict["index"]]
df_cost_dict = df_cost_dict.drop("index",axis=1)
df_cost_dict = df_cost_dict.sort_index()

df_params.columns = ["index"] + fit_param_names
df_params.index = [int(idx.split("/")[1].split(".txt")[0]) for idx in df_params["index"]]
df_params = df_params.drop("index",axis=1)
df_params = df_params.sort_index()

#####

df_params_opt = df_params.loc[df_cost.index[df_cost["cost"]<3]]
# df_params_opt["cost"] = df_cost.loc[df_cost.index[np.log10(df_cost["cost"])<0.5]]["cost"]




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

df_params_best = df_params.loc[df_cost.index[df_cost["cost"]==df_cost["cost"].min()]]
df_params_best.to_csv("fitting/27072024_fitting/fit_results/opt_param.csv")