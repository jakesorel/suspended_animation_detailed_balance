import sys
import os

sys.dont_write_bytecode = True

SCRIPT_DIR = "../../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = "../../../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = "../.."
sys.path.append(os.path.dirname(SCRIPT_DIR))

from cluster_model.cluster_model import Simulate
import numpy as np
import pandas as pd
import threading

try:
    import thread
except ImportError:
    import _thread as thread
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import pickle
import gzip

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

def quit_function(fn_name):
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt

def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            print("Presuming path exists")


if __name__ == "__main__":
    mkdir("../fit_results")
    mkdir("../fit_results/logs")
    mkdir("../fit_results/current_best")
    mkdir("../fit_results/current_best/cost")
    mkdir("../fit_results/current_best/cost_dict")

    mkdir("../fit_results/current_best/log_index")
    mkdir("../fit_results/current_best/opt_param")

    param_dict = {'D_A': 0.2,##ADJUST
                  "D_B":0.28,
                   'D_C': 0.002,  ##ADJUST
                   'k_onA': 1e-3,
                   'k_offA': 5e-3,
                   'k_onB_c': 1e-3,
                   'k_offB_f': 5.4e-3,
                   'k_offB_c': 5.4e-3,
                   'kbind': 0.051794745,
                   'kunbind': 0.02,
                    "kunbind_postNEBD": 0.1,
                   'k_seq': 0.0071968567,
                   'k_rel': 0.01,
                    'k_seq_multiplier':1.,
                   'A_tot': 1.0,
                   'B_tot': 4.666,
                   'psi': 0.174,
                   'L': 134.6,
                   'k_AP': 1e1,
                   'n_clust': 64,
                    'i0':3,
                    'advection_fraction':0.99,
                  "tau_pol":60,
                  "tau_NEBD":60,
                  "tau_anox":600}

    anoxia_dict = {"k_rel_multiplier": 1.0,
                   "kunbind_anoxia": 0.002,
                   "k_AP_multiplier": 0.0}

    t_eval_dict = {'pre_polarisation': {"dt": 10, "tfin": 3e4},
                   'polarisation': {"dt": 10, "tfin": 1e3},
                   'NEBD': {"dt": 10, "tfin": 1e3},
                   'anoxia': {"dt": 10, "tfin": 3720.}}

    sim = Simulate(param_dict, anoxia_dict, t_eval_dict)

    slurm_index = int(sys.argv[1])
    print("Slurm index", slurm_index)

    ##import Joana's data
    # df = pd.read_csv("../data/Intensities_ASI.csv")
    df = pd.read_csv(
        "/Users/cornwaj/PycharmProjects/suspended_animation_detailed_balance/fitting/31012023_initial_fitting/data/Intensities_ASI.csv")

    df["CellCycle_full"] = [nm.split("_")[-1] for nm in df["EmbryoID"]]
    df["CellCycle"] = [nm if nm == "postNEBD" else "preNEBD" for nm in df["CellCycle_full"]]
    df["CellCycle_RNAi"] = ["_".join((a, b)) for (a, b) in zip(df["CellCycle_full"], df["RNAi"])]
    df["StageSimple_RNAi"] = ["_".join((a, b)) for (a, b) in zip(df["StageSimple"], df["RNAi"])]

    MeanMembAntNorm, MeanMembPostNorm, ASI_norm = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for emb in df["EmbryoID"].unique():
        mask = df["EmbryoID"] == emb
        dfi = df[mask]
        MeanMembTot = (dfi["MeanMembAnt"] + dfi["MeanMembPost"]) / 2
        t0_mask = dfi["TimeMin"] < 5
        MeanMembTot = MeanMembTot[t0_mask].values[0] ##normalise total intensity by first frame
        MeanMembAntNorm[mask] = (dfi["MeanMembAnt"]) / MeanMembTot
        MeanMembPostNorm[mask] = (dfi["MeanMembPost"]) / MeanMembTot
        asi = (dfi["MeanMembAnt"] - dfi["MeanMembPost"]) / (dfi["MeanMembAnt"] + dfi["MeanMembPost"])
        ASI_norm[mask] = asi / asi[t0_mask].mean() ##for consistency, normalise by < 5 mins

    df["MeanMembAntNorm"] = MeanMembAntNorm
    df["MeanMembPostNorm"] = MeanMembPostNorm
    df["ASI_new"] = ASI_norm

    fit_param_names = ['k_onA', 'k_offA', 'k_onB_c', 'kbind', 'kunbind', 'k_rel', 'k_seq_multiplier', 'k_rel_multiplier', 'kunbind_anoxia']

    df_out = pd.read_csv("/Users/cornwaj/PycharmProjects/suspended_animation_detailed_balance/fitting/31012023_initial_fitting/data/intensities_processed.csv")
    asi_norm = np.zeros((2,2,12))
    for i, RNAi in enumerate(["ctrlRNAi","cdc42RNAi"]):
        for j, NEBD in enumerate(["early maint.","late maint."]):
            dfi = df_out[(df_out["RNAi"]==RNAi)*(df_out["StageSimple"]==NEBD)]
            asi_norm[i,j] = dfi["ASINorm2tot1"].values

    df_opt = pd.read_csv(
        "/Users/cornwaj/PycharmProjects/suspended_animation_detailed_balance/fitting/31012023_initial_fitting/data/opt_param.csv",
        header=None
    )
    df_opt.columns = fit_param_names
    sns.pairplot(data=df_opt)
    plt.savefig("plots/pairplot.pdf")
    @exit_after(300)
    def run_simulation(log10_fit_params,logger):
        _param_dict = param_dict.copy()
        _anoxia_dict = anoxia_dict.copy()
        for i, nm in enumerate(fit_param_names):
            assert (nm in _param_dict) or (nm in _anoxia_dict), "Names incorrect"
            if nm in _param_dict:
                _param_dict[nm] = 10.0**(log10_fit_params[i])
            else:
                _anoxia_dict[nm] = 10.0**(log10_fit_params[i])
        ##impose the constraint that k_onB_c > k_onB_f implicitly through this, given k_seq_multiplier > 1
        _param_dict["k_seq"] = _anoxia_dict["k_rel_multiplier"]*_param_dict["k_rel"]*_param_dict["k_seq_multiplier"]

        _param_dict_KD = _param_dict.copy()
        _param_dict_KD["B_tot"] = 0

        _param_dict_CR1_mutant = _param_dict.copy()
        _param_dict_CR1_mutant["kbind"] = 0

        ##Simulate CR1 mutant
        sim.initialise_param_dicts(_param_dict_CR1_mutant, _anoxia_dict)
        sim.simulate(pre_polarisation_only=True)
        sim_values_CR1_pre_polarisation = sim.extract_values(sim.y_pre_polarisation).copy()

        ##Simulate WT, from pre and postNEBD
        sim.initialise_param_dicts(_param_dict, _anoxia_dict)
        sim.simulate_pre_and_post()
        sim_values_pre_polarisation = sim.extract_values(sim.y_pre_polarisation).copy()
        sim_values_polarisation = sim.extract_values(sim.y_polarisation).copy()
        sim_values_postNEBD = sim.extract_values(sim.y_postNEBD).copy()
        sim_values_anoxia_preNEBD = sim.extract_values(sim.y_anoxia_preNEBD)
        sim_values_anoxia_postNEBD = sim.extract_values(sim.y_anoxia_postNEBD)
        polarity_preNEBD = sim.get_polarity(sim_values_anoxia_preNEBD)
        polarity_postNEBD = sim.get_polarity(sim_values_anoxia_postNEBD)

        ##Simulate KD, from pre and postNEBD
        sim.initialise_param_dicts(_param_dict_KD, _anoxia_dict)
        sim.simulate_pre_and_post()
        sim_values_polarisation_KD = sim.extract_values(sim.y_polarisation).copy()
        sim_values_postNEBD_KD = sim.extract_values(sim.y_postNEBD).copy()
        sim_values_anoxia_preNEBD_KD = sim.extract_values(sim.y_anoxia_preNEBD)
        sim_values_anoxia_postNEBD_KD = sim.extract_values(sim.y_anoxia_postNEBD)
        polarity_preNEBD_KD = sim.get_polarity(sim_values_anoxia_preNEBD_KD)
        polarity_postNEBD_KD = sim.get_polarity(sim_values_anoxia_postNEBD_KD)

        print("sims complete")
        ###########
        # Data comparison
        ###########

        CR1_membrane_frac = sim_values_CR1_pre_polarisation["A_membrane_frac"][-1]
        B_bound_frac = 1-sim_values_polarisation["b_frac"][0][-1]
        preNEBD_cluster_size_fold_increase = sim_values_anoxia_preNEBD["m_average"][0][-1]/sim_values_anoxia_preNEBD["m_average"][0][0]
        postNEBD_cluster_size_fold_increase = sim_values_anoxia_postNEBD["m_average"][0][-1]/sim_values_anoxia_postNEBD["m_average"][0][0]
        preNEBD_membrane_frac = sim_values_polarisation["A_membrane_frac"][-1]
        postNEBD_membrane_frac = sim_values_postNEBD["A_membrane_frac"][-1]
        surface_area = 4415.84
        N_clusters = surface_area*sim_values_pre_polarisation["p_t"][_param_dict["i0"]:,:,-1].sum(axis=0).mean(axis=0) #Cluster is defined as an oligomer greater than critical level

        model_prediction_ground_truths \
            = {"CR1_membrane_frac":CR1_membrane_frac,
             "B_bound_frac":B_bound_frac,
             "preNEBD_cluster_size_fold_increase":preNEBD_cluster_size_fold_increase,
            "postNEBD_cluster_size_fold_increase":postNEBD_cluster_size_fold_increase,
               "preNEBD_membrane_frac":preNEBD_membrane_frac,
               "postNEBD_membrane_frac":postNEBD_membrane_frac,
               "N_clusters":N_clusters
            }

        ground_truths \
            = {"CR1_membrane_frac":0.05,
             "B_bound_frac":0.2,
             "preNEBD_cluster_size_fold_increase":2.,
            "postNEBD_cluster_size_fold_increase":4.,
               "preNEBD_membrane_frac":0.3,
               "postNEBD_membrane_frac":0.15,
               "N_clusters":400
            }

        ######
        # Processing data
        ######
        #
        # t = sim.t_evals["anoxia"]/60
        # pol_interps = [interp1d(t,c_pol) for c_pol in
        #                [polarity_preNEBD["C_pol"],
        #                 polarity_postNEBD["C_pol"],
        #                 polarity_preNEBD_KD["C_pol"],
        #                 polarity_postNEBD_KD["C_pol"]]]
        #
        # PAR3_A_interps = [interp1d(t,c_t[0]) for c_t in
        #                   [sim_values_anoxia_preNEBD["C_t"],
        #                    sim_values_anoxia_postNEBD["C_t"],
        #                    sim_values_anoxia_preNEBD_KD["C_t"],
        #                    sim_values_anoxia_postNEBD_KD["C_t"]]]
        #
        # PAR3_P_interps = [interp1d(t,c_t[1]) for c_t in
        #                   [sim_values_anoxia_preNEBD["C_t"],
        #                    sim_values_anoxia_postNEBD["C_t"],
        #                    sim_values_anoxia_preNEBD_KD["C_t"],
        #                    sim_values_anoxia_postNEBD_KD["C_t"]]]
        #
        # stagesimpleRNAi_to_index = {'early maint._ctrlRNAi':0,
        #                             'late maint._ctrlRNAi':1,
        #                             'early maint._cdc42RNAi':2,
        #                             'late maint._cdc42RNAi':3}
        #
        # _df = df.copy()
        # _df["pol_model"] = [pol_interps[stagesimpleRNAi_to_index[stagesimpleRNAi]](t) if stagesimpleRNAi in stagesimpleRNAi_to_index else np.nan for (stagesimpleRNAi,t) in zip(df["StageSimple_RNAi"],df["TimeMin"])]
        # _df["PAR3_A_model"] = [PAR3_A_interps[stagesimpleRNAi_to_index[stagesimpleRNAi]](t) if stagesimpleRNAi in stagesimpleRNAi_to_index else np.nan for (stagesimpleRNAi,t) in zip(df["StageSimple_RNAi"],df["TimeMin"])]
        # _df["PAR3_P_model"] = [PAR3_P_interps[stagesimpleRNAi_to_index[stagesimpleRNAi]](t) if stagesimpleRNAi in stagesimpleRNAi_to_index else np.nan for (stagesimpleRNAi,t) in zip(df["StageSimple_RNAi"],df["TimeMin"])]
        #
        #
        # MeanMembAntNorm, MeanMembPostNorm, ASI_norm = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
        # for emb in _df["EmbryoID"].unique():
        #     mask = _df["EmbryoID"] == emb
        #     dfi = _df[mask]
        #     MeanMembTot = (dfi["PAR3_A_model"] + dfi["PAR3_P_model"]) / 2
        #     t0_mask = dfi["TimeMin"] < 5
        #     MeanMembTot = MeanMembTot[t0_mask].values[0] ##normalise total intensity by first frame
        #     MeanMembAntNorm[mask] = (dfi["PAR3_A_model"]) / MeanMembTot
        #     MeanMembPostNorm[mask] = (dfi["PAR3_P_model"]) / MeanMembTot
        #     asi = (dfi["PAR3_A_model"] - dfi["PAR3_P_model"]) / (dfi["PAR3_A_model"] + dfi["PAR3_P_model"])
        #     ASI_norm[mask] = asi / asi[t0_mask].mean() ##for consistency, normalise by < 5 mins
        #
        # _df["MeanMembAntNorm_model"] = MeanMembAntNorm
        # _df["MeanMembPostNorm_model"] = MeanMembPostNorm
        # _df["ASI_new_model"] = ASI_norm


        ##Assemble costs
        cost_dict = {}

        # cost_dict["AnteriorConc"] = np.nanmean(np.abs((_df["MeanMembAntNorm_model"] - _df["MeanMembAntNorm"])))
        # cost_dict["PosteriorConc"] = np.nanmean(np.abs((_df["MeanMembPostNorm_model"] - _df["MeanMembPostNorm"])))
        cost_dict["ASI"] =np.nansum(np.array([np.abs(c_pol[15::30][:12] - asi_norm.reshape(4,12)[i])**2 for i, c_pol in enumerate([polarity_preNEBD["C_pol"],
                        polarity_postNEBD["C_pol"],
                        polarity_preNEBD_KD["C_pol"],
                        polarity_postNEBD_KD["C_pol"]])]))

        for key in ground_truths.keys():
            cost_dict[key] = np.abs(model_prediction_ground_truths[key]-ground_truths[key])**2

        ##impose some minimum concentration
        cost_dict["preNEBD_KD_minconc"] = np.exp(-(sim_values_polarisation_KD["C_t"][0,-1]-0.1)*5)
        cost_dict["postNEBD_KD_minconc"] = np.exp(-(sim_values_postNEBD_KD["C_t"][0,-1]-0.1)*5)
        cost_dict["preNEBD_minconc"] = np.exp(-(sim_values_polarisation["C_t"][0,-1]-0.1)*5)
        cost_dict["postNEBD_minconc"] = np.exp(-(sim_values_postNEBD_KD["C_t"][0,-1]-0.1)*5)

        ##Weight costs

        cost_weighting = {"ASI": 1,
                          "CR1_membrane_frac":1,
                         "B_bound_frac":1,
                         "preNEBD_cluster_size_fold_increase":1/ground_truths["preNEBD_cluster_size_fold_increase"]**2,
                        "postNEBD_cluster_size_fold_increase":1/ground_truths["postNEBD_cluster_size_fold_increase"]**2,
                           "preNEBD_membrane_frac":1,
                           "postNEBD_membrane_frac":1,
                           "N_clusters":1/ground_truths["N_clusters"]**2,
                          "preNEBD_minconc":1,
                          "postNEBD_minconc": 1,
                          "preNEBD_KD_minconc": 1,
                          "postNEBD_KD_minconc": 1,
        }

        cost_weighted = np.array([cost_weighting[key]*cost_dict[key] for key in cost_weighting.keys()])
        cost = cost_weighted.sum()


        current_log = {"log10_fit_params":log10_fit_params,
                       "cost":cost,
                       "cost_dict":cost_dict,
                       "cost_weighted":cost_weighted,
                       "param_dict":_param_dict,
                       "anoxia_dict":_anoxia_dict}

        fig, ax = plt.subplots(figsize=(5,4))
        fig.subplots_adjust(bottom=0.3,left=0.3,right=0.6,top=0.8)
        ax.plot(sim.t_evals["anoxia"]/60,polarity_preNEBD["C_pol"],label="pre")
        ax.plot(sim.t_evals["anoxia"]/60,polarity_postNEBD["C_pol"],label="post")
        ax.plot(sim.t_evals["anoxia"]/60,polarity_preNEBD_KD["C_pol"],label="preKD")
        ax.plot(sim.t_evals["anoxia"]/60,polarity_postNEBD_KD["C_pol"],label="postKD")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        ax.set(xlim=(0,60),ylim=(0.2,1.01),xlabel="t (MIN)",ylabel="Polarity")
        fig.show()

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.subplots_adjust(bottom=0.3, left=0.3, right=0.6, top=0.8)
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_preNEBD["m_average"][0],label="pre")
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_postNEBD["m_average"][0],label="post")
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_preNEBD_KD["m_average"][0],label="preKD")
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_postNEBD_KD["m_average"][0],label="postKD")
        # ax.set(xlim=(0,60))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        ax.set(xlim=(-10,60),xlabel="t (MIN)",ylabel="Cluster size")

        fig.show()

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.subplots_adjust(bottom=0.3, left=0.3, right=0.6, top=0.8)
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_preNEBD["p_t"][5:,0,:].sum(axis=0),label="pre")
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_postNEBD["p_t"][5:,0,:].sum(axis=0),label="post")
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_preNEBD_KD["p_t"][5:,0,:].sum(axis=0),label="preKD")
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_postNEBD_KD["p_t"][5:,0,:].sum(axis=0),label="postKD")
        # ax.set(xlim=(0,60))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        ax.set(xlim=(0,60),xlabel="t (MIN)",ylabel="Cluster number")

        fig.show()

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.subplots_adjust(bottom=0.3, left=0.3, right=0.6, top=0.8)
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_preNEBD["C_t"][0],label="pre")
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_postNEBD["C_t"][0],label="post")
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_preNEBD_KD["C_t"][0],label="preKD")
        ax.plot(sim.t_evals["anoxia"]/60,sim_values_anoxia_postNEBD_KD["C_t"][0],label="postKD")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        ax.set(xlim=(-10,60),xlabel="t (MIN)",ylabel=r"$[A]_{Anterior}$")

        fig.show()
        return cost

    def _run_simulation(log10_fit_params,log):
        try:
            return run_simulation(log10_fit_params,log)
        except:
            return 1e5


    log10_fit_param_lims = {'k_onA':[-4,1],
                          'k_offA':[-0.7,0.7],
                          'k_onB_c':[-4,1],
                          'kbind':[-4,2],
                          'kunbind':[-2,-1.3],
                          'k_seq':[-4,2],
                          'k_rel_multiplier':[-2,0],
                          'kunbind_anoxia':[-2.477,-2.255]}

    log10_fit_params_init = np.array([np.random.uniform(*log10_fit_param_lims[nm]) for nm in fit_param_names])
    log10_fit_params_bounds = np.array([log10_fit_param_lims[nm] for nm in fit_param_names])

    log = []
    res = None
    n_iter = int(1e5)
    lowest_cost = 1e9
    cost_dict = None
    opt_param = None
    log_index = None
    x0 = log10_fit_params_init
    for i in range(n_iter):
        if res is None:
            res = minimize(_run_simulation,
                     x0,
                     args=(log,),
                     method="Nelder-Mead",
                     bounds=log10_fit_params_bounds,
                     options={"maxiter":2,"return_all":True})
        else:
            res = minimize(_run_simulation,
                           x0,
                           args=(log,),
                           method="Nelder-Mead",
                           bounds=log10_fit_params_bounds,
                           options={"maxiter": 2, "return_all": True,"initial_simplex":res["final_simplex"][0]})

        if res.fun < 1e4:##if simulation hits a solving wall or numerical error, randomise
            x0 = res.x
        else:
            idx = int(np.random.random()*len(fit_param_names))
            chosen_mutated = fit_param_names[idx]
            print("Mutating %s"%chosen_mutated)
            new_val = np.random.uniform(*log10_fit_params_bounds[idx])
            x0[idx] = new_val

        costs = np.array([dct["cost"] for dct in log])
        log_index = np.nanargmin(costs)
        cost_dict = log[log_index]["cost_dict"]
        lowest_cost = costs[log_index]
        opt_param = log[log_index]["log10_fit_params"]

        if res.fun < lowest_cost:
            lowest_cost = res.fun
            opt_param = x0.copy()

        with gzip.open("../fit_results/logs/optim_%d.pickle.gz"%slurm_index, "wb") as f:
            pickle.dump(log, f)

        f = open("../fit_results/current_best/cost/%d.txt"%slurm_index,"w")
        f.write(str(lowest_cost) + "\n")
        f.close()

        f = open("../fit_results/current_best/cost_dict/%d.txt"%slurm_index,"w")
        f.write(",".join(list(cost_dict.keys())) + "\n")
        f.write(",".join(np.array(list(cost_dict.values())).astype(str)) + "\n")
        f.close()

        f = open("../fit_results/current_best/log_index/%d.txt"%slurm_index,"w")
        f.write(str(log_index) + "\n")
        f.close()

        f = open("../fit_results/current_best/opt_param/%d.txt"%slurm_index,"w")
        f.write(",".join(list(opt_param.astype(str))) + "\n")
        f.close()


"""
Fits appear to find optima where the 
"""

