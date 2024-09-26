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
from scipy.special import erf
from scipy.interpolate import splrep, BSpline

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

    param_dict = {'D_A': 0.2,
                  "D_B":0.28,
                   'D_C': 0.001,
                   'k_onA': 1e-3,
                   'k_offA': 1.,
                   'k_onB_c': 1e-3,
                   'k_offB_f': 5.4e-3,
                   'k_offB_c': 5.4e-3,
                   'kbind_c': 0.051794745,
                  'kbind_m': 0.051794745,
                   'kunbind': 1/30,
                    "kunbind_postNEBD": 0.1,
                   'k_seq': 0.0071968567,
                   'k_rel': 0.01,
                    'k_seq_multiplier':1.,
                   'A_tot': 1.0,
                   'B_tot': 4.666,
                   'psi': 0.174,
                   'L': 134.6,
                   'k_AP': 1e1,
                   'n_clust': 128,
                    'i0':3,
                    'advection_fraction':0.99,
                  "tau_pol":60,
                  "tau_NEBD":60,
                  "tau_anox":600}


    anoxia_dict = {"k_rel_multiplier": 1.0,
                   "kunbind_anoxia": 0.0042,
                   "k_AP_multiplier": 0.0}

    t_eval_dict = {'pre_polarisation': {"dt": 10, "tfin": 3e4},
                   'polarisation': {"dt": 10, "tfin": 1e3},
                   'NEBD': {"dt": 10, "tfin": 1e3},
                   'anoxia': {"dt": 10, "tfin": 3720.}}

    sim = Simulate(param_dict, anoxia_dict, t_eval_dict)

    slurm_index = int(sys.argv[1])
    print("Slurm index", slurm_index)

    df = pd.read_csv("../data/ASI_normalised.csv",index_col = 0 )

    t_span_data = np.arange(0,62.,2.)
    t_span_data_used = np.arange(0,60.,2.)
    asi_mat = df[t_span_data_used.astype(str)].values
    is_kd = df["KD"].values
    is_early = (df["Stage"] == "early maint.").values

    asi_norm = np.zeros((2,2,len(t_span_data_used)))
    for i,kd in enumerate([False,True]):
        for j,early in enumerate([True,False]):
            mask = (is_kd==kd)*(is_early==early)
            asi_norm[i,j] = asi_mat[mask].mean(axis=0)



    fit_param_names = ['k_onA', 'k_onB_c', 'kbind_c','kbind_m', 'k_rel', 'k_seq_multiplier', 'k_rel_multiplier',"tau_anox"]

    @exit_after(300)
    def run_simulation(log10_fit_params,logger):
        _param_dict = param_dict.copy()
        _anoxia_dict = anoxia_dict.copy()
        for i, nm in enumerate(fit_param_names):
            assert (nm in _param_dict) or (nm in _anoxia_dict), "Names incorrect"
            if nm in _param_dict:
                _param_dict[nm] = 10.0 ** (log10_fit_params[i])
            else:
                _anoxia_dict[nm] = 10.0 ** (log10_fit_params[i])
        # _anoxia_dict["kunbind_anoxia"] = 1/30

        ##impose the constraint that k_onB_c > k_onB_f implicitly through this, given k_seq_multiplier > 1
        _param_dict["k_seq"] = _anoxia_dict["k_rel_multiplier"] * _param_dict["k_rel"] * _param_dict["k_seq_multiplier"]

        _param_dict_KD = _param_dict.copy()
        _param_dict_KD["B_tot"] = 0

        _param_dict_CR1_mutant = _param_dict.copy()
        _param_dict_CR1_mutant["kbind_c"] = 0
        _param_dict_CR1_mutant["kbind_m"] = 0

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
        polarisation_g4 = sim_values_polarisation["p_t"][4:,:,-1].sum()/sim_values_polarisation["p_t"][:,:,-1].sum()*preNEBD_membrane_frac
        postNEBD_g4 = sim_values_postNEBD["p_t"][4:,:,-1].sum()/sim_values_postNEBD["p_t"][:,:,-1].sum()*postNEBD_membrane_frac
        mean_cluster_size = sim_values_polarisation["m_average"][0,-1]


        """
        
        p_{clustered,cell} = p_{clustered,membrane}*p_{membrane}
        """

        model_prediction_ground_truths \
            = {"CR1_membrane_frac":CR1_membrane_frac,
             "B_bound_frac":B_bound_frac,
             "preNEBD_cluster_size_fold_increase":preNEBD_cluster_size_fold_increase,
            "postNEBD_cluster_size_fold_increase":postNEBD_cluster_size_fold_increase,
               "preNEBD_membrane_frac":preNEBD_membrane_frac,
               "postNEBD_membrane_frac":postNEBD_membrane_frac,
               "polarisation_g4":polarisation_g4,
               "postNEBD_g4": postNEBD_g4
               }

        ground_truths \
            = {"CR1_membrane_frac":0.05,
             "B_bound_frac":0.2,
             "preNEBD_cluster_size_fold_increase":2.,
            "postNEBD_cluster_size_fold_increase":4.,
               "preNEBD_membrane_frac":0.3,
               "postNEBD_membrane_frac":0.15,
               "polarisation_g4": 8.6 / 100,
               "postNEBD_g4": 0.4 / 100,
               }

        ##Assemble costs
        cost_dict = {}

        # cost_dict["AnteriorConc"] = np.nanmean(np.abs((_df["MeanMembAntNorm_model"] - _df["MeanMembAntNorm"])))
        # cost_dict["PosteriorConc"] = np.nanmean(np.abs((_df["MeanMembPostNorm_model"] - _df["MeanMembPostNorm"])))
        cost_dict["ASI"] = np.nansum(np.array([np.abs(c_pol[:-12:12] - asi_norm.reshape(4,-1)[i])**2 for i, c_pol in enumerate([polarity_preNEBD["C_pol"],
                        polarity_postNEBD["C_pol"],
                        polarity_preNEBD_KD["C_pol"],
                        polarity_postNEBD_KD["C_pol"]])]))

        for key in ground_truths.keys():
            cost_dict[key] = np.abs(model_prediction_ground_truths[key]-ground_truths[key])**2

        ##impose some minimum concentration
        cost_dict["preNEBD_KD_minconc"] = 0.5-0.5*erf(sim_values_anoxia_preNEBD_KD["C_t"][0].min(axis=-1)-1)
        cost_dict["postNEBD_KD_minconc"] = 0.5-0.5*erf(sim_values_anoxia_postNEBD_KD["C_t"][0].min(axis=-1)-1)
        cost_dict["preNEBD_minconc"] = 0.5-0.5*erf(sim_values_anoxia_preNEBD["C_t"][0].min(axis=-1)-1)
        cost_dict["postNEBD_minconc"] = 0.5-0.5*erf(sim_values_anoxia_postNEBD["C_t"][0].min(axis=-1)-1)
        p = sim_values_polarisation["p_t"][:,0,-1]
        cost_dict["cluster_size_regularisation_preNEBD"] =((erf((np.arange(1,_param_dict["n_clust"]+1)-70)/10)+1)/2 * p/p.sum()).sum()
        p = sim_values_postNEBD["p_t"][:,0,-1]
        cost_dict["cluster_size_regularisation_postNEBD"] =((erf((np.arange(1,_param_dict["n_clust"]+1)-70)/10)+1)/2 * p/p.sum()).sum()
        p = sim_values_polarisation_KD["p_t"][:,0,-1]
        cost_dict["cluster_size_regularisation_preNEBD_KD"] =((erf((np.arange(1,_param_dict["n_clust"]+1)-70)/10)+1)/2 * p/p.sum()).sum()
        p = sim_values_postNEBD_KD["p_t"][:,0,-1]
        cost_dict["cluster_size_regularisation_postNEBD_KD"] =((erf((np.arange(1,_param_dict["n_clust"]+1)-70)/10)+1)/2 * p/p.sum()).sum()

        ##Weight costs

        cost_weighting = {"ASI": 10,
                          "CR1_membrane_frac":1,
                         "B_bound_frac":1.,
                         "preNEBD_cluster_size_fold_increase":1/ground_truths["preNEBD_cluster_size_fold_increase"]**2,
                        "postNEBD_cluster_size_fold_increase":1/ground_truths["postNEBD_cluster_size_fold_increase"]**2,
                           "preNEBD_membrane_frac":4.,
                           "postNEBD_membrane_frac":4.,
                          "preNEBD_minconc":10,
                          "postNEBD_minconc": 10,
                          "preNEBD_KD_minconc": 10,
                          "postNEBD_KD_minconc": 10,
                          "polarisation_g4": 4.,
                          "postNEBD_g4": 4.,
                          "cluster_size_regularisation_preNEBD":4,
                          "cluster_size_regularisation_postNEBD":4,
                          "cluster_size_regularisation_preNEBD_KD":4,
                          "cluster_size_regularisation_postNEBD_KD":4

        }


        cost_weighted = np.array([cost_weighting[key]*cost_dict[key] for key in cost_weighting.keys()])
        cost = cost_weighted.sum()


        current_log = {"log10_fit_params":log10_fit_params,
                       "cost":cost,
                       "cost_dict":cost_dict,
                       "cost_weighted":cost_weighted,
                       "param_dict":_param_dict,
                       "anoxia_dict":_anoxia_dict}
        logger["log"].append(current_log)
        # print(model_prediction_ground_truths)

        logger["costs"] = np.array([dct["cost"] for dct in logger["log"]])
        logger["log_index"] = np.nanargmin(logger["costs"])
        logger["cost_dict"] = logger["log"][logger["log_index"]]["cost_dict"]
        logger["lowest_cost"] = logger["costs"][logger["log_index"]]
        logger["opt_param"] = logger["log"][logger["log_index"]]["log10_fit_params"]
        print(cost)
        if cost < logger["lowest_cost"]:
            logger["lowest_cost"] = cost
            logger["opt_param"] = log10_fit_params

        with gzip.open("../fit_results/logs/optim_%d.pickle.gz"%slurm_index, "wb") as f:
            pickle.dump(logger["log"], f)

        f = open("../fit_results/current_best/cost/%d.txt"%slurm_index,"w")
        f.write(str(logger["lowest_cost"]) + "\n")
        f.close()

        f = open("../fit_results/current_best/cost_dict/%d.txt"%slurm_index,"w")
        f.write(",".join(list(logger["cost_dict"].keys())) + "\n")
        f.write(",".join(np.array(list(logger["cost_dict"].values())).astype(str)) + "\n")
        f.close()

        f = open("../fit_results/current_best/log_index/%d.txt"%slurm_index,"w")
        f.write(str(logger["log_index"]) + "\n")
        f.close()

        f = open("../fit_results/current_best/opt_param/%d.txt"%slurm_index,"w")
        f.write(",".join(list(logger["opt_param"].astype(str))) + "\n")
        f.close()
        return cost

    def _run_simulation(log10_fit_params,logger):
        try:
            return run_simulation(log10_fit_params,logger)
        except KeyboardInterrupt:
            return 1e5


    """
    k_onB_f/k_onB_c = (k_rel_passive/k_seq)* (k_offB_f/k_offB_c)  ###enforces detailed balance.
    We suppose that (k_offB_f/k_offB_c) = 1
    so if we suppose that k_onB_f/k_onB_c < 1 
    then k_rel_passive/k_seq < 1
    i.e. k_seq > k_rel_passive
    
    i.e. k_seq = k_rel_passive*k_seq_multiplier 
    i.e. k_seq = k_rel*k_rel_multiplier*k_seq_multiplier
    where k_seq_multiplier > 1
    
    """

    log10_fit_param_lims = {'k_onA':[-2,0],
                          'k_onB_c':[-3,2],
                          'kbind_c':[-np.infty,np.infty],
                          'kbind_m': [-np.infty, np.infty],
                          'k_rel':[-np.infty,np.infty],
                          'k_seq_multiplier':[0,3], ##to impose the k_onBf/konB_c constraint.
                          'k_rel_multiplier':[-3,0],
                            "tau_anox":[1,3]}
    log10_fit_param_lims_init = log10_fit_param_lims.copy()
    for key,val in log10_fit_param_lims_init.items():
        mn, mx = val
        if np.isinf(mn):
            mn = -2
        if np.isinf(mx):
            mx = 2
        log10_fit_param_lims_init[key] = [mn,mx]

    log10_fit_params_init = np.array([np.random.uniform(*log10_fit_param_lims_init[nm]) for nm in fit_param_names])


    log10_fit_params_bounds = np.array([log10_fit_param_lims[nm] for nm in fit_param_names])

    logger = {"log":[],
              "lowest_cost":1e9,
              "cost_dict":None,
              "opt_param":None,
              "log_index":None}

    # res = None
    # n_iter = int(1e5)
    lowest_cost = 1e9
    x0 = log10_fit_params_init
    # for i in range(n_iter):
    #     if res is None:
    res = minimize(_run_simulation,
             x0,
             args=(logger,),
             method="Nelder-Mead",
             bounds=log10_fit_params_bounds,
             options={"return_all":True,"xatol":1e-9,"fatol":1e-9,"adaptive":True})
    print("COMPLETED")
        # else:
        #     res = minimize(_run_simulation,
        #                    x0,
        #                    args=(logger,),
        #                    method="Nelder-Mead",
        #                    bounds=log10_fit_params_bounds,
        #                    options={"return_all": True,"initial_simplex":res["final_simplex"][0]})
        # #
        # if res.fun < 1e4:##if simulation hits a solving wall or numerical error, randomise
        #     x0 = res.x
        #
        #     logger["costs"] = np.array([dct["cost"] for dct in logger["log"]])
        #     logger["log_index"] = np.nanargmin(logger["costs"])
        #     logger["cost_dict"] = logger["log"][logger["log_index"]]["cost_dict"]
        #     logger["lowest_cost"] = logger["costs"][logger["log_index"]]
        #     logger["opt_param"] = logger["log"][logger["log_index"]]["log10_fit_params"]
        #
        #     with gzip.open("../fit_results/logs/optim_%d.pickle.gz" % slurm_index, "wb") as f:
        #         pickle.dump(logger["log"], f)
        #
        #     f = open("../fit_results/current_best/cost/%d.txt" % slurm_index, "w")
        #     f.write(str(logger["lowest_cost"]) + "\n")
        #     f.close()
        #
        #     f = open("../fit_results/current_best/cost_dict/%d.txt" % slurm_index, "w")
        #     f.write(",".join(list(logger["cost_dict"].keys())) + "\n")
        #     f.write(",".join(np.array(list(logger["cost_dict"].values())).astype(str)) + "\n")
        #     f.close()
        #
        #     f = open("../fit_results/current_best/log_index/%d.txt" % slurm_index, "w")
        #     f.write(str(logger["log_index"]) + "\n")
        #     f.close()
        #
        #     f = open("../fit_results/current_best/opt_param/%d.txt" % slurm_index, "w")
        #     f.write(",".join(list(logger["opt_param"].astype(str))) + "\n")
        #     f.close()
        # else:
        #     idx = int(np.random.random()*len(fit_param_names))
        #     chosen_mutated = fit_param_names[idx]
        #     print("Mutating %s"%chosen_mutated)
        #     new_val = np.random.uniform(*log10_fit_params_bounds[idx])
        #     x0[idx] = new_val



