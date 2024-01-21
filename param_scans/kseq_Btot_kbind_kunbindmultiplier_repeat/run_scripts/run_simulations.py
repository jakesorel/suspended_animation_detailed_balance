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
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
import h5py
import gzip
import shutil
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError



"""
        sim_values = {"p_t": p_t,
                      "b_t": b_t,
                      "C_t": C_t,
                      "B_t": B_t,
                      "m_average":m_average,
                      "b_frac":b_frac,
                      "d_frac":d_frac,
                      "F_t":F_t,
                      "A_membrane_frac":A_membrane_frac,
                      "B_membrane_frac":B_membrane_frac}
"""

def mkdir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            print("Presuming path exists")


if __name__ == "__main__":
    mkdir("../scan_results")
    mkdir("../scan_results/tmp")

    i = 0
    N = 15
    k_seq_range = np.logspace(-3,1,N)
    B_tot_range = np.logspace(-3,2,N)
    kbind_range = np.logspace(-3,1,N)
    kunbind_multiplier = np.logspace(-2,0,N)

    _k_seq,_B_tot,_kbind,_kunbind_multiplier = np.meshgrid(k_seq_range,B_tot_range,kbind_range,kunbind_multiplier,indexing="ij")
    scan_array = np.array((_k_seq.ravel(),
                           _B_tot.ravel(),
                           _kbind.ravel(),
                           _kunbind_multiplier.ravel())).T



    param_dict = {'D_A': 1e-3,
                   'D_C': 1e-4,  ##NOTE THAT FOR NOW, NO DIFFUSION OF B
                   'k_onA': 1e-3,
                   'k_offA': 5e-3,
                   'k_onB_c': 1e-3,
                   'k_offB_f': 5e-3,
                   'k_offB_c': 5e-3,
                   'kbind': 5,
                   'kunbind': 0.02,
                   'k_seq': 0.8,
                   'k_rel': 0.01,
                   'A_tot': 1.0,
                   'B_tot': 1.0,
                   'psi': 0.137,  ##check this!
                   'L': 173.,
                   'k_AP': 1e1,
                   'n_clust': 256,
                    'i0':3,
                    'advection_fraction':0.99,
                  "tau_pol":60,
                  "tau_anox":600}

    anoxia_dict = {"k_rel_multiplier": 1.0,
                   "kunbind_multiplier": 0.1,
                   "k_AP_multiplier": 0.0}

    t_eval_dict = {'pre_polarisation': {"dt": 10, "tfin": 3e4},
                   'polarisation': {"dt": 10, "tfin": 1e3},
                   'anoxia': {"dt": 10, "tfin": 3600*24*3.+10}}

    sim = Simulate(param_dict,anoxia_dict,t_eval_dict)

    total_sim_number = N**4
    n_jobs = N**2
    sims_per_lot = N**2

    slurm_index = int(sys.argv[1])
    print("Slurm index", slurm_index)
    range_to_sample = np.arange(slurm_index*sims_per_lot,(slurm_index+1)*sims_per_lot)

    time_points = np.array(
        [0, 10, 60, 60 * 10, 60 * 60, 2 * 60 * 60, 4 * 60 * 60, 8 * 60 * 60, 16 * 60 * 60, 24 * 60 * 60,
         24 * 60 * 60 * 2, 24 * 60 * 60 * 3])
    time_points_to_interpolate = np.array([0,60,10*60,100*60,1000*60,24 * 60 * 60 * 3])
    time_points_interpolated = np.row_stack([np.arange(t1,t2,int((t2-t1)/60)) for t1,t2 in zip(time_points_to_interpolate[:-1],time_points_to_interpolate[1:]) ]).ravel()
    time_points_interpolated = np.concatenate((time_points_interpolated,(time_points_to_interpolate[-1],)))
    time_steps_interpolated = np.unique((np.round(time_points_interpolated / t_eval_dict["anoxia"]["dt"])).astype(int))
    time_points_interpolated = sim.t_evals["anoxia"][time_steps_interpolated]
    # mkdir("../scan_results/raw")
    # mkdir("../scan_results/raw_tchosen")
    mkdir("../scan_results/summary")
    mkdir("../scan_results/summary_tchosen")
    mkdir("../scan_results/summary_tchosen/by_time")
    for t in time_points_interpolated:
        mkdir("../scan_results/summary_tchosen/by_time/%d" % t)
        # mkdir("../scan_results/raw_tchosen/%d" % t)
    mkdir("../scan_results/summary_tchosen/together")


    def run_simulations(i):
        if not os.path.exists("../scan_results/summary_tchosen/by_time/%d/%i.csv" % (time_points_interpolated[-1], i)):
            _param_dict = param_dict.copy()
            _anoxia_dict = anoxia_dict.copy()

            _param_dict["k_seq"],_param_dict["B_tot"],_param_dict["kbind"],_anoxia_dict["kunbind_multiplier"] = scan_array[i]
            sim.initialise_param_dicts(_param_dict,_anoxia_dict)
            sim.simulate()


            sim_values_anoxia = sim.extract_values(sim.y_anoxia)
            polarity = sim.get_polarity(sim_values_anoxia)
            sim_values_anoxia["C_pol"] = polarity["C_pol"]
            sim_values_anoxia["B_pol"] = polarity["B_pol"]

            raw_dict = {"p_t":sim_values_anoxia["p_t"],"b_t":sim_values_anoxia["b_t"]}


            time_steps = (time_points/t_eval_dict["anoxia"]["dt"]).astype(int)

            columns = 'index','k_seq','B_tot','kbind','kunbind_multiplier','t','C_A_t','C_P_t', 'B_A_t','B_P_t', 'm_A_average','m_P_average', 'b_A_frac','b_P_frac', 'd_P_frac','d_A_frac', 'F_A_t','F_P_t', 'A_membrane_frac', 'B_membrane_frac', 'C_pol', 'B_pol'
            keys = 'C_t', 'B_t', 'm_average', 'b_frac', 'd_frac', 'F_t', 'A_membrane_frac', 'B_membrane_frac', 'C_pol', 'B_pol'
            spatial_outputs = 'C_t', 'B_t', 'm_average', 'b_frac', 'd_frac', 'F_t'
            non_spatial_outputs = 'A_membrane_frac', 'B_membrane_frac', 'C_pol', 'B_pol'
            df = pd.DataFrame(np.row_stack([np.expand_dims(np.ones_like(sim_values_anoxia["C_pol"]),0)*np.expand_dims(np.array((i,_param_dict["k_seq"],_param_dict["B_tot"],_param_dict["kbind"],_anoxia_dict["kunbind_multiplier"])),1)]+[sim.t_evals["anoxia"]]+[sim_values_anoxia[key] for key in keys]).astype(np.float32).T)
            df.columns = columns
            df_chosen_times = df.iloc[time_steps_interpolated]

            #
            # # #This takes up too much space
            # file_path = "../scan_results/raw/" + str(i) + '.h5'
            # #
            # f = h5py.File(file_path, 'w')
            # f.create_dataset("data", data=df.values, compression="gzip")
            # f.close()
            #
            # with open(file_path, 'rb') as f_in:
            #     with gzip.open(file_path + ".gz", 'wb') as f_out:
            #         shutil.copyfileobj(f_in, f_out)
            #
            # os.remove(file_path)

            # for t, ti in zip(time_points_interpolated,time_steps_interpolated):
            #     file_path = "../scan_results/raw_tchosen/%d/"%t + str(i) + '.h5'
            #     f = h5py.File(file_path, 'w')
            #     for key in raw_dict.keys():
            #         f.create_dataset(key, data=raw_dict[key][:,:,ti], compression="gzip")
            #     f.close()
            #
            #     with open(file_path, 'rb') as f_in:
            #         with gzip.open(file_path + ".gz", 'wb') as f_out:
            #             shutil.copyfileobj(f_in, f_out)
            #
            #     os.remove(file_path)

            # df.to_csv("../scan_results/summary/%d.csv"%i)
            df_chosen_times.to_csv("../scan_results/summary_tchosen/together/%d.csv"%i)



            for j in range(len(df_chosen_times)):
                try:
                    df_chosen_times.iloc[j:j+1].to_csv("../scan_results/summary_tchosen/by_time/%d/%i.csv"%(time_points_interpolated[j],i), header=False)
                except OSError:
                    print("OSError, trying again")
                    df_chosen_times.iloc[j:j+1].to_csv("../scan_results/summary_tchosen/by_time/%d/%i.csv"%(time_points_interpolated[j],i), header=False)

                # out = ",".join(df_chosen_times.iloc[j].values.astype(str)) + "\n"
                # file = open("../scan_results/summary_tchosen/by_time/%d/%i.csv"%(time_points_interpolated[j],i),"w+")
                # file.write(out)
                # file.close()

    Parallel(n_jobs=-1,backend="loky", prefer="threads",temp_folder="../scan_results/tmp")(delayed(run_simulations)(i) for i in range_to_sample)
    # run_simulations(5)

    """
    To do: 
    
    - sort out the conda environment etc to run this stuff. 
    - Run the simulations based on some specified t_eval. This should compress the output substantially 
    --> 0s,1s,1min,1hr,2hr,4hr,8hr,16hr,24hr,2day,3day; each linspace with 100
    
    """

