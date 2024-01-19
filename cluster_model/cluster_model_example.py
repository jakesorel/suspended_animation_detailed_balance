from cluster_model.cluster_model import Simulate
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

if __name__ == "__main__":


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

    anoxia_dict = {"k_rel_multiplier": 2.0,
                   "kunbind_multiplier": 1/5,
                   "k_AP_multiplier": 0.0}

    t_eval_dict = {'pre_polarisation': {"dt": 10, "tfin": 1e5},
                   'polarisation': {"dt": 10, "tfin": 1e3},
                   'anoxia': {"dt": 10, "tfin": 1e6}}

    sim = Simulate(param_dict,anoxia_dict,t_eval_dict)
    sim.simulate()
    sim_values_all = sim.extract_values(sim.y)
    polarity = sim.get_polarity(sim_values_all)

    fig, ax = plt.subplots()
    ax.plot(sim.t_eval,polarity["C_pol"])
    fig.show()


    sim_values_pre_polarisation = sim.extract_values(sim.y_pre_polarisation)
    sim_values_polarisation = sim.extract_values(sim.y_polarisation)
    polarity_polarisation = sim.get_polarity(sim_values_polarisation)
    fig, ax = plt.subplots(1,2)
    n_plot = 100
    ti = np.linspace(0,len(sim.t_evals["polarisation"])-1,n_plot).astype(int)
    for i in range(n_plot):
        ax[0].plot(sim_values_polarisation["p_t"][:,0,ti[i]],color=plt.cm.plasma(i/n_plot))
        ax[1].plot(sim_values_polarisation["p_t"][:, 1, ti[i]], color=plt.cm.plasma(i / n_plot))

    fig.show()

    sim_values_anoxia = sim.extract_values(sim.y_anoxia)
    polarity_anoxia = sim.get_polarity(sim_values_anoxia)

    fig, ax = plt.subplots()
    ax.plot(sim.t_evals["anoxia"],polarity_anoxia["C_pol"])
    ax.set(xscale="log")
    fig.show()
