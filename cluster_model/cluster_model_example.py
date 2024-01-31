from cluster_model.cluster_model import Simulate
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


if __name__ == "__main__":


    param_dict = {'D_A': 1e-3,
                   'D_C': 1e-4,  ##NOTE THAT FOR NOW, NO DIFFUSION OF B
                   'k_onA': 1e-3,
                   'k_offA': 5e-3,
                   'k_onB_c': 1e-3,
                   'k_offB_f': 5e-3,
                   'k_offB_c': 5e-3,
                   'kbind': 0.051794745,
                   'kunbind': 0.02,
                    "kunbind_postNEBD": 0.1,
                   'k_seq': 0.0071968567,
                   'k_rel': 0.01,
                   'A_tot': 1.0,
                   'B_tot': 100.0,
                   'psi': 0.137,  ##check this!
                   'L': 173.,
                   'k_AP': 1e1,
                   'n_clust': 256,
                    'i0':3,
                    'advection_fraction':0.99,
                  "tau_pol":60,
                  "tau_NEBD":60,
                  "tau_anox":600}

    anoxia_dict = {"k_rel_multiplier": 1.0,
                   "k_unbind_anoxia": 0.13894954,
                   "k_AP_multiplier": 0.0}

    t_eval_dict = {'pre_polarisation': {"dt": 10, "tfin": 3e4},
                   'polarisation': {"dt": 10, "tfin": 1e3},
                   'NEBD': {"dt": 10, "tfin": 1e3},
                   'anoxia': {"dt": 10, "tfin": 3600*24*1.+10}}


    sim = Simulate(param_dict,anoxia_dict,t_eval_dict)
    sim.simulate(method="LSODA")
    sim_values_all = sim.extract_values(sim.y)
    polarity = sim.get_polarity(sim_values_all)


    fig, ax = plt.subplots()
    ax.plot(sim.t_eval,polarity["C_pol"])
    fig.show()


    sim_values_pre_polarisation = sim.extract_values(sim.y_pre_polarisation)
    sim_values_polarisation = sim.extract_values(sim.y_polarisation)
    sim_values_NEBD = sim.extract_values(sim.y_postNEBD)
    sim_values_anoxia = sim.extract_values(sim.y_anoxia)

    plt.plot(sim_values_NEBD["m_average"].T)
    plt.show()


    polarity_polarisation = sim.get_polarity(sim_values_polarisation)
    fig, ax = plt.subplots(figsize=(4.5,4))
    dt_plot = 300
    n_skip = int(dt_plot/sim.t_eval_dict["anoxia"]["dt"])
    max_time = 60*60
    n_plot = int(max_time/dt_plot)

    t_plot = np.arange(0,max_time,dt_plot)
    for i in range(n_plot):
        ax.plot(np.arange(1,17),sim_values_anoxia["p_t"][:,0,i*n_skip][:16],color=plt.cm.plasma(np.linspace(0,1,n_plot)[i]))
        # ax[1].plot(sim_values_anoxia["b_t"][:, 0, ti[i]][:15], color=plt.cm.plasma(i / n_plot))
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(ylabel=r"$p_{i}^A$",xlabel=r"$i$")

    norm = Normalize(vmin=np.min(t_plot), vmax=np.max(t_plot/60))  # You can set vmin and vmax based on your data

    # Create a ScalarMappable object for the colorbar
    sm = ScalarMappable(cmap='plasma', norm=norm)

    # Create a ScalarMappable object for the colorbar
    sm.set_array([])  # Set an empty array as we only want the colorbar to represent the cmap range

    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time into anoxia')

    fig.subplots_adjust(bottom=0.3,left=0.3,right=0.8,top=0.8)
    fig.savefig("plots/cluster size vs time into anoxia.pdf",dpi=300)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(ylabel=r"$m_A$",xlabel="Time into anoxia (mins)")

    ax.plot(sim.t_evals["anoxia"][:360*2]/60,sim_values_anoxia["m_average"][0][:360*2],color=plt.cm.viridis(0.3))

    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.8, top=0.8)

    fig.savefig("plots/average cluster size vs time into anoxia.pdf",dpi=300)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(ylabel="Percentage A\non membrane",xlabel="Time into anoxia (mins)")

    ax.plot(sim.t_evals["anoxia"][:360*2]/60,sim_values_anoxia["A_membrane_frac"][:360*2]*100,color=plt.cm.viridis(0.3))
    fig.show()

    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.8, top=0.8)

    fig.savefig("plots/A_frac vs time into anoxia.pdf",dpi=300)


    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(ylabel=r"$[B]_{A}$",xlabel="Time into anoxia (mins)")

    ax.plot(sim.t_evals["anoxia"][:360*2]/60,sim_values_anoxia["B_t"][0][:360*2],color=plt.cm.viridis(0.3))

    fig.subplots_adjust(bottom=0.3, left=0.3, right=0.8, top=0.8)

    fig.savefig("plots/B_conc vs time into anoxia.pdf",dpi=300)


    sim_values_anoxia = sim.extract_values(sim.y_anoxia)

    fig, ax = plt.subplots()
    ax.plot(sim.t_evals["anoxia"],polarity_anoxia["C_pol"])
    ax.set(xscale="log")
    fig.show()
