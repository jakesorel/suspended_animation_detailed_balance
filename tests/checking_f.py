import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

D_A = 0
D_B = 0
D_C = 0
k_onA = 0
k_offA = 0
k_onB_c = 0
k_offB_f = 0
k_offB_c = 0
kbind_c = 0.101
kunbind_c = 0.1543
kbind_m = 0.11
kunbind_m = 0.153
k_seq = 0.1
k_rel = 0.155
A_tot = 1.
B_tot = 1.
psi = 0.174
L = 137.
_k_AP = 0.
params = D_A, D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP
n_clust = 128
i0 = 3
params_start = jnp.array(params).ravel()
params_end = jnp.array(params).ravel()
tau = 1.
k_rel_passive = 0.16

y = np.random.random(((n_clust*2 + 1)*2))*0.0001

def f(self, t, y, n_clust, i0, params_start ,params_end ,tau ,k_rel_passive):

    frac_start = jnp.exp(- t /tau)
    frac_end = 1- frac_start
    params = frac_start * params_start + frac_end * params_end
    D_A, D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params

    k_onB_f = k_onB_c * (k_rel_passive * k_offB_f) / (k_seq * k_offB_c)  ###enforces detailed balance.

    _y = y.reshape(n_clust * 2 + 1, 2)
    p, b = _y[:n_clust], _y[n_clust:]

    k_AP_p = jnp.column_stack([jnp.zeros_like(p[:, 0]), jnp.ones_like(p[:, 0]) * _k_AP])
    k_AP = jnp.array([0, _k_AP])

    ##note that p is 1-indexed and b is 0-indexed

    ##Cluster dynamics

    i = np.arange(n_clust) + 1

    fw_rate_per = kbind_c * (i / i0) ** (1 / 3)
    rv_rate_per = kunbind_c

    p1 = p[0]
    fw_rate_cluster = p * jnp.expand_dims(fw_rate_per, 1) * p1
    fw_rate_monomers = p * kbind_m * p1 * jnp.expand_dims(i, 1)
    rv_rate_cluster = p * jnp.expand_dims(i, 1) * rv_rate_per
    rv_rate_monomers = p * jnp.expand_dims(i, 1) * kunbind_m
    rv_rate_active = p * jnp.expand_dims(i, 1) * k_AP_p

    fw_rate = jnp.concatenate((fw_rate_monomers[:i0 - 1], fw_rate_cluster[i0 - 1:-1]))
    rv_rate = jnp.concatenate((rv_rate_monomers[1:i0], rv_rate_cluster[i0:]))

    D_P = jnp.concatenate(
        (jnp.expand_dims(jnp.array(((D_B),)), 1), jnp.expand_dims((i <= i0) * D_A + (i > i0) * D_C, 1)))

    net_rate = fw_rate - rv_rate

    dtp = jnp.concatenate((jnp.array(((0., 0.),)), net_rate - rv_rate_active[1:])) - jnp.concatenate(
        (net_rate - rv_rate_active[1:], (jnp.array(((0., 0.),))))) \
          + D_P[1:] * (jnp.roll(p, -1, axis=1) - 2 * p + jnp.roll(p, 1, axis=1)) / (L / 2) ** 2

    ## loading and unloading
    A = (p.T * i).sum()
    # A_clust = (p[i0-1:]*i[i0-1:]).sum()
    B = (b.T).sum()

    b0 = b[0]
    b1 = b[1]
    _b = b[1:]
    _bm1 = jnp.concatenate((jnp.array(((0., 0.),)), _b[:-1]))
    _bp1 = jnp.concatenate((_b[1:], (jnp.array(((0., 0.),)))))
    pm1 = jnp.concatenate((jnp.array(((0., 0.),)), p[:-1]))

    A_cyto = A_tot - psi * A.mean()
    B_cyto = B_tot - psi * B.mean()

    # p1 = d_{1,0} + d_{1,1}
    # p1 = d_{1,0} + b1
    # if only d_{1,0} can unload, then the fraction that can unload is
    dtp1 = k_onA * A_cyto - k_offA * (p1 - b1) - net_rate.sum(axis=0) - k_AP * p1
    dtp = dtp.at[0].add(dtp1)


    b_load = (k_seq * b0 + k_onB_c * B_cyto) * (jnp.expand_dims(i, 1) * p - _b)
    b_unload = (k_rel + k_offB_c) * _b

    K_plus_i = jnp.expand_dims(kbind_c * (i / i0) ** (1 / 3) * (i > i0) + i * kbind_m * (i <= i0), 1) * jnp.ones((1, 2))
    K_plus_im1 = jnp.concatenate((jnp.array(((0., 0.),)), K_plus_i[:-1]))
    k_minus_i = jnp.expand_dims(k_offB_c * (i > i0) * i + k_offB_f * (i <= i0) * (i > 1) * i, 1) * jnp.ones((1, 2))
    k_minus_i_inc_active = k_minus_i + k_AP_p
    k_minus_ip1_inc_active = jnp.concatenate((k_minus_i_inc_active[1:], (jnp.array(((0., 0.),)))))

    # b_ad_load = K_plus_im1 * (_bm1 * p1 - pm1 * b1) - K_plus_i * p1 * _b
    b_ad_load = K_plus_im1 * (_bm1 * p1 + pm1 * b1) - K_plus_i * p1 * _b

    b_ad_unload = jnp.expand_dims(i, 1) * (k_minus_ip1_inc_active * _bp1 - k_minus_i_inc_active * _b)

    dtb1_add = - (K_plus_i * b1 * p)[:-1].sum() + (k_minus_i * _b)[1:].sum()

    dtb0 = k_onB_f * B_cyto - k_offB_f * b0 + (k_rel * _b - k_seq * b0 * (jnp.expand_dims(i, 1) * p - _b)).sum()
    dt_b = b_ad_load + b_ad_unload + b_load - b_unload

    dtb = jnp.row_stack((dtb0, dt_b)) \
          + D_P * (jnp.roll(b, -1, axis=1) - 2 * b + jnp.roll(b, 1, axis=1)) / (L / 2) ** 2

    dtb = dtb.at[1].add(dtb1_add)

    ## compile
    dty = jnp.concatenate((dtp.ravel(), dtb.ravel()))

    return dty




def f2(self, t, y, n_clust, i0, params_start ,params_end ,tau ,k_rel_passive):

    ###Define the ratio of parameters between params_start to params_end given an exponential temporal decay with timescale tau
    frac_start = jnp.exp(- t /tau)
    frac_end = 1- frac_start
    params = frac_start * params_start + frac_end * params_end

    ##Seperate out the parameters
    D_A, D_B, D_C, k_onA, k_offA, k_onB_c, k_offB_f, k_offB_c, kbind_c, kunbind_c, kbind_m, kunbind_m, k_seq, k_rel, A_tot, B_tot, psi, L, _k_AP = params

    ##Enforce the single detailed balance constraint
    k_onB_f = k_onB_c * (k_rel_passive * k_offB_f) / (k_seq * k_offB_c)

    ##Unpack the variables
    _y = y.reshape(n_clust * 2 + 1, 2)
    p, b = _y[:n_clust], _y[n_clust:]
    _b = b[1:]
    s_p = jnp.row_stack([jnp.array([0.,0.]),p]) ##dummy variable given len(b)-len(p) = 1
    p1 = p[0]
    b0 = b[0]
    b1 = b[1]

    ##Specify removal of due to pPARs occurs only in the posterior.
    k_AP_p = jnp.column_stack([jnp.zeros_like(b[:, 0]), jnp.ones_like(b[:, 0]) * _k_AP])
    _k_AP_p = k_AP_p[1:]
    ##a k_AP reaction removes a monomer (or a dimer) into the cytoplasm. dimer 'choice' is indiscriminate.


    ##Dummy shifts in the parameters for calculations.
    p_m1 = jnp.row_stack([jnp.array([0.,0.]),p[:-1]])
    p_p1 = jnp.row_stack([p[1:],jnp.array([0.,0.])])
    b_m1 = jnp.row_stack([jnp.array([0.,0.]),b[:-1]])
    b_p1 = jnp.row_stack([b[1:],jnp.array([0.,0.])])
    s_p_m1 = jnp.row_stack([jnp.array([0.,0.]),s_p[:-1]])


    ##i is the cluster size.
    i = np.arange(n_clust+1)
    _i = i[1:]

    ##Get the total masses across all oligomeric states for A (PAR3) and B
    A = (p.T * _i).sum()
    B = (b).sum()

    ##Calculate the cytoplasmic concentrations given the above.
    A_cyto = A_tot - psi * A.mean()
    B_cyto = B_tot - psi * B.mean()


    ##Specify the binding and unbinding rates
    Kp_i = (i>=1)*(i<n_clust)*((i*kbind_m)*(i<=i0) + (i/i0)**(2/3)*kbind_c * (i > i0))
    Km_i = (i>1)*((i*kunbind_m)*(i<=i0) + i*kunbind_c * (i > i0))
    km_i = (i>1)*((kunbind_m)*(i<=i0) + kunbind_c * (i > i0)) ##Note that Km_i = km_i * i

    ##Dummy shifts of the above parameters to aid calculations.
    _Kp_i = Kp_i[1:]
    _Km_i = Km_i[1:]
    Kp_i_m1 = jnp.concatenate([jnp.array([0.]),Kp_i[:-1]])
    Km_i_p1 = jnp.concatenate([Km_i[1:],jnp.array([0.])])
    km_i_p1 = jnp.concatenate([km_i[1:],jnp.array([0.])])
    k_AP_p_p1 = jnp.row_stack([k_AP_p[1:],jnp.array([0.,0.])])
    _k_AP_p_p1 = k_AP_p_p1[1:]
    _Kp_i_m1 = Kp_i_m1[1:]
    _Km_i_p1 = Km_i_p1[1:]


    ##Calculate the rate of change of each of the oligomeric states, from the A-perspective
    dtpi = (jnp.expand_dims(_Kp_i_m1,1)*jnp.expand_dims(p1,0)*p_m1 - jnp.expand_dims(_Kp_i,1)*jnp.expand_dims(p1,0)*p) \
           + ((jnp.expand_dims(_Km_i_p1,1) + _k_AP_p_p1*jnp.expand_dims(i[1:],1))*p_p1 - (jnp.expand_dims(_Km_i,1)+ _k_AP_p*jnp.expand_dims(i[:-1],1))*p)

    ##Include mass conservation terms for PAR3 monomers, and also binding and unbinding from cytoplasm
    ##Note there, k_AP is absent, as we assume that all pPAR phosphorylated protein goes directly to the cytoplasm.
    dtp1_add =(jnp.expand_dims(_Km_i,1)*p- jnp.expand_dims(_Kp_i,1)*jnp.expand_dims(p1,0)*p).sum(axis=0) \
              + k_onA*A_cyto - k_offA*(p1-b1)

    ##Combine the additional terms.
    dtpi = dtpi.at[0].add(dtp1_add)

    ##Calculate the rate of change of B-mass of each oligomeric state
    dtbi = jnp.expand_dims(k_seq * b0 + k_onB_c * B_cyto, 0) * (
                jnp.expand_dims(i, 1) * s_p - jnp.expand_dims(i != 0, 1) * b) \
           - (k_rel + k_offB_c) * b * jnp.expand_dims(i != 0, 1) \
           + jnp.expand_dims(Kp_i_m1, 1) * (b_m1 * p1 + s_p_m1 * b1) - jnp.expand_dims(Kp_i, 1) * (p1 * b) \
           + jnp.expand_dims(i, 1) * (
                       (jnp.expand_dims(km_i_p1, 1) + k_AP_p_p1) * b_p1 - (jnp.expand_dims(km_i, 1) + k_AP_p) * b)

    ##Likewise, deal with mass conservation
    dtb0_add = (k_rel * b * jnp.expand_dims(i != 0, 1)
                - jnp.expand_dims(k_seq * b0, 0) * (jnp.expand_dims(i, 1) * s_p - jnp.expand_dims(i != 0, 1) * b)).sum(
        axis=0) \
               + k_onB_f * B_cyto - k_offB_f * b0
    dtb1_add = (jnp.expand_dims(km_i, 1) * b - jnp.expand_dims(Kp_i, 1) * (b1 * s_p)).sum(axis=0)
    ##Note, as above, kAP directly leads to removal into cytoplasm.

    dtbi = dtbi.at[0].add(dtb0_add)
    dtbi = dtbi.at[1].add(dtb1_add)

    ##Calculate diffusion coefficient as a function of size.
    ##Here, len(D_P) = n_clust + 1, where D_P[0] = D_B for simplicity.
    D_P = jnp.concatenate(
        (jnp.expand_dims(jnp.array(((D_B),)), 1), jnp.expand_dims((_i <= i0) * D_A + (_i > i0) * D_C, 1)))

    ##Apply finite difference diffusion
    dtpi += D_P[1:] * (jnp.roll(p, -1, axis=1) - 2 * p + jnp.roll(p, 1, axis=1)) / (L / 2) ** 2
    dtbi += D_P * (jnp.roll(b, -1, axis=1) - 2 * b + jnp.roll(b, 1, axis=1)) / (L / 2) ** 2

    ##Assemble the variables.
    dty = jnp.concatenate((dtpi.ravel(), dtbi.ravel()))


    return dty
