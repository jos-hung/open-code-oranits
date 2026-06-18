#!/usr/bin/env python
"""Experiment parameter grids and model selection.

`ParaConfig` is a convenience container used by the meta-heuristic experiment
scripts. It centralizes parameter sweeps (epochs, population sizes, and
algorithm-specific parameters) and the `models` list that controls which
algorithms are executed in batch runs.

This module is configuration-only; editing values changes experiment behavior
but does not contain runtime logic.
"""


class ParaConfig:

    PATH_SAVE = "results"
    EVAL_PATH_SAVE = "evavalueate_results_20_trials"
    # PATH_SAVE = "results_ARO"

    SEED_GLOBAL = 42
    EPOCH = [1000,]           # Number of generations or epoch in neural network and metaheuristics
    POP_SIZE = [30, ]       # Number of population size in metaheuristics
    N_CPUS_RUN = 16
    N_TRIALS = 15
    # LIST_ALGORITHM_SEEDS = [7, 8, 11, 12, 15, 24, 30, 40, 42, 44, 50, 55, 60, 70, 72]         #  len(LIST_SEEDS) = N_TRIALS
    # LIST_ALGORITHM_SEEDS = [[7]]         #  len(LIST_SEEDS) = N_TRIALS
    # the below is for multiprocess running
    LIST_ALGORITHM_SEEDS = [[7, 8, 11], 
                            [12, 15, 24],
                            [30, 40, 42],
                            [44, 50, 55], 
                            [60, 70, 72]]         #  len(LIST_SEEDS) = N_TRIALS
    # LIST_ALGORITHM_SEEDS = [[30, 40, 42]]  
    ## Evolutionary-based group
    ep_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "bout_size": [0.05],  # percentage of child agents implement tournament selection
    }
    es_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "lamda": [0.75],  # Percentage of child agents evolving in the next generation, default=0.75
    }
    ma_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "pc": [0.85],  # crossover probability
        "pm": [0.15],  # mutation probability
        "p_local": [0.5],  # Probability of local search for each agent, default=0.5
        "max_local_gens": [10],  # Number of local search agent will be created during local search mechanism, default=10
    }
    ga_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "pc": [0.85],  # crossover probability
        "pm": [0.05]  # mutation probability
    }
    de_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "wf": [0.85],  # weighting factor
        "cr": [0.8],  # crossover rate
        "strategy": [1]  # Different variants of DE, default = 0, value in [0, 5]
    }
    jade_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "miu_f": [0.5],  # weighting factor
        "miu_cr": [0.5],  # crossover rate
        "pt": [0.1], # the percent of top best agents (p in the paper)
        "ap": [0.1]  # The Adaptation Parameter control value of f and cr (c in the paper),
    }
    sade_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    sap_de_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "branch": ["ABS"],  # gaussian (absolute) or uniform (relative) method
    }
    fpa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "p_s": [0.8],  # switch probability, default = 0.8
        "levy_multiplier": [0.01],  # multiplier factor of Levy-flight trajectory
    }
    cro_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "po": [0.85],  # The rate between free/occupied at the beginning
        "Fb": [0.9],  # BroadcastSpawner/ExistingCorals rate
        "Fa": [0.1],  # fraction of corals duplicates its self and tries to settle in a different part of the reef
        "Fd": [0.1],  # fraction of the worse health corals in reef will be applied depredation
        "Pd": [0.5],  # Probability of depredation
        "GCR": [0.1],  #
        "gamma_min": [0.02],  #
        "gamma_max": [0.2],  #
        "n_trials": [3],  # number of attempts for a larvar to set in the reef.
    }
    ocro_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "po": [0.85],  # The rate between free/occupied at the beginning
        "Fb": [0.9],  # BroadcastSpawner/ExistingCorals rate
        "Fa": [0.1],  # fraction of corals duplicates its self and tries to settle in a different part of the reef
        "Fd": [0.1],  # fraction of the worse health corals in reef will be applied depredation
        "Pd": [0.5],  # Probability of depredation
        "GCR": [0.1],  #
        "gamma_min": [0.02],  #
        "gamma_max": [0.2],  #
        "n_trials": [3],  # number of attempts for a larvar to set in the reef.
        "restart_count": [20],
    }
    shade_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "miu_f": [0.5],  # initial weighting factor
        "miu_cr": [0.5],  # initial cross-over probability
    }
    lshade_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "miu_f": [0.5],  # initial weighting factor
        "miu_cr": [0.5],  # initial cross-over probability
    }

    ## Swarm-based group
    abc_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "n_limits": [25],  #  Limit of trials before abandoning a food source
    }
    ao_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    aro_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    cgg_aro_01_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    cgg_aro_02_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "n_leaders": [4, ]
    }
    cgg_aro_03_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }

    avoa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "p1": [0.6],  # probability of status transition
        "p2": [0.4],  # probability of status transition
        "p3": [0.6],  # probability of status transition
        "alpha": [0.8],  # probability of 1st best
        "gama": [2.5]  # a factor in the paper
    }
    alo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    acor_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "sample_count": [50],  # Number of Newly Generated Samples
        "intent_factor": [0.5],  # Intensification Factor (Selection Pressure)
        "zeta": [1.0],  # Deviation-Distance Ratio
    }
    agto_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "p1": [0.03],  # p in the paper)
        "p2": [0.8],  # w in the paper
        "beta": [3.0],  # coefficient in updating equation
    }
    ba_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "loudness": [0.8],  # (A_min, A_max): loudness, default = (1.0, 2.0)
        "pulse_rate": [0.95],  # (r_min, r_max): pulse rate / emission rate
        "pf_min": [0.],  # (pf_min, pf_max): pulse frequency
        "pf_max": [10.],
    }
    bfo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "C_s": [0.1, ],  # default: 10, determining the corner between point search in the central point, in [5, 10]
        "C_e": [0.001],  # default: 1.5, determining the number of search cycles, in [0.5, 2]
        "Ped": [0.5, ],  # default: 0.5, parameter for controlling the changes in position
        "Ns": [4, ],  # default: 2, in [1, 2]
        "N_adapt": [2, ],  # c1 and c2 increase the movement intensity of bald eagles towards the best and centre points
        "N_split": [40, ]
    }
    bsa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "ff": [10],  # flight frequency - default = 10
        "pff": [0.8],  # the probability of foraging for food - default = 0.8
        "c1": [1.5],  # [c1, c2]: Cognitive accelerated coefficient, Social accelerated coefficient same as PSO
        "c2": [1.5],
        "a1": [1.0],  # [a1, a2]: The indirect and direct effect on the birds' vigilance behaviours.
        "a2": [1.0],
        "fc": [0.5, ]  # The followed coefficient- default = 0.5
    }
    bes_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "a_factor": [10, ],  # default: 10, determining the corner between point search in the central point, in [5, 10]
        "R_factor": [1.5],  # default: 1.5, determining the number of search cycles, in [0.5, 2]
        "alpha": [2., ],  # default: 2, parameter for controlling the changes in position, in [1.5, 2]
        "c1": [2., ],  # default: 2, in [1, 2]
        "c2": [2., ],  # c1 and c2 increase the movement intensity of bald eagles towards the best and centre points
    }
    beesa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "selected_site_ratio": [0.5],  # (selected_site_ratio, elite_site_ratio)
        "elite_site_ratio": [0.4],
        "selected_site_bee_ratio": [0.1],  # (selected_site_bee_ratio, elite_site_bee_ratio)
        "elite_site_bee_ratio": [2.0],
        "dance_radius": [0.1],  # Bees Dance Radius
        "dance_reduction": [0.99],
    }
    coa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "n_coyotes": [5, ]  # number of coyotes per group
    }
    csa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "p_a": [0.3]  # probability a
    }
    cso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "mixture_ratio": [0.15],  # joining seeking mode with tracing mode
        "smp": [5],  # seeking memory pool, 10 clones  (larger is better but time-consuming)
        "spc": [False],  # self-position considering
        "cdc": [0.8],  # counts of dimension to change  (larger is more diversity but slow convergence)
        "srd": [0.15],  # seeking range of the selected dimension (smaller is better but slow convergence)
        "c1": [0.4],  # same in PSO
        "w_min": [0.5],  # same in PSO
        "w_max": [0.9],
        "selected_strategy": [1],  # 0: best fitness, 1: tournament, 2: roulette wheel, else: random (decrease by quality)
    }
    coatioa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    do_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    dmoa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "peep": [2],
    }
    eho_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "alpha": [0.5],  # a factor that determines the influence of the best in each clan
        "beta": [0.5],  # a factor that determines the influence of the x_center
        "n_clans": [5],  # number of clans
    }
    esoa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    fa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "max_sparks": [50],  # parameter controlling the total number of sparks generated by the pop_size fireworks
        "p_a": [0.04],  # const parameter
        "p_b": [0.8],  # const parameter
        "max_ea": [40],  # maximum explosion amplitude
        "m_sparks": [5],  # number of sparks generated in each explosion generation
    }
    ffa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "gamma": [0.001],  # Light Absorption Coefficient
        "beta_base": [2],  # Attraction Coefficient Base Value
        "alpha": [0.2],  # Mutation Coefficient
        "alpha_damp": [0.99],  # Mutation Coefficient Damp Rate
        "delta": [0.05],  # Mutation Step Size
        "exponent": [2],  # Exponent
    }
    foa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    woa_foa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    fox_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c1": [0.18, ], # the probability of jumping (c1 in the paper)
        "c2": [0.82, ]  # the probability of jumping (c2 in the paper)
    }
    ffo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    goa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c_min": [0.00004], # coefficient c min,
        "c_max": [2.0],     # coefficient c max,
    }
    gwo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    rw_gwo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    gwo_woa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    gjo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    gto_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "A": [0.4, ],   # a position-change-controlling parameter with a range from 0.3 to 0.4, default=0.4
        "H": [2.0, ],       # initial value for specifies the jumping slope function, default=2.0
    }
    hgs_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "PUP": [0.03],  # Switching updating  position probability
        "LH": [1000],  # Largest hunger / threshold
    }
    hho_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    hba_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    ja_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    l_ja_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    mfo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    mpa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    mrfo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "somersault_range": [2, ]  # somersault factor that decides the somersault range of manta rays
    }
    msa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "n_best": [5, ],  # how many of the best moths to keep from one generation to the next
        "partition": [0.5, ],  # The proportional of first partition
        "max_step_size": [1.0, ],  # Max step size used in Levy-flight technique, default=1.0
    }
    mgo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    nmra_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "bp": [0.75],  # breeding probability (0.75)
    }
    ngo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    ooa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    pso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c1": [1.2],  # local coefficient
        "c2": [1.2],  # global coefficient
        "w_min": [0.4],  # weight min factor
        "w_max": [0.9],  # weight max factor
    }
    ppso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    cpso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c1": [2.05],  # local coefficient
        "c2": [2.05],  # global coefficient
        "w_min": [0.4],  # weight min factor
        "w_max": [0.9],  # weight max factor
    }
    clpso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c_local": [1.2],  # local coefficient, default = 1.2
        "w_min": [0.4],  # weight min factor
        "w_max": [0.9],  # weight max factor
        "max_flag": [7],  # Number of times,
    }
    aiw_pso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c1": [2.05],  # local coefficient
        "c2": [2.05],  # global coefficient
        "alpha": [0.4],  # The positive constant
    }
    ldw_pso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c1": [2.05],  # local coefficient
        "c2": [2.05],  # global coefficient
        "w_min": [0.4],  # weight min factor
        "w_max": [0.9],  # weight max factor
    }
    tvac_pso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "ci": [0.5],  # c initial
        "cf": [0.1],  # c final
    }
    pfa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    poa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    sfo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "pp": [0.2, ],  # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        "AP": [4.0, ],  # A = 4, 6,...
        "epsilon": [0.0001, ],  # = 0.0001, 0.001
    }
    sho_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "h_factor": [5],  # default = 5, coefficient linearly decreased from 5 to 0
        "n_trials": [10, ],  # default = 10,
    }
    slo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    srsr_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    ssa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "ST": [0.8],  # ST in [0.5, 1.0], safety threshold value
        "PD": [0.2],  # number of producers
        "SD": [0.1],  # number of sparrows who perceive the danger
    }
    sso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    sspidera_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "r_a": [1.0],  # the rate of vibration attenuation when propagating over the spider web, default=1.0
        "p_c": [0.7],  # controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
        "p_m": [0.1]  # the probability of each value in a dimension mask to be one, default=0.1
    }
    sspidero_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "fp_min": [0.65],  # (fp_min, fp_max): Female Percent, default = (0.65, 0.9)
        "fp_max": [0.9],
    }
    scso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    seaho_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    sto_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    servaloa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    tdo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    tso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    woa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    hi_woa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "feedback_max": [10, ]  # maximum iterations of each feedback, default = 10
    }
    zoa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }

    ## Physics-based group
    aso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "alpha": [10, ],  # Depth weight, default = 10
        "beta": [0.2, ],    # Multiplier weight, default = 0.2
    }
    archoa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c1": [2, ],  # Default belongs [1, 2]
        "c2": [6, ],  # Default belongs [2, 4, 6]
        "c3": [2, ],  # Default belongs [1, 2]
        "c4": [0.5, ],  # Default belongs [0.5, 1]
        "acc_max": [0.9, ],  # Default 0.9
        "acc_min": [0.1, ],  # Default 0.1
    }
    cdo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    eo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    modified_eo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    adaptive_eo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    efo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "r_rate": [0.3],  # default = 0.3     # Like mutation parameter in GA but for one variable, mutation probability
        "ps_rate": [0.85],  # default = 0.85    # Like crossover parameter in GA, crossover probability
        "p_field": [0.1],  # default = 0.1     # portion of population, positive field
        "n_field": [0.45],  # default = 0.45    # portion of population, negative field
    }
    evo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    fla_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "C1": [0.5],  # factor C1, default=0.5
        "C2": [2.0],  # factor C2, default=2.0
        "C3": [0.1],  # factor C3, default=0.1
        "C4": [0.2],  # factor C4, default=0.2
        "C5": [2.0],   # factor C5, default=2.0
        "DD": [0.01],   # factor D in the paper, default=0.01
    }
    hgso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "n_clusters": [2]  # Number of clusters
    }
    mvo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "wep_min": [0.2],  # Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
        "wep_max": [1.0],  # Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
    }
    nro_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    two_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    etwo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    otwo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    wdo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "RT": [3],  # RT coefficient
        "g_c": [0.2],  # gravitational constant
        "alp": [0.4],  # constants in the update equation
        "c_e": [0.4],  # coriolis effect
        "max_v": [0.3],  # maximum allowed speed
    }
    rime_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "sr": [5.0, ]   # Soft-rime parameters, default=5.0
    }

    ## Human-based group
    bro_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "threshold": [3]        # dead threshold, default=3
    }
    bso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "m_clusters": [5, ],  # m: number of clusters
        "p1": [0.2, ],  # probability
        "p2": [0.8, ],  # probability
        "p3": [0.4, ],  # probability
        "p4": [0.5, ],  # probability
        "slope": [20, ],  # k: factor that changing logsig() function's slope
    }
    ca_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "accepted_rate": [0.15],  # probability of accepted rate, default: 0.15
    }
    chio_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "brr": [0.15, ],  # Basic reproduction rate, default=0.15
        "max_age": [10, ],  # Maximum infected cases age, default=10
    }
    fbio_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    gska_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "pb": [0.1, ],  # percent of the best   0.1%, 0.8%, 0.1%
        "kf": [0.5],    # knowledge factor
        "kr": [0.9, ],  # knowledge ratio
        "kg": [5],  # Number of generations effect to D-dimension
    }
    hbo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "degree": [2, ]#  the degree level in Corporate Rank Hierarchy (CRH), default=2
    }
    hco_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "wfp": [0.65, ],  # weight factor for probability of fitness selection, default=0.65
        "wfv": [0.05, ],  # weight factor for velocity update stage, default=0.05
        "c1": [1.4, ],  # acceleration coefficient, same as PSO, default=1.4
        "c2": [1.4, ],  # acceleration coefficient, same as PSO, default=1.4
    }
    ica_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "empire_count": [5, ],  # Number of Empires (also Imperialists)
        "assimilation_coeff": [1.5, ],  # Assimilation Coefficient (beta in the paper)
        "revolution_prob": [0.05, ],  # Revolution Probability
        "revolution_rate": [0.1, ],  # Revolution Rate (mu)
        "revolution_step_size": [0.1, ],  # Revolution Step Size (sigma)
        "zeta": [0.1, ],  # Colonies Coefficient in Total Objective Value of Empires
    }
    lco_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "r1": [2.35]  # step size coefficient
    }
    qsa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    improved_qsa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    saro_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "se": [0.5, ],  # social effect, default = 0.5
        "mu": [15, ],       # maximum unsuccessful search number, default = 15
    }
    ssdo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    spbo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    tlo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    itlo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "n_teachers": [5, ] # number of teachers in class
    }
    toa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    warso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "rr": [0.1, ] # the probability of switching position updating, default=0.1
    }

    ## Bio-based group
    bbo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "p_m": [0.01],      # Mutation probability, default=0.01
        "n_elites": [2]  # Number of elites solution
    }
    bmo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "pl": [5],  # [1, pop_size - 1], barnacle’s threshold
    }
    bboa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    eoa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "p_c": [0.9],  # default = 0.9, crossover probability
        "p_m": [0.01],  # default = 0.01 initial mutation probability
        "n_best": [2],  # default = 2, how many of the best earthworm to keep from one generation to the next
        "alpha": [0.98],  # default = 0.98, similarity factor
        "beta": [0.9],  # default = 1, the initial proportional factor
        "gama": [0.9],  # default = 0.9, a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.
    }
    iwo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "seed_min": [2],  # (Min, Max) Number of Seeds
        "seed_max": [10],
        "exponent": [2],  # Variance Reduction Exponent
        "sigma_start": [1.0],  # (Initial, Final) Value of Standard Deviation
        "sigma_end": [0.01],
    }
    sbo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "alpha": [0.94, ],  # the greatest step size
        "p_m": [0.05, ],  # mutation probability
        "psw": [0.02]  # percent of the difference between the upper and lower limit (Eq. 7)
    }
    sma_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "p_t": [0.03],  # probability threshold
    }
    soa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "fc": [2],  # [1, 5], freequency of employing variable A (A linear decreased from fc to 0), default = 2
    }
    sos_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    tsa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    vcs_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "lamda": [0.5],  # Number of the best will keep (percentage %)
        "sigma": [1.5],  # Weight factor
    }
    who_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "n_explore_step": [3],  # default = 3, number of exploration step
        "n_exploit_step": [3],  # default = 3, number of exploitation step
        "eta": [0.15],  # default = 0.15, learning rate
        "p_hi": [0.9],  # default = 0.9, the probability of wildebeest move to another position based on herd instinct
        "local_alpha": [0.9],  # default = (0.9, 0.3), (alpha 1, beta 1) - control local movement
        "local_beta": [0.3],
        "global_alpha": [0.2],  # default = (0.2, 0.8), (alpha 2, beta 2) - control global movement
        "global_beta": [0.8],
        "delta_w": [2.0],  # default = (2.0, 2.0) , (delta_w, delta_c) - (dist to worst, dist to best)
        "delta_c": [2.0],
    }

    ## System-based group
    gco_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "cr": [0.7],  # Same as DE algorithm  # default: 0.7, crossover-rate
        "wf": [1.25],  # Same as DE algorithm  # default: 1.25, weighting factor
    }
    wca_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "nsr": [4],  # Number of rivers + sea (sea = 1)
        "wc": [2.0],  # Coefficient
    }
    aeo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    aaeo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    eaeo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    maeo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    iaeo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }

    ## Math-based group
    aoa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "alpha": [5],  # fixed parameter, sensitive exploitation parameter, Default: 5,
        "miu": [0.5],  # fixed parameter , control parameter to adjust the search process, Default: 0.5,
        "moa_min": [0.2],  # range min of Math Optimizer Accelerated, Default: 0.2,
        "moa_max": [0.9],  # range max of Math Optimizer Accelerated, Default: 0.9,
    }
    cgo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    circlesa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c_factor": [0.8],  # Threshold factor
    }
    gbo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "pr": [0.5],  # Probability Parameter, default = 0.5
        "beta_min": [0.2],
        "beta_max": [1.2],  # Fixed parameter (no name in the paper), default = (0.2, 1.2)
    }
    info_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    pss_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "acceptance_rate": [0.9],  # the probability of accepting a solution in the normal range, default = 0.9
        "sampling_method": ["LHS"],  # 'LHS': Latin-Hypercube or 'MC': 'MonteCarlo', default = "LHS"
    }
    run_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }

    ## Math-based group
    sca_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    qle_sca_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "alpha": [0.1, ], # the learning rate, default=0.1
        "gama": [0.9, ] # the discount factor, default=0.9
    }
    shio_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }
    ts_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "tabu_size": [5, ], #  Maximum size of the tabu list.
        "neighbour_size": [10, ], # Size of the neighborhood for generating candidate solutions
        "perturbation_scale": [0.05, ]  # Scale of the perturbations for generating candidate solutions.
    }

    ## Music-based group
    hs_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c_r": [0.95],  # Harmony Memory Consideration Rate, default = 0.15
        "pa_r": [0.05],  # Pitch Adjustment Rate, default=0.5
    }

    apo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "neighbor_pairs": [2],
        "pf_max": [0.1],
    }


    # models = [
    #     ## Evolutionary-based
    #     {"name": "APO", "class": "ApoIts", "param_grid": apo_paras},  # Artificial Protozoa Optimizer (APO)
    #     # {"name": "EP", "class": "EpIts", "param_grid": ep_paras},  # Evolutionary Programming (EP)
    #     # {"name": "ES", "class": "EsIts", "param_grid": es_paras},  # Evolution Strategies (ES)
    #     # {"name": "MA", "class": "MaIts", "param_grid": ma_paras},  # Memetic Algorithm (MA)
    #     # {"name": "GA", "class": "GaIts", "param_grid": ga_paras},  # Genetic Algorithm (GA)
    #     # {"name": "DE", "class": "DeIts", "param_grid": de_paras},  # Differential Evolution (DE)
    #     # {"name": "JADE", "class": "JadeIts", "param_grid": jade_paras},  # adaptive differential evolution (JADE)
    #     # {"name": "SADE", "class": "SadeIts", "param_grid": sade_paras},  #  Self-Adaptive Differential Evolution (SADE)
    #     # {"name": "SAP-DE", "class": "SapDeIts", "param_grid": sap_de_paras},  # Differential Evolution with Self-Adaptive Populations (SAP_DE)
    #     # {"name": "FPA", "class": "FpaIts", "param_grid": fpa_paras},  # Flower Pollination Algorithm (FPA)
    #     # {"name": "CRO", "class": "CroIts", "param_grid": cro_paras},  # Coral Reefs Optimization (CRO)
    #     # {"name": "OCRO", "class": "OcroIts", "param_grid": ocro_paras},  # Opposition-based Coral Reefs Optimization (CRO)
    #     {"name": "SHADE", "class": "ShadeIts", "param_grid": shade_paras},  # Success-History Adaptation Differential Evolution (SHADE)
    #     {"name": "L-SHADE", "class": "LshadeIts", "param_grid": lshade_paras},  #  Linear Population Size Reduction Success-History Adaptation Differential Evolution (LSHADE)
    #     #
    #     # ## Swarm-based
    #     # {"name": "ABC", "class": "AbcIts", "param_grid": abc_paras},  # Artificial Bee Colony (ABC)
    #     # {"name": "AO", "class": "AoIts", "param_grid": ao_paras},  # Aquila Optimization (AO)
    #     {"name": "ARO", "class": "AroIts", "param_grid": aro_paras},  # Artificial Rabbits Optimization (ARO)
    #     # {"name": "AVOA", "class": "AvoaIts", "param_grid": avoa_paras},  # African Vultures Optimization Algorithm (AVOA)
    #     # {"name": "ALO", "class": "AloIts", "param_grid": alo_paras},  # Ant Lion Optimizer (ALO)
    #     # {"name": "ACOR", "class": "AcorIts", "param_grid": acor_paras},  # Ant Colony Optimization Continuous (ACOR)
    #     # {"name": "AGTO", "class": "AgtoIts", "param_grid": agto_paras},  # Artificial Gorilla Troops Optimization (AGTO)
    #     # {"name": "BA", "class": "BaIts", "param_grid": ba_paras},  # Bat Algorithm (BA)
    #     # {"name": "BFO", "class": "BfoIts", "param_grid": bfo_paras},  # Adaptive Bacterial Foraging Optimization (ABFO)
    #     # {"name": "BSA", "class": "BsaIts", "param_grid": bsa_paras},  # Bird Swarm Algorithm (BSO)
    #     # {"name": "BES", "class": "BesIts", "param_grid": bes_paras},  # Bald Eagle Search (BES)
    #     # {"name": "BeesA", "class": "BeesaIts", "param_grid": beesa_paras},  # Bees Algorithm (BeesA)
    #     # {"name": "COA", "class": "CoaIts", "param_grid": coa_paras},  # Coyote Optimization Algorithm (COA)
    #     # {"name": "CSA", "class": "CsaIts", "param_grid": csa_paras},  # Cuckoo Search Algorithm (CSA)
    #     # {"name": "CSO", "class": "CsoIts", "param_grid": cso_paras},  # Cat Swarm Optimization (CSO)
    #     # {"name": "CoatiOA", "class": "CoatiOAIts", "param_grid": coatioa_paras},  # Coati Optimization Algorithm (CoatiOA)
    #     # {"name": "DO", "class": "DoIts", "param_grid": do_paras},  # Dragonfly Optimization (DO)
    #     # {"name": "DMOA", "class": "DmoaIts", "param_grid": dmoa_paras},  # Dwarf Mongoose Optimization Algorithm (DMOA)
    #     # {"name": "EHO", "class": "EhoIts", "param_grid": eho_paras},  # Elephant Herding Optimization (EHO)
    #     # {"name": "ESOA", "class": "EsoaIts", "param_grid": esoa_paras},  # Egret Swarm Optimization Algorithm (ESOA)
    #     # {"name": "FA", "class": "FaIts", "param_grid": fa_paras},  # Fireworks Algorithm (FA)
    #     # {"name": "FFA", "class": "FfaIts", "param_grid": ffa_paras},  # Firefly Algorithm (FireflyA)
    #     # {"name": "FOA", "class": "FoaIts", "param_grid": foa_paras},  # Fruit-fly Optimization Algorithm (FOA)
    #     # {"name": "WOA-FOA", "class": "WoaFoaIts", "param_grid": woa_foa_paras},  # Whale Fruit-fly Optimization Algorithm (WFOA)
    #     # {"name": "FOX", "class": "FoxIts", "param_grid": fox_paras},  #Fennec Fox Optimization (FFO)
    #     # {"name": "GOA", "class": "GoaIts", "param_grid": goa_paras},  # Grasshopper Optimization Algorithm (GOA)
    #     # {"name": "GWO", "class": "GwoIts", "param_grid": gwo_paras},  # Grey Wolf Optimizer (GWO)
    #     # {"name": "RW-GWO", "class": "RwGwoIts", "param_grid": rw_gwo_paras},  # Random Walk Grey Wolf Optimizer (RW-GWO)
    #     # {"name": "GWO-WOA", "class": "GwoWoaIts", "param_grid": gwo_woa_paras},  # Hybrid Grey Wolf - Whale Optimization Algorithm (GWO_WOA)
    #     # {"name": "GJO", "class": "GjoIts", "param_grid": gjo_paras},  # Golden jackal optimization (GJO)
    #     # {"name": "GTO", "class": "GtoIts", "param_grid": gto_paras},  # Giant Trevally Optimizer (GTO)
    #     {"name": "HGS", "class": "HgsIts", "param_grid": hgs_paras},  # Hunger Games Search (HGS)
    #     # {"name": "HHO", "class": "HhoIts", "param_grid": hho_paras},  # Harris Hawks Optimization (HHO)
    #     # {"name": "HBA", "class": "HbaIts", "param_grid": hba_paras},  # Honey Badger Algorithm (HBA)
    #     # {"name": "JA", "class": "JaIts", "param_grid": ja_paras},  # Jaya Algorithm (JA)
    #     # {"name": "L-JA", "class": "LjaIts", "param_grid": l_ja_paras},  # Levy-flight Jaya Algorithm (LJA)
    #     # {"name": "MFO", "class": "MfoIts", "param_grid": mfo_paras},  # Moth-flame optimization (MFO)
    #     # {"name": "MPA", "class": "MpaIts", "param_grid": mpa_paras},  # Marine Predators Algorithm (MPA)
    #     # {"name": "MRFO", "class": "MrfoIts", "param_grid": mrfo_paras},  # Manta Ray Foraging Optimization (MRFO)
    #     # {"name": "MSA", "class": "MsaIts", "param_grid": msa_paras},  # Moth Search Algorithm (MSA)
    #     # {"name": "MGO", "class": "MgoIts", "param_grid": mgo_paras},  # Mountain Gazelle Optimizer (MGO)
    #     # {"name": "NMRA", "class": "NmraIts", "param_grid": nmra_paras},  # Naked Mole-rat Algorithm (NMRA)
    #     # {"name": "NGO", "class": "NgoIts", "param_grid": ngo_paras},  # Northern Goshawk Optimization (NGO)
    #     # {"name": "OOA", "class": "OoaIts", "param_grid": ooa_paras},  # Osprey Optimization Algorithm (OOA)
    #     # {"name": "PSO", "class": "PsoIts", "param_grid": pso_paras},  # Particle Swarm Optimization (PSO)
    #     # {"name": "P-PSO", "class": "PpsoIts", "param_grid": ppso_paras},  # Phasor Particle Swarm Optimization (P-PSO)
    #     # {"name": "C-PSO", "class": "CpsoIts", "param_grid": cpso_paras},  # Chaos Particle Swarm Optimization (C-PSO)
    #     # {"name": "CL-PSO", "class": "ClPsoIts", "param_grid": clpso_paras},  # Comprehensive Learning Particle Swarm Optimization (CL-PSO)
    #     # {"name": "AIW-PSO", "class": "AiwPsoIts", "param_grid": aiw_pso_paras},  # Adaptive Inertia Weight Particle Swarm Optimization (AIW-PSO)
    #     # {"name": "LDW-PSO", "class": "LdwPsoIts", "param_grid": ldw_pso_paras},  # Linearly Decreasing inertia Weight Particle Swarm Optimization (LDW-PSO)
    #     # {"name": "TVAC-PSO", "class": "TvacPsoIts", "param_grid": tvac_pso_paras},  # Hierarchical PSO Time-Varying Acceleration (HPSO-TVAC)
    #     # {"name": "PFA", "class": "PfaIts", "param_grid": pfa_paras},  # Pathfinder Algorithm (PFA)
    #     # {"name": "POA", "class": "PoaIts", "param_grid": poa_paras},  # Pelican Optimization Algorithm (POA)
    #     # {"name": "SFO", "class": "SfoIts", "param_grid": sfo_paras},  # Sailfish Optimizer (SFO)
    #     # {"name": "SHO", "class": "ShoIts", "param_grid": sho_paras},  # Spotted Hyena Optimizer (SHO)
    #     # {"name": "SLO", "class": "SloIts", "param_grid": slo_paras},  # Sea Lion Optimization Algorithm (SLO)
    #     # {"name": "SRSR", "class": "SrsrIts", "param_grid": srsr_paras},  # Swarm Robotics Search And Rescue (SRSR)
    #     # {"name": "SSA", "class": "SsaIts", "param_grid": ssa_paras},  # Sparrow Search Algorithm (SpaSA)
    #     # {"name": "SSO", "class": "SsoIts", "param_grid": sso_paras},  # Salp Swarm Optimization (SSO)
    #     # {"name": "SSpiderA", "class": "SspideraIts", "param_grid": sspidera_paras},  # Social Spider Algorithm (SSpiderA)
    #     # {"name": "SSpiderO", "class": "SspideroIts", "param_grid": sspidero_paras},  # Social Spider Optimization (SSpiderO)
    #     # {"name": "SCSO", "class": "ScsoIts", "param_grid": scso_paras},  # Sand Cat Swarm Optimization (SCSO)
    #     # {"name": "SeaHO", "class": "SeaHOIts", "param_grid": seaho_paras},  # Sea-Horse Optimization (SeaHO)
    #     # {"name": "STO", "class": "StoIts", "param_grid": sto_paras},  # Siberian Tiger Optimization (STO)
    #     # {"name": "ServalOA", "class": "ServalOAIts", "param_grid": servaloa_paras},  # Serval Optimization Algorithm (ServalOA)
    #     # {"name": "TDO", "class": "TdoIts", "param_grid": tdo_paras},  # Tasmanian Devil Optimization (TDO)
    #     # {"name": "TSO", "class": "TsoIts", "param_grid": tso_paras},  # Tuna Swarm Optimization (TSO)
    #     {"name": "WOA", "class": "WoaIts", "param_grid": woa_paras},  # Whale Optimization Algorithm (WOA)
    #     # {"name": "HI-WOA", "class": "HiWoaIts", "param_grid": hi_woa_paras},  # Hybrid Improved Whale Optimization Algorithm (HI-WOA)
    #     # {"name": "ZOA", "class": "ZoaIts", "param_grid": zoa_paras},  # Zebra Optimization Algorithm (ZOA)
    #     #
    #     # ## Physics-based
    #     # {"name": "ASO", "class": "AsoIts", "param_grid": aso_paras},  # Atom Search Optimization (ASO)
    #     # {"name": "ArchOA", "class": "ArchoaIts", "param_grid": archoa_paras},  # Archimedes Optimization Algorithm (ArchOA)
    #     # {"name": "CDO", "class": "CdoIts", "param_grid": cdo_paras},  # Chernobyl Disaster Optimizer (CDO)
    #     {"name": "EO", "class": "EoIts", "param_grid": eo_paras},  # Equilibrium Optimizer (EO)
    #     # {"name": "M-EO", "class": "MeoIts", "param_grid": modified_eo_paras},  #  Modified Equilibrium Optimizer (MEO)
    #     # {"name": "A-EO", "class": "AeoIts", "param_grid": adaptive_eo_paras},  #  Adaptive Equilibrium Optimization (AEO)
    #     # {"name": "EFO", "class": "EfoIts", "param_grid": efo_paras},  # Electromagnetic Field Optimization (EFO)
    #     # {"name": "EVO", "class": "EvoIts", "param_grid": evo_paras},  # Energy Valley Optimizer (EVO)
    #     # {"name": "FLA", "class": "FlaIts", "param_grid": fla_paras},  # Fick's Law Algorithm (FLA)
    #     # {"name": "HGSO", "class": "HgsoIts", "param_grid": hgso_paras},  # Henry Gas Solubility Optimization (HGSO)
    #     {"name": "MVO", "class": "MvoIts", "param_grid": mvo_paras},  # Multi-Verse Optimizer (MVO)
    #     # {"name": "NRO", "class": "NroIts", "param_grid": nro_paras},  # Nuclear Reaction Optimization (NRO)
    #     # {"name": "TWO", "class": "TwoIts", "param_grid": two_paras},  # Tug of War Optimization (TWO)
    #     # {"name": "E-TWO", "class": "EtwoIts", "param_grid": two_paras},  # Enhenced Tug of War Optimization (ETWO)
    #     # {"name": "O-TWO", "class": "OtwoIts", "param_grid": two_paras},  # Opossition-based learning version: Tug of War Optimization (OTWO)
    #     # {"name": "TWO", "class": "TwoIts", "param_grid": two_paras},  # Tug of War Optimization (TWO)
    #     # {"name": "WDO", "class": "WdoIts", "param_grid": wdo_paras},  # Wind Driven Optimization (WDO)
    #     {"name": "RIME", "class": "RimeIts", "param_grid": rime_paras},  # physical phenomenon of RIME-ice  (RIME)
    #     #
    #     # # Human-based added
    #     # {"name": "BRO", "class": "BroIts", "param_grid": bro_paras},  # Battle Royale Optimization (BRO)
    #     # {"name": "BSO", "class": "BsoIts", "param_grid": bso_paras},  # Brain Storm Optimization (BSO)
    #     # {"name": "CA", "class": "CaIts", "param_grid": tlo_paras},  # Culture Algorithm (CA)
    #     # {"name": "CHIO", "class": "ChioIts", "param_grid": chio_paras},  # Coronavirus Herd Immunity Optimization (CHIO)
    #     # {"name": "FBIO", "class": "FbioIts", "param_grid": fbio_paras},  # Forensic-Based Investigation Optimization (FBIO)
    #     {"name": "GSKA", "class": "GskaIts", "param_grid": gska_paras},  # Gaining Sharing Knowledge-based Algorithm (GSKA)
    #     # {"name": "HBO", "class": "HboIts", "param_grid": hbo_paras},  # Heap-based optimizer (HBO)
    #     # {"name": "HCO", "class": "HcoIts", "param_grid": hco_paras},  # Human Conception Optimizer (HCO)
    #     # {"name": "ICA", "class": "IcaIts", "param_grid": ica_paras},  # Imperialist Competitive Algorithm (ICA)
    #     # {"name": "LCO", "class": "LcoIts", "param_grid": lco_paras},  # Life Choice-based Optimization (LCO)
    #     # {"name": "QSA", "class": "QsaIts", "param_grid": qsa_paras},  # Queuing search algorithm (QSA)
    #     # {"name": "I-QSA", "class": "IqsaIts", "param_grid": improved_qsa_paras},  # Improved Queuing Search Algorithm (QSA)
    #     # {"name": "SARO", "class": "SaroIts", "param_grid": saro_paras},  # Search And Rescue Optimization (SARO)
    #     # {"name": "SSDO", "class": "SsdoIts", "param_grid": ssdo_paras},  # Social Ski-Driver Optimization (SSDO)
    #     # {"name": "SPBO", "class": "SpboIts", "param_grid": spbo_paras},  #  Student Psychology Based Optimization (SPBO)
    #     {"name": "TLO", "class": "TloIts", "param_grid": tlo_paras},  # Teaching Learning-based Optimization (TLO)
    #     # {"name": "I-TLO", "class": "ItloIts", "param_grid": itlo_paras},  # Improved Teaching-Learning-based Optimization (ImprovedTLO)
    #     # {"name": "TOA", "class": "ToaIts", "param_grid": toa_paras},  # Teamwork Optimization Algorithm (TOA)
    #     # {"name": "WarSO", "class": "WarSOIts", "param_grid": warso_paras},  # War Strategy Optimization (WarSO) algorithm
    #     #
    #     # ## Bio-based
    #     # {"name": "BBO", "class": "BboIts", "param_grid": bbo_paras},  # Biogeography-based optimization (BBO)
    #     # {"name": "BMO", "class": "BmoIts", "param_grid": bmo_paras},  # Barnacles Mating Optimizer (BMO)
    #     # {"name": "BBOA", "class": "BboaIts", "param_grid": bboa_paras},  # Brown-Bear Optimization Algorithm (BBOA)
    #     # {"name": "EOA", "class": "EoaIts", "param_grid": eoa_paras},  # Earthworm Optimisation Algorithm (EOA)
    #     # {"name": "IWO", "class": "IwoIts", "param_grid": iwo_paras},  # Invasive weed colonization (IWO)
    #     # {"name": "SBO", "class": "SboIts", "param_grid": sbo_paras},  # Satin Bowerbird Optimizer (SBO)
    #     {"name": "SMA", "class": "SmaIts", "param_grid": sma_paras},  # Slime Mould Algorithm (SMA)
    #     # {"name": "SOA", "class": "SoaIts", "param_grid": soa_paras},  # Seagull Optimization Algorithm (SOA)
    #     # {"name": "SOS", "class": "SosIts", "param_grid": sos_paras},  # Symbiotic Organisms Search (SOS)
    #     # {"name": "TSA", "class": "TsaIts", "param_grid": tsa_paras},  # Tunicate Swarm Algorithm (TSA)
    #     # {"name": "VCS", "class": "VcsIts", "param_grid": vcs_paras},  # Virus Colony Search (VCS)
    #     # {"name": "WHO", "class": "WhoIts", "param_grid": who_paras},  # Wildebeest Herd Optimization (WHO)
    #     #
    #     # ## System-based
    #     # {"name": "GCO", "class": "GcoIts", "param_grid": gco_paras},  # Germinal Center Optimization (GCO)
    #     # {"name": "WCA", "class": "WcaIts", "param_grid": wca_paras},  # Water Cycle Algorithm (WCA)
    #     # {"name": "AEO", "class": "AeoIts", "param_grid": aeo_paras},  # Artificial Ecosystem-based Optimization (AEO)
    #     {"name": "AAEO", "class": "AAeoIts", "param_grid": aaeo_paras},  # Augmented Artificial Ecosystem Optimization (AAEO)
    #     # {"name": "EAEO", "class": "EAeoIts", "param_grid": eaeo_paras},  # Enhanced Artificial Ecosystem-Based Optimization (EAEO)
    #     # {"name": "MAEO", "class": "MAeoIts", "param_grid": maeo_paras},  # Modified Artificial Ecosystem-Based Optimization (MAEO)
    #     # {"name": "IAEO", "class": "IAeoIts", "param_grid": iaeo_paras},  # Improved Artificial Ecosystem-based Optimization (ImprovedAEO)
    #     #
    #     # ## Math-based
    #     # {"name": "AOA", "class": "AoaIts", "param_grid": aoa_paras},  # Arithmetic Optimization Algorithm (AOA)
    #     # {"name": "CGO", "class": "CgoIts", "param_grid": cgo_paras},  # Chaos Game Optimization (CGO)
    #     # {"name": "CircleSA", "class": "CircleSAIts", "param_grid": circlesa_paras},  # Circle Search Algorithm (CircleSA)
    #     # {"name": "GBO", "class": "GboIts", "param_grid": gbo_paras},  # Gradient-Based Optimizer (GBO)
    #     {"name": "INFO", "class": "InfoIts", "param_grid": info_paras},  # weIghted meaN oF vectOrs (INFO)
    #     # {"name": "PSS", "class": "PssIts", "param_grid": pss_paras},  # Pareto-like Sequential Sampling (PSS)
    #     # {"name": "RUN", "class": "RunIts", "param_grid": run_paras},  # RUNge Kutta optimizer (RUN)
    #     # {"name": "SCA", "class": "ScaIts", "param_grid": sca_paras},  # Sine Cosine Algorithm (SCA)
    #     # {"name": "QLE-SCA", "class": "QleScaIts", "param_grid": qle_sca_paras},  # QLE Sine Cosine Algorithm (QLE-SCA)
    #     # {"name": "SHIO", "class": "ShioIts", "param_grid": shio_paras},  # Hill Climbing (HC)
    #     # {"name": "TS", "class": "TsIts", "param_grid": ts_paras},  # Tabu Search (TS)
    #     #
    #     # ## Music-based group
    #     # {"name": "HS", "class": "HsIts", "param_grid": hs_paras},  # Harmony Search (HS)
    # ]

    models = [
        # The above code snippet appears to be defining a dictionary in Python that contains
        # information about different optimization algorithms. Each dictionary entry includes the name
        # of the algorithm, its class name, and a parameter grid associated with it.
        # {"name": "CGG-ARO-02", "class": "CggAro02Its", "param_grid": cgg_aro_02_paras},  # Chaotic Gaussian-based Global Artificial Rabbits Optimization (CGG-ARO)
        # {"name": "CGG-ARO-03", "class": "CggAro03Its", "param_grid": cgg_aro_03_paras},  # Chaotic Gaussian-based Global Artificial Rabbits Optimization (CGG-ARO)

        # {"name": "APO", "class": "ApoIts", "param_grid": apo_paras},  # Artificial Protozoa Optimizer (APO)
        # {"name": "SHADE", "class": "ShadeIts", "param_grid": shade_paras},  # Success-History Adaptation Differential Evolution (SHADE)
        # {"name": "L-SHADE", "class": "LshadeIts", "param_grid": lshade_paras},  #  Linear Population Size Reduction Success-History Adaptation Differential Evolution (LSHADE)
        # # {"name": "HGS", "class": "HgsIts", "param_grid": hgs_paras},  # Hunger Games Search (HGS)
        # # {"name": "WOA", "class": "WoaIts", "param_grid": woa_paras},  # Whale Optimization Algorithm (WOA)
        # {"name": "EO", "class": "EoIts", "param_grid": eo_paras},  # Equilibrium Optimizer (EO)
        # # {"name": "MVO", "class": "MvoIts", "param_grid": mvo_paras},  # Multi-Verse Optimizer (MVO)
        # # {"name": "RIME", "class": "RimeIts", "param_grid": rime_paras},  # physical phenomenon of RIME-ice  (RIME)
        # # {"name": "GSKA", "class": "GskaIts", "param_grid": gska_paras},  # Gaining Sharing Knowledge-based Algorithm (GSKA)
        # # {"name": "TLO", "class": "TloIts", "param_grid": tlo_paras},  # Teaching Learning-based Optimization (TLO)
        # # {"name": "SMA", "class": "SmaIts", "param_grid": sma_paras},  # Slime Mould Algorithm (SMA)
        # # {"name": "AAEO", "class": "AAeoIts", "param_grid": aaeo_paras},  # Augmented Artificial Ecosystem Optimization (AAEO)
        # # {"name": "INFO", "class": "InfoIts", "param_grid": info_paras},  # weIghted meaN oF vectOrs (INFO)
        # {"name": "ARO", "class": "AroIts", "param_grid": aro_paras},  # Artificial Rabbits Optimization (ARO)
        {"name": "CGG-ARO-01", "class": "CggAro01Its", "param_grid": cgg_aro_01_paras},  # Chaotic Gaussian-based Global Artificial Rabbits Optimization (CGG-ARO)
    ]