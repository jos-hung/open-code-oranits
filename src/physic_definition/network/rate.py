import random
from configs.systemcfg import network_cfg, map_cfg
from configs.config import ParaConfig
import numpy as np
from physic_definition.map.map import Map

SEED = ParaConfig.SEED_GLOBAL
rd_generator = np.random.default_rng(SEED)
import sys

max_ = sys.maxsize
def random_bs_num(interection):
    """
    Generate a list of tuples representing the positions and CPU frequencies of MECs (Mobile Edge Computing servers).
    Args:
        interection (list): A list of positions where MECs can be placed.
    Returns:
        list: A list of tuples, where each tuple contains a position and a CPU frequency for a MEC.
    Raises:
        ValueError: If the provided `interection` list is empty.
    Notes:
        - The function uses a random generator `rd_generator` to select positions and CPU frequencies.
        - The number of MECs (`n_MEC`) and the range of CPU frequencies (`CPU_freq`) are defined in the `network_cfg` configuration.
    """
    
    if len(interection) == 0:
        raise ValueError ("Map is not build")
    position_of_mec = rd_generator.choice(interection, network_cfg['n_MEC'])
    cpu_freq_of_mec = rd_generator.integers(network_cfg['CPU_freq'][0], network_cfg['CPU_freq'][1], network_cfg['n_MEC'])
    return list(zip(position_of_mec, cpu_freq_of_mec))

def chann_rates(distance):
    """
    Calculate the channel rate based on the given distance.
    Parameters:
    distance (float): The distance between the transmitter and receiver.
    Returns:
    float: The calculated channel rate.
    Notes:
    - The function uses a random generator to create a complex channel gain `h`.
    - The transmit power `Pt` is set to 199.526 mW.
    - The noise power spectral density `No` is set to 3.98e-21.
    - The number of channels `m` is set to 10.
    - The total bandwidth `bandwidth` is set to 20 MHz.
    - The path loss exponent `path_loss` is set to 3.
    - The channel gain `h` is normalized and adjusted based on the distance and path loss.
    - The rate is calculated using the Shannon-Hartley theorem.
    """
    
    rate=0
    # h=complex(rd_generator.standard_normal(size=(5,1)),rd_generator.standard_normal(size=(5,1)))
    h = rd_generator.standard_normal(size=(16,)) + 1j * rd_generator.standard_normal(size=(16,))

    h/=np.sqrt(2)
    Pt=199.526*(10**-3)
    No=3.98*(10**-21)
    m=10
    bandwidth=10*(10**6)
    # h=(np.abs(h))**2
    # print(h)
    path_loss=3
    h = np.linalg.norm(h, ord=2)
    
    if round(distance,2) != 0.0:
        h=h/(distance)**path_loss
    else:
        h = max_

    bandwidth_per_channel=bandwidth/m
    rate=bandwidth_per_channel*np.log2(1+((Pt*h)/(bandwidth_per_channel*No)))   
    return rate

def get_rate_and_mec_cpu(v_pos, mec):
    '''the offloading stratey based on greedy: select the MEC with the best channel rate, 
    if there are multiple MECs with the same best channel rate, select the one with the highest CPU frequency. 
    If there are no MECs within the best rate radius, select the MEC with the minimum distance to the vehicle.
    '''
    candidate_mec = []
    min_distance = max_
    mec_min = None
    for m in mec:
        current_mec_point = v_pos.get_dis_to_point(m[0]) #m[0]= a point, m[1]= an integer
        if current_mec_point < network_cfg['best_rate_radius']:
            candidate_mec.append(m)
        if current_mec_point < min_distance:
            min_distance = current_mec_point
            mec_min = m
    select_mec = 0
    distance = 0
    if len(candidate_mec)==0:
        select_mec = mec_min
        distance = min_distance
    else:
        max_cpu = -max_
        for m in candidate_mec:
            if m[1]>max_cpu:
                max_cpu = m[1]
                distance = v_pos.get_dis_to_point(m[0])
                select_mec = m
    return chann_rates(distance), select_mec[1]

            
            



    
    