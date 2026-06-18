#!/usr/bin/env python
# Created by "Thieu" at 19:30, 11/06/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import List
import numpy as np
from mealpy import Optimizer, FloatVar, WOA, GSKA, ARO, AGTO
from mealpy.utils.agent import Agent


class OriginalAPO(Optimizer):
    """
    The original version of: Artificial Protozoa Optimizer (APO)
    """

    def __init__(self, epoch=10000, pop_size=100, neighbor_pairs=1, pf_max=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.neighbor_pairs = self.validator.check_int("neighbor_pairs", neighbor_pairs, [1, 5])
        self.pf_max = self.validator.check_float("pf_max", pf_max, (0, 1.0))
        self.sort_flag = True

    def evolve(self, epoch):
        """
        You can do everything in this function (i.e., Loop through the population multiple times)

        Args:
            epoch (int): The current iteration
        """
        ps = self.pf_max * self.generator.random()
        dr_vector = self.generator.choice(list(range(0, self.pop_size)), int(np.ceil(ps*self.pop_size)), replace=False)

        pop_new = []
        for idx in range(0, self.pop_size):
            p_ah = 0.5 * (1 + np.cos(epoch / self.epoch * np.pi))               # prob of autotrophic and heterotrophic behaviors
            p_dr = 0.5 * (1 + np.cos((1. - (idx+1)/self.pop_size) * np.pi))     # prob of dormancy and reproduction
            # print(p_ah, p_dr)
            if idx in dr_vector:
                if p_dr > self.generator.random():      # Dormancy
                    pos_new = self.problem.generate_solution()        # Eq. 11
                else:                                   # Reproduction
                    idx_vector = self.generator.choice(list(range(self.problem.n_dims)), int(np.ceil(self.problem.n_dims * self.generator.random())), replace=False)
                    Mr = np.zeros(self.problem.n_dims)
                    Mr[idx_vector] = 1
                    rand_vector = self.problem.generate_solution()
                    diff = self.generator.random() * rand_vector * Mr
                    flag = self.generator.choice([1, -1])
                    pos_new = self.pop[idx].solution + flag * diff    # Eq. 13
            else:
                idx_vector = self.generator.choice(list(range(self.problem.n_dims)), int(np.ceil(self.problem.n_dims * (idx+1)/self.pop_size)), replace=False)
                Mf = np.zeros(self.problem.n_dims)
                Mf[idx_vector] = 1
                ff = self.generator.random() * (1. + np.cos(epoch / self.epoch * np.pi))
                if p_ah > self.generator.random():      # Foraging in an autotroph
                    jdx = self.generator.choice(list(range(0, self.pop_size)))
                    epn_list = []
                    for k in range(0, self.neighbor_pairs):
                        if idx == 0:
                            km = idx        # km denotes the k- (k minus)
                            kp = idx + self.generator.choice(list(range(0, self.pop_size - 1 - idx)))       # kp is k+ (k plus)
                        elif idx == self.pop_size - 1:
                            km = self.generator.choice(list(range(0, self.pop_size - 1)))
                            kp = idx
                        else:
                            km = self.generator.choice(list(range(0, idx)))
                            kp = idx + self.generator.choice(list(range(0, self.pop_size-idx)))
                        wa = np.exp(-np.abs(self.pop[km].target.fitness / (self.pop[kp].target.fitness + self.EPSILON)))
                        epn = wa * (self.pop[km].solution - self.pop[kp].solution)
                        epn_list.append(epn)
                    pos_new = self.pop[idx].solution + ff * \
                              (self.pop[jdx].solution - self.pop[idx].solution + np.sum(epn_list, axis=0) / self.neighbor_pairs) * Mf       # Eq. 1
                else:                                   # Foraging in a heterotroph
                    epn_list = []
                    for k in range(0, self.neighbor_pairs):
                        if idx == 0:
                            imk = idx               # imk is i-k (i minus k)
                            ipk = idx + k           # ipk is i+k (i plus k)
                        elif idx == self.pop_size - 1:
                            imk = self.pop_size - 1 - k
                            ipk = idx
                        else:
                            imk = idx - k
                            ipk = idx + k
                        if imk < 0:
                            imk = 0
                        if ipk > self.pop_size - 1:
                            ipk = self.pop_size - 1
                        wh = np.exp(-np.abs(self.pop[imk].target.fitness / (self.pop[ipk].target.fitness + self.EPSILON)))
                        epn = wh * (self.pop[imk].solution - self.pop[ipk].solution)
                        epn_list.append(epn)
                    flag = self.generator.choice([1, -1])
                    x_near = (1 + flag * self.generator.random(self.problem.n_dims) * (1. - epoch / self.epoch)) * self.pop[idx].solution
                    pos_new = self.pop[idx].solution + ff * (x_near - self.pop[idx].solution + np.mean(epn_list, axis=0)) * Mf       # Eq. 7

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class ImprovedAPO(Optimizer):
    """
    The original version of: Improved Artificial Protozoa Optimizer (IAPO)

    TODO: Not done yet.
    """

    def __init__(self, epoch=10000, pop_size=100, neighbor_pairs=1, pf_max=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.neighbor_pairs = self.validator.check_int("neighbor_pairs", neighbor_pairs, [1, 5])
        self.pf_max = self.validator.check_float("pf_max", pf_max, (0, 1.0))
        self.sort_flag = True

    def generate_pcm_next(self, x, rho=0.4):
        # Apply the piecewise chaotic map conditions vectorized for the whole array
        x_new = np.where(x < rho, x / rho,
                         np.where(x < 0.5, (x - rho) / (0.5 - rho),
                                  np.where(x < 1 - rho, (1 - rho - x) / (0.5 - rho),
                                           (1 - x) / rho)))
        return x_new

    def generate_pcm_population(self, pop_size: int = None, rho: float = 0.4) -> List[Agent]:
        """
        Args:
            pop_size (int): number of solutions

        Returns:
            list: population or list of solutions/agents
        """
        if pop_size is None:
            pop_size = self.pop_size

        # Generate random initial values in [0, 1] for the entire population
        x = np.random.uniform(0, 1, (pop_size, self.problem.n_dims))
        # Apply the PCM method to the entire population
        x = self.generate_pcm_next(x, rho)
        # Scale the chaotic values to the range [lb, ub]
        pop_init = self.problem.lb + x * (self.problem.ub - self.problem.lb)
        pop = []
        for idx in range(pop_size):
            agent = Agent(solution=pop_init[idx, :].ravel())
            agent.target = self.get_target(agent.solution)
            pop.append(agent)
        return pop

    def initialization(self):
        """
        Override this method if needed. But the first 2 lines of code is required.
        """
        ### Required code
        if self.pop is None:
            self.pop = self.generate_pcm_population(self.pop_size)

    def evolve(self, epoch):
        """
        You can do everything in this function (i.e., Loop through the population multiple times)

        Args:
            epoch (int): The current iteration
        """
        ps = self.pf_max * self.generator.random()
        dr_vector = self.generator.choice(list(range(0, self.pop_size)), int(np.ceil(ps*self.pop_size)), replace=False)

        pop_new = []
        a1, a2 = 0, 0
        for idx in range(0, self.pop_size):
            p_ah = 0.5 * (1 + np.cos(epoch / self.epoch * np.pi))               # prob of autotrophic and heterotrophic behaviors
            p_dr = 0.5 * (1 + np.cos((1. - (idx+1)/self.pop_size) * np.pi))     # prob of dormancy and reproduction
            # print(p_ah)
            if idx in dr_vector:
                if p_dr > self.generator.random():      # Dormancy
                    pos_new = self.problem.generate_solution()        # Eq. 11
                else:                                   # Reproduction
                    idx_vector = self.generator.choice(list(range(self.problem.n_dims)), int(np.ceil(self.problem.n_dims * self.generator.random())), replace=False)
                    Mr = np.zeros(self.problem.n_dims)
                    Mr[idx_vector] = 1
                    rand_vector = self.problem.generate_solution()
                    diff = self.generator.random() * rand_vector * Mr
                    flag = self.generator.choice([1, -1])
                    pos_new = self.pop[idx].solution + flag * diff   # Eq. 13
            else:
                idx_vector = self.generator.choice(list(range(self.problem.n_dims)), int(np.ceil(self.problem.n_dims * (idx+1)/self.pop_size)), replace=False)
                Mf = np.zeros(self.problem.n_dims)
                Mf[idx_vector] = 1
                ff = self.generator.random() * (1. + np.cos(epoch / self.epoch * np.pi))
                if p_ah > self.generator.random():      # Foraging in an autotroph
                    a1 += 1
                    jdx = self.generator.choice(list(range(0, self.pop_size)))
                    epn_list = []
                    for k in range(0, self.neighbor_pairs):
                        if idx == 0:
                            km = idx        # km denotes the k- (k minus)
                            kp = idx + self.generator.choice(list(range(0, self.pop_size - 1 - idx)))       # kp is k+ (k plus)
                        elif idx == self.pop_size - 1:
                            km = self.generator.choice(list(range(0, self.pop_size - 1)))
                            kp = idx
                        else:
                            km = self.generator.choice(list(range(0, idx)))
                            kp = idx + self.generator.choice(list(range(0, self.pop_size-idx)))
                        wa = np.exp(-np.abs(self.pop[km].target.fitness / (self.pop[kp].target.fitness + self.EPSILON)))
                        epn = wa * (self.pop[km].solution - self.pop[kp].solution)
                        epn_list.append(epn)
                    pos_new = self.pop[idx].solution + ff * \
                              (self.pop[jdx].solution - self.pop[idx].solution + np.sum(epn_list, axis=0) / self.neighbor_pairs) * Mf       # Eq. 1
                else:                                   # Foraging in a heterotroph
                    a2 += 1
                    epn_list = []
                    for k in range(0, self.neighbor_pairs):
                        if idx == 0:
                            imk = idx               # imk is i-k (i minus k)
                            ipk = idx + k           # ipk is i+k (i plus k)
                        elif idx == self.pop_size - 1:
                            imk = self.pop_size - 1 - k
                            ipk = idx
                        else:
                            imk = idx - k
                            ipk = idx + k
                        if imk < 0:
                            imk = 0
                        if ipk > self.pop_size - 1:
                            ipk = self.pop_size - 1
                        wh = np.exp(-np.abs(self.pop[imk].target.fitness / (self.pop[ipk].target.fitness + self.EPSILON)))
                        epn = wh * (self.pop[imk].solution - self.pop[ipk].solution)
                        epn_list.append(epn)
                    flag = self.generator.choice([1, -1])
                    x_near = (1 + flag * self.generator.random(self.problem.n_dims) * (1. - epoch / self.epoch)) * self.pop[idx].solution
                    pos_new = self.pop[idx].solution + ff * (x_near - self.pop[idx].solution + np.mean(epn_list, axis=0)) * Mf       # Eq. 7
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # print(epoch, a1, a2)


# ## Time to test our new optimizer
# def objective_function(solution):
#     return np.sum(solution**2)
#
# problem_dict1 = {
#     "obj_func": objective_function,
#     "bounds": FloatVar(lb=[-100, ]*100, ub=[100, ]*100),
#     "minmax": "min",
#     "log_to": "console"
# }
#
# epoch = 500
# pop_size = 50
#
# model = OriginalAPO(epoch, pop_size, neighbor_pairs=3, pf_max=0.5)
# g_best = model.solve(problem_dict1)
# print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
#
# model2 = ImprovedAPO(epoch, pop_size, neighbor_pairs=3, pf_max=0.5)
# g_best = model2.solve(problem_dict1)
# print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
