#!/usr/bin/env python
# Created by "Thieu" at 18:43, 12/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import List
from mealpy import Optimizer, FloatVar
import numpy as np
from mealpy.utils.agent import Agent
# from mealpy.swarm_based.ARO import OriginalARO


class OriginalARO(Optimizer):
    """
    The original version of: Artificial Rabbits Optimization (ARO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2022.105082
        2. https://www.mathworks.com/matlabcentral/fileexchange/110250-artificial-rabbits-optimization-aro

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ARO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = ARO.OriginalARO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired
    meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        theta = 2 * (1 - epoch/self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * (np.sin(2*np.pi*self.generator.random()))
            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random()*self.problem.n_dims)), replace=False)
            temp[rd_index] = 1
            R = L * temp        # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta      # Eq. 15
            if A > 1:   # detour foraging strategy
                rand_idx = self.generator.integers(0, self.pop_size)
                pos_new = self.pop[rand_idx].solution + R * (self.pop[idx].solution - self.pop[rand_idx].solution) + \
                    np.round(0.5 * (0.05 + self.generator.random())) * self.generator.normal(0, 1)      # Eq. 1
            else:       # Random hiding stage
                gr = np.zeros(self.problem.n_dims)
                rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
                gr[rd_index] = 1        # Eq. 12
                H = self.generator.normal(0, 1) * (epoch / self.epoch)       # Eq. 8
                b = self.pop[idx].solution + H * gr * self.pop[idx].solution       # Eq. 13
                pos_new = self.pop[idx].solution + R * (self.generator.random() * b - self.pop[idx].solution)      # Eq. 11
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, minmax=self.problem.minmax)


class CGG_ARO_01(Optimizer):
    """
    The original version of: Chaotic Gaussian-based Global Artificial Rabbits Optimization (CGG-ARO)
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

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
        x = self.generator.uniform(0, 1, (pop_size, self.problem.n_dims))
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
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        # piecewise chaotic map
        # Levy-flight strategy
        # Gaussian perturbation
        # Global best history
        # PLGG-ARO

        # Chaotic Gaussian-based Global Artificial Rabbits Optimization (CGG-ARO)

        theta = 2 * (1 - epoch / self.epoch)
        pop_new = []
        t1, t2 = 0, 0
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3)
        for idx in range(0, self.pop_size):
            # L = (np.exp(1) - np.exp((epoch / self.epoch) ** 2)) * (np.sin(2 * np.pi * self.generator.random()))
            alpha = (np.cos(2 * self.generator.random()) + 1) * (1 - epoch / self.epoch) - 1

            c_best = list_best[self.generator.choice([0, 1, 2])]

            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
            temp[rd_index] = 1

            # temp = self.generator.choice([0, 1], size=self.problem.n_dims)

            L = (np.exp(1) - np.exp((epoch / self.epoch) ** 2)) * (np.sin(2 * np.pi * self.generator.random()))
            R = L * temp  # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta  # Eq. 15
            if A > 1:  # detour foraging strategy
                ## Exploration phase
                if self.generator.random() > 0.5:
                    sigma = np.std([agent.solution for agent in self.pop], axis=0)  # This is a vector, not a number
                    r1 = self.generator.choice([0, 1], size=self.problem.n_dims)
                    # r1 = self.generator.random()
                    pos_new = self.pop[idx].solution + r1 * self.generator.normal(0, sigma)
                else:
                    r2 = self.generator.choice([0, 1], size=self.problem.n_dims)
                    # r2 = self.generator.random()
                    w = self.generator.choice([0, 1, self.generator.random()], p=[0.2, 0.6, 0.2])
                    dif2 = w * (self.problem.ub + self.problem.lb - self.pop[idx].solution) + (1 - w) * (self.g_best.solution - self.pop[idx].solution)
                    pos_new = self.pop[idx].solution + r2 * dif2

                    # levy_beta = self.generator.uniform(1.0, 3.0)  # self.generator.choice([1, 1.5, 2])  #self.generator.uniform(1.0, 2.0)
                    # levy_beta = self.generator.choice([1, 1.5, 2, 2.5, 3])
                    # levy_multiplier = 0.01 # self.generator.uniform(-0.01, 0.01)
                    # levy_vector = self.get_levy_flight_step(beta=levy_beta, multiplier=levy_multiplier, size=self.problem.n_dims, case=2)
                    #
                    # pos_new = self.pop[idx].solution + temp * levy_vector

                    # pos_new = self.pop[rand_idx].solution + self.generator.random() * (self.g_best.solution - temp * levy_vector)
                    # w = self.generator.random()
                    # pos_new = self.pop[idx].solution + w * levy_vector + self.generator.random() * (1 - w) * (self.g_best.solution - self.pop[idx].solution)


            else:
                ## Exploitation phase
                if self.generator.random() > 0.5:
                    rand_idx = self.generator.integers(0, self.pop_size)
                    #
                    # pos_new = self.pop[rand_idx].solution + R * (self.pop[idx].solution - self.pop[rand_idx].solution) + \
                    #     np.round(0.5 * (0.05 + self.generator.random())) * self.generator.normal(0, 1)      # Eq. 1

                    # pos_new = self.pop[rand_idx].solution + R * (self.g_best.solution - self.pop[rand_idx].solution)

                    # pos_new = self.pop[rand_idx].solution + self.generator.random() * (R * self.g_best.solution - self.pop[rand_idx].solution)

                    # pos_new = self.pop[rand_idx].solution + (2 * self.generator.random() - 1) * (R * self.g_best.solution - self.pop[idx].solution)  # Amazing

                    r1, r2 = self.generator.choice(range(0, self.pop_size), 2, replace=False)
                    pos_new = self.pop[r1].solution + (2 * self.generator.random() - 1) * R * (self.g_best.solution - self.pop[r2].solution)

                    # levy_beta = self.generator.uniform(1.0, 3.0)  # self.generator.choice([1, 1.5, 2])  #self.generator.uniform(1.0, 2.0)
                    # levy_multiplier = self.generator.uniform(-0.01, 0.01)
                    # levy_vector = self.get_levy_flight_step(beta=levy_beta, multiplier=levy_multiplier, size=self.problem.n_dims, case=2)

                    # pos_new = self.pop[idx].solution + self.generator.normal(0, 1) * (1 - self.pop[idx].target.fitness / (self.g_best.target.fitness+self.EPSILON)) * levy_vector

                    # pos_new = self.pop[rand_idx].solution + self.generator.random() * (self.g_best.solution - temp * levy_vector)
                    # w = self.generator.random()
                    # pos_new = self.pop[idx].solution + w * levy_vector + self.generator.random() * (1 - w) * (self.g_best.solution - self.pop[idx].solution)

                else:  # Random hiding stage
                    gr = np.zeros(self.problem.n_dims)
                    rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
                    gr[rd_index] = 1  # Eq. 12
                    H = self.generator.normal(0, 1) * (epoch / self.epoch)  # Eq. 8
                    b = self.pop[idx].solution + H * gr * self.pop[idx].solution  # Eq. 13
                    pos_new = self.pop[idx].solution + R * (self.generator.random() * b - self.pop[idx].solution)  # Eq. 11

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        # print(t1, t2)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, minmax=self.problem.minmax)



        # theta = 2 * (1 - epoch/self.epoch)
        # pop_new = []
        # t1, t2 = 0, 0
        # for idx in range(0, self.pop_size):
        #     # L = (np.exp(1) - np.exp((epoch / self.epoch) ** 2)) * (np.sin(2 * np.pi * self.generator.random()))
        #     alpha = (np.cos(2*self.generator.random())+1) * (1 - epoch/self.epoch) - 1
        #
        #     # Exploration
        #     temp = np.zeros(self.problem.n_dims)
        #     rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
        #     temp[rd_index] = 1
        #     # print(alpha)
        #     if alpha > 0.5 * self.generator.integers(-1,2):
        #         # print("Exploration")
        #         # if self.generator.random() > 0.5:
        #         #     levy_beta = self.generator.uniform(1.0, 2.0)
        #         #     levy_multiplier = self.generator.uniform(-0.1, 0.1)
        #         #     levy_vector = self.get_levy_flight_step(beta=levy_beta, multiplier=levy_multiplier, size=self.problem.n_dims, case=2)
        #         #     pos_new = self.pop[idx].solution + temp * levy_vector
        #         # else:
        #
        #         if self.generator.random() > 0.5:
        #             sigma = np.std([agent.solution for agent in self.pop], axis=0)  # This is a vector, not a number
        #             r1 = self.generator.choice([0, 1], size=self.problem.n_dims)
        #             # r1 = self.generator.random()
        #             pos_new = self.pop[idx].solution + r1 * self.generator.normal(0, sigma)
        #         else:
        #             r2 = self.generator.choice([0, 1], size=self.problem.n_dims)
        #             # r2 = self.generator.random()
        #             w = self.generator.choice([0, 1, self.generator.random()], p=[0.4, 0.4, 0.2])
        #             dif2 = w * (self.problem.ub + self.problem.lb - self.pop[idx].solution) + (1 - w) * (self.g_best.solution - self.pop[idx].solution)
        #             pos_new =  self.pop[idx].solution + r2 * dif2
        #
        #         # sigma = np.std([agent.solution for agent in self.pop], axis=0)  # This is a vector, not a number
        #         # # r1 = self.generator.choice([0, 1], size=self.problem.n_dims)
        #         # # r2 = self.generator.choice([0, 1], size=self.problem.n_dims)
        #         # r1, r2 = self.generator.random(2)
        #         # dif1 = self.generator.normal(0, sigma)
        #         # w = self.generator.choice([0, 1, 0.5])
        #         # dif2 = w * (self.problem.ub + self.problem.lb - self.pop[idx].solution) + (1 - w) * (self.g_best.solution - self.pop[idx].solution)
        #         # idx_rand = self.generator.choice(range(0, self.pop_size))
        #         # pos_new = self.pop[idx_rand].solution + r1 * dif1 + r2 * dif2
        #
        #     # Exploitation
        #     else:
        #         L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * (np.sin(2*np.pi*self.generator.random()))
        #         R = L * temp        # Eq 2
        #         A = 2 * np.log(1.0 / self.generator.random()) * theta      # Eq. 15
        #         if A > 1 :   # detour foraging strategy
        #             t1 += 1
        #             rand_idx = self.generator.integers(0, self.pop_size)
        #             #
        #             # pos_new = self.pop[rand_idx].solution + R * (self.pop[idx].solution - self.pop[rand_idx].solution) + \
        #             #     np.round(0.5 * (0.05 + self.generator.random())) * self.generator.normal(0, 1)      # Eq. 1
        #
        #             # pos_new = self.pop[rand_idx].solution + R * (self.g_best.solution - self.pop[rand_idx].solution)
        #
        #             # pos_new = self.pop[rand_idx].solution + self.generator.random() * (R * self.g_best.solution - self.pop[rand_idx].solution)
        #
        #             # pos_new = self.pop[idx].solution + self.generator.random() * (R * self.g_best.solution - self.pop[idx].solution)  # Amazing
        #
        #             levy_beta = self.generator.uniform(1.0, 3.0)  # self.generator.choice([1, 1.5, 2])  #self.generator.uniform(1.0, 2.0)
        #             levy_multiplier = self.generator.uniform(-0.01, 0.01)
        #             levy_vector = self.get_levy_flight_step(beta=levy_beta, multiplier=levy_multiplier, size=self.problem.n_dims, case=2)
        #             # pos_new = self.pop[rand_idx].solution + self.generator.random() * (self.g_best.solution - temp * levy_vector)
        #             w = self.generator.random()
        #             pos_new = self.pop[idx].solution + w * levy_vector + self.generator.random() * (1 - w) * (self.g_best.solution - self.pop[idx].solution)
        #         else:       # Random hiding stage
        #             t2 += 1
        #             gr = np.zeros(self.problem.n_dims)
        #             rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
        #             gr[rd_index] = 1        # Eq. 12
        #             H = self.generator.normal(0, 1) * (epoch / self.epoch)       # Eq. 8
        #             b = self.pop[idx].solution + H * gr * self.pop[idx].solution       # Eq. 13
        #             pos_new = self.pop[idx].solution + R * (self.generator.random() * b - self.pop[idx].solution)      # Eq. 11
        #     pos_new = self.correct_solution(pos_new)
        #     agent = self.generate_empty_agent(pos_new)
        #     pop_new.append(agent)
        #     if self.mode not in self.AVAILABLE_MODES:
        #         agent.target = self.get_target(pos_new)
        #         self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        # # print(t1, t2)
        # if self.mode in self.AVAILABLE_MODES:
        #     pop_new = self.update_target_for_population(pop_new)
        #     self.pop = self.greedy_selection_population(self.pop, pop_new, minmax=self.problem.minmax)


class CGG_ARO_02(CGG_ARO_01):
    """
    The original version of: Chaotic Gaussian-based Global Artificial Rabbits Optimization (CGG-ARO)
    """

    def __init__(self, epoch=10000, pop_size=100, n_leaders=2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        if n_leaders < int(0.2 * self.pop_size):
            self.n_leaders = n_leaders
        else:
            self.n_leaders = int(0.2 * self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        # piecewise chaotic map
        # Levy-flight strategy
        # Gaussian perturbation
        # Global best history
        # PLGG-ARO
        # Chaotic Gaussian-based Global Artificial Rabbits Optimization (CGG-ARO)

        ## Imagination hunting phase
        pop = []        # This pop new is like a prediction of future move
        for idx in range(0, self.pop_size):
            alpha = (np.cos(2 * self.generator.random()) + 1) * (1 - epoch / self.epoch) - 1
            if self.generator.random() < 0.5:     ## Exploration (move by the wild action of it own)
                sigma = np.std([agent.solution for agent in self.pop], axis=0)  # This is a vector, not a number
                diff = alpha * self.generator.normal(0, sigma)  # This is also a vector, Gaussian perturbation vector
            else:           ## Exploitation (move by the guide of the leader)
                diff = alpha * (self.g_best.solution - self.pop[idx].solution)
            pop.append(self.generate_empty_agent(diff))

        ## Real hunting phase
        theta = 2 * (1 - epoch/self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            # Exploration
            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
            temp[rd_index] = 1
            L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * (np.sin(2*np.pi*self.generator.random()))
            R = L * temp        # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta      # Eq. 15
            if A > 1:   # detour foraging strategy
                if self.generator.random() < 0.5:
                    pos_new = self.pop[idx].solution + self.generator.random() * pop[idx].solution
                else:
                    rand_idx = self.generator.integers(0, self.pop_size)
                    # sigma = np.std([agent.solution for agent in leaders], axis=0)  # This is a vector, not a number
                    # pos_new = leader.solution + R * (self.pop[idx].solution - leader.solution) + R * self.generator.normal(0, sigma)
                    pos_new = self.pop[rand_idx].solution + self.generator.random() * (R * self.g_best.solution - pop[rand_idx].solution)
            else:       # Random hiding stage
                gr = np.zeros(self.problem.n_dims)
                rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
                gr[rd_index] = 1        # Eq. 12
                H = self.generator.normal(0, 1) * (epoch / self.epoch)       # Eq. 8
                b = self.pop[idx].solution + H * gr * self.pop[idx].solution       # Eq. 13
                pos_new = self.pop[idx].solution + R * (self.generator.random() * b - self.pop[idx].solution)      # Eq. 11
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, minmax=self.problem.minmax)


class CGG_ARO_03(CGG_ARO_01):
    """
    The original version of: Chaotic Gaussian-based Global Artificial Rabbits Optimization (CGG-ARO)
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        # piecewise chaotic map
        # Levy-flight strategy
        # Gaussian perturbation
        # Global best history
        # PLGG-ARO
        # Chaotic Gaussian-based Global Artificial Rabbits Optimization (CGG-ARO)

        ## Imagination hunting phase
        pop = []        # This pop new is like a prediction of future move
        for idx in range(0, self.pop_size):
            alpha = (np.cos(2 * self.generator.random()) + 1) * (1 - epoch / self.epoch) - 1
            ## Exploration (move by the wild action of it own)
            # r1, r2, r3 = self.generator.choice(range(0, self.pop_size), 3, replace=False)
            # list_pos = [agent.solution for agent in [self.pop[r1], self.pop[r2], self.pop[r3]]]
            # sigma = np.std(list_pos, axis=0)
            sigma = np.std([agent.solution for agent in self.pop], axis=0)  # This is a vector, not a number
            # mean = np.mean([agent.solution for agent in self.pop], axis=0)
            pos_new = self.pop[idx].solution + alpha * self.generator.normal(0, sigma)  # This is also a vector, Gaussian perturbation vector
            pos_new = self.correct_solution(pos_new)
            pop.append(self.generate_empty_agent(pos_new))

        ## Real hunting phase
        theta = 2 * (1 - epoch/self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            # Exploration
            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
            temp[rd_index] = 1
            L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * (np.sin(2*np.pi*self.generator.random()))
            R = L * temp        # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta      # Eq. 15
            if A > 1:   # detour foraging strategy
                if self.generator.random() < 0.5:
                    ## Exploitation (move by the guide of the leader)
                    pos_new = self.pop[idx].solution + R * (self.g_best.solution - self.pop[idx].solution)
                else:
                    rand_idx = self.generator.integers(0, self.pop_size)
                    sigma = np.std([agent.solution for agent in pop], axis=0)  # This is a vector, not a number
                    pos_new = self.pop[idx].solution + R * (self.pop[idx].solution - self.pop[rand_idx].solution) + R * self.generator.normal(0, sigma)
            else:       # Random hiding stage
                if self.generator.random() < 0.5:
                    pos_new = pop[idx].solution + self.generator.random() * (R * self.g_best.solution - pop[idx].solution)      # Amazing
                else:
                    gr = np.zeros(self.problem.n_dims)
                    rd_index = self.generator.choice(np.arange(0, self.problem.n_dims), int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
                    gr[rd_index] = 1        # Eq. 12
                    H = self.generator.normal(0, 1) * (epoch / self.epoch)       # Eq. 8
                    b = self.pop[idx].solution + H * gr * self.pop[idx].solution       # Eq. 13
                    pos_new = self.pop[idx].solution + R * (self.generator.random() * b - self.pop[idx].solution)      # Eq. 11
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, minmax=self.problem.minmax)


# Time to test our new optimizer
# def objective_function(solution):
#     return np.sum(solution**2)

# problem_dict1 = {
#     "obj_func": objective_function,
#     "bounds": FloatVar(lb=[-100, ]*100, ub=[100, ]*100),
#     "minmax": "min",
#     "log_to": "console"
# }

# from opfunu.cec_based import cec2017
# f3 = cec2017.F132017(ndim=30)
#
# problem_dict1 = {
#     "obj_func": f3.evaluate,
#     "bounds": FloatVar(lb=f3.lb, ub=f3.ub),
#     "minmax": "min",
# }
#
# epoch = 1000
# pop_size = 50

# model = OriginalARO(epoch, pop_size)
# g_best = model.solve(problem_dict1)
# print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

# model2 = CGG_ARO_01(epoch, pop_size)
# g_best = model2.solve(problem_dict1)
# print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

# model3 = CGG_ARO_02(epoch, pop_size)
# g_best = model3.solve(problem_dict1)
# print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

# model4 = CGG_ARO_03(epoch, pop_size)
# g_best = model4.solve(problem_dict1)
# print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
