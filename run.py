# Module: run.py
# Purpose: Main CLI entry point for running simulations and analyses.
#
# This module parses command-line arguments, configures a simple file-backed
# logger (redirects stdout to a timestamped file under `logs/`), and dispatches
# execution to DRL or meta-heuristic routines (e.g. `ddqn`, `many_metaheuristics`).
#
# Notes:
# - This file changes `sys.stdout` to a `Logger` instance so print statements
#   across the codebase are captured in a log file. No runtime logic is altered
#   by the comments here.

import matplotlib.pyplot as plt
# import scienceplots
import matplotlib
# matplotlib.rcParams['font.size']=11
# plt.style.use(["science", "ieee"])
from src import *
import argparse
from src.DRL.rl_run import ddqn
from src.DRL.rl_eval import eval_ddqn
from src.meta_heuristic.script_many_metaheuristics import many_metaheuristics
from src.meta_heuristic.script_statistic import get_statistic_results
from src.meta_heuristic.script_visualize import draw_results, draw_eval_results, draw_compared_greedy_random_drl, csv_compared_greedy_random_meta
import os, sys
from configs.systemcfg import log_configs, DEVICE

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        if self.log:
            self.log.close()
            self.log = None

# Sử dụng Logger để ghi log
if not os.path.exists(log_configs['log_dir']):
    os.makedirs(log_configs['log_dir'])
from datetime import datetime

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Get the current date and time for the log file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./logs/run_log_{timestamp}.log"
logger = Logger(log_filename)
sys.stdout = logger


def parser_argument():
    parser = argparse.ArgumentParser(description='Simulation for ITS Joint task handling and missiong processing paper')

    parser.add_argument('-i', '--input', type=str,choices=['ppo', 'mppo', 'ppo_test', 'A2C', 'ddqn', 'ddqn_ma', 'eval_ddqn', 'many_metaheuristics', 'run_single_agent_ddqn', 'meta_heuristic_proposal', 'None'], required=True, help='Simulation type (DRL or metaheuristic).')
    parser.add_argument('-c', '--compare', type=str, choices=['drls', 'drl_and_meta_heuristic_proposal'], help='Compare simulation btw DRL and metaheuristic.')
    parser.add_argument('-a', '--analysis', type = int, help='Analysis result from many meta_heuristics')
    parser.add_argument('-device', '--cuda', type = int, default=-1)
    parser.add_argument('--verbose', action='store_true', help='Display information during run simulation')

    return parser.parse_args()

def main():
    args = parser_argument()
    DEVICE = args.cuda
    current_module = sys.modules[__name__]
    if args.input == 'None':
        print("No function is selected")
    else:
        getattr(current_module, args.input)(**vars(args))
        
    if args.analysis == 0:
        get_statistic_results()
        draw_results()
    elif args.analysis:
        draw_eval_results(args.analysis)
        draw_compared_greedy_random_drl(args.analysis)
        csv_compared_greedy_random_meta(args.analysis)
    
if __name__ == "__main__":
    main()
    
sys.stdout.log.close()
        