import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))

from trainer.ddqn_trainer import DDQNTrainer
from trainer.ddqn_agent import DDQNAgent
import numpy as np
import torch
import sys
import os
from rl_env import *
from configs.systemcfg import log_configs, mission_cfg, map_cfg, ppo_cfg, DEVICE, ddqn_cfg
from utils import Load, write_config_not_fromfile
import torch.optim as optim
import torch.nn as nn

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
if DEVICE != 'cpu':
    device = torch.device('cuda:'+str(DEVICE) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
if device == "cpu":
    print("cannot train with cpu")
    exit(0)
else:
    print("cuda: ", device)


def create_agent(state_size, action_size, actor_fc1_units=64,
                 actor_fc2_units=32, actor_lr=1e-3, critic_fc1_units=32,
                 critic_fc2_units=32, critic_lr=3e-3, gamma=0.99,
                 num_updates=100, max_eps_length=500, eps_clip=0.3,
                 critic_loss=0.5, entropy_bonus=0.01, batch_size=256, 
                 agent_idx=0, load_from_file=False, ckpt_idx = 0, 
                 ppo=False):

    """
    This function creates an agent with specified parameters for training.

    Arguments:
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
        actor_fc1_units: An integer number of units used in the first FC
            layer for the Actor object.
        actor_fc2_units: An integer number of units used in the second FC
            layer for the Actor object.
        actor_lr: A float designating the learning rate of the Actor's
            optimizer.
        critic_fc1_units: An integer number of units used in the first FC
            layer for the Critic object.
        critic_fc2_units: An integer number of units used in the second FC
            layer for the Critic object.
        critic_lr: A float designating the learning rate of the Critic's
            optimizer.
        gamma: A float designating the discount factor.
        num_updates: Integer number of updates desired for every
            update_frequency steps.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        eps_clip: Float designating range for clipping surrogate objective.
        critic_loss: Float designating initial Critic loss.
        entropy_bonus: Float increasing Actor's tendency for exploration.
        batch_size: An integer for minibatch size.

    Returns:
        agent: An Agent object used for training.
    """
    critic_size = actor_size= state_size 
    if ckpt_idx ==0:
        checkpoint_path = "./checkpoints/agent_" + str(agent_idx) + '.pth'
    else:
        checkpoint_path = "./checkpoints/agent_" + str(agent_idx) +'_'+str(ckpt_idx)+ '.pth'
    if ppo:
        # Create Actor/Critic networks based on designated parameters.
        actor_net = ActorNet(actor_size, action_size, actor_fc1_units,
                            actor_fc2_units).to(device)
        critic_net = CriticNet(critic_size, critic_fc1_units, critic_fc2_units)\
            .to(device)

        # Create copy of Actor/Critic networks for action prediction.
        actor_net_old = ActorNet(actor_size, action_size, actor_fc1_units,
                                actor_fc2_units).to(device)
        critic_net_old = CriticNet(critic_size, critic_fc1_units, critic_fc2_units)\
            .to(device)
        actor_net_old.load_state_dict(actor_net.state_dict())
        critic_net_old.load_state_dict(critic_net.state_dict())

        # Create PolicyNormal objects containing both sets of Actor/Critic nets.
        actor_critic = PolicyNormal(actor_net, critic_net)
        actor_critic_old = PolicyNormal(actor_net_old, critic_net_old)

        # Initialize optimizers for Actor and Critic networks.
        actor_optimizer = torch.optim.Adam(
            actor_net.parameters(),
            lr=actor_lr
        )
        critic_optimizer = torch.optim.Adam(
            critic_net.parameters(),
            lr=critic_lr
        )

        # # Create and return PPOAgent with relevant parameters.
        agent = PPOAgent(
            device=device,
            actor_critic=actor_critic,
            actor_critic_old=actor_critic_old,
            gamma=gamma,
            num_updates=num_updates,
            eps_clip=eps_clip,
            critic_loss=critic_loss,
            entropy_bonus=entropy_bonus,
            batch_size=batch_size,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )

    else:
        agent = DDQNAgent(state_size=state_size, action_size=action_size, checkpoint_path=checkpoint_path, load_model=load_from_file)

    return agent





def create_trainer(env, agents, save_dir, update_frequency=500,
                   max_eps_length=100, score_window_size=100, thread = True, detach_thread = True, type_ = "MAPPOTrainer"):   #change the type of agents
    
    """
    Initializes trainer to train agents in specified environment.

    Arguments:
        env: Environment used for Agent evaluation and training.
        agents: Agent objects used for training.
        save_dir: Path designating directory to save resulting files.
        update_frequency: An integer designating the step frequency of
            updating target network parameters.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        score_window_size: Integer window size used in order to gather
            max mean score to evaluate environment solution.
        
    Returns:
        trainer: A MAPPOTrainer object used to train agents in environment.
        
        
    Note: if update_frequency is small, plase don't use detach_thread.
    """

    # Initialize MAPPOTrainer object with relevant arguments.
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if type_=="DDQNTrainer":
        method = DDQNTrainer
    else:
        method = MAPPOTrainer
    trainer = method(
        env=env,
        agents=agents,
        score_window_size=score_window_size,
        max_episode_length=max_eps_length,
        update_frequency=update_frequency,
        save_dir=save_dir,
        thread = thread,
        detach_thread = detach_thread,
        train_start_factor = 2
    )
    trainer.timestep = 0
    trainer.i_episode = 0

    return trainer
  
def train_agents(env, trainer, n_episodes=100000, target_score=100000,
                 score_window_size=100):
    """
    This function carries out the training process with specified trainer.

    Arguments:
        env: Environment used for Agent evaluation and training.
        trainer: A MAPPOTrainer object used to train agents in environment.
        n_episodes: An integer for maximum number of training episodes.
        target_score: An float max mean target score to be achieved over
            the last score_window_size episodes.
        score_window_size: The integer number of past episode scores
            utilized in order to calculate the current mean scores.
    """

    # Train the agent for n_episodes.
    for i_episode in range(1, n_episodes + 1):

        # Step through the training process.
        trainer.step()
        # Print status of training every 100 episodes.
        if i_episode % 100 == 0:
            scores = np.max(np.array(trainer.score_history), axis=1).tolist()
            trainer.print_status()
        # If target achieved, print and plot reward statistics.
        mean_reward = np.max(
            trainer.score_history[-score_window_size:], axis=1
        ).mean()
        print("e: {} mean_reward {}".format(i_episode, mean_reward))
        # if mean_reward >= target_score:
        if i_episode >0 and i_episode%1000 ==0:
            trainer.save()
            trainer.print_status()
            trainer.plot()
        elif i_episode>0 and i_episode % score_window_size ==0:
            trainer.print_status()
            trainer.plot()
        elif mean_reward >= target_score or i_episode == n_episodes:
            print('Environment is solved.')
            env.close()
            trainer.print_status()
            trainer.plot()
            trainer.save()
            break

def train_agents_ma(env, trainer, n_episodes=100000, target_score=100000,
                 score_window_size=100):
    """
    This function carries out the training process with specified trainer.

    Arguments:
        env: Environment used for Agent evaluation and training.
        trainer: A MAPPOTrainer object used to train agents in environment.
        n_episodes: An integer for maximum number of training episodes.
        target_score: An float max mean target score to be achieved over
            the last score_window_size episodes.
        score_window_size: The integer number of past episode scores
            utilized in order to calculate the current mean scores.
    """

    # Train the agent for n_episodes.
    for i_episode in range(1, n_episodes + 1):

        # Step through the training process.
        trainer.step_ma()
        # Print status of training every 100 episodes.
        if i_episode % 100 == 0:
            scores = np.max(np.array(trainer.score_history), axis=1).tolist()
            trainer.print_status()
        # If target achieved, print and plot reward statistics.
        mean_reward = np.max(
            trainer.score_history[-score_window_size:], axis=1
        ).mean()
        print("e: {} mean_reward {}".format(i_episode, mean_reward))
        # if mean_reward >= target_score:
        if i_episode >0 and i_episode%1000 ==0:
            trainer.save()
            trainer.print_status()
            trainer.plot()
        elif i_episode>0 and i_episode % score_window_size ==0:
            trainer.print_status()
            trainer.plot()
        elif mean_reward >= target_score or i_episode == n_episodes:
            print('Environment is solved.')
            env.close()
            trainer.print_status()
            trainer.plot()
            trainer.save()
            break

def ddqn(**kwargs):
    verbose = kwargs.get('verbose', False)
    load = Load()
    graph, map_information =  load.get_infor()
    task_generator = TaskGenerator(1, map_information)
    config = write_config_not_fromfile(task_generator)
    register_env("its_env", lambda config: ITSEnv(config, verbose=verbose, map__=map_information, generator=task_generator))
    # Initialize environment, extract state/action dimensions and num agents.
    env = ITSEnv(config, verbose=verbose, map__=map_information, generator=task_generator)
    num_agents = config['n_vehicles']
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.shape[0] 
    # Initialize agents for training.
    agents = [create_agent(state_size, action_size, agent_idx=i, load_from_file=False, ckpt_idx=0) for i in range(num_agents)]
    save_dir = os.path.join(os.getcwd(), 'saved_files_global_combine_decay_{}_lr_{}_batch_size_{}_modify_reward_{}_combine_{}_more'.format(ddqn_cfg['epsilon_decay'],ddqn_cfg['learning_rate'],ddqn_cfg['batch_size'],ddqn_cfg['modify_reward'],ddqn_cfg['combine']))
    trainer = create_trainer(env, agents, save_dir, 
                             thread = config['thread'], 
                             detach_thread=config['detach_thread'],
                             score_window_size =  config['score_window_size'],
                             max_eps_length=config['n_miss_per_vec']*config['n_vehicles'],
                             type_='DDQNTrainer',
                             update_frequency=ddqn_cfg['batch_size']/4
                             )

    train_agents(env, trainer, score_window_size = config['score_window_size'])



if __name__ == '__main__':
    # ddqn()
    mppo()

