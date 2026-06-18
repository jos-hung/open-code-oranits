import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import threading
from threading import active_count
import sys
from physic_definition.system_base.ITS_based import TaskGenerator
from configs.systemcfg import avg_reward, ddqn_cfg, mission_cfg
# plt.style.use('dark_background')
import copy

class DDQNTrainer:
    """
    A class for the implementation and utilization of the training process
    steps for the Multi-Agent Proximal Policy Optimization algorithm.

    Attributes:
        env: Environment used for Agent evaluation and training.
        agents: Agent objects being trained in env.
        score_window_size: Integer window size used in order to gather
            mean score to evaluate environment solution.
        max_epsiode_length: An integer for maximum number of timesteps per
            episode.
        update_frequency: An integer designating the step frequency of
            updating target network parameters.
        save_dir: Path designating directory to save resulting files.
    """

    def __init__(self, env, agents, score_window_size, max_episode_length,
                 update_frequency, save_dir, thread = True, detach_thread = True, train_start_factor = 1):
        """Initializes MAPPOTrainer attributes."""
        # Initialize relevant variables for training.
        self.env = env
        self.agents = agents
        self.score_window_size = score_window_size
        self.max_episode_length = max_episode_length
        self.update_frequency = update_frequency
        self.start_train = train_start_factor*update_frequency
        self.save_dir = save_dir
        self.score_history = []
        self.task_history = []
        self.benefit_history = []
        self.episode_length_history = []
        self.timestep = 0
        self.i_episode = 0
        self.thread = thread
        self.detach_thread = detach_thread
        self.eta = 0
        self.max_score = 0
        self.tg = TaskGenerator(15,env.lmap)
        
        self.checkpoints_dir = self.save_dir +'/checkpoints'
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        else:
            print(f"Folder '{self.checkpoints_dir}' existed.")
    def step_env(self, actions, states):
        """
        Realizes actions in environment and returns relevant attributes.

        Parameters:
            actions: Actions array to be realized in the environment.

        Returns:
            states: Array with next state information.
            rewards: Array with rewards information.
            dones: Array with boolean values with 'true' designating the
                episode has finished.
            env_info: BrainInfo object with current environment data.
        """

        # From environment information, extract states and rewards.
        env_info = self.env.step(actions, self.agents, states)
        next_states = env_info[0]
        rewards = env_info[1]
        
        # Evaluate if episode has finished.
        dones = env_info[2]
        truncateds = env_info[3]
        
        action_dict = env_info[5]

        return next_states, rewards, dones, truncateds, env_info[4], action_dict
    
    def step_env_ma(self, actions):
        """
        This is step for multi-action DRL

        Parameters:
            actions: Actions array to be realized in the environment.

        Returns:
            states: Array with next state information.
            rewards: Array with rewards information.
            dones: Array with boolean values with 'true' designating the
                episode has finished.
            env_info: BrainInfo object with current environment data.
        """

        # From environment information, extract states and rewards.
        env_info = self.env.step_ma(actions)
        next_states = env_info[0]
        rewards = env_info[1]
        # Evaluate if episode has finished.
        dones = env_info[2]
        truncateds = env_info[3]
        # action_dict = env_info[5]

        return next_states, rewards, dones, truncateds, env_info#, action_dict
    
    def calculate_r_optimized(self, r):
        sum_squared_diffs = 0
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                sum_squared_diffs += (r[i] - r[j]) ** 2
        return np.sqrt(sum_squared_diffs)
    
    def run_episode(self):
        """
        Runs a single episode in the training process for max_episode_length
        timesteps.

        Returns:
            scores: List of rewards gained at each timestep.
        """      
        # Initialize list to hold reward values at each timestep.
        scores = []
        for i in range(self.env.data['n_vehicles']):
            scores.append([])

        # Restart the environment and gather original states.
        env_info = self.env.reset()
        states = env_info[0]

        # Act and evaluate results and networks for each timestep.
        actionssss = []
        for t in range(self.max_episode_length):
            print(t, self.max_episode_length)
            self.timestep += 1
            # Sample actions for each agent while keeping track of states,
            # actions and log probabilities.
            processed_states, actions, actions_save, log_probs = [], [], [], []
            for idx, state in enumerate(states):
                agent = self.agents[idx]
                observationip = np.reshape(states[state], (1, -1))
                processed_state = torch.from_numpy(observationip).float()
                action, log_prob = agent.get_actions(processed_state, idx)
                if any(torch.equal(processed_state, item) for item in processed_states) \
                    and int(np.argmax(action[1])) in actionssss:
                    continue
                actionssss.append(int(np.argmax(action[1])))
                processed_states.append(processed_state)
                log_probs.append(log_prob)
                actions.append(action)
                actions_save.append(action[1])
            # Realize sampled actions in environment and evaluate new state.
            next_states, rewards, dones, truncated, _, actions = self.step_env(actions, states)

            dones = [dones]*len(states)
            # Add experience to the memories for each agent.
            #avg reward
            # adjust_reward = copy.deepcopy(rewards)
            # keys = list(adjust_reward.keys())
            # for val in range(len(adjust_reward[keys[0]])):
            #     r = []
            #     if t == 0:
            #         old_eta  = 0
            #     for key in adjust_reward:
            #         r.append(adjust_reward[key][val])
                
            #     eta = self.calculate_r_optimized(r)
            #     for key in adjust_reward:
            #         r_change =  adjust_reward[key][val] + 0.05*(eta - old_eta)
            #         adjust_reward[key][val] = r_change
            #     old_eta = eta
            for agent, state, action, log_prob, reward, done, next_state in \
                    zip(self.agents, processed_states, actions, log_probs,
                        rewards, dones, next_states):
                action_save = actions[action]
                state = state.view(-1)
                next_state = next_states[next_state]
                agent.add_memory(state, action_save, rewards[reward], next_state)

            states = next_states
            # Initiate learning for agent if update frequency is observed.
            if self.agents[0].train_start < self.timestep and self.timestep >self.start_train and self.timestep%self.env.data['n_miss_per_vec']==0:
                threads = []
                for idx, agent in enumerate(self.agents):
                    if self.thread == False:
                        agent.train_model()
                    else:
                        update_thread = threading.Thread(target=agent.train_model)
                        if self.detach_thread:
                            update_thread.daemon = True
                            # print("training via a detach thread: {}".format(idx))
                            update_thread.start()
                            pass
                        else:
                            update_thread.start()
                            threads.append(update_thread)
                            pass
                if self.detach_thread==False:
                    for idx, thread in enumerate(threads):
                        # print("update threading {}.... start join".format(idx))
                        thread.join()
            if self.timestep > 0 and self.timestep%1000==0:
                for idx, agent in enumerate(self.agents):
                    agent.update_target_model()
            for idx, reward in rewards.items():
                scores[idx]+=reward
        
            # End episode if desired score is achieved.
            if np.any(dones):
                break
        
        return scores
    
    def do_modify_reward(self,modify_reward):
        change = [0]*len(modify_reward['state'])
        cnt_completed = 0
        for idx, infor in enumerate(modify_reward['modified_infor']):
            for data in infor:
                if data is None:
                    continue
                cnt_completed += 1
        for idx, infor in enumerate(modify_reward['modified_infor']):
            for data in infor:
                if data is None:
                    continue
                vehicle_id, mission_id, n_remove_depends, n_waiting, profit = data
                for aidx, act in enumerate(modify_reward['action']) :
                    #adix chính là thứ tự của action trên mỗi agent
                    if modify_reward['action'][aidx][vehicle_id] == mission_id and modify_reward['current_wards'][aidx][vehicle_id][0] >=0:
                        '''
                        Để tránh việc bỏ qua việc thực hiện thành công dựa vào yếu tố của nó: ví dụ như quãng đường ngắn và gần với xe.
                        Ko nên để reward cho việc xóa bỏ dependce quá cao.
                        #ví dụ việc hoàn thành 1 task: maximum là 200 thì giá trị của remove depend ko nên vuợt quá nó.
                        '''
                        add_reward = ((vehicle_id+1)/mission_cfg['n_vehicle'])*((mission_cfg['n_miss_per_vec']-aidx)*(n_remove_depends)*50 - n_waiting*50)+ cnt_completed*mission_cfg['n_mission']
                        # if add_reward > modify_reward['current_wards'][aidx][vehicle_id][0]:
                        #     add_reward = modify_reward['current_wards'][aidx][vehicle_id][0]
                        # print("------->",vehicle_id, cnt_completed, n_remove_depends, n_waiting, modify_reward['current_wards'][aidx][vehicle_id], add_reward)
                        # print("((mission_cfg['n_miss_per_vec']-aidx)*(n_remove_depends)*50 - n_waiting*50):",((mission_cfg['n_miss_per_vec']-aidx)*(n_remove_depends)*50 - n_waiting*50), add_reward)
                        modify_reward['current_wards'][aidx][vehicle_id][0] = profit + add_reward
                        
                        break
                change[idx] = True
        #update memory
        for idx, state in enumerate(modify_reward['state']):
            for sidx, vehicle in enumerate(state):
                    if change[idx] == True and modify_reward['action'][idx][sidx]!=-1:
                        
                        self.agents[sidx].add_global_memory(state[vehicle], 
                                            modify_reward['action'][idx][sidx],
                                            modify_reward['current_wards'][idx][sidx], 
                                            modify_reward['next_state'][idx][vehicle],
                                            modify_reward['dones'][idx][sidx])
                        self.agents[sidx].add_memory(state[vehicle], 
                                            modify_reward['action'][idx][sidx],
                                            modify_reward['current_wards'][idx][sidx], 
                                            modify_reward['next_state'][idx][vehicle],
                                            modify_reward['dones'][idx][sidx])
                    elif modify_reward['action'][idx][sidx]!=-1:
                        self.agents[sidx].add_memory(state[vehicle], 
                                                modify_reward['action'][idx][sidx],
                                                [-100], 
                                                modify_reward['next_state'][idx][vehicle],
                                                modify_reward['dones'][idx][sidx])
                        self.agents[sidx].add_global_memory(state[vehicle], 
                                                modify_reward['action'][idx][sidx],
                                                [-100], 
                                                modify_reward['next_state'][idx][vehicle],
                                                modify_reward['dones'][idx][sidx])
                        
        return 0
    
    def run_episode_modify_reward(self):
        """
        Runs a single episode in the training process for max_episode_length
        timesteps.

        Returns:
            scores: List of rewards gained at each timestep.
        """      
        # Initialize list to hold reward values at each timestep.
        scores = []
        for i in range(self.env.data['n_vehicles']):
            scores.append([])

        # Restart the environment and gather original states.
        env_info = self.env.reset()
        states = env_info[0]

        # Act and evaluate results and networks for each timestep.
        actionssss = []
        modify_reward = {"step": [], 'state': [], 'action':[], 'current_wards':[], 'next_state':[], 'modified_infor':[], 'dones': []}
        for t in range(self.max_episode_length):
            print(t, self.max_episode_length)
            self.timestep += 1
            # Sample actions for each agent while keeping track of states,
            # actions and log probabilities.
            processed_states, actions, actions_save, log_probs = [], [], [], []
            for idx, state in enumerate(states):
                agent = self.agents[idx]
                observationip = np.reshape(states[state], (1, -1))
                processed_state = torch.from_numpy(observationip).float()
                action, log_prob = agent.get_actions(processed_state, idx)
                if any(torch.equal(processed_state, item) for item in processed_states) \
                    and int(np.argmax(action[1])) in actionssss:
                    continue
                actionssss.append(int(np.argmax(action[1])))
                processed_states.append(processed_state)
                log_probs.append(log_prob)
                actions.append(action)
                actions_save.append(action[1])
            # Realize sampled actions in environment and evaluate new state.
            next_states, rewards, dones, truncated, done_process_infor, actions = self.step_env(actions, states)
            dones = [dones]*len(states)
            modify_reward['step'].append(t)
            modify_reward['state'].append(states)
            modify_reward['action'].append(actions)
            modify_reward['current_wards'].append(rewards)
            modify_reward['next_state'].append(next_states)
            modify_reward['modified_infor'].append(done_process_infor)
            modify_reward['dones'].append(dones)
            # Initiate learning for agent if update frequency is observed.
            if self.agents[0].train_start < self.timestep and self.timestep >self.start_train and self.timestep%self.env.data['n_miss_per_vec']==0:
                threads = []
                for idx, agent in enumerate(self.agents):
                    if self.thread == False:
                        agent.train_model()
                    else:
                        update_thread = threading.Thread(target=agent.train_model)
                        if self.detach_thread:
                            update_thread.daemon = True
                            # print("training via a detach thread: {}".format(idx))
                            update_thread.start()
                            pass
                        else:
                            update_thread.start()
                            threads.append(update_thread)
                            pass
                if self.detach_thread==False:
                    for idx, thread in enumerate(threads):
                        # print("update threading {}.... start join".format(idx))
                        thread.join()
            if self.timestep > 0 and self.timestep%1000==0:
                for idx, agent in enumerate(self.agents):
                    agent.update_target_model()
            for idx, reward in rewards.items():
                scores[idx]+=reward
        
            # End episode if desired score is achieved.
            if np.any(dones):
                break
            states = next_states
        self.do_modify_reward(modify_reward)
        return scores

    def run_episode_ma(self):
        """
        This is multi-action running epochs  
        ALl agent will select the mission, if they select the same action, the agent which select after will be penalty.
        """

        # Initialize list to hold reward values at each timestep.
        n_miss_per_vec = self.env.data['n_miss_per_vec']
        scores = []
        completed_selection = np.array([n_miss_per_vec]*self.env.data['n_vehicles'])
        orders = [0]*self.env.data['n_vehicles']
        
        for i in range(self.env.data['n_vehicles']):
            scores.append([])

        # Restart the environment and gather original states.
        env_info = self.env.reset()
        states = env_info[0]
        
        actions = [0]*self.env.data['n_missions']
        
        tem_memory_action = {}
        for i in range(self.env.data['n_vehicles']):
            tem_memory_action[i] = [[],[],[],[],[]] #state, action, reward, next_state, order
        
        cnt = 0
        max_free_select = 5
        first_queue_list = []
        while (self.env.action_memory == 0).any():
            for idx, state in enumerate(states):
                
                cur_idx = idx
                agent = self.agents[cur_idx]
                observationip = np.reshape(states[state], (1, -1))
                processed_state = torch.from_numpy(observationip).float()
                
                if completed_selection[cur_idx]<=0:
                    continue
                action, log_prob = agent.get_actions(processed_state, cur_idx)
                if agent.epsilon > self.env.generator.random() or cnt > max_free_select:
                    action = self.env.generator.integers(0, agent.action_size)
                else:
                    action = int(np.argmax(action[1]))
                if any(torch.equal(processed_state, item) for item in tem_memory_action[cur_idx][0]) \
                    and any(torch.equal(processed_state, item) for item in tem_memory_action[cur_idx][3]) \
                    and action in tem_memory_action[cur_idx][1]:
                    # print("true------------>")
                    continue
                tem_memory_action[cur_idx][0].append(processed_state) 
                tem_memory_action[cur_idx][3].append(processed_state)
                tem_memory_action[cur_idx][1].append(action)
                if self.env.action_memory[action]:
                    tem_memory_action[cur_idx][2].append(-0.01*avg_reward) 
                    tem_memory_action[cur_idx][4].append(-1)
                    continue
                
                self.env.action_memory[action] = 1
                first_queue_action = len(self.env.missions[action].get_depends()) == 0
                if first_queue_action:
                    first_queue_list.append(action)
                    
                next_state= self.env.get_ma_observations(first_queue_list, move_vehicle_pos = first_queue_action)
                tem_memory_action[cur_idx][2].append(0)

                observationip = np.reshape(next_state[state], (1, -1))
                next_state = torch.from_numpy(observationip).float()
                tem_memory_action[cur_idx][3][-1] = next_state
                tem_memory_action[cur_idx][4].append(orders[cur_idx])
                actions[action] = (orders[cur_idx],cur_idx)
                orders[cur_idx] += 1
                completed_selection[cur_idx] -= 1    
            cnt += 1
            states = self.env.get_ma_observations(first_queue_list)
            # if (completed_selection==0).all():
            #     break
            # Realize sampled actions in environment and evaluate new state.
        while(0 in actions):
            actions.remove(0)
        _, rewards, dones, truncated, _ = self.step_env_ma(actions)

        dones = [dones]*len(states)
        # Add experience to the memories for each agent.
        #ave reward
        #state, action, reward, next_state
        for idx, key in enumerate(tem_memory_action):
            cur_memory = tem_memory_action[key]
            reward = rewards[idx]
            reduce = rewards[idx]/n_miss_per_vec
            if reward/(n_miss_per_vec*100)>1.0:
                reduce = 0
            for i in range(len(cur_memory[0])):
                if cur_memory[2][i] == 0 and cur_memory[4][i]!=-1:
                    self.agents[idx].add_memory(cur_memory[0][i],cur_memory[1][i], reward + cur_memory[2][i]-cur_memory[4][i]*reduce, cur_memory[3][i])
                else:
                    self.agents[idx].add_memory(cur_memory[0][i],cur_memory[1][i], cur_memory[2][i],cur_memory[3][i])
                                    
        # Initiate learning for agent if update frequency is observed.
        print(self.agents[0].train_start, self.timestep,self.start_train)
        if self.agents[0].train_start < self.timestep and self.timestep >self.start_train:
            threads = []
            for idx, agent in enumerate(self.agents):
                if self.thread == False:
                    agent.train_model()
                else:
                    update_thread = threading.Thread(target=agent.train_model)
                    if self.detach_thread:
                        update_thread.daemon = True
                        # print("training via a detach thread: {}".format(idx))
                        update_thread.start()
                        pass
                    else:
                        update_thread.start()
                        threads.append(update_thread)
                        pass
            if self.detach_thread==False:
                for idx, thread in enumerate(threads):
                    # print("update threading {}.... start join".format(idx))
                    thread.join()
        if self.timestep > 0 and self.timestep%5000==0:
            for idx, agent in enumerate(self.agents):
                agent.update_target_model()
        
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards,1).tolist()
        self.timestep += 1
        
        return rewards

    
    def step(self):
        """
        Initiates run of an episode and logs the resulting total rewards and
        episode lengths.
        """

        # Run a single episode in environment.
        self.i_episode += 1
        if ddqn_cfg['modify_reward']:
            scores = self.run_episode_modify_reward()
        else:
            scores = self.run_episode()
            print("Not run episode with modify reward")
            
        
        # Sum the episode rewards for each agent to get the total rewards.
        score_by_agent = np.sum(scores, axis=1)

        # Store total rewards and episode lengths.
        self.score_history.append(score_by_agent)
        # if self.max_score  < max(score_by_agent):
        self.max_score = max(score_by_agent)
        self.episode_length_history.append(len(scores))
        
    def step_ma(self):
        """
        Initiates run of an episode and logs the resulting total rewards and
        episode lengths.
        """

        # Run a single episode in environment.
        self.i_episode += 1
        scores = self.run_episode_ma()

        # Sum the episode rewards for each agent to get the total rewards.
        score_by_agent = np.sum(scores, axis=1)

        # Store total rewards and episode lengths.
        self.score_history.append(score_by_agent)
        self.episode_length_history.append(len(scores))

    def save(self):
        """
        Saves actor_critic for both agents once successful score is achieved.
        """           
        for agent_ix in range(len(self.agents)):
            agent = self.agents[agent_ix]
            filename = self.checkpoints_dir +f'/agent_{agent_ix}_{self.i_episode}.pth'
            agent.save_model(filename)            

    def print_status(self):
        """Prints reward info and episode length stats at current episode."""

        # Calculate necessary statistics.
        mean_reward = np.mean(
            self.score_history[-self.score_window_size:],
            axis=0
        )
        agent_info = ''.join(f'Mean Reward Agent_{i}: {mean_reward[i]:.2f}, '
                             for i in range(len(self.agents)))
        max_mean = np.max(self.score_history[-self.score_window_size:],
                          axis=1).mean()
        mean_eps_len = np.mean(
            self.episode_length_history[-self.score_window_size:]
        ).item()

        # Print current status to terminal.
        print(
            f'\033[1mEpisode {self.i_episode} - '
            f'Mean Max Reward: {max_mean:.2f}\033[0m'
            f'\n\t{agent_info}\n\t'
            f'Mean Total Reward: {mean_reward.sum():.2f}, '
            f'Mean Episode Length {mean_eps_len:.1f}'
        )

    def plot(self):
        """
        Plots moving averages of maximum reward and rewards for each agent.
        Avoids using colors that blend with the background (e.g., white).
        """
        

        # Initialize DataFrame
        columns = [f'Agent {i}' for i in range(len(self.agents))]
        df = pd.DataFrame(self.score_history, columns=columns)
        df['Max'] = df.max(axis=1)

        # Setup figure and axis
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_title('Learning Curve: Multi-Agent DDQN', fontsize=28)
        ax.set_xlabel('Episode', fontsize=21)
        ax.set_ylabel('Score', fontsize=21)

        # Use a colormap that avoids white (tab10 is a good default for up to 10 agents)
        df.rolling(self.score_window_size).mean().iloc[:, :-1].plot(
            ax=ax,
            colormap='tab10',
            legend=True
        )

        # Plot max line in red
        df['Max'].rolling(self.score_window_size).mean().plot(
            ax=ax,
            color='red',
            linewidth=2,
            label='Max Reward'
        )

        # Grid, legend, and layout
        ax.grid(color='gray', linewidth=0.2)
        ax.legend(fontsize=13)
        plt.tight_layout()

        # Save plot and data
        filename = f'scores.png'
        fig.savefig(os.path.join(self.save_dir, filename))
        df.to_csv(os.path.join(self.save_dir, "_reward.csv"))

        plt.close()
