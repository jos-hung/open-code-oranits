import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
# torch.autograd.set_detect_anomaly(True)
# Check if GPU is available
import time
from threading import Lock, active_count
from configs.systemcfg import DEVICE, GLOBAL_SEED

if DEVICE != 'cpu':
    device = torch.device('cuda:'+str(DEVICE) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
if device == 'cpu':
    print("cannot train with cpu")
    exit(0)
else:
    print("cuda: ", device)

    
from configs.systemcfg import ddqn_cfg, eval
class DDQNAgent(nn.Module):
    global_memory = deque(maxlen=ddqn_cfg['maxlen_mem'])
    
    def __init__(self, state_size, action_size, checkpoint_path = './', load_model = False):
        super(DDQNAgent, self).__init__()
        self.load_model = load_model

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = ddqn_cfg['discount_factor']
        self.learning_rate = ddqn_cfg['learning_rate']
        self.epsilon = ddqn_cfg['epsilon']
        self.epsilon_decay = ddqn_cfg['epsilon_decay']
        self.epsilon_min =ddqn_cfg['epsilon_min']
        self.batch_size = ddqn_cfg['batch_size']
        self.train_start = self.batch_size
        self.global_memory = DDQNAgent.global_memory
        self.memory = deque(maxlen=ddqn_cfg['maxlen_mem'])
        # self.memory = deque(maxlen=ddqn_cfg['maxlen_mem'])

        self.model = self.build_model().to(device)
        self.target_model = self.build_model().to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.model_file = checkpoint_path 
        self.generator = np.random.default_rng(GLOBAL_SEED)


        if self.load_model:
            self.model.load_state_dict(torch.load(self.model_file))
            self.epsilon = 0

        self.update_target_model()
        self.lock = Lock()

    def build_model(self):
        layer_size_1 = self.state_size + int(self.state_size*0.3)
        layer_size_2 = int(self.state_size*0.6)        
        layer_size_3 = int(self.state_size*0.2)
        model = nn.Sequential(
            nn.Linear(self.state_size, layer_size_1),
            nn.SELU(),
            nn.Linear(layer_size_1, layer_size_2),
            nn.SELU(),
            nn.Linear(layer_size_2, layer_size_3),
            nn.ELU(),
            nn.Linear(layer_size_3, self.action_size),
            nn.ELU()
        )
        self.softmax = nn.Softmax(dim=1)
        return model
    
    def forward(self, x):
        x = self.model(x)
        #x = self.softmax(x)
        return x

    def save_model(self, name):
        # torch.save(self.model.state_dict(), name)
        pass

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, idx):
        if self.epsilon > self.generator.random():
            return np.random.randint(0, self.action_size)
        else:
            state = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                q_value = self.model(state)
            return torch.argmax(q_value).item()
        
    def get_actions(self, state, vid):
         # Generates actions and log probs from current Normal distribution.
        state = state.to(device)
        with torch.no_grad():
            actions = \
                self(state)
        actions = actions.cpu().detach()
        actions  = [vid, actions]
        return actions, None

    def add_memory(self, state, action, reward, next_state, done = 0):
        if action == -1:
            return
        
        if len(self.memory) >= ddqn_cfg['maxlen_mem']:
            self.memory.popleft()
        self.memory.append((state, action, reward, next_state, done))

    def add_global_memory(self, state, action, reward, next_state, done = 0):
        if action ==-1:
            return 
        
        if len(self.global_memory) >= ddqn_cfg['maxlen_mem']:
            self.global_memory.popleft()
        self.global_memory.append((state, action, reward, next_state, done))
        
    def train_model(self):
        if eval:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print(f"self.learning_rate {self.learning_rate}")
        print(f"epsilon = {self.epsilon}")

        self.lock.acquire()
        try:
            if self.generator.random() > 1-ddqn_cfg['combine'] and len(self.global_memory) >= self.batch_size:
                mini_batch = random.sample(self.global_memory, self.batch_size)

            else:
                mini_batch = random.sample(self.memory, self.batch_size)
                
            states = np.zeros((self.batch_size, self.state_size))
            next_states = np.zeros((self.batch_size, self.state_size))
            actions, rewards, dones = [], [], []

            for i in range(self.batch_size):
                # print("1, data-> ", mini_batch[i][0])
                # print("2, data-> ", mini_batch[i][2])
                # print("3, data-> ", mini_batch[i][3])
                # print("4, data-> ", mini_batch[i][4])
                
                states[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_states[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])  

            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # Tensor 2D
            rewards = torch.FloatTensor(rewards).to(device).reshape([self.batch_size])
            dones = torch.FloatTensor(dones).to(device)
            q_values = self.model(states)  # Q-values từ model chính
            next_q_values = self.target_model(next_states).detach()  

            # Tính Q-target theo DDQN: dùng online model chọn action, target model evaluate
            # Q_target = reward + (1 - done) * gamma * Q_target(s', argmax_a'(Q_online(s', a')))
            best_next_actions = self.model(next_states).detach().argmax(dim=1).unsqueeze(1)
            max_next_q_values = next_q_values.gather(1, best_next_actions).squeeze(1)

            
            target_q_values = rewards + (1 - dones) * self.discount_factor * max_next_q_values

            # Kiểm tra nếu action có giá trị ngoài phạm vi
            assert actions.max().item() < q_values.shape[1], "actions contains invalid indices!"
            assert actions.min().item() >= 0, "actions contains negative indices!"
            current_q_values = q_values.gather(1, actions).squeeze(1)
            self.optimizer.zero_grad()
            loss = self.criterion(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()
        finally:
            self.lock.release()


    def quantile_huber_loss(self, y_true, y_pred):
        quantiles = torch.linspace(1 / (2 * self.action_size), 1 - 1 / (2 * self.action_size), self.action_size).to(device)
        batch_size = y_pred.size(0)
        tau = quantiles.repeat(batch_size, 1)
        e = y_true - y_pred
        huber_loss = torch.where(torch.abs(e) < 0.5, 0.5 * e ** 2, torch.abs(e) - 0.5)
        quantile_loss = torch.abs(tau - (e < 0).float()) * huber_loss
        return quantile_loss.mean()
