import torch as th
from torch import nn
import sys
from configs.systemcfg import DEVICE
# from configs.systemcfg import DEVICE

device = th.device('cuda:'+str(DEVICE) if th.cuda.is_available() else 'cpu')
if device == "cpu":
    print("cannot train with cpu")
    exit(0)
else:
    print("cuda: ", device)


LARGE_VALUE_THRESHOLD = sys.maxsize
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.output_act = output_act
        self.device = device
        self.to(self.device)
    
    def forward(self, state):
        state = state.to(self.device)
        
        if th.isnan(state).any():
            raise ValueError("input has value nan")
        elif (state.abs() > LARGE_VALUE_THRESHOLD).any():
            raise ValueError(f"Input has values larger than {LARGE_VALUE_THRESHOLD}")
        
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        
        if isinstance(self.output_act, nn.Softmax):
            out = self.output_act(self.fc3(out), dim=1)
        elif callable(self.output_act):
            out = self.output_act(self.fc3(out))
        else:
            raise ValueError("output_act must be callable or an instance of nn.Softmax.")
                
        if th.isnan(out).any():
            raise ValueError("nan value error {}.".format(out))
            
        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, hidden_size, output_size=30):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        state = state.to(self.device)
        out = nn.functional.relu(self.fc1(state))
        if out.dim() == 1:
            out = out.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if out.dim() == 2 and action.dim() == 3:
            out = out.unsqueeze(1).expand_as(action)
        out = th.cat([out, action], dim=-1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class ActorCriticNetwork(nn.Module):
    """
    An actor-critic network that shared lower-layer representations but
    have distinct output layers
    """
    def __init__(self, state_dim, action_dim, hidden_size,
                 actor_output_act, critic_output_size=30):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        self.actor_output_act = actor_output_act
        self.device = device
        self.to(self.device)

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        act = self.actor_output_act(self.actor_linear(out))
        val = self.critic_linear(out)
        return act, val