import torch
import torch.distributions as distributions
import torch.nn as nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Actor(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

    def policy(self, s_t):
        s_t = torch.as_tensor(s_t, dtype=torch.float)
        probs = self.model(s_t)
        dist = distributions.Categorical(logits=probs)
        return dist

    def act(self, s_t):
        with torch.no_grad():
            probs = self.policy(s_t)
            a_t = probs.sample()
        return a_t
    
    def compute_loss(self, states, actions, advantages, entropy_coef=0.01):
        actions = torch.tensor(actions, dtype=torch.int64)
        advantages = torch.tensor(advantages)
        actions = actions.unsqueeze(1)
        
        policy = self.policy(states[:-1])
        selected_log_prob = policy.log_prob(actions)
        
        # Calculate entropy
        entropy = policy.entropy().mean()
        
        # Compute loss with entropy term
        policy_loss = torch.mean(-selected_log_prob * advantages)
        total_loss = policy_loss - entropy_coef * entropy

        return total_loss, entropy

    def learn(self, states, actions, advantages):
            actions = torch.tensor(actions, dtype=torch.int64)
            advantages = torch.tensor(advantages)
            actions = actions.unsqueeze(1)

            selected_log_prob = self.policy(states[:-1]).log_prob(actions)
            loss = torch.mean(-selected_log_prob * advantages)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            return loss


class Critic(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),

        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
    
    def forward(self, s_t):
        return self.model(torch.FloatTensor(s_t).unsqueeze(0))

    def learn(self, V_pred, returns):
        returns = torch.tensor(returns)
        #update the view function (critic network) with: td error ** 2
        loss = torch.mean((V_pred - returns)**2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss
