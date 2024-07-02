import numpy as np
from Networks import *

class Agent:
    def __init__(self, actor, timesteps, critic_lr):
        self.actor = actor
        self.critic = Critic(critic_lr)
        self.states = np.empty((timesteps + 1, 6))
        self.actions = np.empty(timesteps)
        self.rewards = np.empty(timesteps)
        self.done = np.empty(timesteps)
        self.pi_losses = []
        self.v_losses = []
        self.timesteps = timesteps

    def store_t_data(self, s_t, r_t, d_t, a_t, t):
        self.actions[t] = a_t
        self.states[t] = s_t
        self.rewards[t] = r_t
        self.done[t] = d_t

    def calculate_returns(self, gamma):
        last_value = self.critic.forward(self.states[-1]).detach().numpy()
        self.rewards[-1] += gamma*(1-self.done[-1]) * last_value
        result = np.empty_like(self.rewards)
        result[-1] = self.rewards[-1]
        #compute recursively first part of td error->update value function
        for t in range(len(self.rewards)-2, -1, -1):
            result[t] = self.rewards[t] + gamma*(1-self.done[t])*result[t+1] #GT->
        return result

    def update_critic(self, returns):
        for i in range(80):
            V_pred = self.critic.forward(self.states)
            v_loss = self.critic.learn(V_pred[:-1], returns)
            self.v_losses.append(v_loss)
        return V_pred

    def calculate_advantages(self, V_pred, lam, gamma):
        V_pred = V_pred.detach().numpy()
        V_pred = np.squeeze(V_pred)
        TD_errors = self.rewards + gamma*(1-self.done) * V_pred[1:] - V_pred[:-1]
        result = np.empty_like(TD_errors)
        result[-1] = TD_errors[-1]
        for t in range(len(TD_errors)-2, -1, -1):
            result[t] = TD_errors[t] + gamma*lam*result[t+1]
        return result

    def compute_loss(self, advantages):
        # magnitudes of the gradients stay within a certain range so that policy steps are always similar size
        advantages = (advantages - advantages.mean())/advantages.std()
        pi_loss, entropy = self.actor.compute_loss(torch.FloatTensor(self.states), self.actions, advantages)
        return pi_loss, entropy

