import torch
import numpy as np
from torch import nn
import seaborn as sns; sns.set()
import json
import websocket
import threading
import time

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
        model = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),

        )
        self.opt = optim.Adam(self.parameters(), lr=lr)
    def act(self, s_t):
        a_t = self.pi(s_t).sample()
        return a_t
    def learn(self, states, actions, advantages):
        actions = torch.tensor(actions)
        advantages = torch.tensor(advantages)

        log_prob = self.pi(states).log_prob(actions)
        loss = torch.mean(-log_prob*advantages)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss

class Critic(nn.Module):
    def __init__(self, lr):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),

        )
        opt = optim.Adam(self.parameters(), lr=lr)
    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output
    
    def learn(self, V_pred, returns):
        returns = torch.tensor(returns)
        loss = torch.mean((V_pred - returns)**2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss

class Agent:
    def __init__(self, actor_lr, critic_lr, timesteps):
        self.actor = Actor(actor_lr)
        self.critic = Critic(critic_lr)
        self.states = np.empty((timesteps, 6))
        self.actions = np.empty(timesteps)
        self.rewards = np.empty(timesteps)
        self.done = np.empty(timesteps)

    def store_t_data(self, s_t, a_t, r_t, d_t):
        self.states[t] = s_t
        self.actions[t] = a_t
        self.rewards[t] = r_t
        self.dones[t] = d_t

    def calculate_returns(self, last_state, gamma):
        self.states[T] = last_state
        last_value = critic.predict(states[-1]).detach().numpy()
        self.rewards[-1] += gamma*(1-dones[-1])*last_value
        result = np.empty_like(rewards)
        result[-1] = rewards[-1]
        for t in range(len(rewards)-2, -1, -1):
            result[t] = rewards[t] + gamma*(1-dones[t])*result[t+1] #GT->
        return result

    def update_critic(self, returns):
        for i in range(80):
            V_pred = self.critic.forward(self.states)
            self.critic.learn(V_pred[:-1], returns)
        return V_pred

    def calculate_advantages(self, V_pred, lam, gamma):
        V_pred = V_pred.detach().numpy()
        TD_errors = rewards + gamma*(1-dones)*V_pred[1:] - V_pred[:-1]
        result = np.empty_like(TD_errors)
        result[-1] = TD_errors[-1]
        for t in range(len(TD_errors)-2, -1, -1):
            result[t] = TD_errors[t] + gamma*lam*result[t+1]
        return result

    def update_actor(self, advantages):
        advantages = (advantages - advantages.mean())/advantages.std()
        pi_loss = actor.learn(states[:-1], actions, advantages)
        return pi_loss

class Game:
    def __init__(self, actor_lr, critic_lr, lam, gamma, timesteps, epochs):
        self.player1 = Agent(actor_lr, critic_lr, timesteps)
        self.player2 = Agent(actor_lr, critic_lr, timesteps)
        self.lam = lam
        self.gamma = gamma
        self.timesteps = timesteps
        self.epochs = epochs
        self.ws = None
        self.state = None
        self.done = False
        self.reward1 = 0
        self.reward2 = 0
        self.score1 = 0
        self.score2 = 0
    
    def on_message(self, message):
        data = json.loads(message)
        self.state = self.deserialize_state(data)

    def on_error(self, error):
        print(f"WebSocket error: {error}")

    def on_close(self):
        print("WebSocket connection closed")

    def on_open(self):
        def run(*args):
            self.ws.send(json.dumps({'t': 'select_game_type', 'game_type': 'pong'}))
            self.ws.send(json.dumps({'t': 'sg'}))
            self.done = False

        threading.Thread(target=run).start()

    def deserialize_state(self, data):
        ball_x = data['ball']['x']
        ball_y = data['ball']['y']
        ball_vx = data['ball']['vx']
        ball_vy = data['ball']['vy']
        paddle1_y = data['p1']['y']
        paddle2_y = data['p2']['y']
        score1 = data['s1']
        score2 = data['s2']
        game_over = data['go']
        if score2 > self.score2:
            self.score2 = score2
            self.reward1 = -1
            self.reward2 = 1
        else:
            reward2 = 0
        if score1 > self.score1:
            self.score1 = score1
            self.reward1 = 1
            self.reward2 = -1
        else:
            self.reward2 = 0
        self.done = game_over
        return (paddle1_y, paddle2_y, ball_x, ball_y, ball_vx, ball_vy)

    def send_action(self, action):
        if action == 1:
            self.ws.send(json.dumps({'t': 'pi', 'p1': "7"}))
        else:
            self.ws.send(json.dumps({'t': 'pi', 'p1': "-7"}))

    
    def run_A2C(self):
        self.ws = websocket.WebSocketApp('ws://localhost:8000/ws/pong/',
            on_message=lambda ws, msg: self.on_message(msg),
            on_error=lambda ws, err: self.on_error(err),
            on_close=lambda ws: self.on_close())
        self.ws.on_open = lambda ws: self.on_open()
        wst = threading.Thread(target=self.ws.run_forever)
        wst.start()
        for epoch in range(epochs):
            s_t_1 = [160, 160, 320, 180, 0, 0]
            s_t_2 = s_t_1
            self.ws.send(json.dumps({'t': 'restart_game'}))
            for t in range(T):
                a_t_1 = self.player1.act(s_t_1)
                self.send_action(a_t_1)
                time.sleep(1)
                s_t_next_1 = self.state
                r_t_1 = self.reward1
                d_t_1 = self.done
                player1.store_t_data(s_t_1, a_t_1, r_t_1, d_t_1)
                s_t_1 = s_t_next_1

                a_t_2 = self.player2.act(s_t_2)
                self.send_action(a_t_1)
                time.sleep(1)
                s_t_next_2 = self.state
                r_t_2 = self.reward2
                d_t_2 = self.done
                player2.store_t_data(s_t_2, a_t_2, r_t_2, d_t_2)
                s_t_2 = s_t_next_2

            returns_2 = self.player2.calculate_returns(s_t_2, self.gamma)
            v_pred_2 = self.player2.update_critic(returns_2)
            advantages_2 = self.player2.calculate_advantages(v_pred_2, self.lam, self.gamma)
            pi_loss_2 = self.player2.update_actor(advantages_2)

            returns_1 = self.player1.calculate_returns(s_t_1, self.gamma)
            v_pred_1 = self.player1.update_critic(returns_1)
            advantages_1 = self.player1.calculate_advantages(v_pred_1, self.lam, self.gamma)
            pi_loss_1 = self.player1.update_actor(advantages_1)

##done flag??
game  = Game(0.001, 0.001, 0.95, 0.99, 4052, 256)
game.run_A2C()               