from Agent import *
from Websocket import *
import time
from colorama import Fore, Back, Style

class Game:
    def __init__(self, lam, gamma, timesteps, actor_p1, actor_p2, critic_lr):
        self.player1 = Agent(actor_p1, timesteps, critic_lr)
        self.player2 = Agent(actor_p2, timesteps, critic_lr)
        self.lam = lam
        self.gamma = gamma
        self.timesteps = timesteps
        self.first = True
        self.ws = WebSocket('ws://localhost:8000/ws/pong/')

    def compute_loss(self, player, s_t):
        player.states[self.timesteps] = s_t
        returns_2 = player.calculate_returns(self.gamma)
        v_pred_2 = player.update_critic(returns_2)
        advantages_2 = player.calculate_advantages(v_pred_2, self.lam, self.gamma)
        pi_loss = player.compute_loss(advantages_2)
        return pi_loss


    async def run_A2C(self):
        if self.first:
            await self.ws.create_connection()
            await self.ws.start_new_game()
            print(f"state: {self.ws.state}")
            d_t = 0
            self.first = False
        if (self.ws.done == True):
            print("new_game")
            await self.ws.restart_game()
            self.ws.done = False
        a_t_1 = 0 
        a_t_2 = 0
        last_update_time = time.time()
        for t in range(self.timesteps):
            self.player1.store_t_data(self.ws.state,self.ws.reward1, self.ws.done, t)
            self.player2.store_t_data(self.ws.state, self.ws.reward2, self.ws.done, t)
            current_time = time.time()
            if current_time - last_update_time >= 1.0:
                a_t_1 = self.player1.actor.act(self.ws.state)
                a_t_2 = self.player2.actor.act(self.ws.state)
                await self.ws.send_action(a_t_1, a_t_2)
                self.player1.store_action(a_t_1, t)
                self.player2.store_action(a_t_2, t)
                last_update_time = current_time
            else:
                # Use the last known state if not updating
                self.player1.store_action(a_t_1, t)
                self.player2.store_action( a_t_2, t)

            await asyncio.sleep(1 / 60)  # Run the loop approxi
        p2_actor_loss, p2_entropy = self.compute_loss(self.player2, self.ws.state)
        p1_actor_loss, p1_entropy = self.compute_loss(self.player1, self.ws.state)
        p1_actor_reward = torch.tensor(self.player1.rewards.sum() / self.timesteps)
        p2_actor_reward = torch.tensor(self.player2.rewards.sum() / self.timesteps)
        return p1_actor_loss, p2_actor_loss, p1_actor_reward, p2_actor_reward, p1_entropy, p2_entropy