from Agent_test import *
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
            self.ws.state = [160.0/290.0, 160.0/290.0, 320.0/640.0, 180.0/360.0, 0.0, 0.0]
            d_t = 0
            self.first = False
        if (self.ws.done == True):
            print("new_game")
            await self.ws.restart_game()
            self.ws.state = [160.0/290.0, 160.0/290.0, 320.0/640.0, 180.0/360.0, 0.0, 0.0]
            self.ws.done = False
            #s_t = [160.0, 160.0, 320.0, 180.0, 0.0, 0.0]
        last_update_time = time.time()
        for t in range(self.timesteps):
            self.player1.store_t_data(self.ws.state,self.ws.reward1, self.ws.done, t)
            self.player2.store_t_data(self.ws.state, self.ws.reward2, self.ws.done, t)
            current_time = time.time()
            #await self.ws.receive_data()
            if current_time - last_update_time >= 1.0:
                #print(f"Timestep {t}: State: {s_t}")
                #print(f"Timestep {t}: actual: {self.ws.state}")

                a_t_1 = self.player1.actor.act(self.ws.state)
                a_t_2 = self.player2.actor.act(self.ws.state)
                await self.ws.send_action(a_t_1, a_t_2)
                self.player1.store_action(a_t_1, t)
                self.player2.store_action(a_t_2, t)

                #print(f"Reward: player1: {self.ws.reward1}, player2: {self.ws.reward2}")
                s_t_next = self.ws.state
                r_t_1 = self.ws.reward1
                r_t_2 = self.ws.reward2
                #self.ws.reward1 = 0
                #self.ws.reward2 = 0
                d_t = self.ws.done

                #s_t = s_t_next

                #print(f"Timestep {t}: Action 1: {a_t_1}, Reward 1: {r_t_1}")
                #print(f"Timestep {t}: Action 2: {a_t_2}, Reward 2: {r_t_2}")

                last_update_time = current_time
            else:
                # Use the last known state if not updating
                self.player1.store_action(0, t)
                self.player2.store_action(0, t)

            await asyncio.sleep(1 / 60)  # Run the loop approxi
        p2_actor_loss = self.compute_loss(self.player2, self.ws.state)
        p1_actor_loss = self.compute_loss(self.player1, self.ws.state)
        return p1_actor_loss, p2_actor_loss