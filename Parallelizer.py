from Game import *
import torch
import json
from concurrent.futures import ThreadPoolExecutor

class Parallelizer:
    def __init__(self, num_parallel_envs, lam, gamma, timesteps, actor_lr, critic_lr, epochs):
        self.num_parallel_envs = num_parallel_envs
        self.games = []
        self.p1_actor = Actor(actor_lr)
        self.p2_actor = Actor(actor_lr)
        self.lam = lam
        self.gamma = gamma
        self.timesteps = timesteps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.losses = []
        self.epochs = epochs
        self.create_games()

    def create_games(self):
        for _ in range(self.num_parallel_envs):
           self.games.append(Game(self.lam, self.gamma, self.timesteps, self.p1_actor, self.p2_actor, self.critic_lr))

    def aggregate_losses(self, losses_list):
        return torch.mean(torch.stack(losses_list))

    def update_network(self, network, aggregated_loss):
        network.opt.zero_grad()
        aggregated_loss.backward()
        network.opt.step()

    async def close(self):
        for game in self.games:
            await game.ws.close_connection()

    async def run(self):
        for _ in range(self.epochs):
            p1_actor_losses = []
            p2_actor_losses = []
            print(f"Epoch: {_}")
            results = await asyncio.gather(*(game.run_A2C() for game in self.games))

            print(f"Results: {results}")
            for result in results:
                print(f"Result: {result}")
                p1_actor_losses.append(result[0])
                p2_actor_losses.append(result[1])

            # Aggregate losses
            agg_p1_actor_loss = self.aggregate_losses(p1_actor_losses)
            agg_p2_actor_loss = self.aggregate_losses(p2_actor_losses)

            # Compute and apply gradients
            self.update_network(self.p1_actor, agg_p1_actor_loss)
            self.update_network(self.p2_actor, agg_p2_actor_loss)

            losses = {
                'p1_actor': agg_p1_actor_loss.item(),
                'p2_actor': agg_p2_actor_loss.item(),
            }
            self.losses.append(losses)
        self.store_model()
        return self.losses

    def store_model(self):
        hyperparameters_str = f"envs_{self.num_parallel_envs}_lam_{self.lam}_gamma_{self.gamma}_t_{self.timesteps}_a_lr_{self.actor_lr}_c_lr_{self.critic_lr}"
        
        # Replace periods with underscores to avoid issues in filenames
        hyperparameters_str = hyperparameters_str.replace('.', '_')
        
        # Save the model state dictionaries with the new filename
        p1_actor_filename = f'./models/p1_actor_{hyperparameters_str}.pth'
        p2_actor_filename = f'./models/p2_actor_{hyperparameters_str}.pth'
        
        torch.save(self.p1_actor.state_dict(), p1_actor_filename)
        torch.save(self.p2_actor.state_dict(), p2_actor_filename)