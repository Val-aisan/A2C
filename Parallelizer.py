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
        self.rewards = []
        self.entropy = []
        self.epochs = epochs
        self.create_games()

    def create_games(self):
        for _ in range(self.num_parallel_envs):
           self.games.append(Game(self.lam, self.gamma, self.timesteps, self.p1_actor, self.p2_actor, self.critic_lr))

    def aggregate_data(self, data):
        return torch.mean(torch.stack(data))

    def update_network(self, network, aggregated_loss):
        network.opt.zero_grad()
        aggregated_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0) 
        network.opt.step()

    async def close(self):
        for game in self.games:
            await game.ws.close_connection()

    async def run(self):
        for _ in range(self.epochs):
            p1_actor_losses = []
            p2_actor_losses = []
            p1_actor_rewards = []
            p2_actor_rewards = []
            p1_entropy = []
            p2_entropy = []
            print(f"Epoch: {_}")
            results = await asyncio.gather(*(game.run_A2C() for game in self.games))

            #result size should be num_envs_par * 6
            for result in results:
                print(f"Result: {result}")
                p1_actor_losses.append(result[0])
                p2_actor_losses.append(result[1])
                p1_actor_rewards.append(result[2])
                p2_actor_rewards.append(result[3])
                p1_entropy.append(result[4])
                p2_entropy.append(result[5])


            # Aggregate losses
            agg_p1_actor_loss = self.aggregate_data(p1_actor_losses)
            agg_p2_actor_loss = self.aggregate_data(p2_actor_losses)
            agg_p1_actor_reward = self.aggregate_data(p1_actor_rewards)
            agg_p2_actor_reward = self.aggregate_data(p2_actor_rewards)
            agg_p1_actor_ent = self.aggregate_data(p1_entropy)
            agg_p2_actor_ent = self.aggregate_data(p2_entropy)

            # Compute and apply gradients
            self.update_network(self.p1_actor, agg_p1_actor_loss)
            self.update_network(self.p2_actor, agg_p2_actor_loss)

            losses = {
                'p1_actor': agg_p1_actor_loss.item(),
                'p2_actor': agg_p2_actor_loss.item(),
            }
            self.losses.append(losses)

            rewards = {
                'p1_actor': agg_p1_actor_reward.item(),
                'p2_actor': agg_p2_actor_reward.item(),
            }
            self.rewards.append(rewards)

            entropy = {
                'p1_actor': agg_p1_actor_ent.item(),
                'p2_actor': agg_p2_actor_ent.item(),
            }
            self.entropy.append(entropy)
        self.store_model()
        #print(f"SIZZ: loss{len(self.losses)} rewards: {len(self.rewards)}, entrop {len(self.entropy)}")
        return self.losses, self.rewards, self.entropy

    def store_model(self):
        hyperparameters_str = f"envs_{self.num_parallel_envs}_lam_{self.lam}_gamma_{self.gamma}_t_{self.timesteps}_a_lr_{self.actor_lr}_c_lr_{self.critic_lr}"
        
        # Replace periods with underscores to avoid issues in filenames
        hyperparameters_str = hyperparameters_str.replace('.', '_')
        
        # Save the model state dictionaries with the new filename
        p1_actor_filename = f'./models/p1_actor_{hyperparameters_str}.pth'
        p2_actor_filename = f'./models/p2_actor_{hyperparameters_str}.pth'
        
        torch.save(self.p1_actor.state_dict(), p1_actor_filename)
        torch.save(self.p2_actor.state_dict(), p2_actor_filename)