import random
import matplotlib.pyplot as plt
import seaborn as sns
from Parallelizer import *

class Supervisor:
    def __init__(self, epochs, num_parallel_envs, number_of_cycles):
        self.epochs = epochs
        self.parallelizers = []
        self.number_of_cycles = number_of_cycles
        self.num_parallel_envs = num_parallel_envs
        self.hyperparameters = []  # Attribute to store the hyperparameters
        self.initialize_parallelizer()

    def initialize_parallelizer(self):
        for _ in range(self.number_of_cycles):
            # Define reasonable ranges for hyperparameters
            lam = random.uniform(0.9, 1.0)  # Lambda (GAE parameter) typically close to 1
            gamma = random.uniform(0.95, 0.99)  # Discount factor
            timesteps = random.randint(200, 300)  # Number of timesteps per update
            actor_lr = random.uniform(1e-5, 1e-3)  # Learning rate for actor
            critic_lr = random.uniform(1e-4, 1e-2)  # Learning rate for critic
            
            # Store the hyperparameters in a dictionary
            hyperparams = {
                'lam': lam,
                'gamma': gamma,
                'timesteps': timesteps,
                'actor_lr': actor_lr,
                'critic_lr': critic_lr
            }
            
            # Append the hyperparameters dictionary to the list
            self.hyperparameters.append(hyperparams)
            
            # Create a new Parallelizer object with randomly chosen hyperparameters
            parallelizer = Parallelizer(self.num_parallel_envs, lam, gamma, timesteps, actor_lr, critic_lr, self.epochs)
            
            # Add the parallelizer to the list
            self.parallelizers.append(parallelizer)
        
    async def run_parallelizer(self):
            results = await asyncio.gather(*(parallelizer.run() for parallelizer  in self.parallelizers))
           # await asyncio.gather(*(parallelizer.close() for parallelizer  in self.parallelizers))
            #print(f"Size loss: {len(parallel2[0])} rewards {len(parallel2[1])}, hyper: {len(parallel2[2])}")
            for i, result in enumerate(results):
                print(f"i: {i}")
                self.plot_data(result[0], "loss", i)
                self.plot_data(result[1], "rewards", i)
                self.plot_data(result[2], "entropy",  i)
            #results[0][0] result [1][0]loss -> n epochs
        
    
    def plot_data(self, dt, matter, i):
        hyperparams = self.hyperparameters[i]
        hyperparameters_str = (
            f"parallel_envs_{self.num_parallel_envs}_"
            f"lam_{hyperparams['lam']:.2f}_"
            f"gamma_{hyperparams['gamma']:.2f}_"
            f"timesteps_{hyperparams['timesteps']}_"
            f"actor_lr_{hyperparams['actor_lr']:.5f}_"
            f"critic_lr_{hyperparams['critic_lr']:.5f}"
        )
        # Replace periods with underscores to avoid issues in filenames
        hyperparameters_str = hyperparameters_str.replace('.', '_')
        filename = f"./plots/{matter}_{hyperparameters_str}.png"
        
        # Extract losses from result
        #print(f"Size before{len(dt)}")
        p1_data = [data['p1_actor'] for data in dt]
        p2_data = [data['p2_actor'] for data in dt]
        #print(f"Size after{len(p1_data)}")
        # Plot the learning curve for p1_actor
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(self.epochs), y=p1_data, label='p1_actor')
        sns.lineplot(x=range(self.epochs), y=p2_data, label='p2_actor')
        plt.xlabel('Epochs')
        plt.ylabel(matter)
        plt.title(f'{matter} ({hyperparameters_str})')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.close()