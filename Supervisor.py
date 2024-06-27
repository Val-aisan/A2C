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
            # timesteps = random.randint(300, 500)
            timesteps = random.randint(1000, 5000)  # Number of timesteps per update
            actor_lr = random.uniform(1e-5, 1e-3)  # Learning rate for actor
            critic_lr = random.uniform(1e-5, 1e-3)  # Learning rate for critic
            
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
            self.plot(results)

    """def plot(self, results):
        for i, result in enumerate(results):
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
            filename = f"learning_curve_{hyperparameters_str}.png"
            
            # Plot the learning curve
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=range(self.epochs), y=result)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Learning Curve ({hyperparameters_str})')
            plt.grid(True)
            plt.savefig(filename)
            plt.close()"""
    
    def plot(self, results):
        for i, result in enumerate(results):
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
            filename = f"learning_curve_{hyperparameters_str}.png"
            
            # Extract losses from result
            p1_losses = [loss['p1_actor'] for loss in result]
            p2_losses = [loss['p2_actor'] for loss in result]
            
            # Plot the learning curve for p1_actor
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=range(self.epochs), y=p1_losses, label='p1_actor')
            sns.lineplot(x=range(self.epochs), y=p2_losses, label='p2_actor')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Learning Curve ({hyperparameters_str})')
            plt.grid(True)
            plt.legend()
            plt.savefig(filename)
            plt.close()

