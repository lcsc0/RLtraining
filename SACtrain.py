from stable_baselines3 import SAC  
from typing import Callable
import os
from carenv import CarEnv
import time
import torch

print('This is the start of the SAC training script')

# Check GPU availability
print(torch.cuda.is_available())  # Should return True if GPU is available

# Setting folders for logs and models
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

print('Connecting to the environment...')
env = CarEnv()

env.reset(seed=None)
print('Environment has been reset as part of the launch')

# Initialize the SAC model
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,  # Learning rate for SAC
    buffer_size=1_000_000,  # Replay buffer size
    batch_size=64,          # Batch size for training updates
    ent_coef="auto",        # Automatically adjust entropy coefficient
    tensorboard_log=logdir,  # Log directory for TensorBoard
    device="cuda"           # Use GPU for training
)

# Training parameters
TIMESTEPS = 750_000  # Total timesteps per iteration
iters = 0

while iters < 4:  # Number of training iterations
    iters += 1
    print(f'Iteration {iters} is about to commence...')
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC")
    print(f'Iteration {iters} has been trained')
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
