from stable_baselines3 import SAC
import os
from carenv import CarEnv
import time
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available

print('Starting SAC training script')

# Setting directories for logs and model saving
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

print('Connecting to the environment...')
env = CarEnv(is_render_enabled=True)
env.render()

env.reset(seed=None)
print('Environment has been reset as part of launch')

# Initialize the SAC model
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,  # Adjusted learning rate for SAC
    buffer_size=1000000,   # Replay buffer size for experience replay
    batch_size=64,         # Batch size for updates
    ent_coef="auto",       # Automatically adjust entropy coefficient
    tensorboard_log=logdir,
    device="cuda"
)

# Training parameters
TIMESTEPS = 250_000  # SAC typically uses shorter timesteps per update
iters = 0
while iters < 4:  # Number of training iterations
    iters += 1
    print(f'Iteration {iters} is about to commence...')
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC")
    print(f'Iteration {iters} has been trained')
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
