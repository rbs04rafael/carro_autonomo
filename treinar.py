from stable_baselines3 import PPO
from wrappers import TradutorDeAcao, TradutorDeObservacao
from callbacks import CustomCallback
import racecar_gym.envs.gym_api 
import gymnasium as gym
import wandb

TOTAL_TIMESTEPS = int(5e5)
TEAM_ENTITY = "rbs04rafael-barbosa-ufrrj"
PROJECT = "projeto-racecar"

run = wandb.init(
    entity=TEAM_ENTITY,
    project=PROJECT,
    sync_tensorboard=True,
    config={
        "policy_type": "MultiInputPolicy",
        "total_timesteps": TOTAL_TIMESTEPS,
        "env_id": "SingleAgentAustria-v0",
    }
)

# For predefined environments:
env = gym.make(
    id='SingleAgentAustria-v0',
    render_mode='rgb_array_follow'
)

#envolvendo os ambientes
env = TradutorDeAcao(env)
env = TradutorDeObservacao(env)

#treinando o modelo 
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log =f"./logs_tensorboard/{run.id}")
callback = CustomCallback()
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=callback)
model.save("ppo_racecar_multiinput")

run.finish()

env.close()