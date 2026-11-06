from stable_baselines3 import PPO
from wrappers import TradutorDeObservacao, TradutorDeAcao
import gymnasium as gym
import racecar_gym.envs.gym_api

env = gym.make("SingleAgentAustria-v0", render_mode= "human")

env = TradutorDeAcao(env)
env = TradutorDeObservacao(env)

model = PPO.load("ppo_racecar_multiinput", env=env)

obs, info = env.reset(options=dict(mode='grid'))
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, trucated, info = env.step(action)
    #done = terminated or trucated
    
env.close()