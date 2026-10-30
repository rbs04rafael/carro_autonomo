from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
import copy
import gymnasium as gym
import racecar_gym.envs.gym_api

#construção do ambiente modificado - wrapper

#necessário devido a forma como a rede do SB3 passa os parâmetros de ação e forma como o ambiente recebe esses dados
class TradutorDeAcao(gym.ActionWrapper): 
    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    
    def action(self, act):
        acao_traduzida = {
            'speed': np.array([act[0]],dtype=np.float32),
            'steering': np.array([act[1]], dtype=np.float32)
        } 

        return acao_traduzida

#necessário devido ao pyTorch lidar apenas com vetores e o "time", no ambiente, é apenas um escalar
class TradutorDeObservacao(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        novo_espaco = copy.deepcopy(env.observation_space.spaces)
        novo_espaco["time"] = spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict(novo_espaco)
    
    def observation(self, obs):
        time_vetor = np.array([obs["time"]], dtype=np.float32)
        obs["time"] = time_vetor
        
        return obs

# For predefined environments:
env = gym.make(
    id='SingleAgentAustria-v0',
    render_mode='human'
)

#envolvendo os ambientes
env = TradutorDeAcao(env)
env = TradutorDeObservacao(env)

#treinando o modelo 
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=int(2e5), progress_bar=True)
model.save("ppo_racecar_multiinput")


done = False
reset_options = dict(mode='grid')
obs, info = env.reset(options=reset_options)

while not done:
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, states = env.step(action)
    done = terminated or truncated

env.close()