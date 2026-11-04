from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import copy
import gymnasium as gym
import racecar_gym.envs.gym_api
import wandb
import yaml



TEAM_ENTITY = "rbs04rafael-barbosa"
PROJECT = "projeto-racecar"

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        
        caminho_cenario = 'scenarios/austria.yml'

        with open(caminho_cenario, 'r') as arquivo:
            cenario = yaml.safe_load(arquivo)
            
        self.time_limit : float = cenario['agents'][0]['task']['params']['time_limit']
        
        self.termino_colisao : int = 0
        self.termino_tempo : int = 0
        self.termino_sucesso : int = 0
        
        self.volta_anterior : int = 1
        self.tempo_inicio_volta : float = 0.0

    def _on_step(self) -> bool:
        infos =  self.locals["infos"][0]
        
        #print(self.locals["infos"][0].keys())
        
        #acompanhamento do tempo médio de cada volta
        if infos.get("lap") > self.volta_anterior:
            tempo_volta  = infos.get("time") - self.tempo_inicio_volta
            
            self.logger.record_mean("custom/tempo_da_volta", tempo_volta)
            
            #essas variáveis são zeradas quando o ep acaba
            self.volta_anterior = infos.get("lap")
            self.tempo_inicio_volta = infos.get("time")
            
        #acompanhamento da contramão
        self.logger.record_mean("custom/taxa_contra_mao", infos.get("wrong_way"))
            
        
        if self.locals["dones"][0]:
            #acompanhamento do progresso por episódio e por rollout usando média
            progresso = infos.get("progress")
            
            if progresso is not None:
                self.logger.record_mean("custom/progresso_medio_ep", progresso)
                
            colisao = infos.get("wall_collision")
            tempo = infos.get("time")
            
            #acompanhamento do motivo de termino por rollout
            if colisao:
                self.termino_colisao += 1
            elif tempo > self.time_limit:
                self.termino_tempo += 1
            else:
                self.termino_sucesso += 1
            
            #continuação do acompanhamento do tempo de cada volta
            self.volta_anterior = 1
            self.tempo_inicio_volta = 0.0
            
            recompensa_total_ep = infos["episode"]["r"]
            
            self.logger.record_mean("custom/recompensa_por_segundo", (recompensa_total_ep / tempo))
        
        #acompanhamento da recompensa média por rollout usando média
        recompensa = self.locals["rewards"][0]
        
        
        if recompensa is not None:
            self.logger.record_mean("custom/recompensa_media_rollout", recompensa)
        
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("custom/termino_por_colisao", self.termino_colisao)
        self.logger.record("custom/termino_por_tempo", self.termino_tempo)
        self.logger.record("custom/termino_por_sucesso", self.termino_sucesso)
        
        total_de_eps = self.termino_tempo + self.termino_colisao + self.termino_sucesso
       
        if total_de_eps > 0: 
            self.logger.record("custom/%_de_colisao", (self.termino_colisao / total_de_eps) )
            self.logger.record("custom/%_de_tempo_limite", (self.termino_tempo / total_de_eps))
            self.logger.record("custom/%_de_sucesso", (self.termino_sucesso / total_de_eps))
        
        self.termino_colisao = 0
        self.termino_tempo = 0
        self.termino_sucesso = 0

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
    render_mode='human',
)

#envolvendo os ambientes
env = TradutorDeAcao(env)
env = TradutorDeObservacao(env)

#treinando o modelo 
model = PPO("MultiInputPolicy", env, verbose=1)
callback = CustomCallback()
model.learn(total_timesteps=int(2e5), progress_bar=True, callback=callback)
model.save("ppo_racecar_multiinput")


done = False
reset_options = dict(mode='grid')
obs, info = env.reset(options=reset_options)

while not done:
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, states = env.step(action)
    done = terminated or truncated

env.close()