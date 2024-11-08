import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

class Env3WGym(gym.Env):
    """
    Modelo do Ambiente em python para detecção de falhas em poços de petróleo.
    O dataset é fornecido como entrada, contendo dados de seis poços com cinco variáveis de entrada
    ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'] e um rótulo identificador de falha [class].
    
    Ações/Recompensas:
    - Rótulo de falha 0: Recompensa de 0.01 se ação 0, senão -1.
    - Rótulo de falha de 1 a 9: Recompensa de -1 se ação 0, senão 1.
    - Rótulo de falha de 101 a 109: Recompensa de -0.1 se ação 0, senão 0.1.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dataset, n_envs = 1):
        super(Env3WGym, self).__init__()
        self.dataset = dataset
        self.n_envs = n_envs
        self.index = 0
        num_features = dataset.shape[1] - 1  # 5 Adjust based on your features
        
        self.action_space = spaces.Discrete(2)  # Actions: 0 or 1
        self.observation_space = spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(num_features,), dtype=np.float32)
        
        self.state = self.dataset[self.index, :-1]
        self.episode_ended = False

    def step(self, action):
        if self.episode_ended:
            return self.reset()
        
        self.index += 1
        reward = self.calculate_reward(action, self.dataset[self.index - 1, -1])
        
        if self.index >= len(self.dataset):
            self.episode_ended = True
        else:
            self.state = self.dataset[self.index, :-1]
        
        done = self.episode_ended
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.index = 0
        self.state = self.dataset[self.index, :-1]
        self.episode_ended = False
        return np.array(self.state, dtype=np.float32)

    def calculate_reward(self, action, class_value):
        if class_value == 0:
            return 0.01 if action == 0 else -1
        elif class_value in range(1, 10):
            return -1 if action == 0 else 1
        elif class_value in range(101, 110):  # Corrigido para refletir o intervalo correto
            return -0.1 if action == 0 else 0.1
        else:
            return 0  # Default reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def create_env(self, dataset_part):
        return Env3WGym(dataset_part)
    
    '''def envs_random(self):
        # Divide o dataset em n partes iguais (ou quase iguais) sem embaralhar
        split_datasets = np.array_split(self.dataset, self.n_envs)
        # Cria os ambientes paralelos com cada parte do dataset
        return make_vec_env(lambda: self.create_env(split_datasets.pop(0)), n_envs=self.n_envs)'''

    def envs_random(self):
        np.random.seed(42)  # Para reprodutibilidade
        shuffled_indices = np.random.permutation(len(self.dataset))
        dataset_shuffled = self.dataset[shuffled_indices]
        split_datasets = np.array_split(dataset_shuffled, self.n_envs)
        return make_vec_env(lambda: self.create_env(split_datasets.pop(0)), n_envs=self.n_envs)
