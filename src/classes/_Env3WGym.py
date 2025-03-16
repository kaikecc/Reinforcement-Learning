import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class Env3WGym(gym.Env):
    """
    Ambiente para detecção de falhas em poços de petróleo.

    O dataset deve conter dados com cinco variáveis de entrada e uma coluna de rótulo, onde:
    Variáveis de entrada (exemplo): ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
    Rótulo [class]:
      - 0: recompensa de 0.01 se ação 0, senão -1;
      - 1 a 9: recompensa de -1 se ação 0, senão 1;
      - 101 a 109: recompensa de -0.1 se ação 0, senão 0.1.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, dataset: np.ndarray) -> None:
        super().__init__()
        self.dataset = np.asarray(dataset)
        self.index = 0
        self.num_features = self.dataset.shape[1] - 1  # Assume que a última coluna é o rótulo

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                             shape=(self.num_features,), dtype=np.float32)
        self.state = self.dataset[self.index, :-1]
        self.episode_ended = False

    def step(self, action: int):
        """
        Executa uma ação e retorna (observação, recompensa, done, info).
        """
        if self.episode_ended:
            return self.reset(), 0.0, True, {}
        
        # Calcula a recompensa com base no rótulo da amostra atual
        reward = self.calculate_reward(action, self.dataset[self.index, -1])
        self.index += 1
        
        if self.index >= len(self.dataset):
            self.episode_ended = True
        else:
            self.state = self.dataset[self.index, :-1]
        
        return np.array(self.state, dtype=np.float32), reward, self.episode_ended, {}

    def reset(self):
        """
        Reinicia o ambiente para o início do dataset.
        """
        self.index = 0
        self.state = self.dataset[self.index, :-1]
        self.episode_ended = False
        return np.array(self.state, dtype=np.float32)

    def calculate_reward(self, action: int, class_value: int) -> float:
        """
        Calcula a recompensa com base na ação tomada e no valor do rótulo.
        """
        if class_value == 0:
            return 0.01 if action == 0 else -1.0
        elif 1 <= class_value < 10:
            return -1.0 if action == 0 else 1.0
        elif 101 <= class_value < 110:
            return -0.1 if action == 0 else 0.1
        else:
            return 0.0  # Recompensa padrão para classes não previstas

    def render(self, mode='human'):
        """
        Renderiza o estado atual.
        """
        print(f"Estado atual: {self.state}")

    def close(self):
        """
        Realiza o fechamento do ambiente, se necessário.
        """
        pass

def make_custom_vec_env(dataset: np.ndarray, n_envs: int, vec_env_type: str = 'dummy'):
    """
    Cria um ambiente vetorizado a partir do dataset, dividindo-o em n_envs partes.
    
    :param dataset: Dataset completo para os ambientes.
    :param n_envs: Número de ambientes paralelos.
    :param vec_env_type: Tipo de vetor de ambiente: 'dummy' ou 'subproc'.
    :return: Um ambiente vetorizado (DummyVecEnv ou SubprocVecEnv).
    """
    # Embaralha o dataset para distribuir as amostras aleatoriamente entre os ambientes
    np.random.seed(42)  # Para reprodutibilidade
    shuffled_indices = np.random.permutation(len(dataset))
    dataset_shuffled = dataset[shuffled_indices]
    
    # Divide o dataset em n_envs partes
    dataset_splits = np.array_split(dataset_shuffled, n_envs)
    
    # Função que cria uma instância do ambiente para cada partição do dataset
    def make_env(i):
        def _init():
            return Env3WGym(dataset_splits[i])
        return _init

    env_fns = [make_env(i) for i in range(n_envs)]
    
    if vec_env_type == 'dummy':
        return DummyVecEnv(env_fns)
    elif vec_env_type == 'subproc':
        return SubprocVecEnv(env_fns)
    else:
        raise ValueError("vec_env_type inválido. Escolha 'dummy' ou 'subproc'.")