import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

class Env3WGym(gym.Env):
    """
    Ambiente para detecção de falhas em poços de petróleo.
    
    O dataset deve conter os dados de 6 poços com 5 variáveis de entrada e 1 coluna de rótulo.
    Variáveis de entrada (exemplo): ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
    Rótulo [class]:
      - 0: recompensa 0.01 se ação 0, senão -1.
      - 1 a 9: recompensa -1 se ação 0, senão 1.
      - 101 a 109: recompensa -0.1 se ação 0, senão 0.1.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dataset: np.ndarray, n_envs: int = 1):
        super().__init__()
        # Converte para array NumPy, caso não seja
        self.dataset = np.asarray(dataset)[:, :-1]  # Remove a coluna de rótulo
        self.n_envs = n_envs
        self.index = 0
        
        num_features = self.dataset.shape[1] - 1  # Última coluna é o rótulo
        self.action_space = spaces.Discrete(2)  # Ações: 0 ou 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        
        self.state = self.dataset[self.index, :-1]
        self.done = False

    def step(self, action: int):
        """
        Executa uma ação e retorna a transição:
        observation, reward, done, info.
        """
        if self.done:
            # Se o episódio terminou, reinicia o ambiente
            return self.reset(), 0.0, True, {}
        
        # Calcula a recompensa com base no rótulo atual
        reward = self.calculate_reward(action, self.dataset[self.index, -1])
        self.index += 1
        
        # Verifica se chegou ao fim do dataset
        if self.index >= len(self.dataset):
            self.done = True
            # Aqui pode-se definir um estado terminal apropriado; neste exemplo, mantemos o último estado
            observation = self.state.copy()
        else:
            self.state = self.dataset[self.index, :-1]
            observation = self.state
        
        return np.array(observation, dtype=np.float32), reward, self.done, {}

    def reset(self):
        """
        Reinicia o ambiente para o início do dataset.
        """
        self.index = 0
        self.state = self.dataset[self.index, :-1]
        self.done = False
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
        Neste exemplo, apenas imprime o estado.
        """
        print(f"Estado atual: {self.state}")

    def close(self):
        """
        Método para limpeza de recursos, se necessário.
        """
        pass

    def create_env(self, dataset_part: np.ndarray):
        """
        Cria uma instância do ambiente para uma parte do dataset.
        """
        return Env3WGym(dataset_part, n_envs=self.n_envs)
    
    def envs_random(self):
        """
        Cria ambientes paralelos a partir do dataset embaralhado.
        
        O dataset é embaralhado para aumentar a variedade dos episódios e
        dividido em n_envs partes iguais (ou quase iguais).
        
        Retorna um ambiente vetorizado usando DummyVecEnv.
        """
        np.random.seed(42)  # Para reprodutibilidade
        shuffled_indices = np.random.permutation(len(self.dataset))
        dataset_shuffled = self.dataset[shuffled_indices]
        split_datasets = np.array_split(dataset_shuffled, self.n_envs)
        
        # Cria uma lista de funções que retornam ambientes com diferentes partes do dataset
        env_fns = [lambda ds=ds: Env3WGym(ds, n_envs=1) for ds in split_datasets]
        return DummyVecEnv(env_fns)
