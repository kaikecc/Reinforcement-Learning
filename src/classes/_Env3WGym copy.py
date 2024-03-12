import gym
from gym import spaces
import numpy as np

class Env3WGym(gym.Env):

    '''
    Essa classe é um Modelo do Ambiente em python para o problema de detecção de falhas em poços de petróleo.
    O dataset é fornecido como entrada para o ambiente com mais de 10 milhões de registros.

     1. O ambiente é um repositório di github https://github.com/petrobras/3W
     2. O ambiente é composto por dados de seis poços de petróleo, com cinco variáveis (observações) de entrada ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'] e um rótulo indentificador de falha [class]
     3. O ambiente é um ambiente de simulação, onde o agente pode escolher entre duas ações: 0 - Não Detectado ou 1 - Detectado para cada observação
     4. A recompensa é calculada com base na ação escolhida e no rótulo de falha [class]:

        Estados:
        - rótulo de falha: 0 - Estado Normal
        - rótulo de falha: 1 a 9 - Estável de Anomalia (Falha)
        - rótulo de falha: 101 a 109 - Transiente de Anomalia (Falha)

        Ações/Recompensas:
        - Se o rótulo de falha for 0, a recompensa é 0.01 se a ação for 0, caso contrário, a recompensa é -1
        - Se o rótulo de falha estiver entre 1 e 9, a recompensa é -1 se a ação for 0, caso contrário, a recompensa é 1
        - Se o rótulo de falha estiver entre 101 e 1098, a recompensa é -0.1 se a ação for 0, caso contrário, a recompensa é 0.1
    
    '''

    metadata = {'render.modes': ['human']}
    
    def __init__(self, dataset):
        super(Env3WGym, self).__init__()
        self.dataset = dataset
        self.index = 0
        num_features = 5  # Adjust based on your features
        
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(2)  # Actions: 0 or 1
        self.observation_space = spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(num_features,), dtype=np.float32)
        
        # Initial state
        self.state = self.dataset[self.index, :-1]
        self.episode_ended = False

    def step(self, action):
        if self.episode_ended:
            return self.reset()

        # Increment the index to move to the next step in the dataset
        self.index += 1
        if self.index >= len(self.dataset) - 1:
            self.episode_ended = True
        
        self.state = self.dataset[self.index, :-1]
        reward = self.calculate_reward(action, self.dataset[self.index - 1, -1])
        done = self.episode_ended
        
        # For Gym environments, additional info can be returned but here we'll just return an empty dict
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
        elif class_value in range(101, 110):
            return -0.1 if action == 0 else 0.1
        else:
            return 0  # Default reward

    def render(self, mode='human'):
        pass  # Implement rendering if necessary for your application

    def close(self):
        pass  # Implement any cleanup necessary
