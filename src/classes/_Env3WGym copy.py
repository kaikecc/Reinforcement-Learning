import gym
from gym import spaces
import numpy as np

class Env3WGym(gym.Env):
    """
   
    Observadores: ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'] 

    Pressão: P-PDG, P-TPT, P-MON-CKP
    Temperatura: T-TPT, T-JUS-CKP

        * A regra geral, entretanto, é que as pressões medidas em pontos mais profundos aumentem 
        e que as pressões nos pontos mais próximos à superfície diminuam. As temperaturas, no geral, costumam
        aumentar em todos os equipamentos.
    Aumento Abrupto de BSW:
        - 0 (Não houve aumento abrupto de BSW)
        - 1 (Houve aumento abrupto de BSW)
        - -1 (Houve diminuição abrupta de BSW)

    Ações/Recompensas:
    - Se Ação 0 e Z-score 0: Recompensa 0.01
    - Se Ação 0 e Z-score 1: Recompensa -1
    - Se Ação 1 e Z-score 0: Recompensa -1
    - Se Ação 1 e Z-score 1: Recompensa 0

    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, array_list):
        super(Env3WGym, self).__init__()
        self.array_list = array_list
        self.array_index = 0 # Indice do array_z (dataset)
        self.index = 0 # Indice do array list
        self.num_datasets = len(array_list) # Tamanho de arrays dentro de array_list
        self.dataset = array_list[0] # Primeiro array de dados
        self.inc_abrupt_bsw = [0, -1, 1, -1, 1]  # Aumento Abrupto de BSW
        
        num_features = self.dataset.shape[1] - 1  # Numero de colunas        
        self.action_space = spaces.Discrete(2)  # Actions: 0 or 1
        self.observation_space = spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(num_features,), dtype=np.float32)       
        
        self.episode_ended = False

    def moving_average_std(self, data, window_size):
        """Calcula a média móvel e o desvio padrão móvel."""
        ma = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        std = np.sqrt(np.convolve(np.square(data - np.pad(ma, (window_size - 1, 0), 'constant', constant_values=np.nan)), np.ones(window_size) / window_size, mode='valid'))
        return ma, std

    def detect_z_score_trends(self, array_data):
        window_size = 1 * 3600
        array_z = np.zeros_like(array_data)
        
        for col in range(array_data.shape[1]):
            data_col = array_data[:, col]
            ma, std = self.moving_average_std(data_col, window_size)
            ma_padded = np.pad(ma, (window_size - 1, 0), 'constant', constant_values=np.nan)
            std_padded = np.pad(std, (window_size - 1, 0), 'constant', constant_values=np.nan)
            z_scores = (data_col - ma_padded) / std_padded
            z_scores_diff = np.diff(z_scores, prepend=np.nan)
            
            z_scores_increasing = np.zeros_like(z_scores)
            if col in [0, 1, 3]:  # Colunas para verificar diminuição
                z_scores_increasing[z_scores_diff < 0] = 1
            else:  # Colunas para verificar aumento
                z_scores_increasing[z_scores_diff > 0] = 1
            
            array_z[:, col] = z_scores_increasing
        
        return array_z

    def update_dataset(self):
        self.dataset = self.array_list[self.array_index]

    def step(self, action):
        if self.episode_ended:
            return self.reset()
        
        reward = self.calculate_reward(action, self.dataset[self.dataset_index, -1])

        self.dataset_index += 1
        if self.dataset_index >= len(self.dataset.shape[0]):
            self.array_index += 1
            if self.array_index >= self.num_datasets:
                self.episode_ended = True
            else:
                self.update_dataset()
                self.array_index = 0
        
        done = self.episode_ended
        
        if not done:
            self.state = self.dataset[self.dataset_index, :-1]
        else:
            self.state = np.zeros(self.observation_space.shape[0])

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.array_index = 0
        self.dataset_index = 0
        self.dataset = self.detect_z_score_trends(self.array_list[0])
        self.state = self.dataset[0, :-1]
        self.episode_ended = False
        return np.array(self.state, dtype=np.float32)

    def calculate_reward(self, action, array_data):
        
        z_score = self.detect_z_score_trends(array_data) # É um array de cinco posições, cada uma representando uma variável

        if action == 0 and z_score == 0:
            return 0.01
        elif action == 0 and z_score == 1:
            return -1
        elif action == 1 and z_score == 0:
            return -1
        else:
            return 0

    def render(self, mode='human'):
        pass

    def close(self):
        pass