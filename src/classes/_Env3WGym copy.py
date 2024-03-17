import gym
from gym import spaces
import numpy as np

class Env3WGym(gym.Env):
    """
   
    - Observadores: ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'] 

    - Pressão: P-PDG, P-TPT, P-MON-CKP
    - Temperatura: T-TPT, T-JUS-CKP

        * A regra geral, entretanto, é que as pressões medidas em pontos mais profundos aumentem 
        e que as pressões nos pontos mais próximos à superfície diminuam. As temperaturas, no geral, costumam
        aumentar em todos os equipamentos.
    
    - Para detecção de Aumento Abrupto de BSW (detect_z_score_trends):
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
        self.inc_abrupt_bsw = np.array([0, -1, 1, -1, 1])  # Definição do padrão para aumento abrupto de BSW
        
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
        window_size = 1 * 3600  # Ajuste para o número de linhas que representa uma hora
        array_z = np.zeros_like(array_data)

        for col in range(array_data.shape[1]):
            data_col = array_data[:, col]
            # Calcular média móvel e desvio padrão móvel
            ma, std = self.moving_average_std(data_col, window_size)
            # Calcular Z-scores
            z_scores = (data_col - np.pad(ma, (window_size - 1, 0), 'constant', constant_values=np.nan)) / \
                    np.pad(std, (window_size - 1, 0), 'constant', constant_values=np.nan)
            # Calcular a diferença dos Z-scores para identificar tendências
            z_scores_diff = np.diff(z_scores, prepend=np.nan)

            # Inicializar array de tendências como zeros
            z_scores_trends = np.zeros_like(z_scores)
            
            # Atribuir -1 para diminuição, 1 para aumento
            z_scores_trends[z_scores_diff < 0] = -1 if col in [0, 1, 3] else 0  # Colunas especificadas para verificar diminuição, se não, mantém 0
            z_scores_trends[z_scores_diff > 0] = 1 if col not in [0, 1, 3] else 0  # Colunas restantes verificam aumento, se não, mantém 0
            
            # Armazenar os resultados
            array_z[:, col] = z_scores_trends
        
        return array_z

    def update_dataset(self):
        self.dataset_index = 0
        self.dataset = self.array_list[self.array_index]

    def step(self, action):
        if self.episode_ended:
            return self.reset()
        
        reward = self.calculate_reward(action, self.dataset[self.dataset_index, -1])

        self.dataset_index += 1
        if self.dataset_index >= self.dataset.shape[0]:
            self.array_index += 1
            if self.array_index >= self.num_datasets:
                self.episode_ended = True
            else:
                self.update_dataset()
                
        
        done = self.episode_ended

        if not done:
            self.state = self.dataset[self.dataset_index, :-1]
        else:
            self.state = np.zeros(self.observation_space.shape[0])

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.array_index = 0
        self.dataset_index = 0
        self.dataset = self.array_list[self.array_index]
        self.state = self.dataset[0, :-1]
        self.episode_ended = False
        return np.array(self.state, dtype=np.float32)

    def calculate_reward(self, action, array_data):

        z_score = self.detect_z_score_trends(array_data)  # É um array de cinco posições, cada uma representando uma variável
                
        # Detectar se houve aumento abrupto de BSW
        # Usamos o operador lógico XOR (^) para comparar os sinais de z_score e inc_abrupt_bsw,
        # seguido de np.all() para verificar se todos os elementos resultantes são False (indicando correspondência de sinais)
        # True (1) em inc_abrupt_bsw indica esperar um valor positivo em z_score para essa variável, e vice-versa.
        aumento_abrupto_bsw = not np.any(z_score ^ (self.inc_abrupt_bsw > 0))
        
        # Definir a recompensa com base na ação e se houve aumento abrupto de BSW
        if aumento_abrupto_bsw == 0 and action == 0:
            reward = 0.01
        elif aumento_abrupto_bsw == 0 and action == 1:
            reward = -1
        elif aumento_abrupto_bsw == 1 and action == 0:
            reward = -1
        elif aumento_abrupto_bsw == 1 and action == 1:
            reward = 1

        return reward
            

    def render(self, mode='human'):
        pass

    def close(self):
        pass