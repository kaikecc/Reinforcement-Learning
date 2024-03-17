import gym
from gym import spaces
import numpy as np
import pandas as pd

class Env3WGym(gym.Env):
    """
   
    - Observadores: ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'] 

    - Pressão: P-PDG, P-TPT, P-MON-CKP
    - Temperatura: T-TPT, T-JUS-CKP

        * A regra geral, entretanto, é que as pressões medidas em pontos mais profundos aumentem 
        e que as pressões nos pontos mais próximos à superfície diminuam. As temperaturas, no geral, costumam
        aumentar em todos os equipamentos.
    
    - Para detecção de Aumento Abrupto de BSW (detect_array_trend_trends):
        - 0 (Não houve aumento abrupto de BSW)
        - 1 (Houve aumento abrupto de BSW)
        - -1 (Houve diminuição abrupta de BSW)

    Ações/Recompensas:
    - Se Ação = 0 AND (self.array_trend[self.dataset_index] XOR self.inc_abrupt_bsw) == 0: Recompensa = 0.01
    - Se Ação = 0 AND (self.array_trend[self.dataset_index] XOR self.inc_abrupt_bsw) == 1: Recompensa = -1
    - Se Ação = 1 AND (self.array_trend[self.dataset_index] XOR self.inc_abrupt_bsw) == 0: Recompensa = -1
    - Se Ação = 1 AND (self.array_trend[self.dataset_index] XOR self.inc_abrupt_bsw) == 1: Recompensa = 1

    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, array_list):
        super(Env3WGym, self).__init__()
        self.array_list = array_list
        self.array_index = 0 # Indice do array_z (dataset)
        self.index = 0 # Indice do array list
        
        
        if isinstance(array_list, np.ndarray):
            self.num_datasets = 1
            self.dataset = array_list
        else:
            self.num_datasets = len(array_list) # Tamanho de arrays dentro de array_list
            self.dataset = array_list[0] # Primeiro array de dados
        # Definição do padrão para aumento abrupto de BSW
        self.inc_abrupt_bsw = np.array([[1, -1, 1, -1, 1], # Padrão original
                                       [1, -1, 1, -1, 0],
                                       [1, -1, 1, 0, 1],
                                       [1, -1, 0, -1, 1],
                                       [1, 0, 1, -1, 1],
                                       [0, 0, 1, -1, 1],
                                       [1, 0, 1, -1, 1],
                                       [0, -1, 1, -1, 1],
                                       [1, -1, 1, -1, -1],
                                       [1, -1, 1, 1, 1],
                                       [1, -1, -1, -1, 1],
                                       [1, 1, 1, -1, 1],
                                       [-1, -1, 1, -1, 1],
                                       [1, -1, 1, -1, 1]])  
        
        self.array_trend = np.zeros_like(self.dataset)  # Inicialização do array de tendências de Z-score
        self.window_size = 6 * 3600   # 1 * 3600 Ajuste para o número de linhas que representa uma hora
        self.margin=0.1
        self.update_dataset() # Atualiza o dataset
        
        num_features = self.dataset.shape[1]  # Numero de colunas        
        self.action_space = spaces.Discrete(2)  # Actions: 0 or 1
        self.observation_space = spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(num_features,), dtype=np.float32)       
        
        self.episode_ended = False

    def moving_average_std(self, data_col):
        """Calcula a média móvel e o desvio padrão móvel usando Pandas."""
        data_series = pd.Series(data_col)
        
        # Calcula a média móvel e o desvio padrão móvel usando a janela definida em self.window_size
        rolling = data_series.rolling(window=self.window_size, min_periods=1)
        ma = rolling.mean()
        std = rolling.std(ddof=0)

        # Substitui NaNs nos cálculos da média e desvio padrão para manter o tamanho consistente
        ma_filled = ma.fillna(method='bfill').fillna(method='ffill')
        std_filled = std.fillna(method='bfill').fillna(method='ffill')

        return ma_filled, std_filled

 
    def detect_trends_with_window(self):
        # Converte self.dataset para um DataFrame Pandas para utilizar as funcionalidades de rolling window
        df = pd.DataFrame(self.dataset)
        array_trend = np.zeros_like(self.dataset, dtype=float)

        for col in df.columns:
            # Calculando a diferença discreta para capturar a tendência
            derivative = df[col].diff()

            # Aplicando a média móvel sobre a diferença discreta com a janela especificada
            moving_average_derivative = derivative.rolling(window=self.window_size).mean()

            # Identificando pontos de aumento, diminuição e estabilidade
            increasing = moving_average_derivative > 0
            decreasing = moving_average_derivative < 0

            # Atribuir 1 para aumento, -1 para diminuição, e 0 para estabilidade
            array_trend[:, col] = np.where(increasing, 1, np.where(decreasing, -1, 0))

        return array_trend



    def update_dataset(self):
        self.dataset_index = 0
        if isinstance(self.array_list, np.ndarray):
            self.dataset = self.array_list
        else:
            self.dataset = self.array_list[self.array_index]
        
        self.array_trend = self.detect_trends_with_window()  # É um array de cinco posições, cada uma representando uma variável

    def step(self, action):
        if self.episode_ended:
            return self.reset()
        
        reward = self.calculate_reward(action)

        self.dataset_index += 1
        if self.dataset_index >= self.dataset.shape[0]:
            self.array_index += 1
            if self.array_index >= self.num_datasets:
                self.episode_ended = True
            else:
                self.update_dataset()
                
        
        done = self.episode_ended

        if not done:
            self.state = self.dataset[self.dataset_index, :]
        else:
            self.state = np.zeros(self.observation_space.shape[0])

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.array_index = 0
        self.dataset_index = 0
        if isinstance(self.array_list, np.ndarray):
            self.dataset = self.array_list
        else:
            self.dataset = self.array_list[self.array_index]

        self.state = self.dataset[0, :]
        self.episode_ended = False
        return np.array(self.state, dtype=np.float32)

    def calculate_reward(self, action):
       
        current_trend = self.array_trend[self.dataset_index, :][np.newaxis, :]  # Make it 2D for comparison

        # Compare the current trend against all patterns in inc_abrupt_bsw
        # This creates a 2D boolean array where each row represents the comparison with one pattern
        pattern_matches = np.all(current_trend == self.inc_abrupt_bsw, axis=1)

        # Check if any of the patterns matches the current trend
        aumento_abrupto_bsw = np.any(pattern_matches)  # Corrected line

        # Initially sets reward to 0 to cover unspecified cases
        reward = 0

        # Adjust the reward based on the action and whether it matches the pattern for abrupt BSW increase
        if action == 0:
            if aumento_abrupto_bsw:
                reward = -1  # Undesired action if there was an abrupt BSW increase
            else:
                reward = 0.01  # Small reward if the action was correct considering no abrupt BSW increase
        elif action == 1:
            if aumento_abrupto_bsw:
                reward = 1  # Positive reward if the action was correct and there was an abrupt BSW increase
            else:
                reward = -1  # Undesired action in the absence of abrupt BSW increase

        return reward

           

    def render(self, mode='human'):
        pass

    def close(self):
        pass