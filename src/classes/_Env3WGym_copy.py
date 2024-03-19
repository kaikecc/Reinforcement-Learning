import gym
from gym import spaces
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

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
            self.window_hour = 5  # Janela de 5 horas
        else:
            self.num_datasets = len(array_list) # Tamanho de arrays dentro de array_list
            self.dataset = array_list[0] # Primeiro array de dados
            self.window_hour = int((len(self.dataset )/3600)/6)
        
        
        
        self.array_trend = np.zeros_like(self.dataset)  # Inicialização do array de tendências de Z-score
        
        self.update_dataset() # Atualiza o dataset
        
        num_features = self.dataset.shape[1] - 1  # Numero de colunas        
        self.action_space = spaces.Discrete(2)  # Actions: 0 or 1
        self.observation_space = spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(num_features,), dtype=np.float32)       
        
        self.episode_ended = False
  

    def detect_trends(self):
        
        columns_to_normalize = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
        
        # Convertendo o array NumPy para um DataFrame pandas
        df = pd.DataFrame(self.dataset, columns=['timestamp', 'P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'])

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Normalizando os dados especificados entre 0 e 1 diretamente no DataFrame original        
        for col in columns_to_normalize:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Calculando a frequência de amostragem mais comum (em segundos)
        sampling_frequency_seconds = df['timestamp'].diff().dt.total_seconds().mode()[0]

        # Convertendo a frequência de amostragem para amostras por hora e ajustando a janela para 6 horas de dados
        samples_per_hour = 3600 / sampling_frequency_seconds
        window_length = int(samples_per_hour * self.window_hour)
        if window_length % 2 == 0:  # Ajustando para ímpar se necessário
            window_length += 1

        # Aplicando o filtro Savitzky-Golay diretamente ao DataFrame
        polyorder = 3  # Ordem do polinômio
        for col in columns_to_normalize:
            df[col] = savgol_filter(df[col], window_length if window_length <= len(df[col]) else len(df[col])//2*2+1, polyorder)

        df_trend = pd.DataFrame(0, index=df.index, columns=columns_to_normalize)
        size = len(df)
        # Calculando a inclinação e classificando as variações
        for col in columns_to_normalize:
            for i in range(window_length, size):
                # Calcula a variação percentual
                percent_change = (df[col].iloc[i] - df[col].iloc[i - window_length]) / df[col].iloc[i - window_length]
                
                # Classifica a variação
                if percent_change > 0.1:
                    df_trend[col].iloc[i] = 1                    
                elif percent_change < -0.1:
                    df_trend[col].iloc[i] = -1
                    

        self.array_trend = df_trend.values  # Convertendo o DataFrame de tendências para um array NumPy

        


    def update_dataset(self):
        self.dataset_index = 0
        if isinstance(self.array_list, np.ndarray):
            self.dataset = self.array_list
        else:
            self.dataset = self.array_list[self.array_index]
            self.window_hour = int((len(self.dataset )/3600)/6)
        
        self.detect_trends()  # É um array de cinco posições, cada uma representando uma variável
        
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
            self.state = self.dataset[self.dataset_index, 1:]
        else:
            self.state = np.zeros(self.observation_space.shape[0]-1)

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.array_index = 0
        self.dataset_index = 0
        if isinstance(self.array_list, np.ndarray):
            self.dataset = self.array_list
        else:
            self.dataset = self.array_list[self.array_index]

        self.state = self.dataset[0, 1:]
        self.episode_ended = False
        return np.array(self.state, dtype=np.float32)

    def calculate_reward(self, action):
       
        pattern_matches = self.array_trend[self.dataset_index, :] != [1, 1, -1, -1, -1] # ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
        # Check if any of the patterns matches the current trend
        aumento_abrupto_bsw  = np.all(pattern_matches)


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