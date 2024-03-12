import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
from torch import nn  # Import the neural network module from PyTorch
import gym
from gym import spaces
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import os
#sys.path.append(os.path.join('..','..'))
from classes._exploration import exploration
from classes._Env3WGym import Env3WGym
from classes._Agent import Agent
from classes._LoadInstances import LoadInstances


events_names = {
    0: 'Normal',
    # 1: 'Abrupt Increase of BSW',
    # 2: 'Spurious Closure of DHSV',
    # 3: 'Severe Slugging',
    4: 'Flow Instability',
    # 5: 'Rapid Productivity Loss',
    # 6: 'Quick Restriction in PCK',
    # 7: 'Scaling in PCK',
    # 8: 'Hydrate in Production Line'
}

event_name = [value for key, value in events_names.items() if key != 0]

path_dataset = '/home/dataset'    

columns = [ 
    'timestamp',      
    'P-PDG',
    'P-TPT',
    'T-TPT',
    'P-MON-CKP',
    'T-JUS-CKP',
    #'P-JUS-CKGL',
    #'T-JUS-CKGL',
    #'QGL',
    'class'
]

well_names = [f'WELL-{i:05d}' for i in range(1, 19)]

instances = LoadInstances(path_dataset)


dataset = instances.load_instance_with_numpy(events_names, columns)    

    
# Definindo a porcentagem para divisão entre treino e teste
train_percentage = 0.8  # 80% para treino

# Inicializando listas para guardar índices de treino e teste
train_indices = []
test_indices = []

# Processamento genérico para cada classe
for event in np.unique(dataset[:, -1]):
    # Selecionando índices para a classe atual        
    class_indices = np.where(dataset[:, -1] == event)[0]
    
    # Logando o número de amostras por classe
    print(f'Número de amostras da classe {event}: {len(class_indices)}')
    
    
    # Dividindo os índices da classe atual em treino e teste
    class_train_indices, class_test_indices = train_test_split(class_indices, train_size=train_percentage, random_state=42)
    
        
    # Adicionando aos índices gerais de treino e teste
    train_indices.extend(class_train_indices)
    test_indices.extend(class_test_indices)

# Convertendo listas para arrays numpy para futura manipulação
train_indices = np.array(train_indices)    
test_temp_indices = np.array(test_indices)       

test_indices, validation_indices = train_test_split(test_temp_indices, test_size=0.5, random_state=42)

# Embaralhando os índices (opcional, dependendo da necessidade)
np.random.shuffle(train_indices)
np.random.shuffle(test_indices)


# Criando conjuntos de dados de treino e teste
dataset_train = dataset[train_indices]
dataset_test = dataset[test_indices]
dataset_validation = dataset[validation_indices]


# Dividindo em features (X) e target (y)
X_train, y_train = dataset_train[:, :-1], dataset_train[:, -1]
X_test, y_test = dataset_test[:, :-1], dataset_test[:, -1]
X_validation, y_validation = dataset_validation[:, :-1], dataset_validation[:, -1]

# Delete a primeira coluna (timestamp) das features
X_train = np.delete(X_train, 0, axis=1)
X_test = np.delete(X_test, 0, axis=1)

# Escalonando as features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validation_scaled = scaler.transform(X_validation[:, 1:])   


# Se necessário, você pode combinar as features escalonadas e o target para formar os datasets finais
dataset_train_scaled = np.column_stack((X_train_scaled, y_train))
dataset_test_scaled = np.column_stack((X_test_scaled, y_test))
dataset_validation_scaled = np.column_stack((X_validation_scaled, y_validation))

fmt = "%Y-%m-%d %H:%M:%S.%f"

# Convertendo a primeira coluna para datetime
timestamps = dataset_validation_scaled[:, 0].astype(str)  # Assegura que estamos lidando com strings
datetimes = np.array([datetime.strptime(ts, fmt) for ts in timestamps])

# Calculando as diferenças em segundos
# As diferenças são calculadas entre timestamps consecutivos, convertendo para 'timedelta64[s]' para obter diferenças em segundos
time_diffs = np.array([(datetimes[i] - datetimes[i-1]).total_seconds() if i > 0 else np.nan for i in range(1, len(datetimes))])

# Encontrando índices onde o incremento excede 1 segundo
split_indices = np.where(time_diffs > 1)[0] + 1  # Correção para alinhar os índices após o cálculo de diferença

# Dividindo o dataset baseado nos índices encontrados
datasets = []
start_idx = 0
for end_idx in split_indices:
    datasets.append(dataset_validation_scaled[start_idx:end_idx, :])
    start_idx = end_idx

# Adicionando o último segmento após o último índice de divisão
datasets.append(dataset_validation_scaled[start_idx:, :])