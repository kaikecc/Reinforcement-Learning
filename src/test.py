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

def class_and_file_generator(data_path, real=False, simulated=False, drawn=False):
    for class_path in data_path.iterdir():
        if class_path.is_dir():
            try:
                class_code = int(class_path.stem)
            except ValueError:
                # Se não for possível converter para int, pule este diretório
                continue

            for instance_path in class_path.iterdir():
                if (instance_path.suffix == '.csv'):
                    if (simulated and instance_path.stem.startswith('SIMULATED')) or \
                       (drawn and instance_path.stem.startswith('DRAWN')) or \
                       (real and (not instance_path.stem.startswith('SIMULATED')) and \
                       (not instance_path.stem.startswith('DRAWN'))):
                        yield class_code, instance_path

def load_instance_with_numpy(data_path, events_names, columns):
    data_path = Path(data_path)
  

    well_names = ['WELL-00001','WELL-00002', 'WELL-00004', 'WELL-00005', 'WELL-00007', 'WELL-00009', 'WELL-00010', 'WELL-00011', 'WELL-00012', 'WELL-00013']

    real_instances = list(class_and_file_generator(data_path, real=True, simulated=False, drawn=False))

    arrays_list = []  # List to store processed NumPy arrays

    for instance in real_instances:
        class_code, instance_path = instance    
        well, _ = instance_path.stem.split('_')  

        

        if class_code in events_names.keys() and well in well_names:

            # Read the entire CSV file into a NumPy array
            with open(instance_path, 'r') as file:
                header = file.readline().strip().split(',')
                indices = [header.index(col) for col in columns]                
                                                
                if 'timestamp' in columns:
                    arr = np.genfromtxt(file, delimiter=',', usecols=indices[1:], dtype=np.float32)
                    timestamp_idx = header.index('timestamp')
                    file.seek(0)
                    file.readline()  # Pula o cabeçalho
                    timestamps = np.genfromtxt(file, delimiter=',', skip_header=0, usecols=timestamp_idx, dtype=str)
                    if isinstance(timestamps, str):                
                        timestamps = np.array([timestamps])

                    fmt = "%Y-%m-%d %H:%M:%S.%f"
                    rounded_timestamps = [datetime.strptime(ts, fmt).strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]
                    
                    # Combinação dos timestamps arredondados e dados numéricos
                    final_data = np.hstack([np.array(rounded_timestamps).reshape(-1, 1), arr])
                    #final_data = final_data[~np.isnan(final_data).any(axis=1)]
                    #final_data[:, 1:-1] = final_data[:, 1:-1].astype(np.float32)
                    #final_data[:, -1] = final_data[:, -1].astype(np.int16)
                    arrays_list.append(final_data)
                else:
                    arr = np.genfromtxt(file, delimiter=',', usecols=indices, dtype=np.float32)
                    arr[:, :-1] = arr[:, :-1].astype(np.float32)  # Convert selected columns to float32
                    arr[:, -1] = arr[:, -1].astype(np.int16)  # Convert 'class' column to int16
                    arr = arr[~np.isnan(arr).any(axis=1)]
                    arrays_list.append(arr)               

                

    # Concatenate all processed NumPy arrays
    final_array = np.concatenate(arrays_list)

    return final_array


path_dataset = '/home/dataset'
events_names = {
    0: 'Normal',
    # 1: 'Abrupt Increase of BSW',
     2: 'Spurious Closure of DHSV',
    # 3: 'Severe Slugging',
    # 4: 'Flow Instability',
    # 5: 'Rapid Productivity Loss',
    # 6: 'Quick Restriction in PCK',
    # 7: 'Scaling in PCK',
    # 8: 'Hydrate in Production Line'
}

columns = [       
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

logging.info(f'Iniciando carregamento do dataset')
dataset = load_instance_with_numpy(data_path = path_dataset, events_names = events_names, columns = columns)    
logging.info(f'Fim carregamento do dataset')

logging.info(f'Iniciando divisão do dataset em treino e teste')

# Filtrar índices que correspondem aos eventos de interesse
condition = np.isin(dataset[:, -1], list(np.unique(dataset[:, -1])))
indices = np.where(condition)[0]

# Definindo a porcentagem para divisão entre treino e teste
train_percentage = 0.8  # 80% para treino

# Inicializando listas para guardar índices de treino e teste
train_indices = []
test_indices = []
unique = np.unique(dataset[:, -1])
# Processamento genérico para cada classe
for event in np.unique(dataset[:, -1]):
    # Selecionando índices para a classe atual
    class_indices = indices[dataset[indices, -1] == event]
    
    # Logando o número de amostras por classe
    print(f'Número de amostras da classe {event}: {len(class_indices)}')
    logging.info(f'Número de amostras da classe {event}: {len(class_indices)}')
    
    # Dividindo os índices da classe atual em treino e teste
    class_train_indices, class_test_indices = train_test_split(class_indices, train_size=train_percentage, random_state=42)
    
    # Logando o número de amostras de treino e teste
    logging.info(f'Número de amostras de treino da classe {event}: {len(class_train_indices)}')
    logging.info(f'Número de amostras de teste da classe {event}: {len(class_test_indices)}')
    
    # Adicionando aos índices gerais de treino e teste
    train_indices.extend(class_train_indices)
    test_indices.extend(class_test_indices)

# Convertendo listas para arrays numpy para futura manipulação
train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

# Embaralhando os índices (opcional, dependendo da necessidade)
np.random.shuffle(train_indices)
np.random.shuffle(test_indices)

# Criando conjuntos de dados de treino e teste
dataset_train = dataset[train_indices]
dataset_test = dataset[test_indices]

# Dividindo em features (X) e target (y)
X_train, y_train = dataset_train[:, :-1], dataset_train[:, -1]
X_test, y_test = dataset_test[:, :-1], dataset_test[:, -1]

# Escalonando as features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)