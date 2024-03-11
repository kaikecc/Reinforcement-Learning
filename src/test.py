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
  

    well_names = ['WELL-00001','WELL-00002', 'WELL-00004', 'WELL-00005', 'WELL-00007']

    real_instances = list(class_and_file_generator(data_path, real=True, simulated=False, drawn=False))

    arrays_list = []  # List to store processed NumPy arrays

    for instance in real_instances:
        class_code, instance_path = instance    
        well, _ = instance_path.stem.split('_')  
             
        if class_code in events_names and well in well_names:

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

columns = [
        #'timestamp',
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
dataset_test = load_instance_with_numpy(data_path = '/home/dataset_test', events_names = events_names, columns = columns)

df = pd.DataFrame(dataset_test, columns = ['timestamp', 'P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'class'])
df.set_index('timestamp', inplace=True)
df[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']] = df[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']].astype('float32')
df['class'] = df['class'].astype(float).astype('int16')
#explora = exploration(df)
#explora.plot_sensor(sensor_columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], _title = 'Plot DQN')