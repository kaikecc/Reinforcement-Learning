import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

import tensorflow as tf
import reverb
from tf_agents.environments import tf_py_environment,  ParallelPyEnvironment
from tf_agents.networks import q_network, sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory as traj
from classes._Env3WGym import Env3WGym  # Asegure-se de que essa classe esteja definida corretamente em outro lugar
from datetime import datetime
from tf_agents.system import multiprocessing as tf_multiprocessing
from tf_agents.specs import tensor_spec
from tf_agents.environments import utils
from pathlib import Path
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from sklearn.preprocessing import StandardScaler
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network
from classes._exploration import exploration
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from torch import nn  # Import the neural network module from PyTorch
import time

def plot_estados(df_env):
    # Contagem de valores para cada classe
    class_counts = df_env['class'].value_counts().sort_index()

    # Agregando contagens de classes
    # Considerando a descrição anterior, parece haver um pequeno equívoco na lógica original
    # para a contagem de classes 'Transiente de anomalia' e 'Estável de anomalia'.
    # Esta versão corrige a lógica de agregação conforme a descrição fornecida.
    normal_count = class_counts.loc[class_counts.index == 0].sum()
    estavel_anomalia_count = class_counts.loc[(class_counts.index >= 1) & (class_counts.index <= 8)].sum()
    transiente_anomalia_count = class_counts.loc[(class_counts.index >= 101) & (class_counts.index <= 108)].sum()

    # Total de amostras
    total = normal_count + estavel_anomalia_count + transiente_anomalia_count

    # Imprimindo as informações
    logging.info('Contagem de amostras por classe:')
    logging.info(f'Normal: {normal_count} - {round(normal_count/total*100, 2)}%')
    logging.info(f'Estável de anomalia: {estavel_anomalia_count} - {round(estavel_anomalia_count/total*100, 2)}%')
    logging.info(f'Transiente de anomalia: {transiente_anomalia_count} - {round(transiente_anomalia_count/total*100, 2)}%')
    logging.info(f'Total: {total}')

    fig, ax = plt.subplots()
    categories = ['Normal', 'Estável de anomalia', 'Transiente de anomalia']
    counts = [normal_count, estavel_anomalia_count, transiente_anomalia_count]
    colors = ['blue', 'orange', 'green']  # Lista de cores para as barras

    ax.bar(categories, counts, color=colors)
    ax.set_ylabel('Quantidade')
    ax.set_title('Quantidade de amostras por classe')
    ax.legend(labels=categories)
    plt.savefig('Reinforcement-Learning/src/imagens/quantidade_por_classe.png')
    #plt.show()

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

def load_instance_with_numpy(data_path):
    data_path = Path(data_path)

    events_names = {
        0: 'Normal',
        #1: 'Abrupt Increase of BSW',
        # 2: 'Spurious Closure of DHSV',
        # 3: 'Severe Slugging',
        4: 'Flow Instability',
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
                arr = np.genfromtxt(file, delimiter=',', usecols=indices, dtype=np.float32)
                                                
                arr = arr[~np.isnan(arr[:, [0, 1, 2, 3, 4, 5]]).any(axis=1)]
                arr[:, :-1] = arr[:, :-1].astype(np.float32)  # Convert selected columns to float32
                arr[:, -1] = arr[:, -1].astype(np.int16)  # Convert 'class' column to int16

                # Adds the processed NumPy array to the list
                arrays_list.append(arr)

    # Concatenate all processed NumPy arrays
    final_array = np.concatenate(arrays_list)

    return final_array

def env3W_dqn(env):
    model = DQN(
        MlpPolicy,
        env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=10000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        max_grad_norm=10,
        tensorboard_log=None,
        policy_kwargs=dict(net_arch=[64, {"vf": [32], "pi": [32]}], activation_fn=nn.ReLU),
        verbose=1,
    )

    model.learn(total_timesteps=int(1.2e5))
    
    model.save('dqn_env3W.pkl')
    env.close()

    return model

def configLog(path):
    # Configuração do Logging
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f'{path}{current_time}_RL-log.txt'    
    logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO, format='[%(levelname)s]\t%(asctime)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')

if __name__ == '__main__':

    path_log = 'Reinforcement-Learning/logs/'
    configLog(path_log)

    
    path_dataset = '/home/dataset'  
    logging.info('Carregando o conjunto de dados 3W')
    dataset = load_instance_with_numpy(data_path = path_dataset)    

    path_exploration = 'Reinforcement-Learning/src/imagens/'
    exploration_raw = exploration(pd.DataFrame(dataset, columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'class']))
    exploration_raw.quartiles_plot(['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], f'{path_exploration}Quartis dos Sensores em Instabilidade de Vazão')
    exploration_raw.heatmap_corr(['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], f'{path_exploration}/Mapa de Calor da Correlação dos Sensores em Instabilidade de Vazão')
    
    # Crie um dataframe com 10000 linhas onde na coluna class é 0 e mais 10000 linhas onde na coluna class é 4
    # Replace 'class' == 0 with 'class' == 0 and 'class' == 4
    indices = np.where((dataset[:, -1] == 0) | (dataset[:, -1] == 4))[0]       
      
    percentage = 0.001  # Change this to the percentage you want

    class_0_indices = indices[dataset[indices, -1] == 0]
    class_4_indices = indices[dataset[indices, -1] == 4]

    logging.info(f'Número de amostras da classe 0: {len(class_0_indices)}')
    logging.info(f'Número de amostras da classe 4: {len(class_4_indices)}')

    num_samples_class_0 = int(len(class_0_indices) * percentage)
    num_samples_class_4 = int(len(class_4_indices) * percentage)
  
    logging.info(f'Número de {percentage} amostras da classe 0: {num_samples_class_0}')
    logging.info(f'Número de {percentage} amostras da classe 4: {num_samples_class_4}')

    selected_rows_class_0 = np.random.choice(class_0_indices, num_samples_class_0, replace=False)
    selected_rows_class_4 = np.random.choice(class_4_indices, num_samples_class_4, replace=False)
    
    selected_rows = np.concatenate([selected_rows_class_0, selected_rows_class_4])
    
    scaler = StandardScaler()
    # Create a new array with the selected rows
    dataset_train = dataset[selected_rows]

    # If needed, shuffle the resulting array
    np.random.shuffle(dataset_train)

    # Assuming the last column is the target variable, and the rest are features
    X_train = dataset_train[:, :-1]
    y_train = dataset_train[:, -1]

    # Use StandardScaler to scale the features
    X_train_scaled = scaler.fit_transform(X_train)

    # Combine the scaled features and the target variable
    dataset_train_scaled = np.column_stack((X_train_scaled, y_train))

    exploration_treinamento = exploration(pd.DataFrame(dataset_train_scaled, columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'class']))
    
    exploration_treinamento.quartiles_plot(['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], f'{path_exploration}Quartis dos Sensores em Instabilidade de Vazão (Treinamento)')
    exploration_treinamento.heatmap_corr(['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], f'{path_exploration}Mapa de Calor da Correlação dos Sensores em Instabilidade de Vazão (Treinamento)')
    # Now you can use the scaled dataset in your main function
    #main(dataset_train_scaled)

    start_time = time.time()
    dqn_model = env3W_dqn(Env3WGym(dataset_train_scaled))
    print("DQN Training Time:", time.time() - start_time)
    