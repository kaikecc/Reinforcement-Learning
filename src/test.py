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
from datetime import datetime
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
#sys.path.append(os.path.join('..'))
from classes._exploration import exploration
from classes._Env3WGym import Env3WGym
from classes._LoadInstances import LoadInstances
from classes._Agent import Agent

if __name__ == '__main__':

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

    event_name = [value for key, value in events_names.items() if key != 0][0]
    

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    log_filename = f'..\\..\\logs\\{current_time}_{event_name}-log.txt'
    # Configuração do Logging
    logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO, format='[%(levelname)s]\t%(asctime)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', force=True, encoding='utf-8')

    path_dataset = '..\\..\\..\\dataset'     

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
    
    logging.info(f'Iniciando carregamento do dataset')
    dataset = instances.load_instance_with_numpy(events_names, columns)    
    logging.info(f'Fim carregamento do dataset')
    
    logging.info(f'Iniciando divisão do dataset em treino e teste')
        
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
        logging.info(f'Número de amostras da classe {event}: {len(class_indices)}')
        
        #O parâmetro random_state=42 garante que essa divisão seja feita de maneira reproducível, ou seja, a função produzirá o mesmo resultado cada vez que for executada com o mesmo estado aleatório. 
        # Dividindo os índices da classe atual em treino e teste
        class_train_indices, class_test_indices = train_test_split(class_indices, train_size=train_percentage) # , random_state=42
        
        # Logando o número de amostras de treino e teste
        logging.info(f'Número de amostras de treino da classe {event}: {len(class_train_indices)}')
        logging.info(f'Número de amostras de teste da classe {event}: {len(class_test_indices)}')
        
        # Adicionando aos índices gerais de treino e teste
        train_indices.extend(class_train_indices)
        test_indices.extend(class_test_indices)

    # Convertendo listas para arrays numpy para futura manipulação
    train_indices = np.array(train_indices)    
    test_temp_indices = np.array(test_indices)       

    test_indices, validation_indices = train_test_split(test_temp_indices, test_size=0.5) # , random_state=42

    # Embaralhando os índices (opcional, dependendo da necessidade)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    

    # Criando conjuntos de dados de treino e teste
    dataset_train = dataset[train_indices]
    dataset_test = dataset[test_indices]
    dataset_validation = dataset[validation_indices]

    logging.info(f'Número de amostras de treino: {len(dataset_train)}')
    logging.info(f'Número de amostras de teste: {len(dataset_test)}')
    logging.info(f'Número de amostras de validação: {len(dataset_validation)}')
    

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
    X_validation_scaled = np.column_stack((X_validation[:, 0], scaler.transform(X_validation[:, 1:])))    
   

    # Se necessário, você pode combinar as features escalonadas e o target para formar os datasets finais
    dataset_train_scaled = np.column_stack((X_train_scaled, y_train))
    dataset_test_scaled = np.column_stack((X_test_scaled, y_test))
    dataset_validation_scaled = np.column_stack((X_validation_scaled, y_validation))

    explora_dataset = exploration(dataset_train_scaled)
    explora_dataset.plot_estados(_title = f'{event_name} - Dataset Treino')

    explora_dataset = exploration(dataset_test_scaled)
    explora_dataset.plot_estados(_title = f'{event_name} - Dataset Teste')

    explora_dataset = exploration(dataset_validation_scaled)
    explora_dataset.plot_estados(_title = f'{event_name} - Dataset Validação')
   
    logging.info(f'Fim divisão do dataset em treino e teste')
    
    logging.info(f'Iniciando treinamento do algoritmo DQN')
    
    start_time = time.time()
    agente = Agent(f'..\\models\\{event_name}_DQN_Env3W')

    agente.env3W_dqn(dataset_train_scaled, n_envs = 5)  
    print(f"Tempo de Treinamento DQN: {round(time.time() - start_time, 2)}s")
    logging.info(f"Tempo de Treinamento DQN: {round(time.time() - start_time, 2)}s")

    logging.info(f'Fim treinamento do algoritmo DQN')

    logging.info(f'Iniciando avaliação do algoritmo DQN conjunto de teste')    
    
    accuracy, dqn_model = agente.env3W_dqn_eval(dataset_test_scaled, n_envs = 5)
    print(f'Acurácia de {accuracy * 100:.2f}% no conjunto de dados de teste usando DQN')
    logging.info(f'Acurácia de {accuracy:.5f} no conjunto de dados de teste usando DQN')
    logging.info(f'Fim avaliação  do algoritmo DQN conjunto de teste')

    if accuracy > 0.8:
        logging.info(f'Iniciando a separação dos grupos de dados para validação individual')
        # Obtendo os índices que ordenariam a primeira coluna
        sort_indices = np.argsort(dataset_validation_scaled[:, 0])

        # Usando esses índices para reordenar todo o array
        dataset_validation_sorted = dataset_validation_scaled[sort_indices]
        
        # Inicializando a lista para armazenar os sub-datasets
        datasets = []
        current_dataset = []

        # Inicializando previous_datetime como None para a primeira comparação
        previous_datetime = None

        for row in dataset_validation_sorted:
            current_datetime = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            
            # Verifica se é a primeira iteração ou se a diferença é maior que 1 hora
            if previous_datetime is None or (current_datetime - previous_datetime).total_seconds() / 3600 > 1:
                # Se não for a primeira iteração e a condição for verdadeira, inicia um novo dataset
                if current_dataset:
                    datasets.append(np.array(current_dataset))
                    current_dataset = []
            
            # Adiciona o registro atual ao dataset corrente
            current_dataset.append(row)
            previous_datetime = current_datetime

        # Não esqueça de adicionar o último dataset após terminar o loop
        if current_dataset:
            datasets.append(np.array(current_dataset))
        

        logging.info(f'Fim da separação dos grupos de dados para validação com {len(datasets)} grupos de instâncias')
        
        count = -1
        acc_total = []
        array_prec_total = []
        for dataset_test in datasets:
            acc = 0
            count += 1
            logging.info(f'Iniciando predição da {count}ª instância para teste usando DQN')
            array_action_pred = []
            for i in range(0, len(dataset_test)):
                obs = dataset_test[i, 1:-1].astype(np.float32)
                action, _states = dqn_model.predict(obs, deterministic=True)  
                array_action_pred.append(action)

                true_action = dataset_test[i, -1]
                if true_action == 0:
                    acc +=  1 if action == 0 else 0
                elif true_action in range(1, 10):
                    acc +=  1 if action == 1 else 0
                elif true_action in range(101, 110):  # Corrigido para refletir o intervalo correto
                    acc +=  1 if action == 1 else 0  
                
                    
            acc_total.append(acc)
            array_prec_total.append(len(array_action_pred))        
            final_acc = int(acc)/len(array_action_pred) * 100
            logging.info(f'Acurácia da {count}ª instância: {final_acc:.3f}%')
            print(f'Acurácia da {count}ª instância: {final_acc:.3f}%')
            expanded_array = np.column_stack((dataset_test, array_action_pred))
            logging.info(f'Fim predição da instância de teste DQN')    
        
            
            df = pd.DataFrame(expanded_array, columns = ['timestamp', 'P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'class', 'action'])
            df.set_index('timestamp', inplace=True)
            df[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']] = df[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']].astype('float32')
            df['class'] = df['class'].astype(float).astype('int16')
            df['action'] = df['action'].astype(float).astype('int16')


            explora = exploration(df)
            explora.plot_sensor(sensor_columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], _title = f'[{count}] - {event_name} - DQN')
        
        logging.info(f'Acurácia: {sum(acc_total)/sum(array_prec_total) * 100:.3f}% no conjunto de dados de validação usando DQN')
        print(f'Acurácia: {sum(acc_total)/sum(array_prec_total) * 100:.3f}% no conjunto de dados de validação usando DQN')

    else:
        logging.info(f'Acurácia insuficiente para validação individual')
        print(f'Acurácia insuficiente para validação individual')
    logging.info(f'Concluído a execução do aprendizado por reforço')