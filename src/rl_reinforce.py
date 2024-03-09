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
from classes._env3W import env3W  # Asegure-se de que essa classe esteja definida corretamente em outro lugar
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

def compute_avg_return(environment, policy, num_episodes):

  logging.info('Computando a média das recompensas...')  
  total_return = 0.0
  for i in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

    logging.info(f'[{i}] - Recompensa: {episode_return}')


  avg_return = total_return / num_episodes
  logging.info(f'Média das recompensas: {avg_return}')
  print(f'Média das recompensas: {avg_return}')
  return avg_return.numpy()[0]

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


def main(df_train):    
    
    # Hiperparâmetros
    num_iterations = 250 # @param {type:"integer"}
    collect_episodes_per_iteration = 1 # @param {type:"integer"}
    replay_buffer_capacity = 10000 # @param {type:"integer"}

    fc_layer_params = (100,)

    learning_rate = 1e-3 # @param {type:"number"}
    log_interval = 25 # @param {type:"integer"}
    num_eval_episodes = 1 # @param {type:"integer"}
    eval_interval = 50 # @param {type:"integer"}

    
    # Criação do Ambiente Python
    env = env3W(df_train)
    train_py_env  = env3W(df_train)
    eval_py_env = env3W(df_train)

    logging.info(f'Primeira linha do dataset de treinamento: {df_train[0]}')
    time_step = train_py_env.reset()
    logging.info(f'Time Step Inicial: {time_step}')

    logging.info(f'Segunda linha do dataset de treinamento: {df_train[1]}')
    next_time_step = train_py_env.step(np.array(1, dtype=np.int32))
    logging.info(f'Próximo Time Step: {next_time_step}')
     
    # Criação do Ambiente TensorFlow
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # AGENTE    
    actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    tf_agent.initialize()

    # FIM AGENTE
        
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        tf_agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        table_name=table_name,
        sequence_length=None,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
        replay_buffer.py_client,
        table_name,
        replay_buffer_capacity
    )

    
    def collect_episode(environment, policy, num_episodes):

        driver = py_driver.PyDriver(
            environment,
            py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True),
            [rb_observer],
            max_episodes=num_episodes)
        initial_time_step = environment.reset()
        driver.run(initial_time_step)

    returns = []
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

        
    for _ in range(num_iterations):

     # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(
      train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)

        # Use data from the buffer and update the agent's network.
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
        trajectories, _ = next(iterator)
        train_loss = tf_agent.train(experience=trajectories)  

        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            logging.info('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            #print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    
    logging.info('Treinamento Concluído!')
    # Plotagem dos Resultados
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Média das Recompensas')
    plt.xlabel('Iterações')
    plt.grid()
    plt.savefig('Reinforcement-Learning/src/imagens/avg_return_reinforce.png')
    plt.show()

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

'''def load_instance_with_chunks(data_path, chunksize):
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

    real_instances = list(class_and_file_generator(data_path, real=True, simulated=False, drawn=False))

    chunk_list = []  # List to store processed chunks

   
    for instance in real_instances:

        class_code, instance_path = instance        

        # Reading the file in chunks
        for chunk in pd.read_csv(instance_path, chunksize=chunksize, usecols=columns):
            
            #well, instance_id = Path(instance_path).stem.split('_')

            # Checks if the columns are valid
            assert all(column in chunk.columns for column in columns), "Some required columns are missing in the file {}: {}".format(str(instance_path), str(chunk.columns.tolist()))

            chunk = chunk.dropna()

            if class_code in events_names:
                # Converting selected columns to float32
                for col in columns[:-1]:  # Last column 'class' is not converted here
                    chunk[col] = chunk[col].astype('float32')

                chunk['class'] = chunk['class'].astype('int16')  # Converting 'class' column to int16

                # Adds the processed chunk to the list
                chunk_list.append(chunk)


            # Concatenates all processed chunks into a single DataFrame        
    df_final = pd.concat(chunk_list, ignore_index=True)

    return df_final '''  

def load_instance_with_numpy(data_path):
    data_path = Path(data_path)

    events_names = {
        0: 'Normal',
        4: 'Flow Instability',
    }

    columns = [        
        'P-PDG',
        'P-TPT',
        'T-TPT',
        'P-MON-CKP',
        'T-JUS-CKP',
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

if __name__ == '__main__':

    # Configuração do Logging
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f'Reinforcement-Learning/logs/{current_time}_RL-log.txt'    
    logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO, format='[%(levelname)s]\t%(asctime)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    
    logging.info('Carregando o dataframe raw')    
    
    dataset = load_instance_with_numpy(data_path ='/home/dataset')    

    exploration_raw = exploration(pd.DataFrame(dataset, columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'class']))
    exploration_raw.quartiles_plot(['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], 'Reinforcement-Learning/src/imagens/Quartis dos Sensores em Instabilidade de Vazão')
    exploration_raw.heatmap_corr(['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], 'Reinforcement-Learning/src/imagens/Mapa de Calor da Correlação dos Sensores em Instabilidade de Vazão')
    
    # Crie um dataframe com 10000 linhas onde na coluna class é 0 e mais 10000 linhas onde na coluna class é 4
    # Replace 'class' == 0 with 'class' == 0 and 'class' == 4
    indices = np.where((dataset[:, -1] == 0) | (dataset[:, -1] == 4))[0]       
      
    percentage = 0.001  # Change this to the percentage you want

    class_0_indices = indices[dataset[indices, -1] == 0]
    class_4_indices = indices[dataset[indices, -1] == 4]

    print(f'Número de amostras da classe 0: {len(class_0_indices)}')
    print(f'Número de amostras da classe 4: {len(class_4_indices)}')
    logging.info(f'Número de amostras da classe 0: {len(class_0_indices)}')
    logging.info(f'Número de amostras da classe 4: {len(class_4_indices)}')

    num_samples_class_0 = int(len(class_0_indices) * percentage)
    num_samples_class_4 = int(len(class_4_indices) * percentage)

    print(f'Número de {percentage} amostras da classe 0: {num_samples_class_0}')
    print(f'Número de {percentage} amostras da classe 4: {num_samples_class_4}')
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
    
    exploration_treinamento.quartiles_plot(['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], 'Reinforcement-Learning/src/imagens/Quartis dos Sensores em Instabilidade de Vazão (Treinamento)')
    exploration_treinamento.heatmap_corr(['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'], 'Reinforcement-Learning/src/imagens/Mapa de Calor da Correlação dos Sensores em Instabilidade de Vazão (Treinamento)')
    # Now you can use the scaled dataset in your main function
    main(dataset_train_scaled)
    