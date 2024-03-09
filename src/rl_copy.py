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
    
    log_interval = 1000
    eval_interval = 5000
    num_iterations = 100000
    learning_rate = 1e-3
    batch_size = 64
    initial_collect_steps = 100
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000
    num_eval_episodes = 1
    

    logging.info('Configuração do Agente e Ambiente')

    env = env3W(df_train)
    train_py_env  = env3W(df_train)
    eval_py_env = env3W(df_train)
    

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1    

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    
    # Reseta o ambiente
    time_step = env.reset()
    logging.info(f'Time step: {time_step}')

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)

    py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
        random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps).run(train_py_env.reset())

    # Preparação para o Treinamento
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    
    iterator = iter(dataset)

    # Otimizações com TF function
    agent.train = common.function(agent.train)
    returns = []
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    logging.info('Iniciando Treinamento...')
    # Loop de Treinamento Simplificado
    for _ in range(num_iterations):

        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))
            logging.info('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)  
            returns.append(avg_return)

    logging.info('Treinamento Concluído!')
    # Plotagem dos Resultados
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Média das Recompensas')
    plt.xlabel('Iterações')
    plt.grid()
    plt.savefig('Reinforcement-Learning/src/imagens/avg_return_2.png')
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

    # Now you can use the scaled dataset in your main function
    main(dataset_train_scaled)
    