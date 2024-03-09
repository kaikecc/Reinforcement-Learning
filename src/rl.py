import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import tensorflow as tf
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


def parallel_compute_avg_return(environment, policy, num_parallel_envs, num_eval_episodes):
    logging.info('Iniciando Avaliação...')
    total_return = 0.0
    num_episodes_per_env = np.ceil(num_eval_episodes / num_parallel_envs).astype(int)

    time_step = environment.reset()
    episode_return = tf.zeros(num_parallel_envs)
    total_episodes = 0

    logging.info(f'Número de Episódios por Ambiente: {num_episodes_per_env}')
    while total_episodes < num_eval_episodes:
        policy_step = policy.action(time_step)
        time_step = environment.step(policy_step.action)
        episode_return += time_step.reward

        # Verifica se algum episódio terminou e atualiza contadores de forma vetorializada
        is_last = time_step.is_last()
        total_return += tf.reduce_sum(episode_return * tf.cast(is_last, tf.float32))
        episode_return *= (1 - tf.cast(is_last, tf.float32))  # Zera retornos dos episódios terminados

        total_episodes += tf.reduce_sum(tf.cast(is_last, tf.int32))
        if total_episodes >= num_eval_episodes:
            break

    avg_return = total_return / num_eval_episodes
    logging.info(f'Recompensa Média: {avg_return}')
    logging.info('Avaliação Concluída!')
    return avg_return
# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

def create_env_constructor(df):
    return lambda: env3W(df)

def main(df_train):    
    
    log_interval = 1000
    eval_interval = 5000
    num_iterations = 10000
    learning_rate = 1e-3
    batch_size = 64
    initial_collect_steps = 100
    #collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000
    num_eval_episodes = 1
    num_parallel_envs = 4

    logging.info('Configuração do Agente e Ambiente')
    # Lista para armazenar os DataFrames de treinamento para cada ambiente paralelo
    dfs_train = [df_train.sample(frac=1, random_state=i).reset_index(drop=True) for i in range(num_parallel_envs)]

    env_constructors = [create_env_constructor(dfs_train[i]) for i in range(num_parallel_envs)]

    
    # Inicializar os ambientes paralelos, cada um com seu próprio DataFrame de treinamento
    parallel_env = ParallelPyEnvironment(env_constructors)
    logging.info('Ambientes paralelos criados com sucesso!')

    # Definição do Ambiente e do Agente
    
    train_env = tf_py_environment.TFPyEnvironment(parallel_env)
    eval_env = train_env  # Reutilização do ambiente para avaliação

    parallel_env.reset()

    logging.info(f'Observation Spec: {parallel_env.time_step_spec().observation}')
    logging.info(f'Reward Spec:: {parallel_env.time_step_spec().reward}')
    logging.info(f'Action Spec: {parallel_env.action_spec()}')

    # Construção da Rede Q
    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(parallel_env.action_spec())
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
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(train_env.time_step_spec(), train_env.action_spec(), q_network=q_net, optimizer=optimizer, td_errors_loss_fn=common.element_wise_squared_loss, train_step_counter=train_step_counter)
    agent.initialize()

    # Otimizações com TF function
    agent.train = common.function(agent.train)
    returns = []
    avg_return = parallel_compute_avg_return(train_env, agent.policy, num_parallel_envs, num_eval_episodes)
    returns = [avg_return]
    print(f'Recompensa Média Inicial: {avg_return}')

    logging.info('Iniciando a coleta de experiências')
    # Reseta o ambiente
    time_step = train_env.reset()
    logging.info(f'Time step: {time_step}')

    # Inicializa o Driver para coleta de experiências
    # Replay Buffer e Driver
    # Inicializa o Replay Buffer com batch_size correto
    replay_buffer = TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=num_parallel_envs,  # Importante para ambientes paralelos
        max_length=replay_buffer_max_length)

    # Inicializa o Driver para coleta de experiências
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)

    # Preparação para o Treinamento
    dataset = replay_buffer.as_dataset(num_parallel_calls=num_parallel_envs, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    logging.info('Iniciando Treinamento...')
    # Loop de Treinamento Simplificado
    for _ in range(num_iterations):
        # Coleta de experiências
        time_step, _ = collect_driver.run(time_step)

        # Amostra um batch de dados do buffer e atualiza a rede do agente
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()
       
        
        if step % log_interval == 0:
            print(f'step = {step}: loss = {train_loss}')            
            logging.info(f'step = {step}: loss = {train_loss}')

        if step % eval_interval == 0:
            print(f'step = {step}: loss = {train_loss}')
            logging.info(f'step = {step}: loss = {train_loss}')
            avg_return = parallel_compute_avg_return(train_env, agent.policy, num_parallel_envs, num_eval_episodes)
            logging.info(f'step = {step}: Média de Recompensa = {avg_return}')
            print(f'step = {step}:Média de Recompensa = {avg_return}')
            returns.append(avg_return)

    logging.info('Treinamento Concluído!')
    # Plotagem dos Resultados
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')

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

def load_instance_with_chunks(data_path, chunksize):
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

    return df_final   

if __name__ == '__main__':

    # Configuração do Logging
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f'Reinforcement-Learning/logs/{current_time}_RL-log.txt'    
    logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO, format='[%(levelname)s]\t%(asctime)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    
    logging.info('Carregando o dataframe raw')    
    df = load_instance_with_chunks(data_path ='/home/dataset', chunksize=1000)    

    
       
    try:
        tf_multiprocessing.enable_interactive_mode()
    except:
        logging.error('Erro ao habilitar o modo interativo do TensorFlow')
        pass
    
    logging.info('Amostrando 10% do dataframe para treinamento...')
    df_train = df.sample(frac=0.01, random_state=42).reset_index(drop=True)  
    

    memory = df_train.memory_usage(deep=True)
    logging.info(f'Memória do dataframe: {memory}')

    logging.info(f'Dataframe info: {df_train.info()}')

    plot_estados(df_train)      
    
    '''try:
        logging.info('Validando o ambiente...')
        utils.validate_py_environment(parallel_env, episodes=5)
        #utils.validate_py_environment(env_py, episodes=5) 
        logging.info('Ambiente validado com sucesso!')       
    except Exception as e:
        logging.error(f'Erro ao validar o ambiente: {e}')'''
        

    main(df_train)

    