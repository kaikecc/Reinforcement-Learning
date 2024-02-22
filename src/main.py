import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.environments import TFPyEnvironment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from classes._env3W import env3W  # Ajuste para o caminho correto da sua classe de ambiente

# Carregar o DataFrame
df = pd.read_csv('analyses data\\real_instances_594.csv')

# Cria instâncias do ambiente
train_py_env = env3W(df)
eval_py_env = env3W(df)

train_env = TFPyEnvironment(train_py_env)
eval_env = TFPyEnvironment(eval_py_env)

# Criação da Rede Q usando QNetwork
fc_layer_params = (100, 50)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

# Configuração do Agente DQN
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()
