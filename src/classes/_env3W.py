import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class env3W(py_environment.PyEnvironment):
    '''
    Essa classe é um Modelo-Baseado em Ambiente para o problema de detecção de falhas em poços de petróleo.

     1. O ambiente é um repositório di github https://github.com/petrobras/3W
     2. O ambiente é composto por dados de seis poços de petróleo, com cinco variáveis (observações) de entrada ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'] e um rótulo indentificador de falha [class]
     3. O ambiente é um ambiente de simulação, onde o agente pode escolher entre duas ações: 0 - Não Detectado ou 1 - Detectado para cada observação
     4. A recompensa é calculada com base na ação escolhida e no rótulo de falha [class]:

        Estados:
        - rótulo de falha: 0 - Estado Normal
        - rótulo de falha: 1 a 8 - Estável de Anomalia (Falha)
        - rótulo de falha: 101 a 108 - Transiente de Anomalia (Falha)

        Ações/Recompensas:
        - Se o rótulo de falha for 0, a recompensa é 1 se a ação for 0, caso contrário, a recompensa é -100
        - Se o rótulo de falha estiver entre 1 e 8, a recompensa é -100 se a ação for 0, caso contrário, a recompensa é 100
        - Se o rótulo de falha estiver entre 101 e 108, a recompensa é -10 se a ação for 0, caso contrário, a recompensa é 10
    
    '''
    def __init__(self, dataframe):
        super(env3W, self).__init__()
        self._dataframe = dataframe
        self._index = 0
        # Ação: 0 - Não Deectado ou 1 - Detectado
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')                  
        num_features = len(['P-PDG', 'P-TPTP', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'])
        self._observation_spec = array_spec.BoundedArraySpec(shape=(num_features,), dtype=np.float32, name='observation')
        self._state = None
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _update_state(self):
        # Seleciona diretamente as colunas necessárias do dataframe
        columns_needed = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
        row = self._dataframe.iloc[self._index][columns_needed]
        self._state = row.values

    def _reset(self):
        self._index = 0
        self._update_state()
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            # Isso deve retornar uma chamada para reset(), não a terminação diretamente
            return self.reset()
        
        if self._index >= len(self._dataframe) - 1:
            # Quando chegar na última linha, marque o episódio como terminado e pare.
            self._episode_ended = True
            reward = self._calculate_reward(action)
            return ts.termination(np.array(self._state, dtype=np.float32), reward=reward)
        
        # Lógica para processar a ação e atualizar o estado aqui
        self._index += 1
        self._update_state()
        reward = self._calculate_reward(action)
        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward=reward)
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=1.0)


    


    def _calculate_reward(self, action):
        class_value = self._dataframe.iloc[self._index]['class']
        if class_value == 0:
            return 1 if action == 0 else -100
        elif class_value in range(1, 9):
            return -100 if action == 0 else 100
        elif class_value in range(101, 109):
            return -10 if action == 0 else 10
        else:
            # Define uma recompensa padrão para qualquer outro valor de class não especificado
            return 0




