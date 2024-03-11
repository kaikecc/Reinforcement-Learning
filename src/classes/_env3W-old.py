
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
import logging

class env3W(py_environment.PyEnvironment):
    '''
    Essa classe é um Modelo-Livre do Ambiente em python para o problema de detecção de falhas em poços de petróleo.
    O dataframe é fornecido como entrada para o ambiente com mais de 10 milhões de registros.

     1. O ambiente é um repositório di github https://github.com/petrobras/3W
     2. O ambiente é composto por dados de seis poços de petróleo, com cinco variáveis (observações) de entrada ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'] e um rótulo indentificador de falha [class]
     3. O ambiente é um ambiente de simulação, onde o agente pode escolher entre duas ações: 0 - Não Detectado ou 1 - Detectado para cada observação
     4. A recompensa é calculada com base na ação escolhida e no rótulo de falha [class]:

        Estados:
        - rótulo de falha: 0 - Estado Normal
        - rótulo de falha: 1 a 8 - Estável de Anomalia (Falha)
        - rótulo de falha: 101 a 108 - Transiente de Anomalia (Falha)

        Ações/Recompensas:
        - Se o rótulo de falha for 0, a recompensa é 0,01 se a ação for 0, caso contrário, a recompensa é -1
        - Se o rótulo de falha estiver entre 1 e 8, a recompensa é -1 se a ação for 0, caso contrário, a recompensa é 1
        - Se o rótulo de falha estiver entre 101 e 108, a recompensa é -0,1 se a ação for 0, caso contrário, a recompensa é 0,1

    5. Implementação de Chunking para lidar com grandes conjuntos de dados
        - self.chunk_size: Define o tamanho de cada chunk de dados.
        - self.current_chunk_index: Controla o índice do chunk atual.
        - self.load_data_chunk(): Um novo método para carregar um chunk de dados baseado no self.current_chunk_index.
        - https://pandas.pydata.org/docs/user_guide/scale.html

    '''
    def __init__(self, dataframe, chunk_size=10000):
        super(env3W, self).__init__()
        self._dataframe = dataframe
        self.chunk_size = chunk_size
        self.current_chunk_index = 0
               

        # Ação: 0 - Não Deectado ou 1 - Detectado
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self.columns_needed = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
        self.load_data_chunk()  # Carrega o primeiro chunk de dados.
        num_features = len(self.columns_needed)
        self._observation_spec = array_spec.BoundedArraySpec(shape=(num_features,), dtype=np.float32, name='observation')
        row = self._dataframe.iloc[self._index][self.columns_needed]
        self._state = row.values
        self._episode_ended = False

    def load_data_chunk(self):
        start_index = self.current_chunk_index * self.chunk_size
        end_index = start_index + self.chunk_size
        self.current_chunk = self._dataframe.iloc[start_index:end_index]
        logging.info(f'Carregando chunk {self.current_chunk_index} de {len(self._dataframe) // self.chunk_size}')
        self._index = 0  # Reset index para o novo chunk
        self._update_state()  # Atualiza o estado para o novo chunk

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


    def _update_state(self):
        self._state = np.array(self.current_chunk.iloc[self._index][self.columns_needed], dtype=np.float32)

    def _reset(self):
        logging.info('Reiniciando o ambiente.')
        self.current_chunk_index = 0
        self.load_data_chunk()
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))


    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        if not 0 <= action <= 1:
            logging.error(f'Ação {action} não é válida. A ação deve ser 0 ou 1.')
            return ts.termination(self._state, reward=-1)

        self._index += 1  # Avança o índice para o próximo estado

        # Verifica se alcançamos o final do chunk atual
        if self._index >= len(self.current_chunk)-1:
            # Verifica se existem mais chunks para processar
            if (self.current_chunk_index + 1) * self.chunk_size < len(self._dataframe):
                self.current_chunk_index += 1
                self.load_data_chunk()
            else:
                # Se não houver mais dados, termina o episódio
                self._episode_ended = True
                reward = self._calculate_reward(action)
                logging.info('Episódio concluído.')
                return ts.termination(self._state, reward)
        else:
            self._update_state()  # Atualiza o estado com o novo índice

        reward = self._calculate_reward(action)
        return ts.transition(self._state, reward=reward, discount=1.0)


    def _calculate_reward(self, action):
        class_value = self.current_chunk.iloc[self._index]['class']
        if class_value == 0:
            return 0.01 if action == 0 else -1
        elif 1 <= class_value <= 8:
            return -1 if action == 0 else 1
        elif 101 <= class_value <= 108:
            return -0.1 if action == 0 else 0.1
        else:
            return 0


