import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class env3W(py_environment.PyEnvironment):
    def __init__(self, dataframe):
        super(env3W, self).__init__()
        self._dataframe = dataframe
        self._index = 0
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(dataframe.shape[1]-1,), dtype=np.float32, name='observation')
        self._state = None
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._index = 0
        self._update_state()
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._index += 1
        if self._index >= len(self._dataframe):
            self._episode_ended = True

        reward = self._calculate_reward(action)

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward=reward)
        else:
            self._update_state()
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=1.0)

    def _update_state(self):
        # Exclui as colunas especificadas e atualiza o estado atual
        row = self._dataframe.iloc[self._index]
        self._state = row.drop(['timestamp', 'label', 'well', 'id', 'class']).values


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




