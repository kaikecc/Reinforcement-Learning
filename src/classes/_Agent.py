from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from classes._Env3WGym import Env3WGym
import logging
import numpy as np

class Agent:
    def __init__(self, path_save):        
        self.path_save = path_save

    def create_env(self, dataset_part):
        return Env3WGym(dataset_part)

    def envs_random(self, dataset_train_scaled, n_envs):
        #np.random.seed(42)  # Para reprodutibilidade
        shuffled_indices = np.random.permutation(len(dataset_train_scaled))
        dataset_shuffled = dataset_train_scaled[shuffled_indices]
        split_datasets = np.array_split(dataset_shuffled, n_envs)

        return make_vec_env(lambda: self.create_env(split_datasets.pop(0)), n_envs=n_envs)

    def env3W_dqn(self, dataset_train_scaled, n_envs):
        envs = self.envs_random(dataset_train_scaled, n_envs)
        model = DQN(
            MlpPolicy, envs,
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
            verbose=1,
            device='auto'
        )
        model.learn(total_timesteps=int(1.2e5))
        model.save(self.path_save)
        envs.close()

    def env3W_dqn_eval(self, dataset_test_scaled, n_envs, n_eval_episodes=1):
        logging.info(f"Avaliando o modelo {self.path_save} com {n_eval_episodes} episódios.")
        envs = self.envs_random(dataset_test_scaled, n_envs)
        model = DQN.load(self.path_save)

        correct_predictions, total_predictions = 0, 0
        for episode in range(n_eval_episodes):
            obs = envs.reset()
            dones = np.array([False] * n_envs)  # Inicializa um array de "done" para cada ambiente
            while not dones.all():  # Continua até que todos os ambientes estejam concluídos
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = envs.step(action)
                correct_predictions += sum(reward > 0 for reward in rewards)
                total_predictions += len(rewards)

        accuracy = correct_predictions / total_predictions if total_predictions else 0
        logging.info(f"Acurácia: {accuracy:.2f}")

        return accuracy, model

