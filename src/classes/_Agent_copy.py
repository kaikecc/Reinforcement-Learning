from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from classes._Env3WGym_copy import Env3WGym
import logging
import numpy as np
import os

class Agent:
    def __init__(self, path_save):        
        self.path_save = path_save

    def create_env(self, dataset_part):
        return Env3WGym(dataset_part)

    def envs_random(self, dataset_train_scaled, n_envs):
        # Ensure the lambda function captures the current value of 'dataset_train_scaled'
        env_lambda = lambda dataset=dataset_train_scaled: self.create_env(dataset)
        return make_vec_env(env_lambda, n_envs=n_envs)

    def env3W_dqn(self, dataset_train_scaled, n_envs):
        envs = self.envs_random(dataset_train_scaled, n_envs)
        model = DQN(
            MlpPolicy, envs,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=10000,
            batch_size=64,
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
            verbose=0,
            device='auto'
        )
        model.learn(total_timesteps=int(1.2e5))
        model.save(f'{self.path_save}_DQN_Env3W')
        envs.close()

    def env3W_dqn_eval(self, dataset_test_scaled, n_envs, n_eval_episodes=1):
        logging.info(f"Evaluating model {self.path_save} with {n_eval_episodes} episodes.")
        envs = self.envs_random(dataset_test_scaled, n_envs)
        model = DQN.load(f'{self.path_save}_DQN_Env3W')

        TP, FP, TN, FN = 0, 0, 0, 0
        reward_thresholds = [0.01, 1.0, -1.0]
        atol = 1e-6

        for _ in range(n_eval_episodes):
            obs = envs.reset()
            dones = np.array([False] * n_envs)
            
            while not dones.all():
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, _infos = envs.step(action)
                
                for reward in rewards:
                    close_values = np.isclose(reward, reward_thresholds, atol=atol)
                    if close_values[0]:
                        TN += 1
                    elif close_values[1]:
                        TP += 1
                    elif close_values[2]:
                        FP += 1
                        FN += 1

        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) else 0
        return accuracy, model
    
    def env3W_ppo(self, dataset_test_scaled, n_envs):
        # Cria ambientes aleatórios a partir do conjunto de dados fornecido
        envs = self.envs_random(dataset_test_scaled, n_envs)

        # Define o caminho para salvar os checkpoints
        checkpoint_dir = os.path.join(self.path_save, 'ppo_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)  # Cria o diretório se não existir

        model = PPO('MlpPolicy', envs, verbose=0,
                    learning_rate=1e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.0,
                    tensorboard_log=os.path.join(self.path_save, 'tensorboard_logs'))

        # Callback para salvar o modelo periodicamente
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir,
                                                  name_prefix='PPO_Env3W')

        # Treina o modelo
        model.learn(total_timesteps=int(1.2e5), callback=checkpoint_callback)

        # Salva o modelo final
        model.save(os.path.join(self.path_save, '_PPO_Env3W'))

        return model

    def env3W_ppo_eval(self, dataset_eval_scaled, n_envs, n_eval_episodes=1):
        """
        Avalia o desempenho do modelo PPO em ambientes gerados a partir de um conjunto de dados de avaliação.
        Agora também calcula a "acurácia" como a proporção de episódios concluídos com sucesso.

        Args:
        - dataset_eval_scaled: Conjunto de dados escalado para avaliação.
        - n_envs: Número de ambientes paralelos para avaliação.
        - n_eval_episodes: Número de episódios de avaliação por ambiente.

        Returns:
        - accuracy: A proporção de episódios concluídos com sucesso.
        """       
        
        logging.info(f"Avaliando o modelo {self.path_save} com {n_eval_episodes} episódios.")
        envs = self.envs_random(dataset_eval_scaled, n_envs)
        model = PPO.load(os.path.join(self.path_save, '_PPO_Env3W'))

         # Inicialização da matriz de confusão
        TP, FP, TN, FN = 0, 0, 0, 0
        # Reduzindo chamadas para np.isclose pré-definindo os valores de recompensa
        reward_thresholds = [0.01, 0.1, 1.0, -1.0, -0.1]
        atol = 1e-6

        for _ in range(n_eval_episodes):
            obs = envs.reset()
            dones = np.array([False] * n_envs)  # Inicializa um array de "done" para cada ambiente
            while not dones.all():  # Continua até que todos os ambientes estejam concluídos
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = envs.step(action)                
                for reward in rewards:
                    # Reduzindo a quantidade de vezes que np.isclose é chamado
                    close_values = np.isclose(reward, reward_thresholds, atol=atol)
                    if close_values[0]:
                        TN += 1
                    elif close_values[1] or close_values[2]:
                        TP += 1
                    elif close_values[3] or close_values[4]:
                        FP += 1  
                        FN += 1

        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) else 0

        return accuracy, model