
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
        np.random.seed(42)  # Para reprodutibilidade
        shuffled_indices = np.random.permutation(len(dataset_train_scaled))
        dataset_shuffled = dataset_train_scaled[shuffled_indices]
        
        split_datasets = np.array_split(dataset_shuffled, n_envs)        
        
        # Crie o ambiente (modifique conforme necessário para seu caso de uso)        
        return make_vec_env(lambda: self.create_env(split_datasets.pop(0)), n_envs= n_envs)

    def env3W_dqn(self, dataset_train_scaled, n_envs):
           
        # Crie o ambiente (modifique conforme necessário para seu caso de uso)        
        envs = self.envs_random(dataset_train_scaled, n_envs)

        # Instanciar o modelo DQN com a política MLP
        model = DQN(
            policy=MlpPolicy,
            env=envs,
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

        # Treinar o modelo
        model.learn(total_timesteps=int(1.2e5))

        # Salvar o modelo
        model.save(self.path_save)

        envs.close()

    def env3W_dqn_eval(self, dataset_test_scaled, n_envs, n_eval_episodes=1):
        
        envs = self.envs_random(dataset_test_scaled, n_envs)
        model = DQN.load(self.path_save)
        # Inicializa listas para armazenar os dados de cada episódio
        correct_predictions_list = []
        total_predictions_list = []

        for episode in range(n_eval_episodes):
            obs = envs.reset()
            done = [False]

            # Reseta contadores para o episódio atual
            correct_predictions = 0
            total_predictions = 0

            while not all(done):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = envs.step(action)

                for reward in rewards:
                    if reward > 0:
                        correct_predictions += 1
                    total_predictions += 1

            # Atualiza a mensagem de log para incluir a recompensa e a porcentagem de acerto
            if total_predictions > 0:  # Evita divisão por zero
                percentage_correct = correct_predictions / total_predictions
                #logging.info(f"[{episode}] Recompensa: {correct_predictions}, Porcentagem de acerto: {percentage_correct:.2f}.")
                #print(f"[{episode}] Recompensa: {correct_predictions}, Porcentagem de acerto: {percentage_correct:.2f}.")
            else:
                logging.info(f"Sem predições.")
                #print(f"[{episode}] Sem predições.")

            # Armazena os dados do episódio atual nas listas
            correct_predictions_list.append(correct_predictions)
            total_predictions_list.append(total_predictions)

        # Calcula a precisão geral
        accuracy = sum(correct_predictions_list) / sum(total_predictions_list) if sum(total_predictions_list) > 0 else 0

        
        return accuracy