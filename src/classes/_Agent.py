from stable_baselines3 import DQN, PPO
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
import os
import csv
import json
from stable_baselines3.common.logger import configure
import tensorflow as tf
import os
from classes._Env3WGym import Env3WGym
import logging
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.logger import Logger
from torch.utils.tensorboard import SummaryWriter  # Importando o TensorBoard

class TensorboardCallback(BaseCallback):
    def __init__(self, model, referencia_top_score, caminho_salvar_modelo="./model_DQN", verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.model = model
        self.referencia_top_score = referencia_top_score  # Uma referência à variável global top_score
        self.caminho_salvar_modelo = caminho_salvar_modelo

    def _on_step(self) -> bool:
        # Recupera de forma segura 'score' de 'infos' se existir, caso contrário, usa None como padrão
        score = self.locals.get("infos")[0].get("score") if self.locals.get("infos") else None

        # Procede apenas com o registro e comparação do score se um score estiver realmente presente
        if score is not None:
            self.logger.record('a_score', score)
            if self.referencia_top_score[0] is None or score > self.referencia_top_score[0]:
                self.referencia_top_score[0] = score  # Atualiza o top_score global
                self.model.save(self.caminho_salvar_modelo)
        
        # Registro condicional no log a cada 10.000 timesteps
        if self.num_timesteps % 10000 == 0:
            self.logger.dump(self.num_timesteps)
        
        return True

class DQNMetricsCSVCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(DQNMetricsCSVCallback, self).__init__(verbose)
        self.save_path = save_path
        # Verifica e cria o diretório se necessário
        os.makedirs(self.save_path, exist_ok=True)
        self.file_path = os.path.join(self.save_path, 'metrics.csv')
        # Verifica se o arquivo já existe para evitar recriar o cabeçalho
        self.file_exists = os.path.isfile(self.file_path)

        # Inicialização dos atributos faltantes
        self.reward_max = float('-inf')
        self.reward_min = float('inf')
        self.reward_sum = 0
        self.reward_count = 0
        self.metrics = []  # Lista para armazenar as métricas coletadas

    def _on_step(self) -> bool:
        # Atualiza e registra o valor de epsilon
        epsilon = self.model.exploration_rate
        self.logger.record('epsilon', epsilon)

        # Atualização das estatísticas de recompensa
        rewards = self.locals['rewards'] if 'rewards' in self.locals else []
        for reward in rewards:
            self.reward_max = max(self.reward_max, reward)
            self.reward_min = min(self.reward_min, reward)
            self.reward_sum += reward
            self.reward_count += 1
        
        # Se houver recompensas, atualiza e registra as métricas de recompensa
        if self.reward_count > 0:
            reward_avg = self.reward_sum / self.reward_count
            self.logger.record('reward_avg', reward_avg)
            self.logger.record('reward_max', self.reward_max)
            self.logger.record('reward_min', self.reward_min)

        # Armazena as métricas coletadas
            self.metrics.append({
                'step': self.num_timesteps,
                'epsilon': epsilon,
                'reward_avg': reward_avg,
                'reward_max': self.reward_max,
                'reward_min': self.reward_min
            })


        # Chama o dump do logger para atualizar as métricas no TensorBoard
        if self.num_timesteps % 10000 == 0:
            self.logger.dump(self.num_timesteps)
            self._save_metrics()

        return True

    def _save_metrics(self):
        # Converte métricas para serem serializáveis em JSON
        serializable_metrics = self._convert_to_serializable(self.metrics)
        
        # Salva as métricas acumuladas em um arquivo JSON
        file_path = os.path.join(self.save_path, 'metrics.json')
        with open(file_path, 'w') as fp:
            json.dump(serializable_metrics, fp, indent=4)
    
    def _convert_to_serializable(self, data):
        # Função para converter recursivamente elementos não serializáveis
        if isinstance(data, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(data)
        elif isinstance(data, (np.float_, np.float16, np.float32, 
                               np.float64)):
            return float(data)
        elif isinstance(data, (np.ndarray,)): # se for um array numpy, converte para lista
            return data.tolist()
        elif isinstance(data, (list,)):
            return [self._convert_to_serializable(item) for item in data]
        elif isinstance(data, (dict,)):
            return {key: self._convert_to_serializable(value) for key, value in data.items()}
        else:
            return data

    
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

        return make_vec_env(lambda: self.create_env(split_datasets.pop(0)), n_envs=n_envs)
    
    def env3W_dqn(self, dataset_test_scaled, n_envs):
        # Cria ambientes aleatórios a partir do conjunto de dados fornecido
        envs = self.envs_random(dataset_test_scaled, n_envs)
        # tensorboard --logdir=tensorboard_logs

        # Ajusta o caminho para subir um nível com os.path.dirname e, em seguida, entra no diretório 'tensorboard_logs'
        logdir_base = os.path.dirname(self.path_save)  # Sobe um nível (para '..\\models\\Abrupt Increase of BSW')
        logdir = os.path.join(logdir_base, 'tensorboard_logs')  # Entra em '..\\models\\Abrupt Increase of BSW\\tensorboard_logs'

        # Cria o diretório se não existir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        replaydir = os.path.join(self.path_save, 'replay_buffer')

        # Cria o diretório se não existir
        if not os.path.exists(replaydir):
            os.makedirs(replaydir)


        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")

        # Cria o diretório se não existir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

         # Cria o diretório se não existir
        if not os.path.exists(replaydir):
            os.makedirs(replaydir)

        # Define o caminho para salvar os checkpoints
        checkpoint_dir = os.path.join(self.path_save, 'dqn_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)  # Cria o diretório se não existir

        model = DQN(
            MlpPolicy, 
            envs,
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
            verbose=1,
            tensorboard_log=logdir,  # Usa o caminho ajustado para os logs do TensorBoard            
            device='auto'
        )

        
        final_model_path = os.path.join(self.path_save, '_DQN')
        # Within env3W_dqn, before starting training
        top_score = [None]  # Use a list for mutability
        top_score = [None]  # Supondo que top_score é gerenciado globalmente
        tensorboard_callback = TensorboardCallback(model=model, referencia_top_score=top_score, caminho_salvar_modelo=final_model_path, verbose=1)

        metrics_callback = DQNMetricsCSVCallback(save_path=final_model_path, verbose=1)
        


        # Treina o modelo
        TIMESTEPS = 100000  # Usando um número inteiro diretamente é mais claro
        for i in range(1, 10):            
            model.learn(total_timesteps=TIMESTEPS, log_interval = 4, reset_num_timesteps=False, tb_log_name="DQN", callback=[metrics_callback, tensorboard_callback])
            # Ajuste para tornar o nome do arquivo salvo mais informativo
            model_path = os.path.join(self.path_save, f'DQN_iteration_{i}_timesteps_{TIMESTEPS*i}')
            model.save(model_path)
       
        # Salva o modelo final
        model.learn(total_timesteps=TIMESTEPS, log_interval = 4, reset_num_timesteps=False, tb_log_name="DQN", callback=[metrics_callback, tensorboard_callback])
        
        model.save(final_model_path)
        logging.info(f"Modelo final salvo em {final_model_path}")
        final_replay_path = os.path.join(replaydir, 'dqn_save_replay_buffer')
        model.save_replay_buffer(final_replay_path)   
        logging.info(f"Replay Buffer de Aprendizado Contínuo salvo em {final_replay_path}")     
        return model, final_replay_path
    
    def env3W_dqn_eval(self, dataset_test_scaled, model, n_envs, n_eval_episodes=1):
        logging.info(f"Avaliando o modelo {self.path_save} com {n_eval_episodes} episódios.")
        envs = self.envs_random(dataset_test_scaled, n_envs)
        #model = DQN.load(f'{self.path_save}_DQN_Env3W')
        # Inicialização da matriz de confusão
        TP, FP, TN, FN = 0, 0, 0, 0

        # Reduzindo chamadas para np.isclose pré-definindo os valores de recompensa
        reward_thresholds = [0.01, 0.1, 1.0, -1.0, -0.1]
        atol = 1e-6

         # Ajusta o caminho para subir um nível com os.path.dirname e, em seguida, entra no diretório 'tensorboard_logs'
        logdir_base = os.path.dirname(self.path_save)  # Sobe um nível (para '..\\models\\Abrupt Increase of BSW')
        logdir = os.path.join(logdir_base, 'tensorboard_logs')  # Entra em '..\\models\\Abrupt Increase of BSW\\tensorboard_logs'
        final_model_path = os.path.join(logdir, 'DQN')
        # Cria uma instância do SummaryWriter para registrar logs
        writer = SummaryWriter(final_model_path)

        for episode in range(n_eval_episodes):
            obs = envs.reset()
            dones = np.array([False] * n_envs)
            
            while not dones.all():
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = envs.step(action)
                
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
            writer.add_scalar('Accuracy', accuracy, episode)
        writer.close() 
        return accuracy
     
    def env3W_ppo(self, dataset_test_scaled, n_envs):
        # Cria ambientes aleatórios a partir do conjunto de dados fornecido
        envs = self.envs_random(dataset_test_scaled, n_envs)
        # tensorboard --logdir=tensorboard_logs

        # Ajusta o caminho para subir um nível com os.path.dirname e, em seguida, entra no diretório 'tensorboard_logs'
        logdir_base = os.path.dirname(self.path_save)  # Sobe um nível (para '..\\models\\Abrupt Increase of BSW')
        logdir = os.path.join(logdir_base, 'tensorboard_logs')  # Entra em '..\\models\\Abrupt Increase of BSW\\tensorboard_logs'

        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")

        # Cria o diretório se não existir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        # Define o caminho para salvar os checkpoints
        checkpoint_dir = os.path.join(self.path_save, 'ppo_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)  # Cria o diretório se não existir

        model = PPO('MlpPolicy', envs, verbose=1,
                    learning_rate=1e-4,
                    n_steps=2048,
                    batch_size=32,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.0,
                    tensorboard_log=logdir)

        # Callback para salvar o modelo periodicamente
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir,
                                                  name_prefix='PPO')

        # Treina o modelo
        TIMESTEPS = 100000  # Usando um número inteiro diretamente é mais claro
        for i in range(1, 10):            
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=checkpoint_callback)
            # Ajuste para tornar o nome do arquivo salvo mais informativo
            model_path = os.path.join(self.path_save, f'PPO_iteration_{i}_timesteps_{TIMESTEPS*i}')
            model.save(model_path)
       
        # Salva o modelo final
        model.save(os.path.join(self.path_save, '_PPO'))
        logging.info(f"Modelo final salvo em {os.path.join(self.path_save, '_PPO')}")        

        return model

    def env3W_ppo_eval(self, dataset_eval_scaled, model, n_envs, n_eval_episodes=1):
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
        #model = PPO.load(os.path.join(self.path_save, '_PPO_Env3W'))

         # Ajusta o caminho para subir um nível com os.path.dirname e, em seguida, entra no diretório 'tensorboard_logs'
        logdir_base = os.path.dirname(self.path_save)  # Sobe um nível (para '..\\models\\Abrupt Increase of BSW')
        logdir = os.path.join(logdir_base, 'tensorboard_logs')  # Entra em '..\\models\\Abrupt Increase of BSW\\tensorboard_logs'
        final_model_path = os.path.join(logdir, 'PPO')
        # Cria o diretório se não existir
        if not os.path.exists(final_model_path):
            os.makedirs(final_model_path)

        # Cria uma instância do SummaryWriter para registrar logs
        writer = SummaryWriter(final_model_path)
        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")

        # Cria o diretório se não existir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        
         # Inicialização da matriz de confusão
        TP, FP, TN, FN = 0, 0, 0, 0
        # Reduzindo chamadas para np.isclose pré-definindo os valores de recompensa
        reward_thresholds = [0.01, 0.1, 1.0, -1.0, -0.1]
        atol = 1e-6

        for episode in range(n_eval_episodes):
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
            writer.add_scalar('Accuracy', accuracy, episode)

        
        # Fechando o SummaryWriter após o registro
        writer.close() 

        return accuracy

    def env3W_a2c(self, dataset_test_scaled, n_envs):
        # Cria ambientes aleatórios a partir do conjunto de dados fornecido
        envs = self.envs_random(dataset_test_scaled, n_envs)

        # Ajusta o caminho para subir um nível com os.path.dirname e, em seguida, entra no diretório 'tensorboard_logs'
        logdir_base = os.path.dirname(self.path_save)
        logdir = os.path.join(logdir_base, 'tensorboard_logs')

        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")

        # Cria o diretório se não existir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        # Define o caminho para salvar os checkpoints
        checkpoint_dir = os.path.join(self.path_save, 'a2c_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = A2C('MlpPolicy', envs, verbose=1,
                    learning_rate=1e-4,
                    n_steps=5, # 5
                    gamma=0.99,
                    gae_lambda=0.95,
                    ent_coef=0.0,
                    tensorboard_log=logdir)

        # Callback para salvar o modelo periodicamente
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir,
                                                name_prefix='A2C')

        # Treina o modelo
        TIMESTEPS = 100000
        for i in range(1, 10):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
            model_path = os.path.join(self.path_save, f'A2C_iteration_{i}_timesteps_{TIMESTEPS*i}')
            model.save(model_path)

        # Salva o modelo final
        model.save(os.path.join(self.path_save, 'A2C'))
        logging.info(f"Modelo final salvo em {os.path.join(self.path_save, '_A2C')}")
        
        return model

    def env3W_a2c_eval(self, dataset_eval_scaled, model, n_envs, n_eval_episodes=1):
        """
        Avalia o desempenho do modelo A2C em ambientes gerados a partir de um conjunto de dados de avaliação.
        Agora também calcula a "acurácia" como a proporção de episódios concluídos com sucesso.

        Args:
        - dataset_eval_scaled: Conjunto de dados escalado para avaliação.
        - model: Modelo A2C carregado para avaliação.
        - n_envs: Número de ambientes paralelos para avaliação.
        - n_eval_episodes: Número de episódios de avaliação por ambiente.

        Returns:
        - accuracy: A proporção de episódios concluídos com sucesso.
        """

        logging.info(f"Avaliando o modelo A2C {self.path_save} com {n_eval_episodes} episódios.")
        envs = self.envs_random(dataset_eval_scaled, n_envs)

        logdir_base = os.path.dirname(self.path_save)
        logdir = os.path.join(logdir_base, 'tensorboard_logs')

        final_model_path = os.path.join(logdir, 'A2C')

        if not os.path.exists(final_model_path):
            os.makedirs(final_model_path)

        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        writer = SummaryWriter(final_model_path)

        TP, FP, TN, FN = 0, 0, 0, 0
        reward_thresholds = [0.01, 0.1, 1.0, -1.0, -0.1]
        atol = 1e-6

        for episode in range(n_eval_episodes):
            obs = envs.reset()
            dones = np.array([False] * n_envs)
            while not dones.all():
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = envs.step(action)
                for reward in rewards:
                    close_values = np.isclose(reward, reward_thresholds, atol=atol)
                    if close_values[0]:
                        TN += 1
                    elif close_values[1] or close_values[2]:
                        TP += 1
                    elif close_values[3] or close_values[4]:
                        FP += 1
                        FN += 1
            accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) else 0
            writer.add_scalar('Accuracy', accuracy, episode)

        writer.close()

        return accuracy
    
    def env3W_dqn_cl(self, model_agent, dataset_cl, replaydir, n_envs):
        '''
        Continual Learning com DQN        
        '''
        envs = self.envs_random(dataset_cl, n_envs)
        # tensorboard --logdir=tensorboard_logs

        # Ajusta o caminho para subir um nível com os.path.dirname e, em seguida, entra no diretório 'tensorboard_logs'
        logdir_base = os.path.dirname(self.path_save)  # Sobe um nível (para '..\\models\\Abrupt Increase of BSW')
        logdir = os.path.join(logdir_base, 'tensorboard_logs')  # Entra em '..\\models\\Abrupt Increase of BSW\\tensorboard_logs'

        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{logdir}'")

        # Cria o diretório se não existir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        # Define o caminho para salvar os checkpoints
        checkpoint_dir = os.path.join(self.path_save, 'dqn-cl_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)  # Cria o diretório se não existir
        
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoint_dir,
                                                  name_prefix='DQN-CL_Env3W')
        
        model_agent.load_replay_buffer(replaydir)
        model_agent.set_env(envs)
        model_agent._last_obs = None
        model_agent.learn(total_timesteps=100000, log_interval = 4, reset_num_timesteps = False, tb_log_name="DQN-CL", callback=checkpoint_callback)
        # Salva o modelo final        
        final_model_path = os.path.join(self.path_save, '_DQN-CL_Env3W')
        model_agent.save(final_model_path)
        final_replay_path = os.path.join(replaydir, 'dqn_save_replay_buffer')
        model_agent.save_replay_buffer(final_replay_path)  
        logging.info(f"Modelo de Aprendizado Contínuo salvo em {replaydir}")  

        return model_agent