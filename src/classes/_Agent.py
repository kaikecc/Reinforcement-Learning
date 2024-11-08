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
from stable_baselines3.common.logger import Logger, KVWriter, TensorBoardOutputFormat
from torch.utils.tensorboard import SummaryWriter  # Importando o TensorBoard
from tensorboard import notebook
import tensorboard.program
import webbrowser  # Importar a biblioteca webbrowser



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

class MetricsCSVCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(MetricsCSVCallback, self).__init__(verbose)
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
        epsilon = None  # Ou 0, dependendo do que faz sentido para sua aplicação

         # Atualiza e registra o valor de epsilon, se aplicável
        if hasattr(self.model, 'exploration_rate'):
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
        if self.num_timesteps % 1000 == 0:
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
    def __init__(self, path_tensorboard, envs_train, envs_test, TIMESTEPS = 10000, port=6006):        
        

        # Cria ambientes aleatórios a partir do conjunto de dados fornecido        
        self.envs_train =   envs_train    
        self.envs_eval = envs_test
        self.TIMESTEPS = TIMESTEPS
              
        self.n_envs_eval = 1

        self.path_tensorboard = path_tensorboard
        self.logdir = os.path.join(self.path_tensorboard, 'tensorboard_logs')
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)            

        self.port = port  # Store the port
        self.launch_tensorboard()
    
    def launch_tensorboard(self):
        # Launch TensorBoard, and specify the log directory
        logging.getLogger('tensorboard').setLevel(logging.ERROR)
        tb = tensorboard.program.TensorBoard()
        try:
            tb.configure(argv=[None, '--logdir', self.logdir, '--port', str(self.port)])
            url = tb.launch()
            print(f"TensorBoard started at {url}")
            logging.info(f"TensorBoard started at {url}")
            # Abrir o TensorBoard no navegador
            webbrowser.open(url)
        except Exception as e:
            print(f"An error occurred while launching TensorBoard: {e}")
            logging.error(f"An error occurred while launching TensorBoard: {e}")
          
    def env3W_dqn(self, path_save):
        '''
        Deep Q Network (DQN) baseia-se em Fitted Q-Iteration (FQI) e 
        faz uso de diferentes truques para estabilizar o aprendizado 
        com redes neurais: usa um buffer de repetição, uma rede alvo e recorte de gradiente.        
        '''        
        
        replaydir = os.path.join(path_save, 'replay_buffer')

        # Cria o diretório se não existir
        if not os.path.exists(replaydir):
            os.makedirs(replaydir)

        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")
        
        model = DQN(
            MlpPolicy, # O modelo de política a ser usado (MlpPolicy, CnnPolicy,…)
            self.envs_train, # O ambiente a ser usado
            learning_rate=1e-3, # Taxa de aprendizado, pode ser uma função do progresso atual restante (de 1 a 0)
            buffer_size=10000, # Tamanho do buffer de repetição
            learning_starts=1000, # Número de etapas de aprendizado para coletar experiências antes de começar a treinar a rede
            batch_size=32, # Tamanho do buffer de repetição
            tau=1.0, #  Coeficiente de atualização suave (“atualização Polyak”, entre 0 e 1) padrão 1 para atualização forçada
            gamma=0.99, # Fator de desconto
            train_freq=1, # Frequência de treinamento, Atualize o modelo a cada train_freq step.
            gradient_steps=1, #  Quantas etapas de gradiente executar após cada implementação (consulte train_freq) Definir como -1significa executar tantas etapas de gradiente quanto as etapas realizadas no ambiente durante a implementação.
            target_update_interval=1000, #  Atualiza a rede de destino a cada target_update_interval etapa do ambiente.
            exploration_fraction=0.1, # Fração de todo o período de treinamento durante o qual a taxa de exploração é reduzida
            exploration_initial_eps=1.0, # Taxa de exploração inicial
            exploration_final_eps=0.01, # Taxa de exploração final
            max_grad_norm=10, # O valor máximo para o recorte do gradiente
            verbose=0,
            tensorboard_log=self.logdir,  # Usa o caminho ajustado para os logs do TensorBoard            
            device='auto'
        )
        
        final_model_path = os.path.join(path_save, '_DQN')
        # Within env3W_dqn, before starting training
        top_score = [None]  # Supondo que top_score é gerenciado globalmente
        tensorboard_callback = TensorboardCallback(model=model, referencia_top_score=top_score, caminho_salvar_modelo=final_model_path, verbose=0)
        metrics_callback = MetricsCSVCallback(save_path=final_model_path, verbose=0)
        
               
        model.learn(total_timesteps=self.TIMESTEPS, log_interval = 4, reset_num_timesteps=True, tb_log_name="DQN", callback=[metrics_callback, tensorboard_callback])
            
               
        model.save(final_model_path)
        logging.info(f"Modelo final salvo em {final_model_path}")
        final_replay_path = os.path.join(replaydir, 'dqn_save_replay_buffer')
        model.save_replay_buffer(final_replay_path)   
        logging.info(f"Replay Buffer de Aprendizado Contínuo salvo em {final_replay_path}")     
        return model, final_replay_path
    
    def env3W_dqn_eval(self, model, path_save, n_eval_episodes=1):
        
        logging.info(f"Avaliando o modelo {path_save} com {n_eval_episodes} episódios.")
        
        # Inicialização da matriz de confusão
        TP, FP, TN, FN = 0, 0, 0, 0

        # Reduzindo chamadas para np.isclose pré-definindo os valores de recompensa
        reward_thresholds = [0.01, 0.1, 1.0, -1.0, -0.1]
        atol = 1e-6
         
        final_model_path = os.path.join(self.logdir, 'DQN')
        # Cria uma instância do SummaryWriter para registrar logs
        writer = SummaryWriter(final_model_path)

        for episode in range(n_eval_episodes):
            
            obs = self.envs_eval.reset()
            dones = np.array([False] * self.n_envs_eval)
            
            while not dones.all():
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = self.envs_eval.step(action)
                
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
     
    def env3W_ppo(self, path_save):
               
        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")

        # Define o caminho para salvar os checkpoints
        checkpoint_dir = os.path.join(path_save, 'ppo_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Configura o modelo PPO
        model = PPO('MlpPolicy', 
                    self.envs_train, verbose=0, 
                    learning_rate=1e-3, 
                    n_steps=128, 
                    batch_size=32, 
                    n_epochs=10, 
                    gamma=0.99, 
                    gae_lambda=0.95, 
                    clip_range=0.2, 
                    ent_coef=0.0, 
                    tensorboard_log=self.logdir)

        # Callback para salvar o modelo periodicamente
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoint_dir, name_prefix='PPO')
        # Callback personalizado para registrar recompensas no TensorBoard
        
        final_model_path = os.path.join(path_save, '_PPO')
        metrics_callback = MetricsCSVCallback(save_path=final_model_path, verbose=0)

        # Preparando callback de avaliação (descomente e ajuste conforme necessário)
        # eval_env = self.envs_random(dataset_test_scaled, 1)
        # eval_callback = EvalCallback(eval_env, best_model_save_path=checkpoint_dir, log_path=checkpoint_dir, eval_freq=5000, deterministic=True, render=False)

        model.learn(total_timesteps=self.TIMESTEPS, reset_num_timesteps=True, tb_log_name="PPO", callback=[checkpoint_callback, metrics_callback])  # Adicione `eval_callback` à lista de callbacks, se estiver usando
            
        # Salva o modelo final
        final_model_path = os.path.join(path_save, '_PPO')
        model.save(final_model_path)
        logging.info(f"Modelo final salvo em {final_model_path}")

        return model
    
    def env3W_ppo_eval(self, model, path_save, n_eval_episodes=1):
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
        
        logging.info(f"Avaliando o modelo {path_save} com {n_eval_episodes} episódios.")
        
        final_model_path = os.path.join(self.logdir, 'PPO')
        # Cria o diretório se não existir
        if not os.path.exists(final_model_path):
            os.makedirs(final_model_path)

        # Cria uma instância do SummaryWriter para registrar logs
        writer = SummaryWriter(final_model_path)
        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")
       
         # Inicialização da matriz de confusão
        TP, FP, TN, FN = 0, 0, 0, 0
        # Reduzindo chamadas para np.isclose pré-definindo os valores de recompensa
        reward_thresholds = [0.01, 0.1, 1.0, -1.0, -0.1]
        atol = 1e-6

        for episode in range(n_eval_episodes):
            obs = self.envs_eval.reset()
            dones = np.array([False] * self.n_envs_eval)  # Inicializa um array de "done" para cada ambiente
            while not dones.all():  # Continua até que todos os ambientes estejam concluídos
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = self.envs_eval.step(action)                
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

    def env3W_a2c(self, path_save):
         
        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")

        model = A2C('MlpPolicy', self.envs_train, verbose=0,
                    learning_rate=1e-3,
                    n_steps=128, 
                    gamma=0.99,
                    gae_lambda=0.95,
                    ent_coef=0.01,
                    tensorboard_log=self.logdir)        
        
        final_model_path = os.path.join(path_save, '_A2C')
        metrics_callback = MetricsCSVCallback(save_path=final_model_path, verbose=0)

       
        model.learn(total_timesteps=self.TIMESTEPS, reset_num_timesteps=True, tb_log_name="A2C", callback=[metrics_callback])
            
        # Salva o modelo final
        model.save(os.path.join(path_save, 'A2C'))
        logging.info(f"Modelo final salvo em {os.path.join(path_save, '_A2C')}")
        
        return model

    def env3W_a2c_eval(self, model, path_save, n_eval_episodes=1):
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

        logging.info(f"Avaliando o modelo A2C {path_save} com {n_eval_episodes} episódios.")              

        final_model_path = os.path.join(self.logdir, 'A2C')

        if not os.path.exists(final_model_path):
            os.makedirs(final_model_path)

        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")

        writer = SummaryWriter(final_model_path)

        TP, FP, TN, FN = 0, 0, 0, 0
        reward_thresholds = [0.01, 0.1, 1.0, -1.0, -0.1]
        atol = 1e-6

        for episode in range(n_eval_episodes):
            obs = self.envs_eval.reset()
            dones = np.array([False] * self.n_envs_eval)
            while not dones.all():
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = self.envs_eval.step(action)
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
    
    def env3W_dqn_cl(self, model_agent, path_save, envs, replaydir, total_timesteps = 10000):
        '''
        Continual Learning com DQN        
        '''
        
        # tensorboard --logdir=tensorboard_logs        

        print(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")
        logging.info(f"Para visualizar os logs do TensorBoard, execute:\ntensorboard --logdir='{self.logdir}'")

        final_model_path = os.path.join(path_save.replace("-real", "-real-CL"), '_DQN-CL')
        # Within env3W_dqn, before starting training
        
        #top_score = [None]  # Supondo que top_score é gerenciado globalmente
        #tensorboard_callback = TensorboardCallback(model=model_agent, referencia_top_score=top_score, caminho_salvar_modelo=final_model_path, verbose=0)
        #metrics_callback = MetricsCSVCallback(save_path=final_model_path, verbose=0)
               
        model_agent.load_replay_buffer(replaydir)
        model_agent.set_env(envs)
        model_agent._last_obs = None
        model_agent.learn(total_timesteps=total_timesteps, log_interval = 4, reset_num_timesteps = False, tb_log_name="DQN-CL")
        
        replaydir_cl = os.path.join(path_save.replace("-real", "-real-CL"), 'replay_buffer-CL')

        if not os.path.exists(replaydir_cl):
            os.makedirs(replaydir_cl)

        # Salva o modelo final
        model_agent.save(final_model_path)
        final_replay_path = os.path.join(replaydir_cl, 'dqn_save_replay_buffer-CL')
        model_agent.save_replay_buffer(final_replay_path)  
        logging.info(f"Modelo de Aprendizado Contínuo salvo em {replaydir}")  

        return model_agent