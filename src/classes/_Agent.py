import os
import json
import csv
from pathlib import Path

import logging
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from torch.utils.tensorboard import SummaryWriter
import tensorboard.program
import webbrowser


# Configuração do logger global
logger = logging.getLogger("global_logger")


# Constantes usadas na avaliação
REWARD_THRESHOLDS: List[float] = [0.0, 0.1, 1.0, -1.0, -0.1]
ATOL: float = 1e-6


class TensorboardCallback(BaseCallback):
    """
    Callback para registrar pontuações no TensorBoard e salvar o modelo quando um novo top score é alcançado.
    """
    def __init__(self, model: Any, referencia_top_score: List[Optional[float]], 
                 caminho_salvar_modelo: str = "./model_DQN", verbose: int = 0):
        super().__init__(verbose)
        self.model = model
        self.referencia_top_score = referencia_top_score
        self.caminho_salvar_modelo = caminho_salvar_modelo

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        score = infos[0].get("score") if infos and "score" in infos[0] else None

        if score is not None:
            self.logger.record('a_score', score)
            if self.referencia_top_score[0] is None or score > self.referencia_top_score[0]:
                self.referencia_top_score[0] = score
                self.model.save(self.caminho_salvar_modelo)

        if self.num_timesteps % 10000 == 0:
            self.logger.dump(self.num_timesteps)
        return True


class MetricsCSVCallback(BaseCallback):
    """
    Callback para registrar métricas de recompensa e salvar os dados em JSON.
    """
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.file_path = os.path.join(self.save_path, 'metrics.csv')
        self.file_exists = os.path.isfile(self.file_path)

        self.reward_max = float('-inf')
        self.reward_min = float('inf')
        self.reward_sum = 0.0
        self.reward_count = 0
        self.metrics: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        epsilon = getattr(self.model, 'exploration_rate', None)
        if epsilon is not None:
            self.logger.record('epsilon', epsilon)

        rewards = self.locals.get('rewards', [])
        for reward in rewards:
            self.reward_max = max(self.reward_max, reward)
            self.reward_min = min(self.reward_min, reward)
            self.reward_sum += reward
            self.reward_count += 1

        if self.reward_count > 0:
            reward_avg = self.reward_sum / self.reward_count
            self.logger.record('reward_avg', reward_avg)
            self.logger.record('reward_max', self.reward_max)
            self.logger.record('reward_min', self.reward_min)
            self.metrics.append({
                'step': self.num_timesteps,
                'epsilon': epsilon,
                'reward_avg': reward_avg,
                'reward_max': self.reward_max,
                'reward_min': self.reward_min
            })

        if self.num_timesteps % 1000 == 0:
            self.logger.dump(self.num_timesteps)
            self._save_metrics()
        return True

    def _save_metrics(self) -> None:
        serializable_metrics = self._convert_to_serializable(self.metrics)
        file_path = os.path.join(self.save_path, 'metrics.json')
        with open(file_path, 'w') as fp:
            json.dump(serializable_metrics, fp, indent=4)

    def _convert_to_serializable(self, data: Any) -> Any:
        if isinstance(data, (np.integer,)):
            return int(data)
        elif isinstance(data, (np.floating,)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._convert_to_serializable(value) for key, value in data.items()}
        else:
            return data


class Agent:
    """
    Classe Agent para treinamento e avaliação de modelos de reinforcement learning.
    Suporta DQN, PPO, A2C e Continual Learning para DQN.
    """
    def __init__(self, path_tensorboard: str, envs_train: Any, envs_test: Any, 
                 TIMESTEPS: Union[int, str] = 10000, port: int = 6006):
        self.envs_train = envs_train
        self.envs_eval = envs_test
        self.TIMESTEPS = int(TIMESTEPS)
        self.n_envs_eval = 1
        self.path_tensorboard = path_tensorboard
        self.logdir = os.path.join(self.path_tensorboard, 'tensorboard_logs')
        os.makedirs(self.logdir, exist_ok=True)
        self.port = port
        self._log_tensorboard_info()
        self.launch_tensorboard()

    def _log_tensorboard_info(self) -> None:
        msg = f"Para visualizar os logs do TensorBoard, execute: tensorboard --logdir='{self.logdir}'"
        logger.info(msg)

    def launch_tensorboard(self) -> None:
        logging.getLogger('tensorboard').setLevel(logging.ERROR)
        tb = tensorboard.program.TensorBoard()
        try:
            tb.configure(argv=[None, '--logdir', self.logdir, '--port', str(self.port)])
            url = tb.launch()
            logger.info(f"TensorBoard started at {url}")
            webbrowser.open(url)
        except Exception as e:
            logger.error(f"Error launching TensorBoard: {e}")

    def _update_confusion_matrix(self, reward: float, counters: Dict[str, int]) -> None:
        """Atualiza a matriz de confusão SEM penalizar FP e FN juntos."""
        if np.isclose(reward, REWARD_THRESHOLDS[0], atol=ATOL):
            counters['TN'] += 1
        elif np.isclose(reward, REWARD_THRESHOLDS[1], atol=ATOL) or np.isclose(reward, REWARD_THRESHOLDS[2], atol=ATOL):
            counters['TP'] += 1
        elif np.isclose(reward, REWARD_THRESHOLDS[3], atol=ATOL):
            counters['FP'] += 1
        elif np.isclose(reward, REWARD_THRESHOLDS[4], atol=ATOL):
            counters['FN'] += 1

    def _evaluate_model(self, model: Any, n_eval_episodes: int, alg_name: str) -> float:
        """
        Método auxiliar para avaliar um modelo em n_eval_episodes episódios,
        calculando a acurácia com base na contagem de TP, TN, FP e FN.
        """
        accuracies = []
        eval_logdir = os.path.join(self.logdir, alg_name)
        os.makedirs(eval_logdir, exist_ok=True)
        writer = SummaryWriter(eval_logdir)

        for episode in range(n_eval_episodes):
            counters = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            obs = self.envs_eval.reset()
            dones = np.array([False] * self.n_envs_eval)
            while not dones.all():
                action, _ = model.predict(obs, deterministic=False)
                obs, rewards, dones, _ = self.envs_eval.step(action)
                for reward in rewards:
                    self._update_confusion_matrix(reward, counters)
            total = sum(counters.values())
            accuracy = (counters['TP'] + counters['TN']) / total if total > 0 else 0.0
            writer.add_scalar('Accuracy', accuracy, episode)
            accuracies.append(accuracy)

        writer.close()
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        logger.info(f"Average accuracy for {alg_name}: {avg_accuracy}")

        # Adicionando registro em CSV
        csv_path = Path("..", "..", "metrics", "model_accuracy_log.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not os.path.exists(csv_path)

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["model", "timestep", "avg_accuracy"])
            writer.writerow([alg_name, self.TIMESTEPS, avg_accuracy])

        return avg_accuracy

    def env3W_dqn(self, path_save: str) -> Tuple[Any, str]:
        replaydir = os.path.join(path_save, 'replay_buffer')
        os.makedirs(replaydir, exist_ok=True)
        model = DQN(
                MlpPolicy,
                self.envs_train,
                learning_rate=1e-3,              # Alterado para igualar o PPO
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=500,
                exploration_fraction=0.2,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.1,
                max_grad_norm=10,
                verbose=0,
                tensorboard_log=self.logdir,
                device='auto'
            )

        final_model_path = os.path.join(path_save, '_DQN')
        top_score = [None]
        tensorboard_callback = TensorboardCallback(model=model, 
                                                referencia_top_score=top_score, 
                                                caminho_salvar_modelo=final_model_path, 
                                                verbose=0)
        metrics_callback = MetricsCSVCallback(save_path=final_model_path, verbose=0)

        model.learn(total_timesteps=self.TIMESTEPS, progress_bar=True, log_interval=4, reset_num_timesteps=True,
                    tb_log_name="DQN", callback=[metrics_callback, tensorboard_callback])
        model.save(final_model_path)
        logger.info(f"Modelo final salvo em {final_model_path}")
        final_replay_path = os.path.join(replaydir, 'dqn_save_replay_buffer')
        model.save_replay_buffer(final_replay_path)
        logger.info(f"Replay buffer salvo em {final_replay_path}")
        return model, final_replay_path

    def env3W_dqn_eval(self, model: Any, path_save: str, n_eval_episodes: int = 5) -> float:
        logger.info(f"Avaliando o modelo DQN em {path_save} com {n_eval_episodes} episódios.")
        return self._evaluate_model(model, n_eval_episodes, "DQN")

    def env3W_ppo(self, path_save: str) -> Any:
        checkpoint_dir = os.path.join(path_save, 'ppo_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        model = PPO(
                'MlpPolicy',
                self.envs_train,
                verbose=0,
                learning_rate=1e-3,              # Mesma taxa de aprendizado do DQN
                n_steps=128,
                batch_size=32,                   # Igual ao DQN
                n_epochs=10,
                gamma=0.99,                      # Mesmo valor de desconto
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                tensorboard_log=self.logdir
            )
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoint_dir, name_prefix='PPO')
        final_model_path = os.path.join(path_save, '_PPO')
        metrics_callback = MetricsCSVCallback(save_path=final_model_path, verbose=0)
        model.learn(total_timesteps=self.TIMESTEPS, progress_bar=True, reset_num_timesteps=True, tb_log_name="PPO", 
                    callback=[checkpoint_callback, metrics_callback])
        model.save(final_model_path)
        logger.info(f"Modelo final salvo em {final_model_path}")
        return model

    def env3W_ppo_eval(self, model: Any, path_save: str, n_eval_episodes: int = 5) -> float:
        logger.info(f"Avaliando o modelo PPO em {path_save} com {n_eval_episodes} episódios.")
        return self._evaluate_model(model, n_eval_episodes, "PPO")

    def env3W_a2c(self, path_save: str) -> Any:
        model = A2C('MlpPolicy', self.envs_train, verbose=0, learning_rate=1e-3, n_steps=128, 
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.01, tensorboard_log=self.logdir)
        final_model_path = os.path.join(path_save, '_A2C')
        metrics_callback = MetricsCSVCallback(save_path=final_model_path, verbose=0)
        model.learn(total_timesteps=self.TIMESTEPS, progress_bar=True, reset_num_timesteps=True, tb_log_name="A2C", 
                    callback=[metrics_callback])
        model.save(final_model_path)
        logger.info(f"Modelo final salvo em {final_model_path}")
        return model

    def env3W_a2c_eval(self, model: Any, path_save: str, n_eval_episodes: int = 5) -> float:
        logger.info(f"Avaliando o modelo A2C em {path_save} com {n_eval_episodes} episódios.")
        return self._evaluate_model(model, n_eval_episodes, "A2C")

    def env3W_dqn_cl(self, model_agent: Any, path_save: str, envs: Any, replaydir: str, 
                      total_timesteps: int = 10000) -> Any:
        logger.info(f"Iniciando Continual Learning para DQN com logs em: {self.logdir}")
        final_model_path = os.path.join(path_save.replace("-real", "-real-CL"), '_DQN-CL')
        model_agent.load_replay_buffer(replaydir)
        model_agent.set_env(envs)
        model_agent._last_obs = None
        model_agent.learn(total_timesteps=total_timesteps, log_interval=4, reset_num_timesteps=False,
                          tb_log_name="DQN-CL")
        replaydir_cl = os.path.join(path_save.replace("-real", "-real-CL"), 'replay_buffer-CL')
        os.makedirs(replaydir_cl, exist_ok=True)
        model_agent.save(final_model_path)
        final_replay_path = os.path.join(replaydir_cl, 'dqn_save_replay_buffer-CL')
        model_agent.save_replay_buffer(final_replay_path)
        logger.info(f"Modelo CL salvo em {final_model_path} e replay buffer em {final_replay_path}")
        return model_agent
