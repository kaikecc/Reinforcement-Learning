import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from itertools import product

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# Add project root to path so we can import classes
sys.path.append(os.path.join('..'))
from classes._LoadInstances import LoadInstances
from classes._Env3WGym import make_custom_vec_env
from classes._Agent import Agent
from classes._Supervised import Supervised
from classes._ValidationModel import ValidationModel


class IsolationForestWrapper:
    """Wrapper para compatibilizar IsolationForest com a interface do pipeline."""
    def __init__(self, **kwargs: Any) -> None:
        self.model = IsolationForest(**kwargs)

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        obs2d = np.atleast_2d(obs)
        pred = self.model.predict(obs2d)
        # -1 = anômalo → ação 1, +1 = normal → ação 0
        action = np.where(pred == -1, 1, 0)
        return action[0], None


def setup_logger(
    directory: str,
    event_name: str,
    type_instance: str,
    ts: int,
    model_type: str
) -> logging.Logger:
    os.makedirs(directory, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{now}_{event_name}-{type_instance}_{model_type}_{ts}.log"
    path = os.path.join(directory, filename)

    logger = logging.getLogger(f"global_logger_{model_type}_{ts}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fmt = logging.Formatter(
        "[%(levelname)s]\t%(asctime)s - %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def train_evaluate_model(
    agente: Agent,
    model_type: str,
    path_model: str,
    ts: int,
    logger: logging.Logger,
    supervised: Optional[Supervised] = None,
    dataset_train: Optional[np.ndarray] = None,
    dataset_test: Optional[np.ndarray] = None,
    if_params: Optional[Dict[str, Any]] = None,
    n_eval_episodes: int = 5,
) -> Optional[Dict[str, Any]]:
    """Treina e avalia um modelo, retornando dicionário de resultados."""
    agente.TIMESTEPS = ts
    logger.info("Treinando %s com %s timesteps", model_type, ts)
    start = time.time()

    try:
        if model_type == "DQN":
            model_agent, _ = agente.env3W_dqn(path_save=path_model)
        elif model_type == "PPO":
            model_agent = agente.env3W_ppo(path_save=path_model)
        elif model_type == "A2C":
            model_agent = agente.env3W_a2c(path_save=path_model)
        elif model_type == "RNA" and supervised is not None:
            model_agent = supervised.keras_train()
        elif model_type == "IF":
            if dataset_train is None or if_params is None:
                raise ValueError("Dados ou parâmetros insuficientes para IF")
            model_agent = IsolationForestWrapper(**if_params)
            model_agent.fit(dataset_train[:, :-1])
        else:
            raise ValueError("Modelo não implementado")
        train_time = time.time() - start
        logger.info("Tempo de treino %s: %.2fs", model_type, train_time)
    except Exception as e:
        logger.error("Erro no treino %s: %s", model_type, e)
        return None

    logger.info("Avaliando %s", model_type)
    start = time.time()
    try:
        if model_type == "DQN":
            accuracy = agente.env3W_dqn_eval(model=model_agent, path_save=path_model, n_eval_episodes=n_eval_episodes)
        elif model_type == "PPO":
            accuracy = agente.env3W_ppo_eval(model=model_agent, path_save=path_model, n_eval_episodes=n_eval_episodes)
        elif model_type == "A2C":
            accuracy = agente.env3W_a2c_eval(model=model_agent, path_save=path_model, n_eval_episodes=n_eval_episodes)
        elif model_type == "RNA" and supervised is not None:
            accuracy = supervised.keras_evaluate(model_agent)
        elif model_type == "IF":
            if dataset_test is None:
                raise ValueError("Dataset de teste não fornecido para IF")
            y_true = np.where(dataset_test[:, -1] == 0, 0, 1)
            preds = model_agent.model.predict(dataset_test[:, :-1])
            y_pred = np.where(preds == -1, 1, 0)
            accuracy = accuracy_score(y_true, y_pred)
        else:
            raise ValueError("Modelo não implementado")
        eval_time = time.time() - start
        logger.info("Acurácia %s: %.5f (avaliado em %.2fs)", model_type, accuracy, eval_time)
        result = {
            "timesteps": ts,
            "model_agent": model_agent,
            "accuracy": accuracy
        }
        if model_type == "IF":
            result["if_params"] = if_params
        return result
    except Exception as e:
        logger.error("Erro na avaliação %s: %s", model_type, e)
        return None


def run_event(
    code: int,
    event: str,
    dataset: np.ndarray,
    models: List[str],
    type_instance: str,
    train_perc: float,
    timesteps_list: List[int],
    instances: LoadInstances,
    if_hyperparams: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    # filtra pelo código de evento
    data_f = dataset[dataset[:, -1] == code]
    if data_f.size == 0:
        logging.warning("Nenhum dado para código %s (%s)", code, event)
        return results
    # separa features de label
    data = data_f[:, :-1]
    # split treino/test/val
    d_train, d_test, d_val = instances.data_preparation(data, train_perc)
    # mantém cópias para IF
    d_train_orig = d_train.copy()
    d_test_orig = d_test.copy()

    # ambientes RL
    envs_train = make_custom_vec_env(d_train, n_envs=5, vec_env_type="dummy")
    envs_test  = make_custom_vec_env(d_test,  n_envs=1, vec_env_type="dummy")
    tensorboard_dir = os.path.join("..", "models", f"{event}-{type_instance}")
    os.makedirs(tensorboard_dir, exist_ok=True)
    agente = Agent(tensorboard_dir, envs_train, envs_test)

    for model_type in models:
        for ts in timesteps_list:
            # caso IF: variação de hiperparâmetros
            if model_type == "IF":
                for hp in if_hyperparams:
                    key = f"IF_ts{ts}_n{hp['n_estimators']}_c{hp['contamination']}"
                    log_dir   = os.path.join("..","logs", event, type_instance, key)
                    model_dir = os.path.join("..","models",  event, type_instance, key)
                    os.makedirs(log_dir, exist_ok=True)
                    os.makedirs(model_dir, exist_ok=True)

                    logger = setup_logger(log_dir, event, type_instance, ts, key)
                    logger.info("IF ts=%s hiper=%s", ts, hp)

                    result = train_evaluate_model(
                        agente, model_type, model_dir, ts, logger,
                        supervised=None,
                        dataset_train=d_train_orig,
                        dataset_test=d_test_orig,
                        if_params=hp
                    )
                    if result:
                        results[key] = result

                    for h in logger.handlers[:]:
                        h.close(); logger.removeHandler(h)
                continue

            # caso RL/RNA
            log_dir   = os.path.join("..","logs", event, type_instance, f"{model_type}_{ts}")
            model_dir = os.path.join("..","models",  event, type_instance, model_type, str(ts))
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            logger = setup_logger(log_dir, event, type_instance, ts, model_type)
            supervised = None
            if model_type == "RNA":
                # ajuste de labels para RNA
                d_train[:, -1] = np.where(d_train[:, -1] == 101, 1, d_train[:, -1])
                d_test[:,  -1] = np.where(d_test[:,  -1] == 101, 1, d_test[:,  -1])
                supervised = Supervised(model_dir, d_train, d_test)

            result = train_evaluate_model(
                agente, model_type, model_dir, ts, logger, supervised
            )
            if result:
                results[f"{model_type}_{ts}"] = result
                logger.info("Validando %s", model_type)
                validation = ValidationModel(model_type, event, ts)
                validation.validation_model(result["accuracy"], d_val, result["model_agent"])
            for h in logger.handlers[:]:
                h.close(); logger.removeHandler(h)

    return results


def main() -> None:
    events_names = {1: "Abrupt Increase of BSW"}
    models = ["DQN", "PPO", "A2C", "RNA", "IF"]
    type_instance = "real"

    # grid de hiperparâmetros para IF
    if_hyperparams = [
        {"n_estimators": n, "contamination": c}
        for n, c in product([50, 100, 200], [0.01, 0.05, 0.1])
    ]

    path_dataset = os.path.join("..", "..", "dataset")
    instances = LoadInstances(path_dataset)
    logging.info("Carregando dataset")
    dataset, _ = instances.load_instance_with_numpy(events_names, type_instance=type_instance)
    logging.info("Fim do carregamento")

    train_perc = 0.8
    timesteps_list = [1000, 10000, 100000, 150000, 300000]

    final_results: Dict[tuple[str, str], Dict[str, Any]] = {}
    for code, event in events_names.items():
        if code == 0:
            continue
        ev_res = run_event(
            code, event, dataset, models,
            type_instance, train_perc,
            timesteps_list, instances,
            if_hyperparams
        )
        for k, v in ev_res.items():
            final_results[(k, event)] = v

    for (model_key, event), res in final_results.items():
        acc = res["accuracy"] * 100
        ts  = res["timesteps"]
        print(f"Resultado final {model_key} em {event} com {ts} timesteps: {acc:.2f}%")

if __name__ == "__main__":
    main()
