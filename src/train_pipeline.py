import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

import numpy as np

# Proposta de algoritmo alternativo - Isolation Forest
# 1.1 Princípio: anomalias são pontos raros e facilmente isolados em poucas
# divisões de árvores binárias. 1.2 Funcionamento: um conjunto de “arbóres de
# isolamento” é treinado de forma aleatória e a profundidade para isolar uma
# instância define seu score de anomalia. 1.3 Vantagens: complexidade O(n log n)
# e independência de rótulos. 1.4 Implementação: utilizamos a classe
# IsolationForest da biblioteca scikit-learn.


class IsolationForestWrapper:
    """Wrapper para compatibilizar IsolationForest com a interface utilizada na
    validação do pipeline."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = IsolationForest(**kwargs)

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Retorna ação equivalente à detecção de anomalia."""
        obs_2d = np.atleast_2d(obs)
        pred = self.model.predict(obs_2d)
        action = np.where(pred == -1, 1, 0)
        return action[0], None

# Add project root to path so we can import classes
sys.path.append(os.path.join('..'))

from classes._LoadInstances import LoadInstances
from classes._Env3WGym import make_custom_vec_env
from classes._Agent import Agent
from classes._Supervised import Supervised
from classes._ValidationModel import ValidationModel


def setup_logger(directory: str, event_name: str, type_instance: str,
                  ts: int, model_type: str) -> logging.Logger:
    """Create and return a logger for a specific training run."""
    os.makedirs(directory, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{current_time}_{event_name}-{type_instance}_{model_type}_{ts}.log"
    full_path = os.path.join(directory, filename)

    logger = logging.getLogger(f"global_logger_{model_type}_{ts}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(full_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter(
        "[%(levelname)s]\t%(asctime)s - %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
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
) -> Optional[Dict[str, Any]]:
    """Train and evaluate a model, returning results as a dictionary."""

    agente.TIMESTEPS = ts
    logger.info(
        "Iniciando treinamento do algoritmo %s com %s timesteps", model_type, ts
    )
    start_time = time.time()
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
            if dataset_train is None or dataset_test is None:
                raise ValueError("Dataset necessário para Isolation Forest")
            model_agent = IsolationForestWrapper(
                n_estimators=100, contamination=0.05, random_state=42
            )
            model_agent.fit(dataset_train[:, :-1])
        else:
            raise ValueError("Modelo não implementado")

        train_time = round(time.time() - start_time, 2)
        logger.info("Tempo de Treinamento %s: %.2fs", model_type, train_time)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao treinar o modelo %s: %s", model_type, exc)
        return None

    logger.info(
        "Iniciando avaliação do algoritmo %s no conjunto de teste", model_type
    )
    start_time = time.time()
    try:
        if model_type == "DQN":
            accuracy = agente.env3W_dqn_eval(model=model_agent, path_save=path_model)
        elif model_type == "PPO":
            accuracy = agente.env3W_ppo_eval(model=model_agent, path_save=path_model)
        elif model_type == "A2C":
            accuracy = agente.env3W_a2c_eval(model=model_agent, path_save=path_model)
        elif model_type == "RNA" and supervised is not None:
            accuracy = supervised.keras_evaluate(model_agent)
        elif model_type == "IF":
            if dataset_test is None:
                raise ValueError("Dataset necessário para Isolation Forest")
            y_true = np.where(dataset_test[:, -1] == 0, 0, 1)
            preds = model_agent.model.predict(dataset_test[:, :-1])
            y_pred = np.where(preds == -1, 1, 0)
            accuracy = accuracy_score(y_true, y_pred)
        else:
            raise ValueError("Modelo não implementado")

        eval_time = round(time.time() - start_time, 2)
        logger.info(
            "Acurácia: %.5f com %s timesteps (avaliado em %.2fs)",
            accuracy,
            ts,
            eval_time,
        )
        return {"timesteps": ts, "model_agent": model_agent, "accuracy": accuracy}
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao avaliar o modelo %s: %s", model_type, exc)
        return None


def run_event(
    code: int,
    event: str,
    dataset: np.ndarray,
    models: list[str],
    type_instance: str,
    train_percentage: float,
    timesteps_list: list[int],
    instances: LoadInstances,
) -> Dict[str, Dict[str, Any]]:
    """Run training and evaluation for a specific event."""

    results: Dict[str, Dict[str, Any]] = {}
    dataset_filtered = dataset[dataset[:, -1] == code]
    if dataset_filtered.size == 0:
        logging.warning("Nenhum dado encontrado para o código %s (%s).", code, event)
        return results

    dataset_filtered = dataset_filtered[:, :-1]
    dataset_train, dataset_test, dataset_val = instances.data_preparation(
        dataset_filtered, train_percentage
    )

    dataset_train_orig = dataset_train.copy()
    dataset_test_orig = dataset_test.copy()

    envs_train = make_custom_vec_env(dataset_train, n_envs=5, vec_env_type="dummy")
    envs_test = make_custom_vec_env(dataset_test, n_envs=1, vec_env_type="dummy")

    tensorboard_dir = os.path.join("..", "models", f"{event}-{type_instance}")
    os.makedirs(tensorboard_dir, exist_ok=True)
    agente = Agent(tensorboard_dir, envs_train, envs_test)

    for model_type in models:
        for ts in timesteps_list:
            dataset_train = dataset_train_orig.copy()
            dataset_test = dataset_test_orig.copy()
            log_dir = os.path.join("..", "..", "logs", f"{event}-{type_instance}")
            model_dir = os.path.join("..", "models", f"{event}-{type_instance}", model_type)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            logger = setup_logger(log_dir, event, type_instance, ts, model_type)
            logger.info(
                "Iniciando execução do algoritmo %s-%s para o evento %s com %s timesteps",
                model_type,
                type_instance,
                event,
                ts,
            )

            supervised = None
            if model_type == "RNA":
                dataset_train[:, -1] = np.where(dataset_train[:, -1] == 101, 1, dataset_train[:, -1])
                dataset_test[:, -1] = np.where(dataset_test[:, -1] == 101, 1, dataset_test[:, -1])
                supervised = Supervised(model_dir, dataset_train, dataset_test)

            result = train_evaluate_model(
                agente,
                model_type,
                model_dir,
                ts,
                logger,
                supervised,
                dataset_train if model_type == "IF" else None,
                dataset_test if model_type == "IF" else None,
            )
            if result:
                results[f"{model_type}_{ts}"] = result

                logger.info("Iniciando a validação do modelo %s", model_type)
                validation = ValidationModel(model_type, event, ts)
                val_type = "IF" if model_type == "IF" else "RNA"
                validation.validation_model(
                    result["accuracy"], dataset_val, result["model_agent"], val_type
                )
                logger.info(
                    "Concluído a execução do algoritmo %s-%s para o evento %s",
                    model_type,
                    type_instance,
                    event,
                )

            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    return results


def main() -> None:
    events_names = {1: "Abrupt Increase of BSW"}
    models = ["DQN", "PPO", "IF"]
    type_instance = "real"

    path_dataset = os.path.join("..", "..", "dataset")
    instances = LoadInstances(path_dataset)

    logging.info("Iniciando carregamento do dataset")
    dataset, _ = instances.load_instance_with_numpy(events_names, type_instance=type_instance)
    logging.info("Fim carregamento do dataset")

    logging.info("Iniciando divisão do dataset em treino, teste e validação")
    train_percentage = 0.8
    timesteps_list = [1000, 10000, 100000, 150000, 300000]

    final_results: Dict[tuple[str, str], Dict[str, Any]] = {}
    for code, event in events_names.items():
        if code == 0:
            continue
        event_results = run_event(
            code,
            event,
            dataset,
            models,
            type_instance,
            train_percentage,
            timesteps_list,
            instances,
        )
        for key, value in event_results.items():
            final_results[(key, event)] = value

    for (model, event), result in final_results.items():
        accuracy = result["accuracy"]
        print(
            f"Final accuracy for {model} on event {event} with {result['timesteps']} timesteps: {accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
