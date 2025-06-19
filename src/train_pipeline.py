import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

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

    envs_train = make_custom_vec_env(dataset_train, n_envs=5, vec_env_type="dummy")
    envs_test = make_custom_vec_env(dataset_test, n_envs=1, vec_env_type="dummy")

    tensorboard_dir = os.path.join("..", "models", f"{event}-{type_instance}")
    os.makedirs(tensorboard_dir, exist_ok=True)
    agente = Agent(tensorboard_dir, envs_train, envs_test)

    for model_type in models:
        for ts in timesteps_list:
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
                agente, model_type, model_dir, ts, logger, supervised
            )
            if result:
                results[f"{model_type}_{ts}"] = result

                logger.info("Iniciando a validação do modelo %s", model_type)
                validation = ValidationModel(model_type, event, ts)
                validation.validation_model(result["accuracy"], dataset_val, result["model_agent"])
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
    models = ["DQN", "PPO"]
    type_instance = "real"

    path_dataset = os.path.join("..", "..", "..", "dataset")
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
