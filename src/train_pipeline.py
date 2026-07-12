import logging
import os
import sys
import time
import argparse
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from itertools import product

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)

SRC_ROOT = PROJECT_ROOT_PATH / "src"
DIGITAL_TWIN_ROOT = PROJECT_ROOT_PATH / "digital_twin"
for import_path in (SRC_ROOT, DIGITAL_TWIN_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from classes._LoadInstances import LoadInstances
from classes._Env3WGym import make_custom_vec_env
from classes._Agent import Agent
from classes._Supervised import Supervised
from classes._ValidationModel import ValidationModel

THREE_W_ROOT = Path(os.environ.get("THREE_W_ROOT", PROJECT_ROOT_PATH.parent / "3W"))

from well_simulator import DigitalTwinWellSimulator


def parse_timesteps_arg(values: List[str]) -> List[int]:
    """Aceita timesteps separados por espaco e/ou virgula."""
    timesteps: List[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                timesteps.append(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"timesteps deve conter inteiros; valor invalido: {part!r}"
                ) from exc
    if not timesteps:
        raise argparse.ArgumentTypeError("informe ao menos um timestep")
    return timesteps


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


def save_training_scaler(instances: LoadInstances, model_dir: str, logger: logging.Logger) -> None:
    scaler = getattr(instances, "scaler", None)
    if scaler is None:
        logger.warning("Scaler de treino nao encontrado; simulador precisara usar scaler online")
        return

    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler de treino salvo em %s", scaler_path)


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
    if_hyperparams: List[Dict[str, Any]],
    enable_tensorboard: bool = False,
    tensorboard_port: int = 6006,
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
    tensorboard_dir = os.path.join(PROJECT_ROOT, "models", f"{event}-{type_instance}")
    os.makedirs(tensorboard_dir, exist_ok=True)
    agente = Agent(
        tensorboard_dir,
        envs_train,
        envs_test,
        launch_tensorboard=enable_tensorboard,
        port=tensorboard_port,
    )

    for model_type in models:
        
        # caso IF: variação de hiperparâmetros
        if model_type == "IF":
            for ts in timesteps_list:
                for hp in if_hyperparams:
                    key = f"IF_ts{ts}_n{hp['n_estimators']}_c{hp['contamination']}"
                    log_dir = os.path.join(PROJECT_ROOT, "logs", event, type_instance, key)
                    model_dir = os.path.join(PROJECT_ROOT, "models", event, type_instance, key)
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

                        # Aqui aplicamos o validation_model para IF
                        logger.info("Validando IF")
                        validation = ValidationModel("IF", event, ts)
                        validation.validation_model(
                            result["accuracy"],
                            d_val,
                            result["model_agent"],
                            type_ml=model_type,
                        )

                    for h in logger.handlers[:]:
                        h.close(); logger.removeHandler(h)
            continue

        for ts in timesteps_list:
            # caso RL/RNA
            log_dir = os.path.join(PROJECT_ROOT, "logs", event, type_instance, f"{model_type}_{ts}")
            model_dir = os.path.join(PROJECT_ROOT, "models", event, type_instance, model_type, str(ts))
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
                if model_type == "RNA":
                    save_training_scaler(instances, model_dir, logger)
                logger.info("Validando %s", model_type)
                validation = ValidationModel(model_type, event, ts)
                validation.validation_model(
                    result["accuracy"],
                    d_val,
                    result["model_agent"],
                    type_ml=model_type,
                )
            for h in logger.handlers[:]:
                h.close(); logger.removeHandler(h)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-digital-twin", action="store_true")
    parser.add_argument("--use-realtime-twin", action="store_true")
    parser.add_argument("--event-code", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=["DQN"])
    parser.add_argument("--type-instance", choices=["real", "simulated", "drawn"], default="real")
    parser.add_argument("--timesteps", nargs="+", default=["150000"])
    parser.add_argument("--train-perc", type=float, default=0.8)
    parser.add_argument("--twin-scenarios", type=int, default=10)
    parser.add_argument("--twin-normal-rows", type=int, default=3600)
    parser.add_argument("--twin-event-rows", type=int, default=3600)
    parser.add_argument("--twin-source", choices=["real", "simulated", "drawn", "any"], default="any")
    parser.add_argument("--twin-noise-std", type=float, default=0.0)
    parser.add_argument("--dataset-path", default=str(THREE_W_ROOT / "dataset"))
    parser.add_argument("--realtime-url", default="http://127.0.0.1:8787")
    parser.add_argument("--realtime-min-rows", type=int, default=1000)
    parser.add_argument("--realtime-poll-seconds", type=float, default=2.0)
    parser.add_argument("--realtime-timeout", type=float, default=300.0)
    parser.add_argument("--realtime-iterations", type=int, default=1)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    args = parser.parse_args()
    args.timesteps = parse_timesteps_arg(args.timesteps)
    return args


def event_names_for(code: int) -> Dict[int, str]:
    all_events = {
        1: "Abrupt Increase of BSW",
        2: "Spurious Closure of DHSV",
        3: "Severe Slugging",
        4: "Flow Instability",
        5: "Rapid Productivity Loss",
        6: "Quick Restriction in PCK",
        7: "Scaling in PCK",
        8: "Hydrate in Production Line",
        9: "Hydrate in Service Line",
    }
    if code not in all_events:
        raise ValueError(f"Evento nao suportado: {code}")
    return {code: all_events[code]}


def load_training_data(
    args: argparse.Namespace,
    events_names: Dict[int, str],
    instances: LoadInstances,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if args.use_realtime_twin:
        return collect_realtime_training_data(args)

    if not args.use_digital_twin:
        logging.info("Carregando dataset 3W original")
        return instances.load_instance_with_numpy(
            events_names,
            type_instance=args.type_instance,
        )

    logging.info("Gerando cenarios fisicos pelo gemeo digital")
    simulator = DigitalTwinWellSimulator(args.dataset_path)
    frame = simulator.simulate_many(
        event_code=args.event_code,
        scenarios=args.twin_scenarios,
        source=args.twin_source,
        normal_source=args.twin_source,
        normal_rows=args.twin_normal_rows,
        event_rows=args.twin_event_rows,
        noise_std=args.twin_noise_std,
    )
    dataset = simulator.to_pipeline_numpy(frame)
    return dataset, [dataset]


def realtime_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def collect_realtime_training_data(args: argparse.Namespace) -> tuple[np.ndarray, list[np.ndarray]]:
    base_url = args.realtime_url.rstrip("/")
    realtime_request(f"{base_url}/api/control", {"action": "start"})
    started = time.time()
    last_count = -1

    while True:
        payload = realtime_request(f"{base_url}/api/samples")
        rows = payload["rows"]
        status = payload["status"]
        row_count = len(rows)
        if row_count != last_count:
            print(
                f"Simulacao em tempo real: {row_count}/{args.realtime_min_rows} "
                f"amostras recebidas"
            )
            last_count = row_count

        classes = {int(row["class"]) for row in rows} if rows else set()
        has_normal = 0 in classes
        has_event = any(value != 0 for value in classes)
        if row_count >= args.realtime_min_rows and has_normal and has_event:
            break
        if not status["running"] and row_count > 0 and status["pointer"] >= status["total_rows"]:
            break
        if time.time() - started > args.realtime_timeout:
            raise TimeoutError(
                "Tempo esgotado esperando amostras do simulador em tempo real"
            )
        time.sleep(args.realtime_poll_seconds)

    import pandas as pd

    frame = pd.DataFrame(rows)
    frame = frame[["timestamp", "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "class", "well", "code"]]
    dataset = frame.to_numpy()
    return dataset, [dataset]


def main() -> None:
    args = parse_args()
    events_names = event_names_for(args.event_code)

    models = args.models # "DQN", "PPO", "A2C", "RNA", "IF"
    if args.use_realtime_twin:
        type_instance = "realtime_twin"
    else:
        type_instance = "digital_twin" if args.use_digital_twin else args.type_instance

    # grid de hiperparâmetros para IF
    if_hyperparams = [
        {"n_estimators": n, "contamination": c}
        for n, c in product([50, 100, 200], [0.01, 0.05, 0.1])
    ]

    instances = LoadInstances(args.dataset_path)
    train_perc = args.train_perc
    timesteps_list = args.timesteps # 1000, 10000, 100000, 150000, 300000

    final_results: Dict[tuple[str, str], Dict[str, Any]] = {}
    iterations = args.realtime_iterations if args.use_realtime_twin else 1
    for iteration in range(1, iterations + 1):
        dataset, _ = load_training_data(args, events_names, instances)
        logging.info("Fim do carregamento")
        iteration_type = type_instance
        if args.use_realtime_twin and iterations > 1:
            iteration_type = f"{type_instance}_iter{iteration}"

        for code, event in events_names.items():
            if code == 0:
                continue
            ev_res = run_event(
                code, event, dataset, models,
                iteration_type, train_perc,
                timesteps_list, instances,
                if_hyperparams,
                enable_tensorboard=args.tensorboard,
                tensorboard_port=args.tensorboard_port,
            )
            for k, v in ev_res.items():
                final_results[(f"{k}_iter{iteration}", event)] = v

    for (model_key, event), res in final_results.items():
        acc = res["accuracy"] * 100
        ts  = res["timesteps"]
        print(f"Resultado final {model_key} em {event} com {ts} timesteps: {acc:.2f}%")

if __name__ == "__main__":
    main()
