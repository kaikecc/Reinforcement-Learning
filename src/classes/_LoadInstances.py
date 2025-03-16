from pathlib import Path
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

logger = logging.getLogger("global_logger")

class LoadInstances:
    """
    Classe para carregamento e preparação de dados a partir de arquivos CSV.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def class_and_file_generator(self, real: bool = False, simulated: bool = False, drawn: bool = False):
        """
        Gera tuplas (class_code, instance_path) para arquivos CSV encontrados
        nos subdiretórios do caminho fornecido.
        """
        valid_prefixes = {"simulated": "SIMULATED", "drawn": "DRAWN"}

        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                try:
                    class_code = int(class_dir.stem)
                except ValueError:
                    continue

                for instance_path in class_dir.glob("*.csv"):
                    prefix = instance_path.stem.split("_")[0]
                    if (simulated and prefix == valid_prefixes["simulated"]) \
                       or (drawn and prefix == valid_prefixes["drawn"]) \
                       or (real and prefix not in valid_prefixes.values()):
                        yield class_code, instance_path

    def load_instance_with_numpy(self, events_names: dict, type_instance: str = "real"):
        """
        Carrega as instâncias dos arquivos CSV, aplicando filtros e retornando um array final concatenado,
        além de uma lista com os arrays individuais.

        :param events_names: dicionário com códigos e nomes dos eventos.
        :param type_instance: tipo de instância: 'real', 'simulated' ou 'drawn'.
        :return: (final_array, arrays_list)
        """
        # Obtém nome do evento (ignorando chave == 0)
        event_name = next((v for k, v in events_names.items() if k != 0), None)

        # Define well names conforme tipo da instância
        instance_map = {
            "real": [f"WELL-{i:05d}" for i in range(1, 19)],
            "simulated": [f"SIMULATED_{i:05d}" for i in range(1, 120)],
            "drawn": [f"DRAWN_{i:05d}" for i in range(1, 19)],
        }
        try:
            well_names = set(instance_map[type_instance])
        except KeyError:
            raise ValueError("type_instance deve ser 'real', 'simulated' ou 'drawn'")

        # Flags para o generator
        type_flags = {
            "real":      {"real": True, "simulated": False, "drawn": False},
            "simulated": {"real": False, "simulated": True, "drawn": False},
            "drawn":     {"real": False, "simulated": False, "drawn": True}
        }[type_instance]

        columns = ["timestamp", "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "class"]

        # Lista de instâncias
        instances = list(self.class_and_file_generator(**type_flags))
        logger.info(f"Total de {len(instances)} instâncias {type_instance} encontradas.")

        arrays_list = []
        for class_code, instance_path in instances:
            # Extrai well name
            if type_instance == "real":
                well, _ = instance_path.stem.split("_", 1)
            else:
                well = instance_path.stem

            # Verifica se pertence ao conjunto de wells e classes desejadas
            if class_code in events_names and well in well_names:
                try:
                    df = pd.read_csv(instance_path, usecols=columns)
                except Exception as e:
                    logger.error(f"Erro ao ler {instance_path}: {e}")
                    continue

                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)

                # Processa timestamp
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

                arrays_list.append(df.to_numpy())

        logger.info(f"Total de {len(arrays_list)} instâncias {type_instance} carregadas para o evento {event_name}.")

        # Concatena tudo num único array (caso haja instâncias)
        final_array = np.concatenate(arrays_list) if arrays_list else np.array([])
        return final_array, arrays_list

    @staticmethod
    def apply_undersampling(X: np.ndarray, y: np.ndarray):
        """
        Aplica undersampling para balancear as classes do dataset.
        Retorna (X_undersampled, y_undersampled).
        """
        logger.info("Iniciando o processo de undersampling.")
        dataset = np.column_stack((X, y))
        classes = np.unique(dataset[:, -1])
        datasets_by_class = {lbl: dataset[dataset[:, -1] == lbl] for lbl in classes}
        min_class_size = min(len(data) for data in datasets_by_class.values())
        logger.info(f"Tamanho da menor classe: {min_class_size}")

        undersampled_datasets = []
        for lbl, data in datasets_by_class.items():
            undersampled_data = resample(data, replace=False, n_samples=min_class_size, random_state=42)
            undersampled_datasets.append(undersampled_data)
            logger.info(f"Classe {lbl} foi undersampled para {min_class_size} instâncias.")

        undersampled_dataset = np.vstack(undersampled_datasets)
        np.random.shuffle(undersampled_dataset)
        logger.info("Dataset final undersampled e embaralhado.")

        X_undersampled, y_undersampled = undersampled_dataset[:, :-1], undersampled_dataset[:, -1]
        return X_undersampled, y_undersampled

    def data_preparation(self, dataset: np.ndarray, train_percentage: float):
        """
        Divide o dataset em treino, teste e validação (70-15-15, por exemplo, caso train_percentage=0.7),
        removendo a coluna timestamp (primeira) das features de treino e teste, e aplica MinMaxScaler.

        :param dataset: array com todas as colunas (features + target), sendo a última coluna o target.
        :param train_percentage: fração (0-1) para treino por classe; o restante é dividido em teste e validação.
        :return: (dataset_train_scaled, dataset_test_scaled, dataset_validation_scaled)
        """
        # Seleciona índices de cada classe
        train_indices, test_indices = [], []
        classes = np.unique(dataset[:, -1])

        for event in classes:
            class_indices = np.where(dataset[:, -1] == event)[0]
            logger.info(f"Número de amostras da classe {event}: {len(class_indices)}")

            train_idx, test_idx = train_test_split(
                class_indices, train_size=train_percentage, random_state=42
            )
            logger.info(f"Treino classe {event}: {len(train_idx)} | Teste classe {event}: {len(test_idx)}")

            train_indices.extend(train_idx)
            test_indices.extend(test_idx)

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Divide teste em teste e validação (50/50 do 'teste')
        test_indices, val_indices = train_test_split(test_indices, test_size=0.5, random_state=42)

        # Cria subsets
        dataset_train = dataset[train_indices]
        dataset_test = dataset[test_indices]
        dataset_val = dataset[val_indices]

        logger.info(f"Registros treino: {len(dataset_train)} | teste: {len(dataset_test)} | val: {len(dataset_val)}")

        # Separa X e y
        X_train, y_train = dataset_train[:, :-1], dataset_train[:, -1]
        X_test, y_test = dataset_test[:, :-1], dataset_test[:, -1]
        X_val, y_val = dataset_val[:, :-1], dataset_val[:, -1]

        # Remove timestamp (coluna 0) de treino e teste
        X_train_no_ts = np.delete(X_train, 0, axis=1)
        X_test_no_ts = np.delete(X_test, 0, axis=1)

        # Escalona (MinMax no range -1..1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train_no_ts)
        X_test_scaled = scaler.transform(X_test_no_ts)

        # Validação: mantém timestamp na 1ª coluna
        X_val_ts = X_val[:, 0].reshape(-1, 1)
        X_val_no_ts = np.delete(X_val, 0, axis=1)
        X_val_scaled_only = scaler.transform(X_val_no_ts)
        X_val_scaled = np.hstack((X_val_ts, X_val_scaled_only))

        # Concatena com o target
        dataset_train_scaled = np.column_stack((X_train_scaled, y_train))
        dataset_test_scaled = np.column_stack((X_test_scaled, y_test))
        dataset_val_scaled = np.column_stack((X_val_scaled, y_val))

        logger.info("Fim da preparação do dataset (treino, teste, validação).")
        return dataset_train_scaled, dataset_test_scaled, dataset_val_scaled
