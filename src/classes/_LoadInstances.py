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

                 # Adiciona a coluna 'well' e 'code' com o nome do poço
                df['well'] = well
                df['code'] = class_code

                # Processa timestamp
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

                arrays_list.append(df.to_numpy())

        logger.info(f"Total de {len(arrays_list)} instâncias {type_instance} selecionadas.")

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
        Divide o dataset em treino, teste e validação (70-15-15, por exemplo, caso train_percentage=0.7).
        - A penúltima coluna (índice -2) é 'class' (target).
        - A última coluna (índice -1) é 'well'.
        - A primeira coluna (índice 0) é 'timestamp'.

        Objetivos:
        * Treino e teste não possuem 'timestamp' nem 'well'.
        * Validação mantém 'timestamp' e 'well' no final.
        * Aplica MinMaxScaler somente nas colunas numéricas (sem 'timestamp' e sem 'well').

        :param dataset: array (2D) com todas as colunas (features + class + well).
                        As duas últimas colunas são: [-2]=class, [-1]=well.
        :param train_percentage: fração (0-1) para treino por classe; 
                                o restante é dividido em teste e validação.
        :return: (dataset_train_scaled, dataset_test_scaled, dataset_val_scaled)
                Cada um contendo suas colunas + a coluna target no final.
                No caso da validação, também com 'timestamp' e 'well'.
        """
        logger.info("Iniciando data_preparation com 3 splits (treino, teste, validação).")

        # A penúltima coluna é o target
        y = dataset[:, -2]
        # A última coluna é well
        well = dataset[:, -1]
        # As colunas anteriores às duas últimas são as features
        X = dataset[:, :-2]  # features (inclui timestamp em X[:,0])

        # Identifica classes únicas a partir do penúltimo índice (target)
        classes = np.unique(y)

        # Seleciona índices de cada classe
        train_indices, test_indices = [], []
        for event in classes:
            class_indices = np.where(y == event)[0]
            logger.info(f"Número de amostras da classe {event}: {len(class_indices)}")

            # Split estratificado para cada classe (treino e teste)
            train_idx, test_idx = train_test_split(
                class_indices, train_size=train_percentage, random_state=42
            )
            logger.info(f"Treino classe {event}: {len(train_idx)} | Teste classe {event}: {len(test_idx)}")

            train_indices.extend(train_idx)
            test_indices.extend(test_idx)

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Divide o conjunto de teste em teste e validação (50/50 do 'teste')
        test_indices, val_indices = train_test_split(test_indices, test_size=0.5, random_state=42)

        # Cria subsets
        X_train, y_train, well_train = X[train_indices], y[train_indices], well[train_indices]
        X_test, y_test, well_test   = X[test_indices],  y[test_indices],  well[test_indices]
        X_val, y_val, well_val      = X[val_indices],   y[val_indices],   well[val_indices]

        logger.info(f"Registros treino: {len(X_train)} | teste: {len(X_test)} | val: {len(X_val)}")

        # ============ TREINO E TESTE ============
        # Remove timestamp (índice 0) e não inclui well (pois well está fora de X, era a última do dataset)
        # No entanto, se a "well" estivesse dentro de X, precisaríamos removê-la explicitamente.
        # Como well está separada, não há nada a remover de X diretamente para well.
        # Caso haja outra(s) colunas a remover, faça aqui.

        # Exemplo de colunas que devem ser escalonadas: 
        # (assumindo que a col 0 é timestamp e não deve ser escalonada)
        # Então col_to_scale = range(1, X_train.shape[1])
        X_train_no_ts = np.delete(X_train, 0, axis=1)
        X_test_no_ts  = np.delete(X_test, 0, axis=1)

        # Aplica MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train_no_ts)
        X_test_scaled  = scaler.transform(X_test_no_ts)

        # Concatena com o target no final (treino e teste NÃO têm 'well' nem 'timestamp')
        dataset_train_scaled = np.column_stack((X_train_scaled, y_train))
        dataset_test_scaled  = np.column_stack((X_test_scaled,  y_test))

        # ============ VALIDAÇÃO ============
        # Mantemos timestamp (coluna 0) e 'well' (fora de X, mas no final)
        # Precisamos apenas escalar as colunas internas (1..end) de X_val,
        # exceto a 0 (timestamp) e sem 'well' (porque well está fora de X).

        # 1) Separa timestamp
        timestamp_val = X_val[:, 0].reshape(-1, 1)

        # 2) Remove timestamp para escalar as colunas restantes
        X_val_no_ts = np.delete(X_val, 0, axis=1)
        # Escalona
        X_val_scaled_temp = scaler.transform(X_val_no_ts)

        # 3) Reconstrói X_val com timestamp (col 0) + colunas escalonadas
        X_val_scaled = np.hstack((timestamp_val, X_val_scaled_temp))

        # 4) Concatena com target e well (mantendo well no final)
        #    Formato final: [timestamp, colunas escalonadas, target, well]
        dataset_val_scaled = np.column_stack((X_val_scaled, y_val, well_val))

        logger.info("Fim da preparação do dataset (treino, teste, validação).")

        return dataset_train_scaled, dataset_test_scaled, dataset_val_scaled

