from pathlib import Path
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample


# Obtém o logger global
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
        valid_prefixes = {
            'simulated': 'SIMULATED',
            'drawn': 'DRAWN'
        }
        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                try:
                    class_code = int(class_dir.stem)
                except ValueError:
                    continue

                for instance_path in class_dir.glob("*.csv"):
                    prefix = instance_path.stem.split('_')[0]
                    if (simulated and prefix == valid_prefixes['simulated']) or \
                       (drawn and prefix == valid_prefixes['drawn']) or \
                       (real and prefix not in valid_prefixes.values()):
                        yield class_code, instance_path

    def load_instance_with_numpy(self, events_names: dict, type_instance: str = 'real'):
        """
        Carrega as instâncias dos arquivos CSV e retorna um array final concatenado
        e uma lista com os arrays individuais.
        
        :param events_names: dicionário com códigos e nomes dos eventos.
                             O nome do evento é extraído do primeiro par (chave != 0).
        :param type_instance: tipo de instância: 'real', 'simulated' ou 'drawn'
        :return: tuple (final_array, arrays_list)
        """
        event_name = next((value for key, value in events_names.items() if key != 0), None)

        # Define os "well names" de acordo com o tipo da instância
        if type_instance == 'real':
            well_names = {f'WELL-{i:05d}' for i in range(1, 19)}
        elif type_instance == 'simulated':
            well_names = {f'SIMULATED_{i:05d}' for i in range(1, 120)}
        elif type_instance == 'drawn':
            well_names = {f'DRAWN_{i:05d}' for i in range(1, 19)}
        else:
            raise ValueError("type_instance deve ser 'real', 'simulated' ou 'drawn'")

        flags = {
            'real': {'real': True, 'simulated': False, 'drawn': False},
            'simulated': {'real': False, 'simulated': True, 'drawn': False},
            'drawn': {'real': False, 'simulated': False, 'drawn': True}
        }[type_instance]

        # Definindo as colunas a serem lidas
        columns = ['timestamp', 'P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'class']

        instances = list(self.class_and_file_generator(**flags))
        logger.info(f'Total de {len(instances)} instâncias {type_instance} encontradas.')

        arrays_list = []
        for class_code, instance_path in instances:
            # Identifica o nome do poço a partir do nome do arquivo
            if type_instance == 'real':
                well, _ = instance_path.stem.split('_', 1)
            else:
                well = instance_path.stem

            if class_code in events_names and well in well_names:
                try:
                    df = pd.read_csv(instance_path, usecols=columns)
                except Exception as e:
                    logger.error(f"Erro ao ler {instance_path}: {e}")
                    continue

                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)

                # Adiciona a coluna 'well' com o nome do poço
                df['well'] = well

                # Processa a coluna timestamp, se presente
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
                arr = df.to_numpy()
                arrays_list.append(arr)

        logger.info(f'Total de {len(arrays_list)} instâncias {type_instance} carregadas para o evento {event_name}.')
        final_array = np.concatenate(arrays_list) if arrays_list else np.array([])

        return final_array, arrays_list

    @staticmethod
    def apply_undersampling(X: np.ndarray, y: np.ndarray):
        """
        Aplica undersampling para balancear as classes do dataset.
        """
        logger.info("Iniciando o processo de undersampling.")
        dataset = np.column_stack((X, y))
        classes = np.unique(dataset[:, -1])
        datasets_by_class = {label: dataset[dataset[:, -1] == label] for label in classes}
        min_class_size = min(len(data) for data in datasets_by_class.values())
        logger.info(f"Tamanho da menor classe: {min_class_size}")

        undersampled_datasets = []
        for label, data in datasets_by_class.items():
            undersampled_data = resample(data, replace=False, n_samples=min_class_size, random_state=42)
            undersampled_datasets.append(undersampled_data)
            logger.info(f"Classe {label} foi undersampled para {min_class_size} instâncias.")

        undersampled_dataset = np.vstack(undersampled_datasets)
        np.random.shuffle(undersampled_dataset)
        logger.info("Dataset final undersampled e embaralhado.")

        X_undersampled = undersampled_dataset[:, :-1]
        y_undersampled = undersampled_dataset[:, -1]

        return X_undersampled, y_undersampled

    def data_preparation(self, dataset: np.ndarray, train_percentage: float):
        """
        Divide o dataset em treino, teste e validação, aplicando escalonamento das features.
        
        Observação: 
          - Para treino e teste, a coluna timestamp (primeira coluna) é removida antes do escalonamento.
          - No dataset de validação, a coluna timestamp é mantida (ou seja, dataset_validation_scaled
            mantém todas as colunas originais).
          - A coluna de índice 6 não será escalonada (ajuste conforme a sua necessidade real).
        
        :param dataset: array contendo os dados, onde a última coluna é o target.
        :param train_percentage: percentual (0-1) para divisão de treino por classe.
        :return: tuple (dataset_train_scaled, dataset_test_scaled, dataset_validation_scaled)
        """
        train_indices = []
        test_indices = []

        # Divisão dos índices por classe
        for event in np.unique(dataset[:, -1]):
            class_indices = np.where(dataset[:, -1] == event)[0]
            logger.info(f'Número de amostras da classe {event}: {len(class_indices)}')

            class_train_idx, class_test_idx = train_test_split(
                class_indices, 
                train_size=train_percentage, 
                random_state=42
            )
            logger.info(f'Número de amostras de treino da classe {event}: {len(class_train_idx)}')
            logger.info(f'Número de amostras de teste da classe {event}: {len(class_test_idx)}')

            train_indices.extend(class_train_idx)
            test_indices.extend(class_test_idx)

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Divide os índices de teste em teste e validação
        test_indices, validation_indices = train_test_split(
            test_indices, test_size=0.5, random_state=42
        )

        dataset_train = dataset[train_indices]
        dataset_test = dataset[test_indices]
        dataset_validation = dataset[validation_indices]

        logger.info(f'Número de registros de treino: {len(dataset_train)}')
        logger.info(f'Número de registros de teste: {len(dataset_test)}')
        logger.info(f'Número de registros de validação: {len(dataset_validation)}')

        # Separa features (X) e target (y)
        X_train, y_train = dataset_train[:, :-1], dataset_train[:, -1]
        X_test, y_test = dataset_test[:, :-1], dataset_test[:, -1]
        X_validation, y_validation = dataset_validation[:, :-1], dataset_validation[:, -1]

        # Remove a coluna timestamp (primeira) para treino e teste
        X_train_no_ts = np.delete(X_train, 0, axis=1)
        X_test_no_ts = np.delete(X_test, 0, axis=1)

        # Determina quais colunas serão escalonadas.
        # Exemplo: queremos pular a coluna de índice 5 (a 6ª em zero-based).
        n_cols = X_train_no_ts.shape[1]  # total de colunas (já sem timestamp)
        col_to_scale = list(range(n_cols))
        col_to_scale.remove(5)  # Remova o índice da coluna que não deseja escalar

        # Copiamos os arrays para que possamos inserir os valores escalonados sem sobrescrever o original
        X_train_scaled = X_train_no_ts.copy()
        X_test_scaled = X_test_no_ts.copy()

        # Ajuste do scaler somente nas colunas definidas
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X_train_no_ts[:, col_to_scale])

        X_train_scaled[:, col_to_scale] = scaler.transform(X_train_no_ts[:, col_to_scale])
        X_test_scaled[:, col_to_scale] = scaler.transform(X_test_no_ts[:, col_to_scale])

        # Para validação, mantemos a coluna timestamp
        X_validation_ts = X_validation[:, 0].reshape(-1, 1)
        X_validation_numeric = np.delete(X_validation, 0, axis=1)
        
        # Escalona somente as colunas escolhidas no conjunto de validação
        X_validation_numeric_scaled = X_validation_numeric.copy()
        X_validation_numeric_scaled[:, col_to_scale] = scaler.transform(
            X_validation_numeric[:, col_to_scale]
        )

        # Junta a coluna timestamp (não escalada) com as demais colunas escaladas
        X_validation_scaled = np.hstack((X_validation_ts, X_validation_numeric_scaled))

        # Combina novamente features e alvo
        dataset_train_scaled = np.column_stack((X_train_scaled, y_train))
        dataset_test_scaled = np.column_stack((X_test_scaled, y_test))
        dataset_validation_scaled = np.column_stack((X_validation_scaled, y_validation))

        logger.info('Fim da divisão do dataset em treino e teste (com partial scaling)')
        return dataset_train_scaled, dataset_test_scaled, dataset_validation_scaled
