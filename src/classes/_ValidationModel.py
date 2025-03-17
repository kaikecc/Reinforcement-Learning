import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from classes._exploration import Exploration

logger = logging.getLogger("global_logger")


class ValidationModel:
    """
    Classe responsável por validar um modelo de aprendizado, realizando:
      - separação de datasets com base em um gap de tempo,
      - preprocessamento de observações,
      - predição (tanto para modelo RL quanto RNA),
      - cálculo de métricas de acurácia (TN, TP, FP, FN),
      - plot e salvamento de resultados.
    """

    def __init__(self, model_name: str, event_name: str) -> None:
        """
        :param model_name: Nome do modelo (e.g. 'DQN', 'RNA', etc.)
        :param event_name: Nome do evento (e.g. 'Abrupt Increase of BSW')
        """
        self.model_name = model_name
        self.event_name = event_name
        logger.info("ValidationModel criado para o modelo '%s' e evento '%s'.",
                    self.model_name, self.event_name)

    def separate_datasets(self, dataset_validation_sorted: np.ndarray) -> List[np.ndarray]:
        """
        Separa o dataset em grupos de instâncias com base em um gap temporal superior a 1 hora.

        :param dataset_validation_sorted: Array 2D já ordenado por timestamp na coluna 0.
        :return: Lista de arrays (cada array é um "grupo" de dados).
        """
        datasets = []
        current_dataset = []
        previous_datetime = None

        for row in dataset_validation_sorted:
            current_datetime = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            # Se for o primeiro ou se o gap > 1 hora, "fecha" o dataset atual e inicia um novo
            if previous_datetime is None or (current_datetime - previous_datetime).total_seconds() / 3600 > 1:
                if current_dataset:
                    datasets.append(np.array(current_dataset))
                    current_dataset = []
            current_dataset.append(row)
            previous_datetime = current_datetime

        # Último bloco
        if current_dataset:
            datasets.append(np.array(current_dataset))

        return datasets

    def preprocess_observation(self, row: np.ndarray) -> np.ndarray:
        """
        Pré-processa uma linha de observação para float32,
        removendo a primeira e as duas últimas colunas (timestamp e [class, well]).

        Formato esperado de 'row':
          [timestamp, P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, class, well]

        :param row: Uma linha do dataset (np.ndarray).
        :return: Vetor de floats (obs) para ser usado na predição (somente sensores).
        """
        # row[1:-2] pega do índice 1 até (len - 2), excluindo timestamp (0) e excluindo class e well
        obs = row[1:-2].astype(np.float32)
        return obs

    def create_batches(self, data: np.ndarray, batch_size: int) -> Generator[np.ndarray, None, None]:
        """
        Gera lotes (batches) dos dados de tamanho batch_size.

        :param data: Array de dados.
        :param batch_size: Tamanho do lote.
        :return: Um gerador de lotes (slices) do array.
        """
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    def predict_and_evaluate(self, model: Any, dataset_test: np.ndarray, batch_size: int = 32) -> List[int]:
        """
        Realiza a predição das ações para o dataset de teste.

        :param model: Modelo de predição (pode ser RL ou RNA).
        :param dataset_test: Array 2D com as instâncias de teste (8 colunas).
        :param batch_size: Tamanho do lote para RNA (se self.model_name == 'RNA').
        :return: Lista com as ações previstas (inteiros).
        """
        array_action_pred = []

        # Se não for RNA (i.e. modelo RL), faz predição 1 a 1
        if self.model_name != 'RNA':
            for row in dataset_test:
                obs = self.preprocess_observation(row)
                # model.predict(obs, deterministic=True) -> (acao, estado_interno)
                action = model.predict(obs, deterministic=True)[0]
                array_action_pred.append(action)
        else:
            # Para RNA, processa em lotes
            for batch in self.create_batches(dataset_test, batch_size):
                obs_batch = np.array([self.preprocess_observation(r) for r in batch])
                # Supondo que model.predict(obs_batch) retorne array 2D com probabilidades
                batch_predictions = np.argmax(model.predict(obs_batch, verbose=0), axis=1)
                array_action_pred.extend(batch_predictions)

        return array_action_pred

    def create_and_filter_df(self, dataset_test: np.ndarray, array_action_pred: List[int]) -> pd.DataFrame:
        """
        Cria e formata um DataFrame com colunas na ordem:
        [timestamp, P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, class, action, well].

        `dataset_test` deve ter 8 colunas:
        [timestamp, P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, class, well]
        `array_action_pred` é a ação prevista, adicionada como penúltima coluna (action).
        """

        # 1) dataset_test tem 8 colunas. A última é 'well'.
        #    Separamos 'well' para reinserir depois.
        temp_no_well = dataset_test[:, :-1]  # shape Nx7, sem a coluna 'well'
        well_column = dataset_test[:, -1]    # shape Nx1, que contém 'well'

        # 2) Agora adicionamos array_action_pred como penúltima coluna
        #    e depois recolocamos 'well' como última coluna:
        merged_data = np.column_stack((temp_no_well, array_action_pred, well_column))
        # Assim, merged_data terá 9 colunas:
        # [0] timestamp
        # [1] P-PDG
        # [2] P-TPT
        # [3] T-TPT
        # [4] P-MON-CKP
        # [5] T-JUS-CKP
        # [6] class
        # [7] action
        # [8] well

        df = pd.DataFrame(
            merged_data,
            columns=[
                'timestamp',   # idx 0
                'P-PDG',       # idx 1
                'P-TPT',       # idx 2
                'T-TPT',       # idx 3
                'P-MON-CKP',   # idx 4
                'T-JUS-CKP',   # idx 5
                'class',       # idx 6
                'action',      # idx 7
                'well'         # idx 8
            ]
        )

        # Ajusta o índice para timestamp
        df.set_index('timestamp', inplace=True)

        # Converte colunas de sensores para float32
        sensor_cols = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
        df[sensor_cols] = df[sensor_cols].astype('float32')

        # Converte 'class' e 'action' para int16
        df['class'] = df['class'].astype(float).astype('int16')
        df['action'] = df['action'].astype(float).astype('int16')

        # Caso queira converter 'well' para string (se houver tipos mistos):
        # df['well'] = df['well'].astype(str)

        return df

    def calculate_accuracy(self, df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
        """
        Calcula a acurácia e as taxas TN, TP, FP, FN em relação ao total de previsões.

        :param df: DataFrame contendo colunas 'class' e 'action'.
        :return: (accuracy, TN_rate, TP_rate, FP_rate, FN_rate).
        """
        total = len(df)
        if total == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        TN = len(df[(df['class'] == 0) & (df['action'] == 0)])
        TP = len(df[(df['class'] != 0) & (df['action'] == 1)])
        FP = len(df[(df['class'] == 0) & (df['action'] == 1)])
        FN = len(df[(df['class'] != 0) & (df['action'] == 0)])

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        TN_rate = TN / total
        TP_rate = TP / total
        FP_rate = FP / total
        FN_rate = FN / total

        return accuracy, TN_rate, TP_rate, FP_rate, FN_rate

    def calculate_evaluation_metrics(
        self, TP_rate: float, FP_rate: float, TN_rate: float, FN_rate: float
    ) -> Tuple[float, float, float]:
        """
        Calcula precisão, recall e F1-score a partir das taxas (TP_rate, FP_rate, etc).

        :param TP_rate: Taxa de Verdadeiro Positivo (TP / total).
        :param FP_rate: Taxa de Falso Positivo (FP / total).
        :param TN_rate: Taxa de Verdadeiro Negativo (TN / total).
        :param FN_rate: Taxa de Falso Negativo (FN / total).
        :return: (precision, recall, f1_score).
        """
        TP = TP_rate
        FP = FP_rate
        FN = FN_rate

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1_score

    def validation_model(
        self,
        accuracy: float,
        dataset_validation_scaled: np.ndarray,
        model: Any,
        type_ml: str = 'Supervised'
    ) -> None:
        """
        Valida o modelo com base na acurácia fornecida e nos dados de validação escalados.

        :param accuracy: Acurácia global (fora do escopo, ex. acurácia em outro conjunto).
        :param dataset_validation_scaled: Dados de validação escalados (array 2D).
        :param model: Modelo (RL ou RNA) que possui método .predict().
        :param type_ml: Tipo de aprendizado, 'Supervised' ou outro.
        """
        min_acc_threshold = 0.8
        if accuracy > min_acc_threshold:
            logger.info(
                "Iniciando validação individual. Acurácia=%.3f > %.2f",
                accuracy, min_acc_threshold
            )

            if type_ml == 'Supervised':
                # Ordena pelo timestamp (coluna 0)
                sort_indices = np.argsort(dataset_validation_scaled[:, 0])
                dataset_sorted = dataset_validation_scaled[sort_indices]
                # Separa por gap de 1 hora
                datasets = self.separate_datasets(dataset_sorted)
                logger.info("Separados %d grupos de instâncias para validação.", len(datasets))
            else:
                # Caso não supervisionado, consideramos tudo como 1 único grupo
                datasets = [dataset_validation_scaled]
                logger.info("Usando dataset único para validação (não supervisionado).")

            acc_total = []
            accuracy_vals, test_acc_vals, TN_vals, TP_vals = [], [], [], []
            precision_vals, recall_vals, f1_vals = [], [], []

            all_dfs = []

            for count, dataset_test in enumerate(datasets, start=1):
                logger.info(
                    "Predição da %dª instância de validação com modelo '%s'.",
                    count, self.model_name
                )
                array_action_pred = self.predict_and_evaluate(model, dataset_test)

                df = self.create_and_filter_df(dataset_test, array_action_pred)
                # Mantém cópia com 'well' para métricas por well
                df_copy = df.copy()

                # Remove a última coluna (que é 'well') antes de calcular acurácia
                # para evitar conversão indevida (well = string).
                df = df.iloc[:, :-1]  # drop da coluna 'well'

                acc, TN_rate, TP_rate, FP_rate, FN_rate = self.calculate_accuracy(df)
                acc_total.append(acc)

                precision, recall, f1_score = self.calculate_evaluation_metrics(
                    TP_rate, FP_rate, TN_rate, FN_rate
                )

                logger.info(
                    "Instância %d: Acurácia=%.3f%%, TN=%.3f%%, TP=%.3f%%",
                    count, acc * 100, TN_rate * 100, TP_rate * 100
                )

                accuracy_vals.append(acc * 100)
                test_acc_vals.append(accuracy * 100)  # se deseja plotar a acurácia global
                TN_vals.append(TN_rate * 100)
                TP_vals.append(TP_rate * 100)

                additional_labels = [
                    f'Acurácia (Dataset Teste): {accuracy * 100:.1f}%',
                    f'Acurácia: {acc * 100:.1f}%',
                    f'TN: {TN_rate * 100:.1f}%',
                    f'TP: {TP_rate * 100:.1f}%',
                    f'Precision: {precision:.3f}',
                    f'Recall: {recall:.3f}',
                    f'F1 Score: {f1_score:.3f}'
                ]

                precision_vals.append(precision)
                recall_vals.append(recall)
                f1_vals.append(f1_score)

                logger.info(
                    "Instância %d: Precision=%.3f, Recall=%.3f, F1=%.3f.",
                    count, precision, recall, f1_score
                )

                # Plot dos sensores
                '''expl = Exploration(df)
                expl.plot_sensor(
                    sensor_columns=['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'],
                    title=f'[{count}] - {self.event_name} - {self.model_name}',
                    additional_labels=additional_labels,
                    model=self.model_name
                )'''

                all_dfs.append(df_copy)

            # Plota métricas gerais
            self.plot_and_save_metrics(
                len(datasets), accuracy_vals, test_acc_vals, TN_vals, TP_vals
            )

            final_validation_accuracy = (sum(acc_total) / len(acc_total)) * 100
            logger.info(
                "Acurácia final: %.3f%% no conjunto de dados de validação",
                final_validation_accuracy
            )
            logger.info(
                "Precision=%.3f, Recall=%.3f, F1=%.3f",
                np.mean(precision_vals), np.mean(recall_vals), np.mean(f1_vals)
            )
            print(f"Acurácia final: {final_validation_accuracy:.3f}% no conjunto de dados de validação")

            # Concatena todos os dataframes para avaliar métricas por well
            df_all = pd.concat(all_dfs, axis=0)
            if 'well' in df_all.columns:
                well_metrics_df = self.evaluate_metrics_by_well(df_all)
                self.plot_accuracy_by_well(well_metrics_df)

        else:
            logger.info(
                "Acurácia=%.3f <= %.2f, insuficiente para validação individual",
                accuracy, min_acc_threshold
            )
            print("Acurácia insuficiente para validação individual")

    def plot_and_save_metrics(
        self,
        num_datasets: int,
        accuracy_values: List[float],
        acc_values: List[float],
        TN_values: List[float],
        TP_values: List[float]
    ) -> None:
        """
        Plota e salva as métricas de acurácia por instância de validação.

        :param num_datasets: Quantidade de datasets (instâncias de validação).
        :param accuracy_values: Lista de acurácias por dataset (em %).
        :param acc_values: Lista de acurácias "globais" (teste) (em %), se aplicável.
        :param TN_values: Lista de taxas de TN (em %).
        :param TP_values: Lista de taxas de TP (em %).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        count_iterations = list(range(1, num_datasets + 1))

        ax.plot(count_iterations, accuracy_values, marker='o', color='blue', label='Acurácia (Validação)')
        ax.plot(count_iterations, acc_values, marker='o', color='red', label='Acurácia (Teste)')
        ax.plot(count_iterations, TN_values, marker='o', color='green', label='TN')
        ax.plot(count_iterations, TP_values, marker='o', color='purple', label='TP')

        ax.set(
            title='Métricas de Acurácia por Instância de Validação',
            xlabel='Instâncias de Validação',
            ylabel='Métricas de Acurácia (%)'
        )
        ax.legend()
        ax.grid(True)

        # Caminhos para salvar
        images_path = Path('..', '..', 'img', 'metrics')
        metrics_path = Path('..', 'metrics')
        images_path.mkdir(parents=True, exist_ok=True)
        metrics_path.mkdir(parents=True, exist_ok=True)

        # Salva figura
        fig_save_path = images_path / f"Métricas_{self.event_name}_{self.model_name}.png"
        plt.savefig(fig_save_path)
        logger.info("Gráfico de métricas salvo em %s", fig_save_path)
        plt.close()

        # Salva dados num arquivo texto
        txt_save_path = metrics_path / f"Métricas_{self.event_name}_{self.model_name}.txt"
        with open(txt_save_path, 'w') as txt_file:
            txt_file.write("Count Iteration, Accuracy (%), ACC (%), TN (%), TP (%)\n")
            for i in range(len(count_iterations)):
                txt_file.write(f"{count_iterations[i]}, {accuracy_values[i]}, {acc_values[i]}, "
                               f"{TN_values[i]}, {TP_values[i]}\n")
        logger.info("Métricas numéricas salvas em %s", txt_save_path)

    def evaluate_metrics_by_well(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula as métricas de avaliação (acurácia, precisão, recall e F1-score)
        para cada well presente na coluna 'well' do DataFrame.

        Args:
            df (pd.DataFrame): DataFrame com os dados e que deve conter a coluna 'well'.

        Returns:
            pd.DataFrame: DataFrame contendo as métricas calculadas para cada well.
        """
        if 'well' not in df.columns:
            logger.error("Coluna 'well' não encontrada no DataFrame.")
            return pd.DataFrame()
        
        metrics_list = []
        # Agrupa os dados por well
        for well, group in df.groupby('well'):
            # Calcula acurácia no DataFrame do grupo
            acc, TN_rate, TP_rate, FP_rate, FN_rate = self.calculate_accuracy(group)
            precision, recall, f1_score = self.calculate_evaluation_metrics(TP_rate, FP_rate, TN_rate, FN_rate)

            metrics_list.append({
                'well': well,
                'total': len(group),
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })

        metrics_df = pd.DataFrame(metrics_list)
        logger.info("Métricas calculadas para %d wells.", len(metrics_df))
        return metrics_df

    def plot_accuracy_by_well_old(self, metrics_df: pd.DataFrame) -> None:
        """
        Plota um gráfico de barras com a acurácia (em %) de cada well.

        Args:
            metrics_df (pd.DataFrame): DataFrame contendo as métricas calculadas para cada well.
        """
        if metrics_df.empty:
            logger.error("Nenhum dado de métricas por well para plotar.")
            return

        plt.figure(figsize=(10, 6))
        # Multiplica a acurácia por 100 para exibir em porcentagem
        plt.bar(metrics_df['well'], metrics_df['accuracy'] * 100, color='blue')
        plt.title(f'Acurácia por Well - {self.event_name} - {self.model_name}')
        plt.xlabel('Well')
        plt.ylabel('Acurácia (%)')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')

        images_path = Path('..', '..', 'img', 'metrics')
        images_path.mkdir(parents=True, exist_ok=True)

        save_path = images_path / f'Acuracia_por_well_{self.event_name}_{self.model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Gráfico de acurácia por well salvo em %s", save_path)

    def plot_accuracy_by_well(self, metrics_df: pd.DataFrame) -> None:
        """
        Plota um gráfico de barras da acurácia por well, com layout melhorado para uso acadêmico.
        """
        if metrics_df.empty:
            logger.error("Nenhum dado de métricas por well para plotar.")
            return

        sns.set_theme(style="whitegrid")  # "whitegrid" ou "ticks", a gosto

        fig, ax = plt.subplots(figsize=(6, 4))  # ajuste conforme necessidade

        # Multiplica a acurácia por 100 para exibir em porcentagem
        accuracies_percent = metrics_df["accuracy"] * 100

        # Plota as barras
        bars = ax.bar(
            metrics_df["well"], 
            accuracies_percent, 
            color="royalblue",  # escolha de cor mais suave
            edgecolor="black"   # borda para dar contraste
        )

        # Título e rótulos
        ax.set_title(
            f"Acurácia por Well - {self.event_name} - {self.model_name}",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Well", fontsize=11)
        ax.set_ylabel("Acurácia (%)", fontsize=11)

        # Ajusta limite do eixo Y para dar espaço às anotações
        ax.set_ylim(0, 110)

        # Remove linhas superiores e à direita, para visual mais “clean”
        sns.despine(ax=ax)

        # Adiciona anotações (valores) no topo de cada barra
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",              # formatação em porcentagem
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),                 # deslocamento vertical
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10,
                color="black"
            )

        # Ajuste de espaçamento para evitar cortes de labels
        plt.tight_layout()

        # Usando a mesma convenção de pastas do restante do código
        images_path = Path('..', '..', 'img', 'metrics')
        images_path.mkdir(parents=True, exist_ok=True)

        save_path = images_path / f'Acuracia_por_well_{self.event_name}_{self.model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        logger.info("Gráfico de acurácia por well salvo em %s", save_path)