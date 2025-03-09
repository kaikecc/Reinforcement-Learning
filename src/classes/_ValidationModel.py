import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classes._exploration import Exploration



# Obtém o logger global
logger = logging.getLogger("global_logger")

# Parameter for accuracy
ACCURACY_THRESHOLD = 0.01

class ValidationModel:
    def __init__(self, model_name: str, event_name: str) -> None:
        self.model_name = model_name
        self.event_name = event_name
        # Cria um logger específico para a classe
        
        logger.info("Initialized ValidationModel for model '%s' and event '%s'.", 
                         self.model_name, self.event_name)

    def separate_datasets(self, dataset_validation_sorted: np.ndarray) -> List[np.ndarray]:
        """
        Separa o dataset validado em grupos de instâncias com base em um gap temporal superior a 1 hora.
        """
        logger.debug("Starting separation of datasets based on time gaps.")
        try:
            timestamps = pd.to_datetime(dataset_validation_sorted[:, 0], format='%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.error("Error converting timestamps: %s", e)
            raise

        diffs = timestamps.diff().total_seconds() / 3600
        split_indices = np.where(diffs > 1)[0]
        if len(split_indices) > 0:
            datasets = np.split(dataset_validation_sorted, split_indices)
        else:
            datasets = [dataset_validation_sorted]
        logger.info("Separated dataset into %d groups based on time gap.", len(datasets))
        return datasets

    def preprocess_observation(self, row: np.ndarray) -> np.ndarray:
        """
        Pré-processa uma linha de observação convertendo os dados (exceto o timestamp e a classe) para float32.
        """
        logger.debug("Preprocessing observation: %s", row)
        return row[1:-1].astype(np.float32)

    def create_batches(self, data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
        """
        Gera lotes (batches) dos dados.
        """
        logger.debug("Creating batches of size %d.", batch_size)
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def predict_and_evaluate(self, model: Any, dataset_test: List[Any], batch_size: int = 32) -> List[Any]:
        """
        Realiza a predição das ações para o dataset de teste, tratando de forma diferenciada o modelo RNA.
        """
        dataset_test = dataset_test[:, 1:]  # Remove timestamp column
        logger.debug("Starting prediction on dataset with %d instances.", len(dataset_test))
        array_action_pred = []

        if self.model_name != 'RNA':
            for index, row in enumerate(dataset_test):
                obs = self.preprocess_observation(row)
                try:
                    action = model.predict(obs, deterministic=True)[0]
                except Exception as e:
                    logger.error("Prediction error at index %d: %s", index, e)
                    raise
                array_action_pred.append(action)
        else:
            for batch_index, batch in enumerate(self.create_batches(dataset_test, batch_size)):
                logger.debug("Processing batch %d with %d instances.", batch_index + 1, len(batch))
                obs_batch = np.array([self.preprocess_observation(row) for row in batch])
                try:
                    batch_predictions = np.argmax(model.predict(obs_batch, verbose=0), axis=1)
                except Exception as e:
                    logger.error("Batch prediction error in batch %d: %s", batch_index + 1, e)
                    raise
                array_action_pred.extend(batch_predictions)

        logger.info("Completed prediction; total predictions: %d", len(array_action_pred))
        return array_action_pred

    def create_and_filter_df(self, dataset_test: np.ndarray, array_action_pred: List[Any]) -> pd.DataFrame:
        """
        Cria e formata um DataFrame a partir do dataset de teste e das predições geradas.
        """
        logger.debug("Creating DataFrame from test dataset and predictions.")
        df = pd.DataFrame(
            np.column_stack((dataset_test, array_action_pred)),
            columns=['timestamp', 'P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'class', 'action']
        )
        df.set_index('timestamp', inplace=True)
        sensor_cols = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
        df[sensor_cols] = df[sensor_cols].astype('float32')
        df['class'] = df['class'].astype(float).astype('int16')
        df['action'] = df['action'].astype(float).astype('int16')
        logger.info("DataFrame created with %d entries.", len(df))
        return df

    def calculate_accuracy(self, df: pd.DataFrame) -> Tuple[float, int, int, int, int]:
        """
        Calcula a acurácia e os contadores de TN, TP, FP e FN a partir do DataFrame.
        """
        total = len(df)
        if total == 0:
            logger.warning("Empty DataFrame passed to calculate_accuracy; returning zeros.")
            return 0.0, 0, 0, 0, 0

        TN = len(df[(df['class'] == 0) & (df['action'] == 0)])
        TP = len(df[(df['class'] != 0) & (df['action'] == 1)])
        FP = len(df[(df['class'] == 0) & (df['action'] == 1)])
        FN = len(df[(df['class'] != 0) & (df['action'] == 0)])
        accuracy = (TP + TN) / total
        logger.debug("Accuracy calculation: TP=%d, TN=%d, FP=%d, FN=%d, Accuracy=%.3f", 
                            TP, TN, FP, FN, accuracy)
        return accuracy, TN, TP, FP, FN

    def calculate_evaluation_metrics(
        self, TP: int, FP: int, TN: int, FN: int
    ) -> Tuple[float, float, float]:
        """
        Calcula as métricas de precisão, recall e F1-score.
        """
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        logger.debug("Evaluation metrics: Precision=%.3f, Recall=%.3f, F1=%.3f", 
                          precision, recall, f1_score)
        return precision, recall, f1_score

    def validation_model(self, accuracy: float, dataset_validation_scaled: np.ndarray, model: Any,
                         type_ml: str = 'Supervised') -> None:
        """
        Valida o modelo com base na acurácia e nos dados de validação escalados.
        """
        logger.info("Starting validation for model '%s' on event '%s' with accuracy %.3f.", 
                         self.model_name, self.event_name, accuracy)
        if accuracy > ACCURACY_THRESHOLD:
            logger.info(f"Model accuracy is sufficient for validation (%.3f > {ACCURACY_THRESHOLD}).", accuracy)
            if type_ml == 'Supervised':
                sort_indices = np.argsort(dataset_validation_scaled[:, 0])
                dataset_validation_sorted = dataset_validation_scaled[sort_indices]
                datasets = self.separate_datasets(dataset_validation_sorted)
                logger.info("Validation dataset separated into %d groups.", len(datasets))
            else:
                datasets = dataset_validation_scaled
                logger.info("Non-supervised learning: using provided dataset without separation.")

            acc_total = []
            accuracy_values, acc_values, TN_values, TP_values = [], [], [], []
            precision_values, recall_values, f1_score_values = [], [], []

            for count, dataset in enumerate(datasets):
                dataset_test = dataset[:, :-1] # Remove timestamp and class columns
                logger.info("Starting prediction for instance %d using model '%s'.", count + 1, self.model_name)
                array_action_pred = self.predict_and_evaluate(model, dataset_test)
                df = self.create_and_filter_df(dataset_test, array_action_pred)
                total = len(df)
                acc_val, TN_count, TP_count, FP_count, FN_count = self.calculate_accuracy(df)
                acc_total.append(acc_val)

                tn_rate = (TN_count / total) * 100 if total > 0 else 0
                tp_rate = (TP_count / total) * 100 if total > 0 else 0

                precision, recall, f1_score = self.calculate_evaluation_metrics(TP_count, FP_count, TN_count, FN_count)

                logger.info("Instance %d: Accuracy=%.3f%%, TN=%.3f%%, TP=%.3f%%", 
                                 count + 1, acc_val * 100, tn_rate, tp_rate)

                accuracy_values.append(acc_val * 100)
                acc_values.append(accuracy * 100)  # Confirmar o contexto de uso
                TN_values.append(tn_rate)
                TP_values.append(tp_rate)

                additional_labels = [
                    f'Acurácia (Dataset Teste): {accuracy * 100:.1f}%',
                    f'Acurácia: {acc_val * 100:.1f}%',
                    f'TN: {tn_rate:.1f}%',
                    f'TP: {tp_rate:.1f}%',
                    f'Precision: {precision:.3f}',
                    f'Recall: {recall:.3f}',
                    f'F1 Score: {f1_score:.3f}'
                ]

                precision_values.append(precision)
                recall_values.append(recall)
                f1_score_values.append(f1_score)

                logger.info("Instance %d: Precision=%.3f, Recall=%.3f, F1 Score=%.3f.", 
                                 count + 1, precision, recall, f1_score)

                logger.info("Starting plotting for instance %d.", count + 1)
                explora = Exploration(df)
                explora.plot_sensor(
                    sensor_columns=['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'],
                    title=f'[{count}] - {self.event_name} - {self.model_name}',
                    additional_labels=additional_labels,
                    model=self.model_name
                )

                logger.info("Completed plotting for instance %d.", count + 1)

            logger.info("Starting to plot and save overall validation metrics for model '%s'.", self.model_name)
            self.plot_and_save_metrics(len(datasets), accuracy_values, acc_values, TN_values, TP_values, downsample=False)
            logger.info("Completed plotting and saving metrics for model '%s'.", self.model_name)

            final_validation_accuracy = (sum(acc_total) / len(acc_total)) * 100
            logger.info("Final validation accuracy: %.3f%%", final_validation_accuracy)
            logger.info("Overall metrics: Precision=%.3f, Recall=%.3f, F1 Score=%.3f.", 
                             sum(precision_values) / len(precision_values),
                             sum(recall_values) / len(recall_values),
                             sum(f1_score_values) / len(f1_score_values))
            print(f'Acurácia final: {final_validation_accuracy:.3f}% no conjunto de dados de validação')
        else:
            logger.info("Model accuracy insufficient for individual validation (%.3f <= 0.8).", accuracy)
            print('Acurácia insuficiente para validação individual')

    def plot_and_save_metrics(
        self, 
        num_datasets: int, 
        accuracy_values: List[float], 
        acc_values: List[float],
        TN_values: List[float], 
        TP_values: List[float], 
        max_points: int = 50,
        downsample: bool = False
    ) -> None:
        """
        Plota e salva as métricas de validação de forma otimizada.
        Se o número de instâncias for maior que max_points e downsample for True, os dados serão amostrados para
        reduzir o tempo de plotagem.

        Args:
            num_datasets (int): Número total de instâncias.
            accuracy_values (List[float]): Lista de acurácias (validação) em %.
            acc_values (List[float]): Lista de acurácias (teste) em %.
            TN_values (List[float]): Lista de taxas de TN em %.
            TP_values (List[float]): Lista de taxas de TP em %.
            max_points (int, opcional): Número máximo de pontos a serem plotados. Padrão é 50.
            downsample (bool, opcional): Se True, realiza downsampling quando necessário; se False, plota todos os pontos.
        """
        logger.debug("Plotting and saving metrics for %d instances.", num_datasets)
        
        if downsample and num_datasets > max_points:
            indices = np.linspace(0, num_datasets - 1, max_points, dtype=int)
            accuracy_sampled = [accuracy_values[i] for i in indices]
            acc_sampled = [acc_values[i] for i in indices]
            TN_sampled = [TN_values[i] for i in indices]
            TP_sampled = [TP_values[i] for i in indices]
            count_iterations = [i + 1 for i in indices]
            logger.info("Downsampled data from %d to %d points for plotting.", num_datasets, max_points)
        else:
            accuracy_sampled = accuracy_values
            acc_sampled = acc_values
            TN_sampled = TN_values
            TP_sampled = TP_values
            count_iterations = list(range(1, num_datasets + 1))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(count_iterations, accuracy_sampled, marker='o', color='blue', label='Acurácia (Validação)')
        ax.plot(count_iterations, acc_sampled, marker='o', color='red', label='Acurácia (Teste)')
        ax.plot(count_iterations, TN_sampled, marker='o', color='green', label='TN')
        ax.plot(count_iterations, TP_sampled, marker='o', color='purple', label='TP')

        ax.set(
            title='Métricas de Acurácia por Instância de Validação',
            xlabel='Instâncias de Validação',
            ylabel='Métricas de Acurácia (%)'
        )
        ax.legend()
        ax.grid(True)

        images_path = Path('..', '..', 'img', 'metrics')
        metrics_path = Path('..', 'metrics')
        images_path.mkdir(parents=True, exist_ok=True)
        metrics_path.mkdir(parents=True, exist_ok=True)

        save_path = images_path / f'Métricas_{self.event_name}_{self.model_name}.png'
        plt.savefig(save_path)
        logger.info("Saved metrics plot to %s", save_path)
        plt.close()

        metrics_file = metrics_path / f'Métricas_{self.event_name}_{self.model_name}.txt'
        with open(metrics_file, 'w') as txt_file:
            txt_file.write('Count Iteration, Accuracy (%), ACC (%), TN (%), TP (%)\n')
            for i in range(len(count_iterations)):
                txt_file.write(f'{count_iterations[i]}, {accuracy_sampled[i]}, {acc_sampled[i]}, {TN_sampled[i]}, {TP_sampled[i]}\n')
        logger.info("Saved metrics data to %s", metrics_file)


