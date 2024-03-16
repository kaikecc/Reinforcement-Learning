import logging
import numpy as np
import pandas as pd
from datetime import datetime
from classes._exploration import exploration
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

class ValidationModel():
    def __init__(self, model_name, event_name):
        self.model_name = model_name
        self.event_name = event_name

    def separate_datasets(self, dataset_validation_sorted):
        datasets = []
        current_dataset = []
        previous_datetime = None

        for row in dataset_validation_sorted:
            current_datetime = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            if previous_datetime is None or (current_datetime - previous_datetime).total_seconds() / 3600 > 1:
                if current_dataset:
                    datasets.append(np.array(current_dataset))
                    current_dataset = []
            current_dataset.append(row)
            previous_datetime = current_datetime

        if current_dataset:
            datasets.append(np.array(current_dataset))
        return datasets

    def preprocess_observation(self, row):
        obs = row[1:-1].astype(np.float32)        
        return obs

    def create_batches(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

    def predict_and_evaluate(self, model, dataset_test, batch_size=32):
        
        array_action_pred = []

        if self.model_name != 'RNA':
            for row in dataset_test:
                obs = self.preprocess_observation(row)
                action = model.predict(obs, deterministic=True)[0]
                array_action_pred.append(action)                
        else:
            # Processamento em lote para RNA
            batches = self.create_batches(dataset_test, batch_size)
            for batch in batches:
                # Preprocessa cada observação no lote
                obs_batch = np.array([self.preprocess_observation(row) for row in batch])
                # Ajusta a entrada para o formato esperado pelo RNN se necessário
                #obs_batch = np.expand_dims(obs_batch, axis=1)  # Ajuste conforme a necessidade do seu modelo RNN
                batch_predictions = np.argmax(model.predict(obs_batch, verbose=0), axis=1)
                array_action_pred.extend(batch_predictions)
               

        return array_action_pred


    def create_and_filter_df(self, dataset_test, array_action_pred):
        df = pd.DataFrame(np.column_stack((dataset_test, array_action_pred)),
                        columns=['timestamp', 'P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'class', 'action'])
        df.set_index('timestamp', inplace=True)
        df[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']] = df[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']].astype('float32')
        df['class'], df['action'] = df['class'].astype(float).astype('int16'), df['action'].astype(float).astype('int16')
        return df

    def calculate_accuracy(self, df):
        # Calcula o total de previsões para o cálculo da acurácia
        total_de_previsoes = len(df)
        
        # Se não houver previsões, retorna zero para todas as métricas
        if total_de_previsoes == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Calcula TN, TP, FP, FN
        TN = len(df[(df['class'] == 0) & (df['action'] == 0)])
        TP = len(df[(df['class'] != 0) & (df['action'] == 1)])
        FP = len(df[(df['class'] == 0) & (df['action'] == 1)])
        FN = len(df[(df['class'] != 0) & (df['action'] == 0)])
        
        # Converte os valores para taxas dividindo pelo total de previsões
        TN_rate = TN / total_de_previsoes
        TP_rate = TP / total_de_previsoes
        FP_rate = FP / total_de_previsoes
        FN_rate = FN / total_de_previsoes

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        
        return accuracy, TN_rate, TP_rate, FP_rate, FN_rate


    def calculate_evaluation_metrics(self, TP, FP, TN, FN):
        # Calcula precisão, recall e F1-score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1_score
    
    def validation_model(self, accuracy, dataset_validation_scaled, model):
        """Valida o modelo com base na acurácia fornecida e nos dados de validação escalados."""
        if accuracy > 0.8:
            logging.info('Iniciando a separação dos grupos de dados para validação individual')
            sort_indices = np.argsort(dataset_validation_scaled[:, 0])
            dataset_validation_sorted = dataset_validation_scaled[sort_indices]

            datasets = self.separate_datasets(dataset_validation_sorted)
            logging.info(f'Fim da separação dos grupos de dados para validação com {len(datasets)} grupos de instâncias')

            acc_total = []
            accuracy_values, acc_values, TN_values, TP_values = [], [], [], []

            for count, dataset_test in enumerate(datasets):
                logging.info(f'Iniciando predição da {count + 1}ª instância de validação usando {self.model_name}')
                array_action_pred = self.predict_and_evaluate(model, dataset_test)
                
                df = self.create_and_filter_df(dataset_test, array_action_pred)
                acc, TN, TP, FP, FN = self.calculate_accuracy(df)
                acc_total.append(acc)

                precision, recall, f1_score = self.calculate_evaluation_metrics(TP, FP, TN, FN)


                logging_details = f'Acurácia da {count + 1}ª instância: {acc * 100:.3f}%, ' \
                                f'Verdadeiro Negativo: {TN * 100:.3f}%, Verdadeiro Positivo: {TP * 100:.3f}%'
                logging.info(logging_details)

                accuracy_values.append(acc * 100)
                acc_values.append(accuracy * 100)  # Revisar se este uso está correto
                TN_values.append(TN * 100)
                TP_values.append(TP * 100)

                additional_labels = [
                f'Acurácia (Dataset Teste): {accuracy * 100:.1f}%', 
                f'Acurácia: {acc  * 100:.1f}%',  
                f'TN: {TN * 100:.1f}%',  
                f'TP: {TP * 100:.1f}%', 
                f'Precision: {precision:.3f}',
                f'Recall: {recall:.3f}',
                f'F1 Score: {f1_score:.3f}' 
                ]

                explora = exploration(df)
                explora.plot_sensor(sensor_columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'],
                                     _title = f'[{count}] - {self.event_name} - {self.model_name}', additional_labels =  additional_labels, model = self.model_name)

            # Plotagem e salvamento das métricas
            self.plot_and_save_metrics(len(datasets), accuracy_values, acc_values, TN_values, TP_values)

            final_validation_accuracy = sum(acc_total) / len(acc_total) * 100
            logging.info(f'Acurácia final: {final_validation_accuracy:.3f}% no conjunto de dados de validação')
        else:
            logging.info('Acurácia insuficiente para validação individual')
            print('Acurácia insuficiente para validação individual')

    def plot_and_save_metrics(self, num_datasets, accuracy_values, acc_values, TN_values, TP_values):
        """Plota e salva as métricas de validação."""
        fig, ax = plt.subplots(figsize=(10, 6))
        count_iterations = list(range(1, num_datasets + 1))

        ax.plot(count_iterations, accuracy_values, marker='o', color='blue', label='Acurácia (Validação)')
        ax.plot(count_iterations, acc_values, marker='o', color='red', label='Acurácia (Teste)')
        ax.plot(count_iterations, TN_values, marker='o', color='green', label='TN')
        ax.plot(count_iterations, TP_values, marker='o', color='purple', label='TP')

        ax.set(title='Métricas de Acurácia por Instância de Validação', xlabel='Instâncias de Validação', ylabel='Métricas de Acurácia (%)')
        ax.legend()
        ax.grid(True)

        images_path = Path(f'..\\..\\img\\metrics')
        metrics_path = Path(f'..\\metrics\\')
        images_path.mkdir(parents=True, exist_ok=True)
        metrics_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(images_path / f'Métricas_{self.event_name}_{self.model_name}.png')
        plt.close()

        with open(metrics_path / f'Métricas_{self.event_name}_{self.model_name}.txt', 'w') as txt_file:
            txt_file.write('Count Iteration, Accuracy (%), ACC (%), TN (%), TP (%)\n')
            for i in range(len(count_iterations)):
                txt_file.write(f'{count_iterations[i]}, {accuracy_values[i]}, {acc_values[i]}, {TN_values[i]}, {TP_values[i]}\n')
