import logging
import numpy as np
import pandas as pd
from datetime import datetime
from classes._exploration import exploration


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
        if self.model_name == 'RNN':
            obs = np.expand_dims(obs, axis=0)
        return obs

    def predict_and_evaluate(self, model, dataset_test):
        '''
            TP (Verdadeiro Positivo): Ação correta (1) e falha (1)
            TN (Verdadeiro Negativo): Ação correta (0) e não falha (0)
            FP (Falso Positivo): Ação incorreta (1) e não falha (0)
            FN (Falso Negativo): Ação incorreta (0) e falha (1)
        
        '''
                
        acc = 0
        array_action_pred = []
        # Inicialização da matriz de confusão
        TP, FP, TN, FN = 0, 0, 0, 0
        
        for row in dataset_test:
            obs = self.preprocess_observation(row)
            action = model.predict(obs, deterministic=True)[0] if self.model_name != 'RNN' else np.argmax(model.predict(obs, verbose=0), axis=1)
            array_action_pred.append(action)

            if (row[-1] in range(1, 10) and action == 1) or (row[-1] in range(101, 110) and action == 1):                
                TP += 1
            elif row[-1] == 0 and action == 0:
                TN += 1
            elif row[-1] == 0 and action == 1:
                FP += 1
            elif (row[-1] in range(1, 10) and action == 0) or (row[-1] in range(101, 110) and action == 0):
                FN += 1           

            # Calculando a acurácia
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            
        return acc, array_action_pred

    def create_and_filter_df(self, dataset_test, array_action_pred):
        df = pd.DataFrame(np.column_stack((dataset_test, array_action_pred)),
                        columns=['timestamp', 'P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'class', 'action'])
        df.set_index('timestamp', inplace=True)
        df[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']] = df[['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']].astype('float32')
        df['class'], df['action'] = df['class'].astype(float).astype('int16'), df['action'].astype(float).astype('int16')
        return df

    def calculate_accuracy(self, df):
        numerator_normal = len(df[(df['class'] == 0) & (df['action'] == 0)])
        denominator_normal = len(df[df['class'] == 0])
        TN = (numerator_normal / denominator_normal) if denominator_normal > 0 else 0

        numerator_falha = len(df[(df['class'] != 0) & (df['action'] == 1)])
        denominator_falha = len(df[df['class'] != 0])
        TP = (numerator_falha / denominator_falha) if denominator_falha > 0 else 0

        return TN, TP

    def validation_model(self, accuracy, dataset_validation_scaled, model):
        if accuracy > 0.8:
            logging.info('Iniciando a separação dos grupos de dados para validação individual')
            sort_indices = np.argsort(dataset_validation_scaled[:, 0])
            dataset_validation_sorted = dataset_validation_scaled[sort_indices]

            datasets = self.separate_datasets(dataset_validation_sorted)
            logging.info(f'Fim da separação dos grupos de dados para validação com {len(datasets)} grupos de instâncias')

            acc_total, array_prec_total = [], []
            for count, dataset_test in enumerate(datasets):
                logging.info(f'Iniciando predição da {count}ª instância de validação usando {self.model_name}')
                acc, array_action_pred = self.predict_and_evaluate(model, dataset_test)
                
                acc_total.append(acc)
                array_prec_total.append(len(array_action_pred))
                

                df = self.create_and_filter_df(dataset_test, array_action_pred)
                TN, TP = self.calculate_accuracy(df)

                logging.info(f'Acurácia da {count}ª instância: {acc  * 100:.3f}%')
                logging.info(f'Verdadeiro Negativo na {count}ª instância: {TN * 100:.3f}%')
                logging.info(f'Verdadeiro Positivo na {count}ª instância: {TP * 100:.3f}%')
                logging.info(f'Fim predição da instância de teste {self.model_name}')

                # Aqui você adicionaria o código para a plotagem, se necessário
                additional_labels = [
                f'Acurácia (Teste): {accuracy * 100:.1f}%', 
                f'Acurácia (Validação): {acc  * 100:.1f}%',  
                f'TN: {TN * 100:.1f}%',  
                f'TP: {TP * 100:.1f}%' 
                ]

                explora = exploration(df)
                explora.plot_sensor(sensor_columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP'],
                                     _title = f'[{count}] - {self.event_name} - {self.model_name}', additional_labels =  additional_labels, model = self.model_name)

            final_validation_accuracy = sum(acc_total) / len(acc_total) * 100
            logging.info(f'Acurácia: {final_validation_accuracy:.3f}% no conjunto de dados de validação')

        else:
            logging.info(f'Acurácia insuficiente para validação individual')
            print(f'Acurácia insuficiente para validação individual')
