from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd  # Adicionado para melhorar a leitura de arquivos
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LoadInstances:

    '''
    Exemplo de uso:
    timestamp,P-PDG,P-TPT,T-TPT,P-MON-CKP,T-JUS-CKP,P-JUS-CKGL,T-JUS-CKGL,QGL,class
    2017-02-01 02:02:07.000000,0.000000e+00,1.009211e+07,1.190944e+02,1.609800e+06,8.459782e+01,1.564147e+06,,0.000000e+00,0
    2017-02-01 02:02:08.000000,0.000000e+00,1.009200e+07,1.190944e+02,1.618206e+06,8.458997e+01,1.564148e+06,,0.000000e+00,0
    2017-02-01 02:02:09.000000,0.000000e+00,1.009189e+07,1.190944e+02,1.626612e+06,8.458213e+01,1.564148e+06,,0.000000e+00,0
    2017-02-01 02:02:10.000000,0.000000e+00,1.009178e+07,1.190944e+02,1.635018e+06,8.457429e+01,1.564148e+06,,0.000000e+00,0
    '''
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        
    def class_and_file_generator(self, real=False, simulated=False, drawn=False):
        for class_path in self.data_path.iterdir():
            if class_path.is_dir():
                try:
                    class_code = int(class_path.stem)
                except ValueError:
                    continue

                for instance_path in class_path.iterdir():
                    if instance_path.suffix == '.csv':
                        prefix = instance_path.stem.split('_')[0]
                        if ((simulated and prefix == 'SIMULATED') or 
                            (drawn and prefix == 'DRAWN') or 
                            (real and prefix not in ['SIMULATED', 'DRAWN'])):
                            yield class_code, instance_path
    
    def load_instance_with_numpy(self, events_names):
        well_names = [f'WELL-{i:05d}' for i in range(1, 19)]
        columns = [ 
            'timestamp',      
            'P-PDG',
            'P-TPT',
            'T-TPT',
            'P-MON-CKP',
            'T-JUS-CKP',
            #'P-JUS-CKGL',
            #'T-JUS-CKGL',
            #'QGL',
            'class'
        ]
        real_instances = list(self.class_and_file_generator(real=True))
        logging.info(f'Total de  {len(real_instances)} instâncias reais encontradas.')
        arrays_list = []

        for class_code, instance_path in real_instances:
            well, _ = instance_path.stem.split('_')
            
            if class_code in events_names and well in well_names:
                df = pd.read_csv(instance_path, usecols=columns + (['timestamp'] if 'timestamp' in columns else []))
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)
                
                if 'timestamp' in columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
                    arr = df.to_numpy()
                else:
                    arr = df.to_numpy(dtype=np.float32)
                    arr[:, -1] = arr[:, -1].astype(np.int16)
                
                arrays_list.append(arr)
        
        logging.info(f'Total de {len(arrays_list)} instâncias reais carregadas para o evento {events_names}.')
        final_array = np.concatenate(arrays_list) if arrays_list else np.array([])
        
        return final_array
    
    # Função para aplicar undersampling no dataset
    def apply_undersampling(X, y):
        logging.info("Iniciando o processo de undersampling.")
        
        # Concatenar os arrays de features e target para facilitar o resampling
        dataset = np.column_stack((X, y))
        
        # Separar os datasets por classe
        datasets_by_class = {label: dataset[dataset[:, -1] == label] for label in np.unique(dataset[:, -1])}
        
        # Encontrar o tamanho da menor classe
        min_class_size = min(len(datasets_by_class[label]) for label in datasets_by_class)
        logging.info(f"Tamanho da menor classe: {min_class_size}")
        
        # Aplicar undersampling em cada classe para igualar ao tamanho da menor classe
        undersampled_datasets = []
        for label in datasets_by_class:
            undersampled_data = resample(datasets_by_class[label], replace=False, n_samples=min_class_size, random_state=42)
            undersampled_datasets.append(undersampled_data)
            logging.info(f"Classe {label} foi undersampled para {min_class_size} instâncias.")
        
        
        # Combinar todos os datasets undersampled em um único conjunto
        undersampled_dataset = np.vstack(undersampled_datasets)
        
        # Embaralhar o dataset final para garantir uma distribuição aleatória
        np.random.shuffle(undersampled_dataset)
        logging.info("Dataset final undersampled e embaralhado.")
        
        # Separar novamente em X e y
        X_undersampled, y_undersampled = undersampled_dataset[:, :-1], undersampled_dataset[:, -1]
        
        return X_undersampled, y_undersampled
    
    def data_preparation(self, dataset, train_percentage):

        # Inicializando listas para guardar índices de treino e teste
        train_indices = []
        test_indices = []

        # Processamento genérico para cada classe
        for event in np.unique(dataset[:, -1]):
            # Selecionando índices para a classe atual        
            class_indices = np.where(dataset[:, -1] == event)[0]
            
            # Logando o número de amostras por classe
            print(f'Número de amostras da classe {event}: {len(class_indices)}')
            logging.info(f'Número de amostras da classe {event}: {len(class_indices)}')
            
            #O parâmetro random_state=42 garante que essa divisão seja feita de maneira reproducível, ou seja, a função produzirá o mesmo resultado cada vez que for executada com o mesmo estado aleatório. 
            # Dividindo os índices da classe atual em treino e teste
            class_train_indices, class_test_indices = train_test_split(class_indices, train_size=train_percentage) # , random_state=42
            
            # Logando o número de amostras de treino e teste
            logging.info(f'Número de amostras de treino da classe {event}: {len(class_train_indices)}')
            logging.info(f'Número de amostras de teste da classe {event}: {len(class_test_indices)}')
            
            # Adicionando aos índices gerais de treino e teste
            train_indices.extend(class_train_indices)
            test_indices.extend(class_test_indices)

        # Convertendo listas para arrays numpy para futura manipulação
        train_indices = np.array(train_indices)    
        test_temp_indices = np.array(test_indices)       

        test_indices, validation_indices = train_test_split(test_temp_indices, test_size=0.5) # , random_state=42

        # Embaralhando os índices (opcional, dependendo da necessidade)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        

        # Criando conjuntos de dados de treino e teste
        dataset_train = dataset[train_indices]
        dataset_test = dataset[test_indices]
        dataset_validation = dataset[validation_indices]

        logging.info(f'Número de registros de treino: {len(dataset_train)}')
        logging.info(f'Número de registros de teste: {len(dataset_test)}')
        logging.info(f'Número de registros de validação: {len(dataset_validation)}')
        

        # Dividindo em features (X) e target (y)
        X_train, y_train = dataset_train[:, :-1], dataset_train[:, -1]
        X_test, y_test = dataset_test[:, :-1], dataset_test[:, -1]
        X_validation, y_validation = dataset_validation[:, :-1], dataset_validation[:, -1]

        # Delete a primeira coluna (timestamp) das features
        X_train = np.delete(X_train, 0, axis=1)
        X_test = np.delete(X_test, 0, axis=1)

        # Escalonando as features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_validation_scaled = np.column_stack((X_validation[:, 0], scaler.transform(X_validation[:, 1:])))    
    
        #X_train_undersampled, y_train_undersampled = apply_undersampling(X_train_scaled, y_train)
        #X_test_undersampled, y_test_undersampled = apply_undersampling(X_test_scaled, y_test) # Se desejar aplicar no teste também


        # Se necessário, você pode combinar as features escalonadas e o target para formar os datasets finais
        dataset_train_scaled = np.column_stack((X_train_scaled, y_train))
        dataset_test_scaled = np.column_stack((X_test_scaled, y_test))
        #dataset_train_scaled = np.column_stack((X_train_undersampled, y_train_undersampled))
        #dataset_test_scaled = np.column_stack((X_test_undersampled, y_test_undersampled))
        dataset_validation_scaled = np.column_stack((X_validation_scaled, y_validation))

        logging.info(f'Número de registros de treino undersampling: {len(dataset_train_scaled)}')
        logging.info(f'Número de registros de teste undersampling: {len(dataset_test_scaled)}')
        logging.info(f'Número de registros de validação: {len(dataset_validation_scaled)}')       
        logging.info(f'Fim divisão do dataset em treino e teste')

        return dataset_train_scaled, dataset_test_scaled, dataset_validation_scaled
