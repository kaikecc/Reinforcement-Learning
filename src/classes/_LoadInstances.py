
from pathlib import Path
import numpy as np
from datetime import datetime


class LoadInstances:
    def __init__(self, data_path):        
        self.data_path = Path(data_path) 
        
    def class_and_file_generator(self, real=False, simulated=False, drawn=False):
        for class_path in self.data_path.iterdir():
            if class_path.is_dir():
                try:
                    class_code = int(class_path.stem)
                except ValueError:
                    # Se não for possível converter para int, pule este diretório
                    continue

                for instance_path in class_path.iterdir():
                    if (instance_path.suffix == '.csv'):
                        if (simulated and instance_path.stem.startswith('SIMULATED')) or \
                        (drawn and instance_path.stem.startswith('DRAWN')) or \
                        (real and (not instance_path.stem.startswith('SIMULATED')) and \
                        (not instance_path.stem.startswith('DRAWN'))):
                            yield class_code, instance_path
    
    def load_instance_with_numpy(self, events_names, columns):
        
        
        well_names = [f'WELL-{i:05d}' for i in range(1, 19)]        
        real_instances = list(self.class_and_file_generator(real=True, simulated=False, drawn=False))

        arrays_list = []  # List to store processed NumPy arrays

        for instance in real_instances:
            class_code, instance_path = instance    
            well, _ = instance_path.stem.split('_')  
                
            if class_code in events_names.keys() and well in well_names:

                # Read the entire CSV file into a NumPy array
                with open(instance_path, 'r') as file:
                    header = file.readline().strip().split(',')
                    indices = [header.index(col) for col in columns]
                                                    
                    if 'timestamp' in columns:
                        arr = np.genfromtxt(file, delimiter=',', usecols=indices[1:], dtype=np.float32)
                        arr[np.isinf(arr)] = np.nan
                        timestamp_idx = header.index('timestamp')
                        # Primeiro, vamos carregar todos os timestamps como um array numpy
                        file.seek(0)
                        file.readline()  # Skip the header
                        timestamps = np.genfromtxt(file, delimiter=',', skip_header=0, usecols=timestamp_idx, dtype=str)
                        if isinstance(timestamps, str):
                            timestamps = np.array([timestamps])

                        # Converta os timestamps para o formato desejado antes de aplicar o filtro
                        fmt = "%Y-%m-%d %H:%M:%S.%f"
                        rounded_timestamps = np.array([datetime.strptime(ts, fmt).strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps])

                        # Agora, aplicamos o mesmo filtro usado em 'arr' para 'rounded_timestamps'
                        # Precisamos determinar quais linhas em 'arr' NÃO serão removidas por conter NaN
                        not_nan_rows = ~np.isnan(arr).any(axis=1)

                        # Filtramos 'arr' e 'rounded_timestamps' usando este índice
                        arr_filtered = arr[not_nan_rows]
                        rounded_timestamps_filtered = rounded_timestamps[not_nan_rows]

                        # Agora, ambos 'arr_filtered' e 'rounded_timestamps_filtered' estão alinhados
                        # e podemos concatená-los sem enfrentar o problema de dimensões incompatíveis
                        final_data = np.hstack([rounded_timestamps_filtered.reshape(-1, 1), arr_filtered])

                        arrays_list.append(final_data)
                    else:
                        arr = np.genfromtxt(file, delimiter=',', usecols=indices, dtype=np.float32)
                        # Tratando NaN e valores infinitos antes da conversão
                        arr[np.isinf(arr)] = np.nan
                        arr = arr[~np.isnan(arr).any(axis=1)]
                        arr[:, :-1] = arr[:, :-1].astype(np.float32)  # Convert selected columns to float32
                        arr[:, -1] = arr[:, -1].astype(np.int16)  # Convert 'class' column to int16
                        
                        arrays_list.append(arr)               

        # Concatenate all processed NumPy arrays
        final_array = np.concatenate(arrays_list) if arrays_list else np.array([])

        return final_array
