from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd  # Adicionado para melhorar a leitura de arquivos
import logging

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
    
    def load_instance_with_numpy(self, events_names, columns):
        well_names = [f'WELL-{i:05d}' for i in range(1, 19)]
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
        
        final_array = np.concatenate(arrays_list) if arrays_list else np.array([])
        
        return final_array
