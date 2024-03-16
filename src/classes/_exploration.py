import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
from collections import OrderedDict
import logging

class exploration():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def quartiles_plot(self, sensors, _title):

        '''directory = os.path.dirname(_title)
        if not os.path.exists(directory):
            os.makedirs(directory)'''

        # Garantir que os valores de 'class' estejam no formato correto
        self.dataframe['class'] = self.dataframe['class'].fillna('-1', inplace=False)
        self.dataframe['class'] = self.dataframe['class'].astype(float).astype(int).astype(str)
        
        base_colors = {'Normal': 'lightgreen', 'Estável de Anomalia': 'lightcoral', 'Transiente de Anomalia': 'lightyellow', 'Não Rotulado': 'lightgrey'}
        legend_class = {
            '0': 'Normal', 
            '-1': 'Não Rotulado',
            **{str(i): 'Estável de Anomalia' for i in range(1, 9)},
            **{str(100 + i): 'Transiente de Anomalia' for i in range(1, 9)}
        }
        
        # Aplicar cores com base em legend_class para manter consistência
        class_colors = {cls: base_colors[label] for cls, label in legend_class.items()}
        
        # Gerar patches para a legenda de forma otimizada
        unique_labels = np.unique(list(legend_class.values()))
        patches = [mpatches.Patch(color=base_colors[label], label=label) for label in unique_labels]
        

        # Calculando a frequência das classes e identificando as top classes
        class_counts = self.dataframe['class'].value_counts()
        top_classes = class_counts.nlargest(100).index

        # Filtrando os dados para incluir apenas as top classes
        filtered_data = self.dataframe[self.dataframe['class'].isin(top_classes)]

        # Criação de um DataFrame vazio com as mesmas colunas do DataFrame original para armazenar os dados filtrados
        #filtered_data_without_outliers = pd.DataFrame(columns=self.dataframe.columns)
        filtered_data_near_median = pd.DataFrame(columns=self.dataframe.columns)


        # Filtrando outliers para cada sensor e classe
        '''for var in sensors:
            for cls in filtered_data['class'].unique():
                subset = filtered_data[filtered_data['class'] == cls][var]
                Q1 = subset.quantile(0.25)
                Q3 = subset.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Filtragem dos dados dentro dos limites para cada var e cls
                condition = (filtered_data['class'] == cls) & (filtered_data[var] >= lower_bound) & (filtered_data[var] <= upper_bound)
                filtered_subset = filtered_data[condition]
                filtered_data_without_outliers = pd.concat([filtered_data_without_outliers, filtered_subset])'''
        
        for var in sensors:
            for cls in self.dataframe['class'].unique():
                subset = self.dataframe[self.dataframe['class'] == cls][var]
                median = subset.median()
                margin = 0.25 * median
                lower_bound = median - margin
                upper_bound = median + margin
                condition = (self.dataframe['class'] == cls) & \
                            (self.dataframe[var] >= lower_bound) & \
                            (self.dataframe[var] <= upper_bound)
                filtered_subset = self.dataframe.loc[condition]
                filtered_data_near_median = pd.concat([filtered_data_near_median, filtered_subset], ignore_index=True)
        
        
        # Removendo duplicatas após concatenação
        #filtered_data_without_outliers.drop_duplicates(inplace=True)
        filtered_data_near_median.drop_duplicates(inplace=True)

        # Convertendo o DataFrame filtrado para um array NumPy
        filtered_dataset_numpy = filtered_data_near_median.to_numpy()
        

        # Definindo o layout dos subplots para ter dois gráficos por linha
        n_vars = len(sensors)
        ncols = 2
        nrows = int(np.ceil(n_vars / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows), squeeze=False)
        axes = axes.flatten()

        # Iterar sobre cada sensor para criar os gráficos
        for i, var in enumerate(sensors):
            class_palette = {cls: class_colors.get(cls, 'gray') for cls in filtered_data['class'].unique()}
            sns.boxplot(x='class', y=var, data=filtered_data, ax=axes[i], palette=class_palette, showfliers=False)
            axes[i].set_title(f'Distribuição de {var} por Classificação')
            axes[i].set_xlabel('Classificação')
            axes[i].set_ylabel('Pressão (Pa)' if var in ['P-PDG', 'P-TPT', 'P-MON-CKP'] else 'Temperatura (°C)')
            axes[i].tick_params(axis='x', rotation=0)
            axes[i].grid(True)
            

        # Esconder eixos vazios se o número de sensores não preencher completamente a última linha
        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes[j])


        # Dicionário para armazenar os resultados
        quartiles_results = {}

        logging.info('Quartis para cada sensor e classe:')
        # Iterar sobre cada sensor
        for var in sensors:
            quartiles_results[var] = {}
            for cls in filtered_data['class'].unique():
                data = filtered_data[filtered_data['class'] == cls][var]
                quartiles = data.quantile([0.25, 0.5, 0.75]).to_dict()
                quartiles_results[var][cls] = quartiles
                logging.info(f'{var} - {legend_class[cls]}: {quartiles}')
        
        

        plt.tight_layout()
        plt.figlegend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4, title='Rotulagem de Observação')
        plt.subplots_adjust(bottom=0.15, top=0.95)
        
        plt.grid(True)
        plt.savefig(f"..\\..\\img\\{_title}.png", dpi=300, bbox_inches='tight')
        
        #plt.show()
        plt.close()
        return filtered_dataset_numpy
 
    def heatmap_corr(self, columns_of_interest, title):
     
        #title = 'Mapa de Calor do sensores em Operação Normal'
        # Selecionando apenas as colunas de interesse
        #columns_of_interest = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
        data_selected = self.dataframe[columns_of_interest]

        # Calculando a matriz de correlação
        correlation_matrix = data_selected.corr()

        # Criando o mapa de calor
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        #plt.title(title)
        plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
        #plt.show()
          
    def plot_sensor(self, sensor_columns, _title, additional_labels, model):
        # Substituindo valores de 'class' e convertendo 'timestamp'
        self.dataframe['timestamp'] = pd.to_datetime(self.dataframe.index)
        #replace_values = {101: -1, 102: -1, 103: -1, 104: -1, 105: -1, 106: -1, 107: -1, 108: -1, 109: -1}
        #self.dataframe['class'] = self.dataframe['class'].replace(replace_values)
        
        # troque valores faltantes NaN por -1 da coluna 'class'
        self.dataframe['class'] = self.dataframe['class'].fillna(-1)


        # Definindo o eixo X e configurações de plotagem
        x_hours = self.dataframe['timestamp']
        #sensor_columns = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']  # Atualizado para incluir 'class'
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'black']  # Atualizado para corresponder ao número de colunas 
        class_colors = {0: 'lightgreen', 
                        1: 'lightcoral', 
                        2: 'lightcoral',
                        3: 'lightcoral',
                        4: 'lightcoral',
                        5: 'lightcoral',
                        6: 'lightcoral',
                        7: 'lightcoral',
                        8: 'lightcoral',
                        9: 'lightcoral',
                        101: 'lightyellow', 
                        102: 'lightyellow',
                        103: 'lightyellow',
                        104: 'lightyellow',
                        105: 'lightyellow',
                        106: 'lightyellow',
                        107: 'lightyellow',
                        108: 'lightyellow',
                        109: 'lightyellow',
                        -1: 'lightgrey'}
        
        legend_class = {0: 'Normal', 
                        101: 'Transiente de Anomalia', 
                        102: 'Transiente de Anomalia', 
                        103: 'Transiente de Anomalia', 
                        104: 'Transiente de Anomalia', 
                        105: 'Transiente de Anomalia', 
                        106: 'Transiente de Anomalia', 
                        107: 'Transiente de Anomalia', 
                        108: 'Transiente de Anomalia', 
                        109: 'Transiente de Anomalia', 
                        1: 'Estável de Anomalia',
                        2: 'Estável de Anomalia',
                        3: 'Estável de Anomalia',
                        4: 'Estável de Anomalia',
                        5: 'Estável de Anomalia',
                        6: 'Estável de Anomalia',
                        7: 'Estável de Anomalia',
                        8: 'Estável de Anomalia',
                        9: 'Estável de Anomalia',
                        -1: 'Não Rotulado'}

        instance_label_dict = {
            0: 'Operação Normal',
            1: 'Aumento Abrupto de BSW',
            2: 'Fechamento Espúrio de DHSV',
            3: 'Intermitência Severa',
            4: 'Instabilidade na Vazão',
            5: 'Perda Rápida de Produtividade',
            6: 'Restrição Rápida em PCK',
            7: 'Incrustações em PCK',
            8: 'Hidrato na Linha de Produção'
        }


        plt.figure(figsize=(6, 12))  # Ajustado para acomodar todos os subplots
        patches = [mpatches.Patch(color=class_colors[cls], label=label) for cls, label in legend_class.items()]

        for i, column in enumerate(sensor_columns):
            ax = plt.subplot(len(sensor_columns), 1, i + 1)  # Ajustado para criar um subplot para cada coluna
            ax.plot(x_hours, self.dataframe[column], color=colors[i]) # comment  label=column)
            ax.set_title(column)
            ax.set_xlabel('Tempo (h)')
            ax.set_ylabel('Pressão (Pa)' if column in ['P-PDG', 'P-TPT', 'P-MON-CKP'] else 'Temperatura (°C)')
            ax.grid(True)
            #ax.legend()

                    
        # Pintando a área de fundo de acordo com a classe
            
            start_idx = 0
            for j in range(1, len(self.dataframe)):
                if self.dataframe.iloc[j]['class'] != self.dataframe.iloc[j-1]['class'] or j == len(self.dataframe) - 1 or self.dataframe.iloc[j]['action'] != self.dataframe.iloc[j-1]['action']:
                    end_idx = j
                    cls = self.dataframe.iloc[start_idx]['class']
                    action = self.dataframe.iloc[start_idx]['action']
                    
                    # Aplica a cor baseada em 'class' em toda a extensão vertical
                    class_color = class_colors.get(cls, 'lightgrey')
                    ax.axvspan(self.dataframe.iloc[start_idx]['timestamp'], self.dataframe.iloc[end_idx]['timestamp'], color=class_color, alpha=0.5, ymin=0.0, ymax=0.5)
                    
                    # Cor baseada em 'action', aplicada apenas à metade superior do gráfico
                    if action == 1 or action == 0:
                        action_color = 'magenta' if action == 1 else 'cyan'
                        ax.axvspan(self.dataframe.iloc[start_idx]['timestamp'], self.dataframe.iloc[end_idx]['timestamp'], color=action_color, alpha=0.5, ymin=0.5, ymax=1)
                    else:
                        action_color = '#807fff'
                        ax.axvspan(self.dataframe.iloc[start_idx]['timestamp'], self.dataframe.iloc[end_idx]['timestamp'], color=action_color, alpha=0.5, ymin=0.5, ymax=1)
                    start_idx = j


            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # Rotacionando os ticks do eixo X especificamente para este subplot
            for label in ax.get_xticklabels():
                label.set_rotation(90)


        #plt.suptitle(instance_label_dict[self.dataframe['label'].values[0]], fontsize=16, y=1.02)  # y é ajustado para evitar sobreposição com o topo do subplot superior


        plt.tight_layout()

        unique_patches_dict = OrderedDict()

        for patch in patches:  # 'patches' é sua lista original de objetos Patch
            color = patch.get_facecolor()  # Ou outra função apropriada para obter a cor
            label = patch.get_label()
            key = (color, label)
            
            if key not in unique_patches_dict:
                unique_patches_dict[key] = patch

        unique_patches = list(unique_patches_dict.values())
        
        # Adicionando legenda de classes ao gráfico com ajuste de posição
        plt.figlegend(handles=unique_patches, loc='upper right', title='Rotulagem de Observação', bbox_to_anchor=(1.4, 1), bbox_transform=plt.gcf().transFigure)

        # Criação de manipuladores para a segunda legenda
        action_patches = [mpatches.Patch(color='magenta', label='Detectado'),
                        mpatches.Patch(color='cyan', label='Não-Detectado')]

        # Adicionando a segunda legenda ao gráfico
        plt.figlegend(handles=action_patches, loc='upper right', title=f'Identificação de Falha ({model})', bbox_to_anchor=(1.4, 0.9), bbox_transform=plt.gcf().transFigure)

        # Criação de manipuladores para a terceira legenda sem especificar cores
        
        #additional_patches = [mpatches.Patch(color='none', label=label) for label in additional_labels]
        additional_patches = [mpatches.Patch(color='white', alpha=0, label=label) for label in additional_labels]

        # Adicionando a terceira legenda ao gráfico
        plt.figlegend(handles=additional_patches, loc='upper right', title='Métricas', bbox_to_anchor=(1.45, 0.83), bbox_transform=plt.gcf().transFigure)
        


        folder = _title.split(" - ")[1] + " - " + _title.split(" - ")[2]
        directory = f'..\\..\\img\\{folder}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.grid(True)
        plt.savefig(f"..\\..\\img\\{folder}\\{_title}.png", dpi=300, bbox_inches='tight')        
        plt.close()
        #plt.show()
    
    def plot_estados(self, _title):
        # Contagem de valores para cada classe
        class_counts = self.dataframe['class'].value_counts().sort_index()
        # soma os valores de 1 a 9 e 101 a 109
        rare_class_counts_A = class_counts[(class_counts.index > 0) & (class_counts.index < 10)].sum()
        rare_class_counts_B = class_counts[(class_counts.index > 10)].sum()
        rare_class_counts_C = class_counts[(class_counts.index == 0)].sum()

        # Total de amostras
        total = rare_class_counts_A + rare_class_counts_B + rare_class_counts_C

        logging.info(f'Normal: {rare_class_counts_C} - {round(rare_class_counts_C/total*100, 2)}%')
        logging.info(f'Transiente de anomalia: {rare_class_counts_B} - {round(rare_class_counts_B/total*100, 2)}%')
        logging.info(f'Estável de anomalia: {rare_class_counts_A} - {round(rare_class_counts_A/total*100, 2)}%')
        logging.info(f'Total: {rare_class_counts_A + rare_class_counts_B + rare_class_counts_C}')

        fig, ax = plt.subplots()
        bars = ax.bar(['Normal', 'Estável de anomalia', 'Transiente de anomalia'], [rare_class_counts_C, rare_class_counts_A, rare_class_counts_B], label=['Normal', 'Estável de anomalia', 'Transiente de anomalia'])

        ax.set_ylabel('Quantidade')
        ax.set_title('Quantidade de registros por rótulos')
        ax.legend()

        # Adiciona o valor acima de cada barra
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

        # Salva a figura
        plt.grid(True)
        plt.savefig(f"..\\..\\img\\{_title}.png", dpi=300, bbox_inches='tight')
        plt.close()