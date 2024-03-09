import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

class exploration():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def quartiles_plot(self, sensors, title):
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

        # Iterar sobre cada sensor
        for var in sensors:
            quartiles_results[var] = {}
            for cls in filtered_data['class'].unique():
                data = filtered_data[filtered_data['class'] == cls][var]
                quartiles = data.quantile([0.25, 0.5, 0.75]).to_dict()
                quartiles_results[var][cls] = quartiles
        
        print(quartiles_results)

        plt.tight_layout()
        plt.figlegend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4, title='Classificação')
        plt.subplots_adjust(bottom=0.15, top=0.95)

        plt.savefig(f"{title}.jpg", dpi=100, bbox_inches='tight')
        plt.grid(True)
        #plt.show()
 
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
        plt.savefig(f"{title}.png", dpi=100, bbox_inches='tight')
        #plt.show()