import os
import logging
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from collections import OrderedDict

logger = logging.getLogger("global_logger")

class Exploration:
    def __init__(self, dataframe):
        # Trabalha com uma cópia para não modificar o DataFrame original
        self.dataframe = dataframe.copy()

    def quartiles_plot(self, sensors, title):
        """
        Plota boxplots com os quartis para cada sensor, filtrando os valores que estão próximos à mediana.
        Retorna os dados filtrados como array NumPy.
        """
        df = self.dataframe.copy()
        # Garante que a coluna 'class' esteja preenchida e no formato string
        df['class'] = df['class'].fillna('-1').astype(float).astype(int).astype(str)
        
        base_colors = {
            'Normal': 'lightgreen',
            'Estável de Anomalia': 'lightcoral',
            'Transiente de Anomalia': 'lightyellow',
            'Não Rotulado': 'lightgrey'
        }
        legend_class = {
            '0': 'Normal', 
            '-1': 'Não Rotulado',
            **{str(i): 'Estável de Anomalia' for i in range(1, 9)},
            **{str(100 + i): 'Transiente de Anomalia' for i in range(1, 9)}
        }
        
        # Parâmetros de formatação
        title_size = 8
        axis_label_size = 7
        tick_label_size = 4
        legend_title_size = 7
        legend_label_size = 7

        # Mapeia classes para cores e gera os patches da legenda
        class_colors = {cls: base_colors[label] for cls, label in legend_class.items()}
        unique_labels = np.unique(list(legend_class.values()))
        patches = [mpatches.Patch(color=base_colors[label], label=label) for label in unique_labels]

        # Filtra para manter apenas as top classes (por frequência)
        top_classes = df['class'].value_counts().nlargest(100).index
        filtered_data = df[df['class'].isin(top_classes)]
        
        # Otimização: calcula medianas e margens para todos os sensores de forma vetorizada
        medians = df.groupby('class')[sensors].transform('median')
        margins = 0.25 * medians
        lower_bounds = medians - margins
        upper_bounds = medians + margins
        # Gera uma máscara que identifica linhas em que pelo menos um sensor está dentro dos limites
        mask = ((df[sensors] >= lower_bounds) & (df[sensors] <= upper_bounds)).any(axis=1)
        filtered_data_near_median = df[mask].drop_duplicates()
        filtered_dataset_numpy = filtered_data_near_median.to_numpy()

        # Configura o layout dos subplots
        n_vars = len(sensors)
        ncols = 5
        nrows = int(np.ceil(n_vars / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6, 3 * nrows))
        axes = axes.flatten()

        # Gera os gráficos de boxplot para cada sensor
        for i, var in enumerate(sensors):
            class_palette = {cls: class_colors.get(cls, 'gray') for cls in filtered_data['class'].unique()}
            sns.boxplot(
                x='class', y=var, data=filtered_data, hue='class',
                palette=class_palette, showfliers=False, ax=axes[i], legend=False
            )
            axes[i].set_title(var, fontsize=title_size)
            axes[i].set_xlabel('Classificação', fontsize=axis_label_size)
            ylabel = 'Pressão (Pa)' if var in ['P-PDG', 'P-TPT', 'P-MON-CKP'] else 'Temperatura (°C)'
            axes[i].set_ylabel(ylabel, fontsize=axis_label_size)
            axes[i].tick_params(axis='x', rotation=0, labelsize=tick_label_size)
            axes[i].tick_params(axis='y', rotation=0, labelsize=tick_label_size)
            axes[i].grid(True)

        # Remove subplots vazios, se houver
        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes[j])

        # Cálculo dos quartis por sensor e classe (operação pouco custosa em comparação ao resto)
        quartiles_results = {}
        logging.info('Quartis para cada sensor e classe:')
        for var in sensors:
            quartiles_results[var] = {}
            for cls in filtered_data['class'].unique():
                data = filtered_data[filtered_data['class'] == cls][var]
                quartiles = data.quantile([0.25, 0.5, 0.75]).to_dict()
                quartiles_results[var][cls] = quartiles
                logging.info(f'{var} - {legend_class.get(cls, cls)}: {quartiles}')

        plt.tight_layout()
        plt.figlegend(
            handles=patches, loc='upper center', bbox_to_anchor=(0.5, 0.05),
            ncol=4, title='Rotulagem de Observação', title_fontsize=legend_title_size, fontsize=legend_label_size
        )
        plt.subplots_adjust(bottom=0.15, top=0.95)
        plt.grid(True)
        save_path = os.path.join("..", "..", "img", f"{title}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return filtered_dataset_numpy

    def heatmap_corr(self, columns_of_interest, title):
        """
        Gera e salva um mapa de calor da correlação entre as colunas de interesse.
        """
        df = self.dataframe.copy()
        data_selected = df[columns_of_interest]
        correlation_matrix = data_selected.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        save_path = os.path.join(f"{title}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sensor(self, sensor_columns, title, additional_labels, model):
        """
        Plota séries temporais dos sensores com áreas de fundo coloridas de acordo com mudanças nos rótulos e ações.
        """
        df = self.dataframe.copy()
        # Converte o índice em timestamp se necessário
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df.index)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['class'] = df['class'].fillna(-1)

        x_hours = df['timestamp']
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'black']
        class_colors = {
            0: 'lightgreen', 
            1: 'lightcoral', 2: 'lightcoral', 3: 'lightcoral', 4: 'lightcoral',
            5: 'lightcoral', 6: 'lightcoral', 7: 'lightcoral', 8: 'lightcoral', 9: 'lightcoral',
            101: 'lightyellow', 102: 'lightyellow', 103: 'lightyellow', 104: 'lightyellow',
            105: 'lightyellow', 106: 'lightyellow', 107: 'lightyellow', 108: 'lightyellow',
            109: 'lightyellow', -1: 'lightgrey'
        }
        legend_class = {
            0: 'Normal', 
            101: 'Transiente de Anomalia', 102: 'Transiente de Anomalia', 103: 'Transiente de Anomalia',
            104: 'Transiente de Anomalia', 105: 'Transiente de Anomalia', 106: 'Transiente de Anomalia',
            107: 'Transiente de Anomalia', 108: 'Transiente de Anomalia', 109: 'Transiente de Anomalia',
            1: 'Estável de Anomalia', 2: 'Estável de Anomalia', 3: 'Estável de Anomalia',
            4: 'Estável de Anomalia', 5: 'Estável de Anomalia', 6: 'Estável de Anomalia',
            7: 'Estável de Anomalia', 8: 'Estável de Anomalia', 9: 'Estável de Anomalia',
            -1: 'Não Rotulado'
        }
        
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

        plt.figure(figsize=(6, 12))
        patches = [mpatches.Patch(color=class_colors[cls], label=label) for cls, label in legend_class.items()]

        # Segmenta as mudanças de 'class' ou 'action' de forma vetorizada
        df['segment'] = (df['class'].ne(df['class'].shift()) | df['action'].ne(df['action'].shift())).cumsum()
        segments = df.groupby('segment').agg(
            start=('timestamp', 'first'),
            end=('timestamp', 'last'),
            class_val=('class', 'first'),
            action_val=('action', 'first')
        ).reset_index()

        for i, column in enumerate(sensor_columns):
            ax = plt.subplot(len(sensor_columns), 1, i + 1)
            ax.plot(x_hours, df[column], color=colors[i % len(colors)])
            ax.set_title(column)
            ylabel = 'Pressão (Pa)' if column in ['P-PDG', 'P-TPT', 'P-MON-CKP'] else 'Temperatura (°C)'
            ax.set_xlabel('Tempo (h)')
            ax.set_ylabel(ylabel)
            ax.grid(True)

            # Desenha os intervalos com base nos segmentos
            for _, row in segments.iterrows():
                cls = row['class_val']
                action = row['action_val']
                ax.axvspan(row['start'], row['end'], color=class_colors.get(cls, 'lightgrey'),
                           alpha=0.5, ymin=0.0, ymax=0.5)
                if action in [0, 1]:
                    action_color = 'cyan' if action == 0 else 'magenta'
                else:
                    action_color = '#807fff'
                ax.axvspan(row['start'], row['end'], color=action_color,
                           alpha=0.5, ymin=0.5, ymax=1)

            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            for label_text in ax.get_xticklabels():
                label_text.set_rotation(90)

        plt.tight_layout()
        # Garante que as legendas sejam únicas
        unique_patches_dict = OrderedDict()
        for patch in patches:
            key = (patch.get_facecolor(), patch.get_label())
            if key not in unique_patches_dict:
                unique_patches_dict[key] = patch
        unique_patches = list(unique_patches_dict.values())

        plt.figlegend(
            handles=unique_patches, loc='upper right', title='Rotulagem de Observação',
            bbox_to_anchor=(1.4, 1), bbox_transform=plt.gcf().transFigure
        )
        action_patches = [
            mpatches.Patch(color='magenta', label='Detectado'),
            mpatches.Patch(color='cyan', label='Não-Detectado')
        ]
        plt.figlegend(
            handles=action_patches, loc='upper right', title=f'Identificação de Falha ({model})',
            bbox_to_anchor=(1.4, 0.9), bbox_transform=plt.gcf().transFigure
        )
        additional_patches = [mpatches.Patch(color='white', alpha=0, label=lbl) for lbl in additional_labels]
        plt.figlegend(
            handles=additional_patches, loc='upper right', title='Métricas',
            bbox_to_anchor=(1.45, 0.83), bbox_transform=plt.gcf().transFigure
        )
        
        folder_parts = title.split(" - ")
        folder = f"{folder_parts[1]} - {folder_parts[2]}" if len(folder_parts) >= 3 else title
        directory = os.path.join("..", "..", "img", folder)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        save_path = os.path.join(directory, f"{title}.png")
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_estados(self, title):
        """
        Plota um gráfico de barras com a contagem de registros por rótulo e salva a figura.
        """
        df = self.dataframe.copy()
        class_counts = df['class'].value_counts().sort_index()
        rare_class_counts_A = class_counts[(class_counts.index > 0) & (class_counts.index < 10)].sum()
        rare_class_counts_B = class_counts[(class_counts.index > 10)].sum()
        rare_class_counts_C = class_counts[(class_counts.index == 0)].sum()
        total = rare_class_counts_A + rare_class_counts_B + rare_class_counts_C

        logging.info(f'Normal: {rare_class_counts_C} - {round(rare_class_counts_C / total * 100, 2)}%')
        logging.info(f'Transiente de anomalia: {rare_class_counts_B} - {round(rare_class_counts_B / total * 100, 2)}%')
        logging.info(f'Estável de anomalia: {rare_class_counts_A} - {round(rare_class_counts_A / total * 100, 2)}%')
        logging.info(f'Total: {total}')

        plt.figure()
        bars = plt.bar(
            ['Normal', 'Estável de anomalia', 'Transiente de anomalia'],
            [rare_class_counts_C, rare_class_counts_A, rare_class_counts_B]
        )
        plt.ylabel('Quantidade')
        plt.title('Quantidade de registros por rótulos')
        plt.legend(['Normal', 'Estável de anomalia', 'Transiente de anomalia'])

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2., 1.05 * height,
                f'{int(height)}', ha='center', va='bottom'
            )

        save_path = os.path.join("..", "..", "img", f"{title}.png")
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
