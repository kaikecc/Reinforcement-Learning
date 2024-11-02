import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import os
from datetime import datetime
import matplotlib.dates as mdates

class EDA:
    def __init__(self, df, df_uom):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input must be a pandas DataFrame.")
        self.df = df
        self.columns = df.columns
        self.numeric_columns = df.select_dtypes(include=['number']).columns
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        self.datetime_columns = df.select_dtypes(include=['datetime']).columns
        self.bool_columns = df.select_dtypes(include=['bool']).columns
        self.columns_types = {
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'datetime_columns': self.datetime_columns,
            'bool_columns': self.bool_columns
        }

        self.dict_uom = df_uom.set_index('Name').to_dict()['UnitOfMeasure']

        self.uoms = {

            'pound-force per square inch' : 'psi',
            'inch per second': 'in/sec',
            'percent' : '%',
            'foot': 'ft',
            'US gallon per minute' : 'US gal/min',
            'pound' : 'lb',
            'inH2O' : 'inH2O',
            'Fahrenheit': '°F',
            'barrel per day' : 'bbl/d',
            'ampere': 'A',
            'volt': 'V',
            'hertz' : 'Hz'
        }

    def get_columns(self):
        return self.columns

    def get_columns_types(self):
        return self.columns_types

    
    def get_summary(self):
        """
        Função para analisar dados de diferentes equipamentos em um DataFrame.
        
        Args:
        df (pandas.DataFrame): DataFrame contendo os dados dos equipamentos.
        
        Returns:
        None: A função imprime resultados diretamente e não retorna nada.
        """
        # Iterando sobre cada equipamento único
        for equipment in self.df['equipment'].unique():
            # Filtrando os dados para o equipamento atual
            equipment_data = self.df[self.df['equipment'] == equipment]
            
            # Removendo colunas que contêm apenas valores nulos
            clean_data = equipment_data.dropna(axis=1, how='all')
            
            # Verificando as colunas remanescentes e suas primeiras linhas para entender melhor os dados
            print(f'Dados para o equipamento: {equipment}')
            
            # Convertendo 'TimeStamp' para datetime para facilitar a manipulação
            if 'TimeStamp' in clean_data.columns:
                clean_data['TimeStamp'] = pd.to_datetime(clean_data['TimeStamp'])
            
            # Calculando estatísticas descritivas para as colunas numéricas, excluindo 'TimeStamp'
            numeric_columns = clean_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'TimeStamp' in numeric_columns:
                numeric_columns.remove('TimeStamp')
            numeric_data = clean_data[numeric_columns]
            
            # Calculando estatísticas descritivas
            descriptive_stats = numeric_data.describe()
            display(descriptive_stats.head()) 

    def plot_vibration_temperature_scatter(self, df, combinations, titles, colors):
        """
        Função para criar scatter plots para combinações de dados de vibração e temperatura.

        Args:
        df (pandas.DataFrame): DataFrame contendo os dados.
        combinations (list of tuples): Lista de tuplas com as colunas para plotar (vibração, temperatura).
        titles (list of str): Títulos para cada scatter plot.
        colors (list of str): Cores para cada scatter plot.

        Returns:
        None: A função apenas exibe os gráficos.
        """
        # Verificar se todas as colunas especificadas nas combinações existem no DataFrame
        cols = [col for combo in combinations for col in combo]
        if not all(col in df.columns for col in cols):
            raise ValueError("Uma ou mais colunas especificadas não existem no DataFrame.")
        
        # Criar figura e eixos para os scatter plots
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

        # Plotar cada combinação com nova cor
        for ax, (vibration, temperature), title, color in zip(axs.flatten(), combinations, titles, colors):
            ax.scatter(df[vibration], df[temperature], alpha=0.5, color=color)
            ax.set_xlabel(vibration)
            ax.set_ylabel(temperature)
            ax.set_title(title)
            ax.grid(True)

        # Ajustar layout
        plt.tight_layout()
        plt.show()

    def get_correlation(self):
        return self.df.corr()

    def plot_correlation(self):
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.show()

    def plot_distribution(self):
        for column in self.numeric_columns:
            sns.histplot(self.df[column].dropna())
            plt.show()

    def plot_categorical(self):
        for column in self.categorical_columns:
            sns.catplot(x=column, kind="count", data=self.df)
            plt.show()

    def plot_datetime(self):
        for column in self.datetime_columns:
            self.df[column].hist()
            plt.show()

    def plot_bool(self):
        for column in self.bool_columns:
            sns.catplot(x=column, kind="count", data=self.df)
            plt.show()

    def plot_pairplot(self):
        sns.pairplot(self.df.dropna())
        plt.show()
    
    def plot_histograms(self,data, columns):
        # Lista de cores para os histogramas
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'violet', 'lightgrey', 'orange', 'lime', 'pink', 'teal']
        
        num_cols = len(columns)
        num_rows = (num_cols + 1) // 2  # Calculate the number of rows needed, assuming 2 columns per row

        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, num_rows * 3), constrained_layout=True)
        fig.suptitle('Distribution of Numeric Columns', fontsize=16)

        # Loop over the columns to create histograms
        for i, (col, ax) in enumerate(zip(columns, axes.flatten())):
            data[col].dropna().hist(ax=ax, bins=30, alpha=1.0, color=colors[i % len(colors)])  # Use modulo for color cycling
            ax.set_title(col)
            ax.set_ylabel('Frequency')

        # Turn off any unused subplots
        if num_cols % 2 != 0:
            axes.flatten()[-1].axis('off')  # Hide last subplot if the number of columns is odd

        plt.show()

    def plot_scatter(self):
        if len(self.numeric_columns) > 1:
            for column in self.numeric_columns[1:]:
                sns.scatterplot(x=self.df[column], y=self.df[self.numeric_columns[0]])
                plt.show()

    def plot_all(self):
        self.plot_correlation()
        self.plot_distribution()
        self.plot_categorical()
        self.plot_datetime()
        self.plot_bool()
        self.plot_pairplot()
        self.plot_boxplot()
        self.plot_scatter()
       
    def get_missing(self):
        return self.df.isnull().sum()

    def plot_missing(self):
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.show()
    
    def simple_boxplots(self, data, columns):
        num_cols = len(columns)
        num_rows = (num_cols + 1) // 2  # Calculate the number of rows needed, assuming 2 columns per row

        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 6 * num_rows), constrained_layout=True)
        fig.suptitle('Boxplots for Numeric Variables', fontsize=16)

        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'violet', 'lightgrey', 'orange', 'lime', 'pink', 'teal']

        # Ensure axes is always a 2D array for consistency
        if num_rows == 1:
            axes = [axes]

        # Loop over the number of specified columns and create a boxplot for each
        for i, column in enumerate(columns):
            row, col = divmod(i, 2)
            sns.boxplot(ax=axes[row][col], y=data[column], color=colors[i % len(colors)])
            axes[row][col].set_title(column)
            axes[row][col].set_xlabel(column.split(' - ')[-1])  # Splitting to shorten x labels if needed
            axes[row][col].grid()

        # Turn off any unused subplots
        if num_cols % 2 != 0:
            axes[-1][-1].axis('off')  # Hide last subplot if the number of columns is odd

        plt.show()

    def calculate_deviation(self, df_element, start_date=None, end_date=None):
        # Convert start_date and end_date from string to datetime if provided as strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date, errors='coerce')
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date, errors='coerce')

        # Filter the DataFrame for the specified date range
        if start_date is not None and end_date is not None:
            df_element = df_element[(df_element['TimeStamp'] >= start_date) & (df_element['TimeStamp'] <= end_date)]

        # Ensure that the required columns are present
        required_columns = ['TimeStamp', 'equipment', 'Head - Expected', 'Head - Calculated']
        if not all(col in df_element.columns for col in required_columns):
            raise ValueError("Missing one or more required columns in the DataFrame")

        # Create a copy to avoid SettingWithCopyWarning when modifying df_element
        df_element = df_element.copy()

        # Convert 'Head - Expected' and 'Head - Calculated' to numeric, treating non-numeric as NaN
        df_element['Head - Expected'] = pd.to_numeric(df_element['Head - Expected'], errors='coerce')
        df_element['Head - Calculated'] = pd.to_numeric(df_element['Head - Calculated'], errors='coerce')
       

        # Sort DataFrame by TimeStamp
        df_element.sort_values('TimeStamp', inplace=True)

        # Create separate datasets for 'Head - Expected' and 'Head - Calculated'
        expected = df_element[['TimeStamp', 'equipment', 'Head - Expected']].dropna()
        calculated = df_element[['TimeStamp', 'equipment', 'Head - Calculated']].dropna()

        # Perform an asof merge to find the nearest matches within the same 'equipment'
        merged = pd.merge_asof(expected, calculated, on='TimeStamp', by='equipment', suffixes=('_expected', '_calculated'), direction='nearest')

        # Calculate the percentage deviation and handle division by zero
        merged['deviation'] = ((merged['Head - Calculated'] - merged['Head - Expected']) / merged['Head - Expected']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        return merged

    def add_status_background(self, ax, status_periods_df, start_date, end_date):
        # Aplicando um fundo cinza para todo o intervalo de dados
        ax.axvspan(start_date, end_date, facecolor='lightgrey', alpha=0.5)  # Background para o intervalo de dados
        
        # Adicionando coloração específica para os status
        for index, row in status_periods_df.iterrows():
            if row['status'] == 'RUNNING':
                ax.axvspan(max(row['start'], start_date), min(row['end'], end_date), facecolor='lightgreen', alpha=0.5)
            elif row['status'] == 'STOPPED':
                ax.axvspan(max(row['start'], start_date), min(row['end'], end_date), facecolor='lightcoral', alpha=0.5)
     
    def preprocess_status_pump(self, df_element, start_date=None, end_date=None):
        # Convert start_date and end_date from string to datetime if they are provided as strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date, errors='coerce')
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date, errors='coerce')

        # Filter the DataFrame for Status - Pump within the specified date range
        df_element['TimeStamp'] = pd.to_datetime(df_element['TimeStamp'], errors='coerce')
        if start_date:
            df_element = df_element[df_element['TimeStamp'] >= start_date]
        if end_date:
            df_element = df_element[df_element['TimeStamp'] <= end_date]

        df_element = df_element.sort_values(by='TimeStamp')

        df_element.replace('None', np.nan, inplace=True)
        df_element.dropna(subset=['Status - Pump'], inplace=True)

        # Identify changes in status and create a column for status change groups
        df_element['StatusChange'] = df_element['Status - Pump'].ne(df_element['Status - Pump'].shift()).cumsum()

        # Group by status change and calculate the start and end time for each status period
        status_periods_df = df_element.groupby('StatusChange').agg(
            start=('TimeStamp', 'min'),
            end=('TimeStamp', 'max'),
            status=('Status - Pump', 'first')
        ).reset_index(drop=True)

        # Calculate the duration of each status period in hours
        status_periods_df['Duration'] = (status_periods_df['end'] - status_periods_df['start']).dt.total_seconds() / 3600

        # Calculate total running and stopped hours
        running_hours = status_periods_df[status_periods_df['status'] == 'RUNNING']['Duration'].sum()
        stopped_hours = status_periods_df[status_periods_df['status'] == 'STOPPED']['Duration'].sum()

        return status_periods_df, running_hours, stopped_hours

    def def_multiplots(self, elements = [], attributes=[], start_date=None, end_date=None):
        
        if not elements:
            elements = self.df['equipment'].unique()
         
        if not attributes:
            print("No attributes specified for plotting.")
            return
        
        num_attributes = len(attributes)
      
        deviation = False
        if 'deviation' in attributes:
            attributes.remove('deviation')
            deviation = True
          
         # Convert start_date and end_date from string to datetime if they are provided as strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date, errors='coerce')
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date, errors='coerce')

        # Set default start_date and end_date if not provided or if conversion failed (coerce resulted in NaT)
        if start_date is None or pd.isna(start_date):
            start_date = self.df['TimeStamp'].min()
        else:
            # Validate start_date against the DataFrame's range
            if start_date < self.df['TimeStamp'].min():
                print(f"Provided start_date {start_date} is earlier than the earliest TimeStamp in the data. Adjusting to earliest available date.")
                start_date = self.df['TimeStamp'].min()

        if end_date is None or pd.isna(end_date):
            end_date = self.df['TimeStamp'].max()
        else:
            # Validate end_date against the DataFrame's range
            if end_date > self.df['TimeStamp'].max():
                print(f"Provided end_date {end_date} is later than the latest TimeStamp in the data. Adjusting to latest available date.")
                end_date = self.df['TimeStamp'].max()
          
        for element in elements:

            _, axs = plt.subplots(num_attributes, 1, figsize=(15, 3 * num_attributes), sharex=True)
             
            if num_attributes == 1:
                axs = [axs]  # Ensure axs is iterable in the case of a single subplot

            df_element = self.df[self.df['equipment'] == element]
            
            status_periods_df, running_hours, stopped_hours  = self.preprocess_status_pump(df_element, start_date=start_date, end_date=end_date)
            running_patch = mpatches.Patch(color='lightgreen', label=f'RUNNING: {running_hours:.2f} h')
            stopped_patch = mpatches.Patch(color='lightcoral', label=f'STOPPED: {stopped_hours:.2f} h')
            #other_patch = mpatches.Patch(color='lightgrey', label=f'NaN: {other_hours:.2f} h')
            unique_patches = [running_patch, stopped_patch]
            
            for ax, attribute in zip(axs, attributes):
                              
                filtered_df = df_element.loc[(df_element['TimeStamp'] >= start_date) & 
                             (df_element['TimeStamp'] <= end_date), ['TimeStamp', attribute]]
                
                filtered_df.replace('None', np.nan, inplace=True)
                filtered_df = filtered_df.dropna(subset=[attribute])
                
                if filtered_df.empty:
                    ax.text(0.5, 0.5, f"No data for {attribute} in the given time range.", fontsize=12, ha='center')
                    continue
                 
                ax.plot(filtered_df['TimeStamp'], filtered_df[attribute], label=attribute, alpha=0.7, color='black') # colors[attributes.index(attribute) % len(colors)]
                ax.set_title(f'{attribute} between {start_date.strftime("%m-%d-%Y")} and {end_date.strftime("%m-%d-%Y")}') 
                ax.set_ylabel(f'[{self.uoms[self.dict_uom[attribute]]}]')
                self.add_status_background(ax, status_periods_df, start_date, end_date)
                #ax.legend()
                ax.grid(True)

            if deviation:
                deviation_df = self.calculate_deviation(df_element, start_date,end_date)                
                axs[-1].plot(deviation_df['TimeStamp'], deviation_df['deviation'], label='Deviation Head', alpha=0.7, color='black')
                axs[-1].set_title(f'Deviation Head between {start_date.strftime("%m-%d-%Y")} and {end_date.strftime("%m-%d-%Y")} ')
                axs[-1].set_ylabel('Deviation Head - [%]')
                self.add_status_background(axs[-1], status_periods_df, start_date, end_date)
                #axs[-1].legend()
                axs[-1].grid(True)
            
            axs[-1].set_xlabel('Time')
            plt.figlegend(handles=unique_patches, loc='upper right', title='Status - Pump', bbox_to_anchor=(1.15, 1), bbox_transform=plt.gcf().transFigure)
            plt.xticks(rotation=45)
            plt.tight_layout()            
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f'..\\..\\img\\{element}'   
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"{path}\\Trend_{element}_{current_time}.png", dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory

    def set_axis_limits(self, df, column='deviation', expansion_factor=2.75):
        if df[column].isnull().all():
            return (0, 1)  # Default limits when data is completely null
        
        Q1, Q3 = df[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1

        # Handle cases where IQR is zero
        if IQR == 0:
            IQR = df[column].std() if df[column].std() != 0 else 1  # Use standard deviation if available

        whisker_spread = 1.5 * IQR
        whisker_gap = whisker_spread * 0.1
        upper_whisker = Q3 + whisker_spread
        lower_whisker = Q1 - whisker_spread

        upper_limit = upper_whisker + whisker_gap
        lower_limit = lower_whisker - whisker_gap

        range_width = upper_limit - lower_limit
        upper_limit += range_width * expansion_factor
        lower_limit -= range_width * expansion_factor

        # Ensure limits are finite
        upper_limit = np.nan_to_num(upper_limit, nan=10, posinf=10, neginf=-10)
        lower_limit = np.nan_to_num(lower_limit, nan=-10, posinf=10, neginf=-10)

        return lower_limit, upper_limit

    def validate_date(self, date, default, df_min, df_max):
        if pd.isna(date):
            return default
        elif date < df_min:
            print(f"Adjusting date from {date} to {df_min}")
            return df_min
        elif date > df_max:
            print(f"Adjusting date from {date} to {df_max}")
            return df_max
        return date

    def plot_boxplot(self, df,  title, ylabel, xlabel, path, current_time, axis_x = 'equipment'):
        plt.figure(figsize=(12, 8))
        bp = sns.boxplot(x=axis_x, y='deviation', data=df, color='white', fliersize=0, showfliers=False)

        for box in bp.artists:
            box.set_edgecolor('black')
        for line in bp.lines:
            line.set_color('darkorange')

        lower_limit, upper_limit = self.set_axis_limits(df)
        plt.ylim(lower_limit, upper_limit)
        plt.axhline(y=3, color='red', linestyle='--', label='Hi: +3%', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', label='Ref.: 0%', linewidth=1)
        plt.axhline(y=-6, color='red', linestyle='--', label='Lo: -6%', linewidth=1)
        plt.legend()

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')

        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f"{path}/Boxplot_{current_time}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def boxplot_monthly(self, elements=[], start_date=None, end_date=None):
        start_date = pd.to_datetime(start_date, errors='coerce')
        end_date = pd.to_datetime(end_date, errors='coerce')

        start_date = self.validate_date(start_date, self.df['TimeStamp'].min(), self.df['TimeStamp'].min(), self.df['TimeStamp'].max())
        end_date = self.validate_date(end_date, self.df['TimeStamp'].max(), self.df['TimeStamp'].min(), self.df['TimeStamp'].max())

        if not elements:
            elements = self.df['equipment'].unique()

        df_filtered = self.df[(self.df['TimeStamp'] >= start_date) & (self.df['TimeStamp'] <= end_date)]

        
        for element in elements:
            all_data = pd.DataFrame()
            path = f'../../img/{element}'
            df_element = df_filtered[df_filtered['equipment'] == element]
            for month, group in df_element.groupby(df_element['TimeStamp'].dt.to_period("M")):
                
                deviation_df = self.calculate_deviation(group, start_date, end_date)
                deviation_df['Month'] = month.strftime('%Y-%m')
                all_data = pd.concat([all_data, deviation_df], ignore_index=True)

            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.plot_boxplot(all_data, f"Deviation for {element} - {month.strftime('%Y-%m')}", 'Deviation Head (%)', 'Month', path, current_time, axis_x ='Month')
        
    def boxplot_deviation(self, elements=[], start_date=None, end_date=None):
        start_date = pd.to_datetime(start_date, errors='coerce')
        end_date = pd.to_datetime(end_date, errors='coerce')

        start_date = self.validate_date(start_date, self.df['TimeStamp'].min(), self.df['TimeStamp'].min(), self.df['TimeStamp'].max())
        end_date = self.validate_date(end_date, self.df['TimeStamp'].max(), self.df['TimeStamp'].min(), self.df['TimeStamp'].max())

        if not elements:
            elements = self.df['equipment'].unique()

        
        all_data = pd.DataFrame()
        for element in elements:
            
            df_element = self.df[(self.df['equipment'] == element) & (self.df['TimeStamp'] >= start_date) & (self.df['TimeStamp'] <= end_date)]
            deviation_df = self.calculate_deviation(df_element, start_date, end_date)
            all_data = pd.concat([all_data, deviation_df], ignore_index=True)

            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f'../../img/{element}'
            self.plot_boxplot(deviation_df, f"Deviation of {element} between {start_date} and {end_date}",
                        'Head Calculated - Head Expected (%)', 'Water Injection Pumps', path, current_time)

        # Plotting all elements together
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = '../../img/all_elements'
        self.plot_boxplot(all_data, f"Deviation Head of all WIJs between {start_date} and {end_date}",
                    'Head Calculated - Head Expected (%)', 'Water Injection Pumps', path, current_time)

    def plot_density_by_status(self, df_subset, attribute, element, path):
        
        df_subset['Status - Pump'] = df_subset['Status - Pump'].fillna(method='ffill').fillna(method='bfill')

        running_df = df_subset[df_subset['Status - Pump'] == 'RUNNING']
        stopped_df = df_subset[df_subset['Status - Pump'] == 'STOPPED']

        # Verificação se há dados suficientes para plotagem
        if running_df.empty and stopped_df.empty:
            print(f"No data available for {element} - {attribute} in either RUNNING or STOPPED state.")
            return

        plt.figure(figsize=(10, 6))
        if not running_df.empty:
            sns.kdeplot(data=running_df[attribute].dropna(), fill=True, label=f"{attribute} - RUNNING", color="green")
        if not stopped_df.empty:
            sns.kdeplot(data=stopped_df[attribute].dropna(), fill=True, label=f"{attribute} - STOPPED", color="red")

        plt.title(f'Density Plot for {element} - {attribute} by Pump Status')
        uom = self.uoms[self.dict_uom[attribute]]
        plt.xlabel(f'{uom}')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        # Save the plot
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"{path}/DensityPlotGrouping_{element}_{attribute}_by_status_{current_time}.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 5))

        # Subplot 1: Density plot for RUNNING status
        plt.subplot(1, 3, 1)
        if not running_df.empty:
            sns.kdeplot(data=running_df[attribute].dropna(), fill=True, label=f"{attribute} - RUNNING", color="green")
            plt.title(f'RUNNING Status - Pump of {attribute}')
            plt.xlabel(uom)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)

        # Subplot 2: Density plot for STOPPED status
        plt.subplot(1, 3, 2)
        if not stopped_df.empty:
            sns.kdeplot(data=stopped_df[attribute].dropna(), fill=True, label=f"{attribute} - STOPPED", color="red")
            plt.title(f'STOPPED Status - Pump of {attribute}')
            plt.xlabel(uom)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)

        # Subplot 3: Density plot for all data
        plt.subplot(1, 3, 3)
        if not running_df.empty:
            sns.kdeplot(data=running_df[attribute].dropna(), fill=True, label=f"{attribute} - RUNNING", color="green")
        if not stopped_df.empty:
            sns.kdeplot(data=stopped_df[attribute].dropna(), fill=True, label=f"{attribute} - STOPPED", color="red")
        plt.title(f'Grouping Status - Pump of {attribute}')
        plt.xlabel(uom)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        # Save the plot
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"{path}/DensityPlot_{element}_{attribute}_by_status_{current_time}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def density_plot(self, elements=[], attributes=[], start_date=None, end_date=None):
        if not attributes:
            print("No attributes specified for plotting.")
            return

        # Handle datetime conversion efficiently
        start_date = pd.to_datetime(start_date, errors='coerce') if isinstance(start_date, str) else start_date
        end_date = pd.to_datetime(end_date, errors='coerce') if isinstance(end_date, str) else end_date

        # Set default dates based on the DataFrame if not specified
        start_date = start_date if start_date is not None else self.df['TimeStamp'].min()
        end_date = end_date if end_date is not None else self.df['TimeStamp'].max()

        # Filter the DataFrame once based on the timestamp
        filtered_df = self.df[(self.df['TimeStamp'] >= start_date) & (self.df['TimeStamp'] <= end_date)]

        if not elements:
            elements = self.df['equipment'].unique()
        
        for element in elements:
            # Prepare the path and check if it exists
            path = f'../../img/{element}'  # Ensure this path is correct for your file system
            os.makedirs(path, exist_ok=True)

            for attribute in attributes:
                df_subset = filtered_df[filtered_df['equipment'] == element]

                if df_subset.empty:
                    print(f"No data for {element} - {attribute} in the given time range.")
                    continue

                self.plot_density_by_status(df_subset, attribute, element, path)

                attribute_data = df_subset[attribute].dropna()

                if attribute_data.empty:
                    print(f"No valid data for {element} - {attribute} after removing NAs.")
                    continue

                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=attribute_data, fill=True, label=f"{attribute}")
                plt.title(f'Density Plot for {element} - {attribute}')
                uom = self.uoms[self.dict_uom[attribute]]
                plt.xlabel(f'{uom}')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True)

                # Save each individual plot
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                plt.savefig(f"{path}/DensityPlot_{element}_{attribute}_{current_time}.png", dpi=300, bbox_inches='tight')
                plt.close()
               
    def density_elements(self, attributes=[], start_date=None, end_date=None):
        # Check if no attributes are provided and exit the method early
        if not attributes:
            print("No attributes specified for plotting.")
            return

        # Convert start_date and end_date to pandas datetime objects if they are strings
        start_date = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        end_date = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        # Set default values for start_date and end_date if None
        start_date = start_date if start_date is not None else self.df['TimeStamp'].min()
        end_date = end_date if end_date is not None else self.df['TimeStamp'].max()

        # Filter the dataframe for the given date range
        filtered_df = self.df[(self.df['TimeStamp'] >= start_date) & (self.df['TimeStamp'] <= end_date)]

        # Get sorted unique equipment identifiers
        unique_elements = sorted(filtered_df['equipment'].unique())

        # Create the directory if it doesn't exist
        path = '../../img/all_elements'
        if not os.path.exists(path):
            os.makedirs(path)

        # Define a list of colors for the plots
        colors = ['b', 'cyan', 'purple', 'yellow', 'orange', 'brown', 'pink', 'gray', 'olive', 'green']
        i = 0
        # Plotting loop
        for attribute in attributes:
            plt.figure(figsize=(10, 6))
            for element in unique_elements:
                df_subset = filtered_df[filtered_df['equipment'] == element]
                if not df_subset.empty:
                    sns.kdeplot(data=df_subset[attribute], fill=True, label=f"{element}", color=colors[i])
                i = i + 1
            
            uom = self.uoms[self.dict_uom[attribute]]    
            plt.xlabel(uom)    
            plt.ylabel('Density')
            plt.title(f'Density plot for {attribute} from {start_date.strftime("%m-%d-%Y")} to {end_date.strftime("%m-%d-%Y")}')
            plt.legend(title='Elements', title_fontsize='13', fontsize='11', loc='upper right')
            plt.grid(True)
            plt.tight_layout()

            # Save the plot with a timestamp
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f"{path}/DensityPlot_{attribute}_{current_time}.png", dpi=300, bbox_inches='tight')
            plt.close()           
           
    def prepare_data(self):
        #self.df['TimeStamp'] = pd.to_datetime(self.df['TimeStamp'])
        #self.df['Value'] = pd.to_numeric(self.df['Value'], errors='coerce')
        self.identify_pre_stop_periods()

    def identify_pre_stop_periods(self):
       
        self.df['Status'] = self.df['Value'].apply(lambda x: 'STOPPED' if x == 'STOPPED' else ('RUNNING' if x == 'RUNNING' else 'UNKNOWN'))
        stop_times = self.df[self.df['Status'] == 'STOPPED']['TimeStamp']
        pre_stop_start_times = stop_times - pd.Timedelta(hours=8)

        # Marcar os períodos
        self.df['Group'] = 'Normal-Running'  # Default group
        for start, stop in zip(pre_stop_start_times, stop_times):
            self.df.loc[(self.df['TimeStamp'] >= start) & (self.df['TimeStamp'] < stop), 'Group'] = 'Pre-Stop'

    def plot_ridge_density(self, attributes=[], start_date=None, end_date=None):
        # Identificar os períodos de pré-parada primeiro
        self.identify_pre_stop_periods()

        # Se não houver atributos especificados, não há nada a plotar
        if not attributes:
            print("No attributes specified for plotting.")
            return

        # Converter datas de início e término, se fornecidas, ou usar os valores mínimos e máximos
        start_date = pd.to_datetime(start_date) if start_date else self.df['TimeStamp'].min()
        end_date = pd.to_datetime(end_date) if end_date else self.df['TimeStamp'].max()

        # Obter os períodos de pré-parada
        stop_times = self.df[self.df['Status'] == 'STOPPED']['TimeStamp']
        pre_stop_start_times = stop_times - pd.Timedelta(hours=8)

        # Filtrar os períodos de pré-parada para excluir aqueles com menos de 8 horas de diferença
        valid_pre_stop_start_times = []
        for i in range(len(pre_stop_start_times)):
            if i == 0 or (pre_stop_start_times.iloc[i] - pre_stop_start_times.iloc[i-1] >= pd.Timedelta(hours=8)):
                valid_pre_stop_start_times.append(pre_stop_start_times.iloc[i])
        
        pre_stop_start_times = valid_pre_stop_start_times
        # Configurando o estilo do plot
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Criar uma nova figura para cada atributo
        for attribute in attributes:
            # Filtrar o dataframe pelo atributo e intervalo de datas
            df_subset = self.df[(self.df['AttributeName'] == attribute) &
                                (self.df['TimeStamp'] >= start_date) &
                                (self.df['TimeStamp'] <= end_date)]

            # Preparar o DataFrame para o FacetGrid
            facet_data = pd.DataFrame()
            for start in pre_stop_start_times:
                # Filtrar o DataFrame para o período de pré-parada
                temp = df_subset[(df_subset['TimeStamp'] >= start) & (df_subset['TimeStamp'] < start + pd.Timedelta(hours=8))]
                if not temp.empty:
                    temp['Period'] = start.strftime("%m-%d-%Y %H:%M")
                    facet_data = pd.concat([facet_data, temp])

            # Verificar se existem dados após filtragem
            if facet_data.empty:
                print(f"No 'Pre-Stop' data for attribute {attribute} in the given time range.")
                continue
            
            
            # Cria uma instância de FacetGrid
            g = sns.FacetGrid(facet_data, row="Period", hue="Group", aspect=8, height=1, palette=['red'])
            
            # Desenha as densidades de KDE para 'Value'
            g.map(sns.kdeplot, 'Value', clip_on=False, shade=True, alpha=1, lw=1.5, bw_adjust=.5)

            # Usa linhas para dividir os plots e adicionar rótulos
            g.map(plt.axhline, y=0, lw=2, clip_on=False)            
            g.set_titles("{row_name}")
            g.set(yticks=[])
            g.despine(bottom=True, left=True)

            # Mostrar o plot
            plt.show()
           
    def density_compare_plot(self, elements=[], attributes=[], start_date=None, end_date=None):
        if not elements:
            elements = self.df['Element'].unique()

        if not attributes:
            print("No attributes specified for plotting.")
            return

        self.prepare_data()

        start_date = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        end_date = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        start_date = start_date if start_date is not None else self.df['TimeStamp'].min()
        end_date = end_date if end_date is not None else self.df['TimeStamp'].max()

        filtered_df = self.df[(self.df['TimeStamp'] >= start_date) & (self.df['TimeStamp'] <= end_date)]

        colors = {'Pre-Stop': 'red', 'Normal-Running': 'blue'}
       

        #num_plots = len(attributes) * len(elements) * 2 + 1  # Todos os subplots individuais mais o subplot combinado
        num_cols = 3  # Duas colunas de subplots
        num_rows = 1  # Calcula o número de linhas necessário


        fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 4 * num_rows))
        ax_idx = 0  # Índice para o subplot atual

        for element in elements:
            for attribute in attributes:
                # Calcula os limites globais para os eixos
                min_value = filtered_df[filtered_df['AttributeName'] == attribute]['Value'].min()
                max_value = filtered_df[filtered_df['AttributeName'] == attribute]['Value'].max()
                for group in ['Pre-Stop', 'Normal-Running']:
                    df_subset = filtered_df[(filtered_df['Element'] == element) & 
                                            (filtered_df['AttributeName'] == attribute) &
                                            (filtered_df['Group'] == group)]

                    if df_subset.empty:
                        print(f"No data for {element} - {attribute} - {group} in the given time range.")
                        continue

                    ax = axes.flat[ax_idx]
                    ax_idx += 1  # Atualizar índice do subplot

                    sns.kdeplot(data=df_subset['Value'], fill=True, color=colors[group], label=f"{attribute} - {group}", ax=ax)
                    ax.set_title(f'Density Plot for {element} - {attribute} [{group}]')
                    uom = df_subset["UnitOfMeasure"].iloc[0] if not df_subset.empty and "UnitOfMeasure" in df_subset.columns else "UOM not specified"
                    ax.set_xlabel(f'{self.uoms[uom]}')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True)
                    ax.set_xlim(min_value, max_value)

                df_subset = filtered_df[(filtered_df['Element'] == element) & (filtered_df['AttributeName'] == attribute)]

                if df_subset.empty:
                    print(f"No data for {element} - {attribute} in the given time range.")
                    continue

                
                sns.kdeplot(data=df_subset['Value'], fill=True, color = 'green', label=f"{attribute}")
                ax.set_title(f'Density Plot for {element} - {attribute}')
                uom = df_subset["UnitOfMeasure"].iloc[0] if not df_subset.empty and "UnitOfMeasure" in df_subset.columns else "UOM not specified"
                ax.set_xlabel(f'{self.uoms[uom]}')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True)
                ax.set_xlim(min_value, max_value)

        # Adicionando o subplot combinado
        ax = axes.flat[ax_idx]  # Seleciona o último subplot
        for element in elements:
            for attribute in attributes:
                for group in ['Pre-Stop', 'Normal-Running']:
                    df_subset = filtered_df[(filtered_df['Element'] == element) & 
                                            (filtered_df['AttributeName'] == attribute) &
                                            (filtered_df['Group'] == group)]

                    if not df_subset.empty:
                        sns.kdeplot(data=df_subset['Value'], fill=True, color=colors[group], label=f"{attribute} - {group}", ax=ax)

        ax.set_title(f'Density Plot for {element} - {attribute}')
        ax.set_xlabel(f'{self.uoms[uom]}')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(min_value, max_value)

        plt.tight_layout()
        plt.show()



    