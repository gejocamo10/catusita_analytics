import pandas as pd
import os
from datetime import datetime
import numpy as np
from pathlib import Path
from utils.process_data.config import DATA_PATHS

class SunatProcessor:
    def __init__(self):
        self.raw_data_path = DATA_PATHS['raw_sunat']
        self.processed_path = DATA_PATHS['process'] / 'sunat_consolidated.csv'

    def process_file(self, file_path):
        try:
            # Read the Excel file, specifically the "Resumen" sheet, starting from row 2
            df = pd.read_excel(file_path, sheet_name='Resumen', header=1, usecols="B:O")

            df = df.iloc[0:19]
            
            # Drop empty rows and columns
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            # Ensure the first column is named correctly
            first_col_name = df.columns[0]
            
            # Melt the dataframe to get the desired format
            melted_df = pd.melt(df, 
                               id_vars=[first_col_name],
                               var_name='fecha',
                               value_name='value')
            
            # Drop rows with NaN values
            melted_df = melted_df.dropna()
            
            # Convert fecha to datetime if it's not already
            melted_df['fecha'] = pd.to_datetime(melted_df['fecha'])
            
            # Extract year and month from datetime
            melted_df['year'] = melted_df['fecha'].dt.year
            melted_df['mes'] = melted_df['fecha'].dt.month
            
            # Rename columns to match desired output
            melted_df = melted_df.rename(columns={first_col_name: 'Descripcion'})
            
            # Ensure value column is numeric
            melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce')
            
            # Drop any remaining rows with NaN values
            melted_df = melted_df.dropna()
            
            return melted_df[['Descripcion', 'mes', 'value', 'year', 'fecha']]
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def process_all(self):
        """Process all files in the raw data directory"""
        all_dfs = []
        
        # Process each Excel file in the folder
        for file in os.listdir(self.raw_data_path):
            if file.endswith('.xlsx') or file.endswith('.xls'):
                file_path = os.path.join(self.raw_data_path, file)
                df = self.process_file(file_path)
                if df is not None and not df.empty:
                    all_dfs.append(df)
        
        # Concatenate all dataframes
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            return final_df
        else:
            return pd.DataFrame()