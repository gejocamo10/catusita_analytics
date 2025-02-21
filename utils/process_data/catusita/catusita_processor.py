import os
import pandas as pd
import requests
from utils.process_data.catusita.config import (
    COLUMN_RENAME_MAPPING, KITS_RENAME_MAPPING, 
    FILTER_COLUMNS, COLUMNS_TO_KEEP, FILTER_DATE
)
from utils.process_data.config import DATA_PATHS
from utils.process_data.catusita.utils import (
    format_column_names, clean_string_columns, clean_article_names
)

class CatusitaProcessor:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.api_url = "http://api.catusita.com:8083/api/sales/forDate"
        # http://api.catusita.com:8083/api/sales/forDate?Date1=20250101&Date2=20250115

    def fetch_data_from_api(self):
        """Obtiene datos desde la API y los convierte en un DataFrame."""
        params = {"Date1": self.start_date, "Date2": self.end_date}
        headers = {"Accept": "application/json"}

        try:
            response = requests.get(self.api_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            return pd.DataFrame(data)
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con la API: {e}")
            return pd.DataFrame()

    def read_main_data(self):
        """Obtiene los datos desde la API y los estructura en un DataFrame compatible."""
        df_catusita = self.fetch_data_from_api()
        
        if df_catusita.empty:
            print("No se obtuvieron datos desde la API.")
            return df_catusita
        
        df_catusita['fecha'] = pd.to_datetime(df_catusita['fecha'], format='%Y-%m-%d')
        df_catusita['transacciones'] = 1
        
        df_catusita = df_catusita.drop(columns=['document', 'codeSupply'], errors='ignore')
        
        df_catusita = format_column_names(df_catusita).rename(columns=COLUMN_RENAME_MAPPING)
        df_catusita = clean_article_names(df_catusita)
        df_catusita = clean_string_columns(df_catusita)
        
        df_catusita = df_catusita[(df_catusita[FILTER_COLUMNS] >= 0).all(axis=1)]
        df_catusita.drop_duplicates(inplace=True)
        
        df_catusita = df_catusita[df_catusita['fecha'].dt.weekday != 6]
        
        return df_catusita
    
    def process_kits_and_blacklist(self, df):
        """Process kits and blacklist filtering"""
        kits_file_path = DATA_PATHS['kits_file']
        df_kits = pd.read_excel(kits_file_path)
        df_kits = df_kits.rename(columns={
            "Código KIT (Sin historial)": "articulo_madre",
            "Código 1": "articulo_1",
            "Código 2": "articulo_2",
            "Código 3": "articulo_3"
        })

        blacklist_file_path = DATA_PATHS['blacklist_file']
        df_blacklist = pd.read_excel(blacklist_file_path)
        df_blacklist = df_blacklist.rename(columns={'codigo': 'articulo'})

        kit_mothers = set(df_kits['articulo_madre'].str.lower())
        
        mask_kits = df['articulo'].str.lower().isin(kit_mothers)
        df_kits_rows = df[mask_kits]
        df_non_kits = df[~mask_kits]

        expanded_rows = []
        for _, row in df_kits_rows.iterrows():
            kit_match = df_kits[df_kits['articulo_madre'].str.lower() == row['articulo'].lower()]
            kit_row = kit_match.iloc[0]
            for i in range(1, 4):
                component = kit_row[f'articulo_{i}']
                if pd.notna(component) and component.strip() != '':
                    new_row = row.copy()
                    new_row['articulo'] = component.lower()
                    expanded_rows.append(new_row)

        if expanded_rows:
            df_expanded_kits = pd.DataFrame(expanded_rows)
            df_final = pd.concat([df_non_kits, df_expanded_kits], ignore_index=True)
        else:
            df_final = df_non_kits

        df_final = df_final[~df_final['articulo'].isin(df_blacklist['articulo'])]
        df_final['articulo'] = df_final['articulo'].str.lower()

        return df_final
    
    def process_data(self):
        """Realiza el procesamiento completo de los datos."""
        df_catusita = self.read_main_data()
        if df_catusita.empty:
            return df_catusita
        
        df_catusita = self.process_kits_and_blacklist(df_catusita)
        df_catusita = df_catusita[COLUMNS_TO_KEEP]
        
        return df_catusita

    def save_data(self, df):
        """Guarda los datos procesados en un archivo CSV."""
        output_path = DATA_PATHS['process']
        output_file = os.path.join(output_path, 'catusita_consolidated.csv')
        df.to_csv(output_file, index=False)
        print(f"✅ Datos guardados en {output_file}")
