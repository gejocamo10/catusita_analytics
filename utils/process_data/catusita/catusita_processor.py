import os
import pandas as pd
from utils.process_data.catusita.config import (
    PATHS, COLUMN_RENAME_MAPPING, KITS_RENAME_MAPPING, 
    FILTER_COLUMNS, COLUMNS_TO_KEEP, FILTER_DATE
)
from utils.process_data.config import DATA_PATHS
from utils.process_data.catusita.utils import (
    format_column_names, clean_string_columns, clean_article_names
)

class CatusitaProcessor:
    def __init__(self):
        self.base_path = DATA_PATHS['raw_catusita']
     
    def _get_full_path(self, relative_path):
        """Helper method to construct full path from base path and relative path"""
        return os.path.join(self.base_path, relative_path.lstrip('/'))
     
    def read_main_data(self):
        """Read and combine all sheets from the main Excel file"""
        file_path = self._get_full_path(PATHS['input_file'])
        
        df_catusita = pd.read_excel(file_path, sheet_name="Sheet1")
        lista_columnas = df_catusita.columns.tolist()
        
        excel_file = pd.ExcelFile(file_path)
        list_hojas = excel_file.sheet_names[1:]
        
        for hoja in list_hojas:
            df_catusita_hoja = pd.read_excel(file_path, sheet_name=hoja, header=None)
            df_catusita_hoja.columns = lista_columnas
            df_catusita = pd.concat([df_catusita, df_catusita_hoja], ignore_index=True)
        
        return df_catusita

    def read_lt_data(self):
        """Read lead time data"""
        file_path_lt = self._get_full_path(PATHS['lt'])
        df_lt = pd.read_csv(file_path_lt)
        df_lt = df_lt.rename(columns={"fuente_de_suministro": "fuente_suministro"})
        return df_lt

    def process_kits_and_blacklist(self, df):
        """Process kits and blacklist filtering"""
        kits_file_path = self._get_full_path(PATHS['kits_file'])
        df_kits = pd.read_excel(kits_file_path)
        df_kits = df_kits.rename(columns={
            "Código KIT (Sin historial)": "articulo_madre",
            "Código 1": "articulo_1",
            "Código 2": "articulo_2",
            "Código 3": "articulo_3"
        })

        blacklist_file_path = self._get_full_path(PATHS['blacklist_file'])
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
        """Main processing function"""
        df_catusita = self.read_main_data()
        df_catusita = format_column_names(df_catusita).rename(columns=COLUMN_RENAME_MAPPING)
     
        df_catusita['fecha'] = pd.to_datetime(df_catusita['fecha'], format='%Y-%m-%d')
        df_catusita['transacciones'] = 1
     
        df_catusita["cia"].replace("Pagina 1 de 1", pd.NA, inplace=True)
        df_catusita.dropna(how='all', inplace=True)
        df_catusita = clean_article_names(df_catusita)
        df_catusita = clean_string_columns(df_catusita)
     
        df_catusita = df_catusita[(df_catusita[FILTER_COLUMNS] >= 0).all(axis=1)]
        df_catusita.drop_duplicates(inplace=True)
        df_catusita = df_catusita[df_catusita['fecha'].dt.weekday != 6]
     
        df_catusita = self.process_kits_and_blacklist(df_catusita)
        df_catusita = df_catusita[COLUMNS_TO_KEEP]
        
        df_lt = self.read_lt_data()
        df_catusita = pd.merge(df_catusita, df_lt[["fuente_suministro", "LT_meses"]], on="fuente_suministro", how="left")
        df_catusita = df_catusita.rename(columns={"LT_meses": "lt"})
        
        return df_catusita

    def save_data(self, df):
        """Save processed data"""
        output_path = DATA_PATHS['process']
        output_file = os.path.join(output_path, 'catusita_consolidated.csv')
        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
        df.to_csv(output_file, index=False)