import pandas as pd
from pathlib import Path
import os
from utils.process_data.config import DATA_PATHS
from utils.process_data.sunarp.config import FILE_CATEGORIES
from utils.process_data.sunarp.utils import clean_dataframe, melt_dataframe,get_year_from_filename
#
class SunarpProcessor:
    def __init__(self):
        self.raw_data_path = DATA_PATHS['raw_sunarp']
        self.processed_path = DATA_PATHS['process'] / 'sunarp_consolidated.csv'
        self.file_categories = self.categorize_files()

    def categorize_files(self):
        """Categorize files based on their names"""
        categorized_files = FILE_CATEGORIES.copy()
        
        for file in os.listdir(self.raw_data_path):
            if file.endswith(".xlsx"):
                full_path = os.path.join(self.raw_data_path, file)
                for category in categorized_files.keys():
                    if category in file:
                        categorized_files[category].append(full_path)
        
        return categorized_files

    def find_first_data_row(self, file, sheet_name):
        """Encuentra la primera fila con datos en un archivo Excel"""
        df_temp = pd.read_excel(file, sheet_name=sheet_name, header=None)
        for i, row in df_temp.iterrows():
            if row.notna().sum() > 2:  # Consideramos fila válida si tiene al menos 3 valores no vacíos
                return i
        return 0  # En caso de error, asumimos que empieza desde la primera fila

    def process_livianos(self):
        dataframes = []
        for file in self.file_categories['Livianos']:
            sheet_name = "Oficina Reg I al XIII"
            start_row = self.find_first_data_row(file, sheet_name)
            df = pd.read_excel(file, sheet_name=sheet_name, skiprows=start_row)
            df = clean_dataframe(df, ['OFICINA', 'CLASE', 'MARCA'])
            df["TIPO"] = "Livianos"
            df = melt_dataframe(df, ["OFICINA", "TIPO", "CLASE", "MARCA", "MODELO"])
            df["ANIO"] = get_year_from_filename(file)
            dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

    def process_pesados(self):
        dataframes = []
        for file in self.file_categories['Pesados']:
            sheet_name = "Oficina y Clase"
            start_row = self.find_first_data_row(file, sheet_name)
            df = pd.read_excel(file, sheet_name=sheet_name, skiprows=start_row)
            df = clean_dataframe(df, ['OFICINA', 'CLASE', 'MARCA'])
            df["TIPO"] = "Pesados"
            df = melt_dataframe(df, ["OFICINA", "TIPO", "CLASE", "MARCA", "MODELO"])
            df["ANIO"] = get_year_from_filename(file)
            dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

    
    def process_hibridos(self):
        dataframes = []
        for file in self.file_categories['Hibridos']:
            sheet_name = "Marca y Modelo"
            start_row = self.find_first_data_row(file, sheet_name)
            df = pd.read_excel(file, sheet_name=sheet_name, skiprows=start_row)
            df = clean_dataframe(df, ['CLASE', 'MARCA'])
            df["OFICINA"] = "LIMA"
            df["TIPO"] = "Hibridos y Electricos"
            df = df.drop(columns="TECNOLOGIA")
            df = melt_dataframe(df, ["OFICINA", "TIPO", "CLASE", "MARCA", "MODELO"])
            df["ANIO"] = get_year_from_filename(file)
            dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

    def process_remolques(self):
        dataframes = []
        for file in self.file_categories['Remolques']:
            for sheet_name in ["Oficina Reg", "Oficina Registral"]:
                try:           
                    start_row = self.find_first_data_row(file, sheet_name)
                    df = pd.read_excel(file, sheet_name=sheet_name, skiprows=start_row)
                    df = clean_dataframe(df, ['OFICINA', 'MARCA'])
                    df["CLASE"] = "Remolques y SemiR"
                    df["TIPO"] = "Remolques y SemiR"
                    df = melt_dataframe(df, ["OFICINA", "TIPO", "CLASE", "MARCA", "MODELO"])
                    df["ANIO"] = get_year_from_filename(file)
                    dataframes.append(df)
                    break
                except Exception:
                    if sheet_name == "Oficina Registral":
                        raise ValueError(f"Could not process file {file} with any known sheet name")
                    continue
        return pd.concat(dataframes, ignore_index=True)

    def process_menores(self, tipo):
        dataframes = []
        for file in self.file_categories['Menores']:
          
            sheet_name = f"Oficina x {tipo}"
            start_row = self.find_first_data_row(file, sheet_name)
            df = pd.read_excel(file, sheet_name=sheet_name, skiprows=start_row)
            df = clean_dataframe(df, ['OFICINA REGISTRAL', 'MARCA'])
            df = df.rename(columns={"OFICINA REGISTRAL": "OFICINA"})
            df["TIPO"] = "Menores"
            df["CLASE"] = tipo
            df = melt_dataframe(df, ["OFICINA", "TIPO", "CLASE", "MARCA", "MODELO"])
            df["ANIO"] = get_year_from_filename(file)
            dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

    def process_all(self):
        df_livianos = self.process_livianos()  #bien
        df_pesados = self.process_pesados()    #bien
        df_hibridos = self.process_hibridos() #mal
        df_remolques = self.process_remolques() #bien
        df_menores1 = self.process_menores("Motocicletas") #bien
        df_menores2 = self.process_menores("Trimotos")  #bien
        df_consolidado = pd.concat([
            df_livianos, df_pesados,
            df_hibridos, 
            df_remolques,
            df_menores1, df_menores2
        ], ignore_index=True)
        df_consolidado = df_consolidado[df_consolidado["VENTAS"].fillna(0) > 0]
        df_consolidado["ANIO"] = df_consolidado["ANIO"].astype(int)

        # Dictionary to convert Spanish month names to numbers
        mes_to_num = {
            'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
        }
        
        # Create fecha column
        df_consolidado['fecha'] = df_consolidado['ANIO'].astype(str) + '-' + df_consolidado['MES'].map(mes_to_num) + '-01'
        # print(df_consolidado[df_consolidado['ANIO']==2024])
        
        # Group by all columns except VENTAS
        df_consolidado = df_consolidado.groupby(
            df_consolidado.columns.drop('VENTAS').tolist()
        ).agg({'VENTAS': 'sum'}).reset_index()

        return df_consolidado