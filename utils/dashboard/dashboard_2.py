import pandas as pd
import numpy as np
import yfinance as yf
import math
from typing import Tuple, Optional

class DataProcessor:
    def __init__(self, path: str):
        self.path = path
        self.results_models_comparison = None
        self.df_inventory = None
        self.df_rfm = None
        self.tipo_de_cambio_df = None
        self.df_products = None
        self.back_order = None
        self.closing_prices = None
        self.long_format = None
        self.merged_df_tc_final = None
        self.df_merged = None
        self.result_precio = None
        self.margin_result = None
        self.margin_result_fuente = None
        self.df1_final = None
        self.df_download = None
        self.df_dashboard_by_fuente = None
        self.df_dashboard = None

    def load_data(self) -> None:
        self.results_models_comparison = pd.read_csv(f"{self.path}/data/cleaned/predictions.csv")
        self.df_inventory = pd.read_excel(f"{self.path}/data/raw/catusita/inventory.xlsx")
        self.df_rfm = pd.read_csv(f"{self.path}/data/process/df_rfm.csv")
        self.tipo_de_cambio_df = pd.read_excel(f"{self.path}/data/raw/catusita/saldo de todo 04.11.2024.2.xls", skiprows=2)
        self.df_products = pd.read_csv(f"{self.path}/data/process/catusita_consolidated.csv")
        try:
            self.back_order = pd.read_excel(f"{self.path}/data/raw/catusita/backorder12_12.xlsx")
        except FileNotFoundError:
            self.back_order = pd.DataFrame()
        

    def preprocess_exchange_rates(self) -> None:
        self.tipo_de_cambio_df = self.tipo_de_cambio_df[['Código','Mnd','Fob','Ult. Fecha','Ult. Compra']]
        self.tipo_de_cambio_df.columns = ['codigo', 'moneda', 'monto', 'ultima_fecha', 'ultima_compra']
        self.tipo_de_cambio_df = self.tipo_de_cambio_df.copy()
        self.tipo_de_cambio_df['codigo'] = self.tipo_de_cambio_df['codigo'].astype(str)
        self.tipo_de_cambio_df = self.tipo_de_cambio_df.dropna(subset=['ultima_fecha'])
        self.tipo_de_cambio_df['codigo'] = self.tipo_de_cambio_df['codigo'].str.lower()
        self.tipo_de_cambio_df = self.tipo_de_cambio_df[self.tipo_de_cambio_df['ultima_fecha'].notna()]
        self.tipo_de_cambio_df['ultima_fecha'] = pd.to_datetime(self.tipo_de_cambio_df['ultima_fecha'], errors='coerce')

    def get_currency_data(self) -> None:
        start = self.tipo_de_cambio_df['ultima_fecha'].min().date()
        end = self.tipo_de_cambio_df['ultima_fecha'].max().date()
        currency_pairs = ['PENUSD=X', 'EURUSD=X', 'JPYUSD=X', 'GBPUSD=X']
        data = yf.download(currency_pairs, start=start, end=end)
        self.closing_prices = data['Close']
        self.closing_prices.columns = [col.split('.')[0] for col in self.closing_prices.columns]

    def process_currency_data(self) -> None:
        self.long_format = self.closing_prices.reset_index().melt(id_vars='Date', var_name='Currency Pair', value_name='Closing Price')
        self.long_format['Currency Pair'] = self.long_format['Currency Pair'].str.replace('=X', '', regex=False)
        self.long_format = self.long_format.dropna(subset=['Closing Price'])
        
        full_date_range = pd.date_range(start=self.long_format['Date'].min(), end=self.long_format['Date'].max(), freq='D')
        currency_pairs = self.long_format['Currency Pair'].unique()
        complete_index = pd.MultiIndex.from_product([full_date_range, currency_pairs], names=['Date', 'Currency Pair'])
        df_full = pd.DataFrame(index=complete_index).reset_index()
        
        self.long_format = df_full.merge(self.long_format, on=['Date', 'Currency Pair'], how='left')
        self.long_format['Closing Price'] = self.long_format.groupby('Currency Pair')['Closing Price'].fillna(method='ffill')
        self.long_format = self.long_format.rename(columns={'Closing Price': 'tc'})

    def merge_exchange_rates(self) -> None:
        merged_df_tc = pd.merge(self.tipo_de_cambio_df, self.long_format, left_on='ultima_fecha', right_on='Date', how='left')
        merged_df_tc['monto'] = pd.to_numeric(merged_df_tc['monto'], errors='coerce')
        merged_df_tc['tc'] = pd.to_numeric(merged_df_tc['tc'], errors='coerce')
        
        def convert_to_usd(row):
            if pd.isna(row['Currency Pair']) or row['moneda'] == 'USD':
                return row['monto']
            currency_pair_map = {'SOL': 'PENUSD', 'EUR': 'EURUSD', 'JPY': 'JPYUSD', 'GBP': 'GBPUSD'}
            if row['moneda'] in currency_pair_map and row['Currency Pair'] == currency_pair_map[row['moneda']]:
                return row['monto'] / row['tc'] if row['moneda'] == 'SOL' else row['monto'] * row['tc']
            return 0

        merged_df_tc['monto_usd'] = merged_df_tc.apply(convert_to_usd, axis=1)
        merged_df_tc = merged_df_tc[merged_df_tc['monto_usd'] != 0]
        self.merged_df_tc_final = merged_df_tc[['codigo', 'ultima_fecha', 'monto_usd', 'ultima_compra']]
        self.merged_df_tc_final = self.merged_df_tc_final[self.merged_df_tc_final['monto_usd'].notna()]

    def process_inventory(self) -> None:
        self.df_inventory = self.df_inventory.copy()
        self.df_inventory.columns = ['cia', 'date', 'codigo', 'descripcion', 'um', 'stock']
        
        self.df_inventory = self.df_inventory[
            (self.df_inventory['date'] != 'Periodo') & 
            (self.df_inventory['date'].notna())
        ]

        self.df_inventory['date'] = pd.to_datetime(self.df_inventory['date'], format='%Y%m%d')
        # max_date = self.results_models_comparison['date'].max()
        # self.df_inventory = self.df_inventory[
        #     (self.df_inventory['date'] == max_date) & 
        #     (self.df_inventory['codigo'].notna())
        # ]
        self.df_inventory.loc[:, 'codigo'] = self.df_inventory['codigo'].str.lower()

    def merge_dataframes(self) -> None:
        self.df_merged = self.results_models_comparison.copy()
        self.df_merged = self.df_merged.rename(columns={'sku':'articulo'})
        self.df_merged = self.df_merged.merge(
            self.df_products[['articulo', 'fuente_suministro','lt']].drop_duplicates(), 
            how='left', 
            on='articulo'
        )
        self.df_merged['date'] = pd.to_datetime(self.df_merged['date'])
        self.df_inventory['date'] = pd.to_datetime(self.df_inventory['date'])
        self.df_merged = self.df_merged.merge(
            self.df_inventory[['codigo', 'stock', 'date']].drop_duplicates(), 
            how='left', 
            left_on=['articulo', 'date'], 
            right_on=['codigo', 'date']
        )
        self.df_merged['stock'] = self.df_merged['stock'].fillna(0)
        self.df_merged = self.df_merged.drop(columns='codigo')

    # def calculate_risk(self) -> None:
    #     self.df_merged['index_riesgo'] = self.df_merged['stock'] / (self.df_merged['caa'] / self.df_merged['lt_x'])
    #     self.df_merged['riesgo'] = pd.cut(
    #         self.df_merged['index_riesgo'], 
    #         bins=[-float('inf'), 1, 1.2, 1.5, float('inf')],
    #         labels=['Rojo', 'Naranja', 'Amarillo', 'Verde'], 
    #         right=False
    #     )
    #     self.df_merged['ranking_riesgo'] = self.df_merged['index_riesgo'].rank(method='dense', ascending=True).fillna(0).astype(int)

    def process_prices(self) -> None:
        df_precio = self.df_products[['articulo', 'cantidad', 'venta_pen', 'fecha']].copy()
        df_precio['fecha'] = pd.to_datetime(df_precio['fecha'], errors='coerce')
        # df_precio = df_precio[df_precio['fecha'].dt.year == 2024]
        df_precio['precio'] = df_precio['venta_pen'] / df_precio['cantidad']
        self.result_precio = df_precio.groupby('articulo').agg(precio=('precio', 'mean')).reset_index()

    def calculate_margin(self) -> None:
        df_margen = self.df_products[['articulo', 'fuente_suministro','costo', 'venta_pen', 'fecha']].copy()
        df_margen['fecha'] = pd.to_datetime(df_margen['fecha'], errors='coerce')
        # df_margen = df_margen[df_margen['fecha'].dt.year == 2024]
        df_margen['margen'] = df_margen['venta_pen'] / df_margen['costo'] - 1
        self.margin_result = df_margen.groupby('articulo').agg(
            total_venta_pen=('venta_pen', 'sum'),
            mean_margen=('margen', 'mean')
        ).reset_index().sort_values(by='total_venta_pen', ascending=False)
        self.margin_result_fuente = df_margen.groupby('articulo').agg(
            mean_margen=('margen', 'mean')
        ).reset_index()

    def create_df1_final(self) -> None:
        df1 = self.df_merged[['fuente_suministro', 'date', 'articulo','real', 'catusita', 'caa','lt_x']].copy()
        df1 = df1.rename(columns={'catusita': 'venta_sin_recomendacion', 'caa': 'venta_con_recomendacion'})
        self.df1_final = df1.merge(self.result_precio, how='left', on='articulo')
        self.df1_final = self.df1_final[['fuente_suministro', 'date', 'articulo', 'venta_sin_recomendacion', 'venta_con_recomendacion','real', 'precio','lt_x']]
        
        self.df1_final['ingreso_sin_recomendacion'] = np.where(
            self.df1_final['venta_sin_recomendacion'] < self.df1_final['real'],
            self.df1_final['venta_sin_recomendacion'] * self.df1_final['precio'],
            self.df1_final['real'] * self.df1_final['precio']
        )
        
        self.df1_final['venta_con_recomendacion'] = np.where(
            self.df1_final['venta_con_recomendacion'] < self.df1_final['real'],
            self.df1_final['venta_con_recomendacion'] * self.df1_final['precio'],
            self.df1_final['real'] * self.df1_final['precio']
        )
        
        self.df1_final['ingreso_sin_recomendacion_ajustado'] = self.df1_final['ingreso_sin_recomendacion'] / (self.df1_final['lt_x'] * 0.83)
        self.df1_final['ingreso_con_recomendación_ajustado'] = self.df1_final['venta_con_recomendacion'] / (self.df1_final['lt_x'] * 0.83)
        
        penusd_tc = self.long_format[self.long_format['Currency Pair'] == 'PENUSD'].groupby('Date')['tc'].last().reset_index()
        self.df1_final = self.df1_final.merge(penusd_tc, how='left', left_on='date', right_on='Date')
        self.df1_final['tc'] = 1/self.df1_final['tc']
        self.df1_final['ingreso_usd_sin_recomendacion'] = self.df1_final['ingreso_sin_recomendacion_ajustado'] / self.df1_final['tc']
        self.df1_final['ingreso_usd_con_recomendacion'] = self.df1_final['ingreso_con_recomendación_ajustado'] / self.df1_final['tc']
        self.df1_final = self.df1_final[['fuente_suministro', 'date', 'articulo', 'lt_x', 'ingreso_usd_sin_recomendacion', 'ingreso_usd_con_recomendacion', 'tc']]
        # self.df1_final = self.df1_final.drop_duplicates()

    def create_final_dataframe(self) -> None:
        # last_date = self.df_merged['date'].max()
        # df_merged_last = self.df_merged[self.df_merged['date'] == last_date].copy()
        df_merged_last = self.df_merged.copy()
        
        df_merged_last['demanda_mensual'] = df_merged_last['caa'] / df_merged_last['lt_x']
        self.df_download = df_merged_last[['articulo', 'stock', 'caa', 'demanda_mensual', 'corr_sd', 'lt_x']]
        self.df_download = self.df_download.copy()
        self.df_download['meses_proteccion'] = self.df_download['corr_sd'] / self.df_download['demanda_mensual']
        self.df_download = self.df_download[['articulo', 'stock', 'caa', 'demanda_mensual', 'meses_proteccion', 'lt_x']]
        self.df_download = self.df_download.merge(self.margin_result[['articulo', 'mean_margen']], how='left', on='articulo')
        self.df_download = self.df_download.merge(self.merged_df_tc_final, how='left', left_on='articulo', right_on='codigo')

    def add_compra_real(self) -> None:
        df_predicciones = pd.read_csv(f"{self.path}/data/cleaned/predictions.csv")
        df_inventory2 = pd.read_excel(f"{self.path}/data/raw/catusita/inventory.xlsx")
        
        df_predicciones = df_predicciones.rename(columns={'sku': 'articulo'})
        
        df_inventory2 = df_inventory2[
            (df_inventory2['FECHA AL'] != 'Periodo') & 
            (df_inventory2['FECHA AL'].notna())
        ]
        
        df_inventory2['FECHA AL'] = pd.to_datetime(df_inventory2['FECHA AL'], format='%Y%m%d')
        # max_date = df_inventory2['FECHA AL'].max()
        # df_inventory2 = df_inventory2[df_inventory2['FECHA AL'] == max_date]
        df_inventory2['FECHA AL'] = df_inventory2['FECHA AL'].dt.strftime('%d/%m/%Y')
        df_inventory2['CODIGO'] = df_inventory2['CODIGO'].str.lower()
        
        df_predicciones['date'] = pd.to_datetime(df_predicciones['date'], format='%Y-%m-%d')
        df_predicciones['date'] = df_predicciones['date'].dt.strftime('%d/%m/%Y')
        
        merged_df = df_predicciones.merge(
            df_inventory2[['FECHA AL', 'CODIGO', 'STOCK']], 
            how='left', 
            left_on=['articulo'], 
            right_on=['CODIGO']
        )
        
        merged_df['STOCK'] = merged_df['STOCK'].fillna(0)
        
        if not self.back_order.empty:
            merged_df = merged_df.merge(self.back_order, how='left', on='articulo')
            merged_df['backorder'] = merged_df['backorder'].fillna(0)
        else:
            merged_df['backorder'] = 0
        
        merged_df['sobrante'] = np.maximum(merged_df['STOCK'] + merged_df['backorder'] - merged_df['caa_lt'], 0)
        merged_df['nueva_compra_sugerida'] = np.maximum(merged_df['caa'] - merged_df['sobrante'], 0)
        # merged_df['nueva_compra_sugerida'] = np.ceil(merged_df['nueva_compra_sugerida']).astype(int)
        merged_df['nueva_compra_sugerida'] = np.ceil(merged_df['nueva_compra_sugerida']).fillna(0).astype(int)

        merge_columns = merged_df[['articulo', 'nueva_compra_sugerida', 'caa', 'backorder']].copy()
        
        self.df_download = self.df_download.merge(merge_columns, how='left', on='articulo')
        self.df_download['compra_sugerida'] = self.df_download['nueva_compra_sugerida'].fillna(0)
        self.df_download['backorder'] = self.df_download['backorder'].fillna(0)
        
        mask = self.df_download['demanda_mensual'] != 0
        self.df_download.loc[mask, 'meses_proteccion'] = (
            self.df_download.loc[mask, 'meses_proteccion'] * 
            (self.df_download.loc[mask, 'compra_sugerida'] / self.df_download.loc[mask, 'demanda_mensual'])
        )
        
        columns_to_drop = ['codigo', 'nueva_compra_sugerida', 'caa']
        for col in columns_to_drop:
            if col in self.df_download.columns:
                self.df_download = self.df_download.drop(columns=[col])

    def finalize_processing(self) -> None:
        self.df_download = self.df_download.rename(columns={'caa_x': 'compras_recomendadas'})
        self.df_download = self.df_download.drop_duplicates()
        # self.df_download['compras_recomendadas'] = self.df_download['compras_recomendadas'].apply(lambda x: math.ceil(x / 50) * 50)
        self.df_download['compras_recomendadas'] = self.df_download['compras_recomendadas'].fillna(0).apply(lambda x: math.ceil(x / 50) * 50)
        self.df_download['costo_compra'] = self.df_download['monto_usd'] * self.df_download['compras_recomendadas']

        df1_final_filled = self.df1_final.fillna(0)
        df1_final_grouped = df1_final_filled.groupby(['articulo', 'fuente_suministro']).agg({
            'ingreso_usd_sin_recomendacion': 'sum',
            'ingreso_usd_con_recomendacion': 'sum'
        }).reset_index()

        self.df_download = self.df_download.merge(
            df1_final_grouped[['articulo', 'fuente_suministro']], 
            how='left', 
            on='articulo'
        )

        self.df_download = self.df_download.merge(
            self.df_rfm, 
            left_on='articulo',
            right_on='sku',
            how='left'
        )
        self.df_download['rfm'] = self.df_download['rfm'].fillna(0).astype(int)

        df1_final_grouped['ganancia_oportunidad'] = (
            df1_final_grouped['ingreso_usd_con_recomendacion'] - 
            df1_final_grouped['ingreso_usd_sin_recomendacion']
        )

        df1_final_grouped_fs = df1_final_grouped.groupby(['fuente_suministro']).agg({
            'ganancia_oportunidad': 'sum'
        }).reset_index()

        df1_final_grouped_fs = df1_final_grouped_fs.sort_values(
            by='ganancia_oportunidad', 
            ascending=False
        ).reset_index(drop=True)
        
        df1_final_grouped_fs['hierarchy'] = df1_final_grouped_fs.index + 1

        self.df_download = self.df_download.merge(
            df1_final_grouped_fs[['fuente_suministro', 'hierarchy']], 
            how='left', 
            on='fuente_suministro'
        )

        # self.df_download['venta_acumulada'] = self.df_download['demanda_mensual'] * self.df_download['monto_usd'] * self.df_download['lt_x']
        # self.df_download['deficit'] = self.df_download['venta_acumulada'] - (self.df_download['stock'] + self.df_download['backorder']) * self.df_download['monto_usd']
        # self.df_download['deficit'] = self.df_download['deficit'].apply(lambda x: max(x, 0))  # El déficit no puede ser negativo
        # self.df_download['urgency'] = self.df_download['deficit'].rank(method='min', ascending=False).fillna(0).astype(int)

        self.df_download.loc[self.df_download['demanda_mensual'] < 0, 'demanda_mensual'] = 0
        self.df_download.loc[self.df_download['compras_recomendadas'] < 0, 'compras_recomendadas'] = 0
        # self.df_download['demanda_mensual'] = self.df_download['demanda_mensual'].fillna(0) 
        # self.df_download['compras_recomendadas'] = self.df_download['compras_recomendadas'].fillna(0) 

        # Calcular riesgo
        self.df_download['holgura'] = self.df_download['stock'] / self.df_download['demanda_mensual']
        self.df_download['consumiendo_proteccion'] = (self.df_download['holgura'] < self.df_download['meses_proteccion']).astype(int)
        self.df_download['quebro'] = (self.df_download['holgura'] <= 0).astype(int)
        self.df_download['va_a_quebrar'] = ((self.df_download['stock'] + self.df_download['backorder']) < self.df_download['demanda_mensual']).astype(int)
        self.df_download['verde'] = (
            (self.df_download['quebro'] == 0) & 
            (self.df_download['consumiendo_proteccion'] == 0) & 
            (self.df_download['va_a_quebrar'] == 0)
        ).astype(int)
        self.df_download['amarillo'] = (
            (self.df_download['consumiendo_proteccion'] == 1) & 
            (self.df_download['quebro'] == 0)
        ).astype(int)
        self.df_download['rojo'] = (
            (self.df_download['quebro'] == 1) |
            (self.df_download['va_a_quebrar'] == 1) 
            ).astype(int)
        self.df_download['riesgo'] = self.df_download.apply(
            lambda row: 'rojo' if row['rojo'] == 1 else 
                        'amarillo' if row['amarillo'] == 1 else 
                        'verde',
            axis=1
        )

        # filtrar solo las importantes para la tabla por fuente de suministro
        self.df_download['demanda_mensual_usd'] = self.df_download['demanda_mensual'] * self.df_download['monto_usd']
        df_temp = self.df_download.copy()
        df_temp = df_temp[(df_temp['rfm']==3) & (df_temp['riesgo']=='rojo')]
        df_temp = df_temp.groupby('fuente_suministro').agg(
            recomendacion=('costo_compra', 'sum'),
            demanda_mensual_usd = ('demanda_mensual_usd','sum')
        ).reset_index()
        df_temp2 = self.df_download.groupby('fuente_suministro').agg(
            lead_time=('lt_x', 'first'),
            riesgo=('riesgo', lambda x: x.mode()[0] if not x.mode().empty else None), # moda
        ).reset_index()
        self.df_dashboard_by_fuente = df_temp2.merge(df_temp, how='left',on='fuente_suministro')
        self.df_dashboard_by_fuente['recomendacion'] = self.df_dashboard_by_fuente['recomendacion'].fillna(0).astype(int)
        self.df_dashboard_by_fuente['demanda_mensual_usd'] = self.df_dashboard_by_fuente['demanda_mensual_usd'].fillna(0).astype(int)
        
        # filtrar rfm = 3 para tabla de dashboard
        # self.df_dashboard = self.df_download[self.df_download['rfm']==3]
        
        # dar formato
        self.df_dashboard_by_fuente['riesgo_color'] = self.df_dashboard_by_fuente['riesgo'].map({
            'verde': '🟢',
            'amarillo': '🟡',
            #'naranja': '🟠',
            'rojo': '🔴'
        })
        self.df_download = self.df_download[[
            'articulo','fuente_suministro','stock','compras_recomendadas','demanda_mensual','meses_proteccion',
            'riesgo','lt_x','mean_margen','ultima_fecha','monto_usd',
            'ultima_compra','costo_compra','rfm','backorder'
        ]]
        self.df_dashboard = self.df_download[[
            'articulo','fuente_suministro','stock','backorder','rfm','riesgo',
            'monto_usd','ultima_compra','compras_recomendadas','costo_compra'
        ]]
        self.df_dashboard_by_fuente = self.df_dashboard_by_fuente[[
            'fuente_suministro',
            'lead_time',
            'recomendacion',
            'demanda_mensual_usd'
        ]]
        # columns mapping
        display_columns = {
            'articulo': 'Artículo',
            'fuente_suministro': 'Fuente Suministro',
            'stock': 'Inventario',
            'backorder': 'Backorder',
            'compras_recomendadas': 'Compras Recomendadas',
            'rfm':'Importancia RFM',
            'riesgo': 'Alerta',
            'monto_usd': 'Monto USD',
            'ultima_compra': 'Última Compra',
            'demanda_mensual': 'Demanda Mensual',
            'lt_x': 'Lead Time',
            'mean_margen': 'Margen',
            'meses_proteccion': 'Meses proteccion',
            'ultima_fecha': 'Ultima fecha',
            'costo_compra': 'Recomendacion USD'
            }
        display_columns_fuente = {
            'fuente_suministro': 'Fuente de Suministro',
            'lead_time': 'Lead Time',
            'recomendacion': 'Recomendacion USD',
            'demanda_mensual_usd': 'Demanda Mensual USD',
            'mean_margen': 'Margen Promedio'
        }
        
        self.df_dashboard = self.df_dashboard.rename(columns=display_columns)
        self.df_download = self.df_download.rename(columns=display_columns)
        self.df_dashboard_by_fuente = self.df_dashboard_by_fuente.rename(columns=display_columns_fuente)

    def process_all(self) -> None:
        self.load_data()
        self.preprocess_exchange_rates()
        self.get_currency_data()
        self.process_currency_data()
        self.merge_exchange_rates()
        self.process_inventory()
        self.merge_dataframes()
        self.process_prices()
        self.calculate_margin()
        self.create_df1_final()
        self.create_final_dataframe()
        self.add_compra_real()
        self.finalize_processing()

if __name__ == "__main__":
    from pathlib import Path
    # Usar pathlib para definir y manejar la ruta base
    base_path = Path('C:/Users/YOGA/Desktop/repositories/caa/catusita/catusita_predictions')
    
    # Inicializar el procesador
    processor = DataProcessor(base_path)
    processor.process_all()
    
    # Definir rutas para guardar los archivos
    cleaned_path = base_path / 'data' / 'cleaned'
    cleaned_path.mkdir(parents=True, exist_ok=True)  # Crear el directorio si no existe
    
    # Guardar los DataFrames
    processor.df_dashboard.to_csv(cleaned_path / 'dashboard.csv', index=False)
    processor.df_dashboard_by_fuente.to_csv(cleaned_path / 'dashboard_by_fuente.csv', index=False)
    processor.df_download.to_csv(cleaned_path / 'download.csv', index=False)