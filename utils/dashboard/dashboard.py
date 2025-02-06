import pandas as pd
import numpy as np
import yfinance as yf
import math
from typing import Tuple, Optional

class DataProcessor:
    def __init__(self, path: str):
        self.path = path
        self.df_predictions = None # results_models_comparison
        self.df_inventory = None
        self.df_rfm = None
        self.df_tc = None # tipo_de_cambio_df
        self.df_products = None
        self.df_backorder = None # back_order
        self.df_closing_prices = None # closing_prices
        self.df_long_format = None # long_format
        self.df_tc_final = None # merged_df_tc_final
        self.df_merged = None
        # self.df_precio_result = None # result_precio
        self.df_margin_result = None # margin_result
        self.df_download = None # dffinal2
        self.df_dashboard_by_fuente = None # dffinal3
        self.df_dashboard = None


    def load_and_process_data(self) -> None:
        self.df_predictions = pd.read_csv(f"{self.path}/data/cleaned/predictions.csv")
        self.df_inventory = pd.read_excel(f"{self.path}/data/raw/catusita/inventory.xlsx")
        self.df_rfm = pd.read_csv(f"{self.path}/data/process/df_rfm.csv")
        self.df_tc = pd.read_excel(f"{self.path}/data/raw/catusita/saldo de todo 04.11.2024.2.xls", skiprows=2)
        self.df_products = pd.read_csv(f"{self.path}/data/process/catusita_consolidated.csv")
        try:
            self.df_backorder = pd.read_excel(f"{self.path}/data/raw/catusita/backorder12_12.xlsx")
        except FileNotFoundError:
            self.df_backorder = pd.DataFrame()

        ### to_datime
        self.df_tc['Ult. Fecha'] = pd.to_datetime(self.df_tc['Ult. Fecha'], errors='coerce')
        self.df_products['fecha'] = pd.to_datetime(self.df_products['fecha'])
        self.df_predictions['date'] = pd.to_datetime(self.df_predictions['date'])
        self.df_inventory['FECHA AL'] = pd.to_datetime(self.df_inventory['FECHA AL'], format='%Y%m%d')
        
        ### processing data for raw tables
        ## df_tc
        self.df_tc = self.df_tc[['Código','Mnd','Fob','Ult. Fecha','Ult. Compra']]
        self.df_tc.columns = ['codigo', 'moneda', 'monto', 'ultima_fecha', 'ultima_compra']
        self.df_tc['codigo'] = self.df_tc['codigo'].astype(str)
        self.df_tc = self.df_tc.dropna(subset=['ultima_fecha'])
        self.df_tc['codigo'] = self.df_tc['codigo'].str.lower()
        self.df_tc = self.df_tc[self.df_tc['ultima_fecha'].notna()]
        
        ## df_product
        self.df_products['fecha_mensual'] = self.df_products['fecha'].dt.to_period('M').dt.to_timestamp()       
        # crear variable precio
        self.df_products['precio'] = self.df_products['venta_pen'] / self.df_products['cantidad']
        # crear variable margen
        self.df_products['margen'] = self.df_products['venta_pen'] / self.df_products['costo'] - 1
        self.df_margin_result = self.df_products.groupby('articulo').agg(
            total_venta_pen=('venta_pen', 'sum'),
            mean_margen=('margen', 'mean')
        ).reset_index().sort_values(by='total_venta_pen', ascending=False)
        # agregar por fecha_mensual, articulo, fuente_suministro 
        self.df_products = self.df_products.groupby(['fecha_mensual', 'articulo', 'fuente_suministro']).agg({
            'codigo': 'first', 
            'cantidad': 'sum',
            'transacciones': 'sum',
            'venta_pen': 'sum', 
            'costo': 'mean',
            'precio': 'mean',
            'lt': 'first'
        }).reset_index().rename(columns={'venta_pen': 'total_venta_pen','margen': 'mean_margen'})

        ## df_predictions
        self.df_predictions = self.df_predictions.rename(columns={'sku': 'articulo'})

        ## df_inventory
        self.df_inventory.columns = ['cia', 'date', 'codigo', 'descripcion', 'um', 'stock']
        self.df_inventory.loc[:, 'codigo'] = self.df_inventory['codigo'].str.lower()
        self.df_inventory = self.df_inventory.groupby(['date','codigo','descripcion','um']).agg(
            {
                'stock':'sum'
            }
        ).reset_index()



    def processing_price_usd(self) -> None:        
        # Get currency data 
        start = self.df_tc['ultima_fecha'].min().date()
        end = self.df_tc['ultima_fecha'].max().date()
        currency_pairs = ['PENUSD=X', 'EURUSD=X', 'JPYUSD=X', 'GBPUSD=X']
        data = yf.download(currency_pairs, start=start, end=end)
        self.df_closing_prices = data['Close']
        self.df_closing_prices.columns = [col.split('.')[0] for col in self.df_closing_prices.columns]

        # process currency data
        self.df_long_format = self.df_closing_prices.reset_index().melt(id_vars='Date', var_name='Currency Pair', value_name='Closing Price')
        self.df_long_format['Currency Pair'] = self.df_long_format['Currency Pair'].str.replace('=X', '', regex=False)
        self.df_long_format = self.df_long_format.dropna(subset=['Closing Price'])
        
        full_date_range = pd.date_range(start=self.df_long_format['Date'].min(), end=self.df_long_format['Date'].max(), freq='D')
        currency_pairs = self.df_long_format['Currency Pair'].unique()
        complete_index = pd.MultiIndex.from_product([full_date_range, currency_pairs], names=['Date', 'Currency Pair'])
        df_temp = pd.DataFrame(index=complete_index).reset_index()
        
        self.df_long_format = df_temp.merge(self.df_long_format, on=['Date', 'Currency Pair'], how='left')
        self.df_long_format['Closing Price'] = self.df_long_format.groupby('Currency Pair')['Closing Price'].ffill()
        self.df_long_format = self.df_long_format.rename(columns={'Closing Price': 'tc'})

        # merge exchange rates
        df_tc_merged = pd.merge(self.df_tc, self.df_long_format, left_on='ultima_fecha', right_on='Date', how='left')
        df_tc_merged['monto'] = pd.to_numeric(df_tc_merged['monto'], errors='coerce')
        df_tc_merged['tc'] = pd.to_numeric(df_tc_merged['tc'], errors='coerce')

        def convert_to_usd(row):
            if pd.isna(row['Currency Pair']) or row['moneda'] == 'USD':
                return row['monto']
            currency_pair_map = {'SOL': 'PENUSD', 'EUR': 'EURUSD', 'JPY': 'JPYUSD', 'GBP': 'GBPUSD'}
            if row['moneda'] in currency_pair_map and row['Currency Pair'] == currency_pair_map[row['moneda']]:
                return row['monto'] / row['tc'] if row['moneda'] == 'SOL' else row['monto'] * row['tc']
            return 0
        
        df_tc_merged['monto_usd'] = df_tc_merged.apply(convert_to_usd, axis=1)
        df_tc_merged = df_tc_merged[df_tc_merged['monto_usd'] != 0]
        self.df_tc_final = df_tc_merged[['codigo', 'ultima_fecha', 'monto_usd', 'ultima_compra']]
        self.df_tc_final = self.df_tc_final[self.df_tc_final['monto_usd'].notna()]

    def merge_dataframes(self) -> None:
        # cleaning df_inventory with df_products
        max_date = self.df_inventory['date'].max()
        self.df_inventory = self.df_inventory[
            (self.df_inventory['date'] != 'Periodo') & 
            (self.df_inventory['date'].notna())&
            (self.df_inventory['date']==max_date)
        ]
        df_inventory_final=pd.concat(
            [
                pd.DataFrame(self.df_inventory['codigo'].unique(), columns=['codigo']),
                pd.DataFrame(self.df_products['articulo'].unique(), columns=['codigo'])
            ], 
            ignore_index=True
        ).drop_duplicates()
        self.df_inventory = df_inventory_final.merge(
            self.df_inventory[['date','codigo','stock']].drop_duplicates(),
            how='left',
            on='codigo'
        )
        self.df_inventory['stock']=self.df_inventory['stock'].fillna(0)
        self.df_inventory['date'] = self.df_inventory['date'].fillna(max_date)
        # merging df_predictions, df_products and df_invetory
        self.df_merged = self.df_predictions.copy()
        self.df_merged = self.df_merged.merge(
            self.df_products[['articulo','fuente_suministro']].drop_duplicates(), 
            how='left', 
            on = 'articulo'
        )
        self.df_merged = self.df_merged.merge(
            self.df_inventory[['codigo','stock']], 
            how='left', 
            left_on=['articulo'],
            right_on=['codigo']
        )
        
        # merging df_tc_final, df_backorder, rfm and df_margin_result
        self.df_merged = self.df_merged.merge(
            self.df_margin_result[['articulo', 'mean_margen']], 
            how='left', 
            on='articulo'
        )
        self.df_merged = self.df_merged.merge(
            self.df_tc_final, 
            how='left', 
            left_on='articulo', 
            right_on='codigo'
        )
        if not self.df_backorder.empty:
            self.df_merged = self.df_merged.merge(
                self.df_backorder, 
                how='left', 
                on='articulo'
            )
        else:
            self.df_merged['backorder'] = np.nan
        self.df_merged = self.df_merged.merge(
            self.df_rfm, 
            left_on='articulo',
            right_on='sku',
            how='left'
        )
        self.df_merged['rfm'] = self.df_merged['rfm'].fillna(0).astype(int)
        self.df_merged = self.df_merged.drop_duplicates()

    def adding_final_variables(self) -> None:
        # adding demanda_mensual and meses_proteccion    
        self.df_merged['demanda_mensual'] = self.df_merged['caa'] / self.df_merged['lt']
        self.df_merged['meses_proteccion'] = self.df_merged['corr_sd'] / self.df_merged['demanda_mensual']

        # adding compra_sugerida
        self.df_merged['sobrante'] = np.maximum(self.df_merged['stock'] + self.df_merged['backorder'] - self.df_merged['caa_lt'], 0)
        self.df_merged['compra_sugerida'] = np.maximum(self.df_merged['caa'] - self.df_merged['sobrante'], 0)
        self.df_merged['compra_sugerida'] = np.ceil(self.df_merged['compra_sugerida']).astype('Int64')
        self.df_merged['compra_sugerida'] = self.df_merged['compra_sugerida'].astype("Float64")
        self.df_merged['meses_proteccion'] = self.df_merged['meses_proteccion'].astype("Float64")
        # mask = self.df_merged['demanda_mensual'] != 0
        # self.df_merged.loc[mask, 'meses_proteccion'] = (
        #     self.df_merged.loc[mask, 'meses_proteccion'] * 
        #     (self.df_merged.loc[mask, 'compra_sugerida'].fillna(0) / self.df_merged.loc[mask, 'demanda_mensual'])
        # ).astype("Float64")

        # adding costo_compra and compras_recomendadas
        self.df_merged = self.df_merged.rename(columns={'caa': 'compras_recomendadas'})
        # self.df_merged = self.df_merged.rename(columns={'compra_sugerida': 'compras_recomendadas'})
        self.df_merged.loc[self.df_merged['demanda_mensual'] < 0, 'demanda_mensual'] = 0
        self.df_merged.loc[self.df_merged['compras_recomendadas'] < 0, 'compras_recomendadas'] = 0
        self.df_merged['compras_recomendadas'] = self.df_merged['compras_recomendadas'].apply(
            lambda x: math.ceil(x / 50) * 50 if pd.notna(x) else pd.NA
        )
        self.df_merged['costo_compra'] = self.df_merged['monto_usd'] * self.df_merged['compras_recomendadas']

        # Calcular riesgo
        self.df_merged['holgura'] = (self.df_merged['stock'] / self.df_merged['demanda_mensual']).fillna(0)
        self.df_merged['consumiendo_proteccion'] = (self.df_merged['holgura'] < self.df_merged['meses_proteccion']).astype('Int64')
        self.df_merged['quebro'] = (self.df_merged['holgura'] <= 0).astype('Int64')
        self.df_merged['va_a_quebrar'] = ((self.df_merged['stock'] + self.df_merged['backorder']) < self.df_merged['demanda_mensual']).astype('Int64')
        self.df_merged['verde'] = (
            (self.df_merged['quebro'] == 0) & 
            (self.df_merged['consumiendo_proteccion'] == 0) & 
            (self.df_merged['va_a_quebrar'] == 0)
        ).astype('Int64')
        self.df_merged['amarillo'] = (
            (self.df_merged['consumiendo_proteccion'] == 1) & 
            (self.df_merged['quebro'] == 0)
        ).astype('Int64')
        self.df_merged['rojo'] = (
            (self.df_merged['quebro'] == 1) |
            (self.df_merged['va_a_quebrar'] == 1) 
        ).astype('Int64')
        self.df_merged['riesgo'] = self.df_merged.apply(
            lambda row: 'rojo' if pd.notna(row.get('rojo')) and row['rojo'] == 1 else 
                        'amarillo' if pd.notna(row.get('amarillo')) and row['amarillo'] == 1 else 
                        ('verde' if pd.notna(row.get('rojo')) or pd.notna(row.get('amarillo')) else np.nan),
            axis=1
        )

        # modificar period2 (caa_lt)
        self.df_merged['caa_lt'] += self.df_merged['corr_sd'] * self.df_merged['rfm'].map({3: 0.4, 2: 0.3, 1: 0.2, 0: 0.1}).fillna(0)

        # filtrar solo las importantes para la tabla por fuente de suministro
        self.df_merged['demanda_mensual_usd'] = self.df_merged['demanda_mensual'] * self.df_merged['monto_usd']
        df_temp = self.df_merged.copy()
        df_temp = df_temp[(df_temp['rfm'] == 3) & (df_temp['riesgo'] == 'rojo')]
        df_temp = df_temp.groupby('fuente_suministro').agg(
            recomendacion=('costo_compra', 'sum'),
            demanda_mensual_usd=('demanda_mensual_usd', 'sum')
        ).reset_index()
        df_temp_2 = self.df_merged.groupby('fuente_suministro').agg(
            lead_time=('lt', 'first'),
            riesgo=('riesgo', lambda x: x.mode()[0] if not x.mode().empty else None)
        ).reset_index()
        self.df_dashboard_by_fuente = df_temp_2.merge(df_temp, how='left', on='fuente_suministro')
        self.df_dashboard_by_fuente['recomendacion'] = pd.to_numeric(self.df_dashboard_by_fuente['recomendacion'], errors='coerce').fillna(0).astype(int)
        self.df_dashboard_by_fuente['demanda_mensual_usd'] = pd.to_numeric(self.df_dashboard_by_fuente['demanda_mensual_usd'], errors='coerce').fillna(0).astype(int)

    def formatting(self) -> None:
        # dar formato
        self.df_dashboard_by_fuente['riesgo_color'] = self.df_dashboard_by_fuente['riesgo'].map({
            'verde': '🟢',
            'amarillo': '🟡',
            #'naranja': '🟠',
            'rojo': '🔴'
        })
        self.df_download = self.df_merged[[
            'date','articulo','fuente_suministro','stock','compras_recomendadas','demanda_mensual',
            'meses_proteccion','riesgo','lt','mean_margen','ultima_fecha','monto_usd',
            'ultima_compra','costo_compra','rfm','backorder','holgura','quebro','va_a_quebrar','consumiendo_proteccion'
        ]]
        self.df_dashboard = self.df_merged[[
            'date','articulo','fuente_suministro','stock','backorder','rfm','riesgo',
            'demanda_mensual','monto_usd','ultima_compra','compras_recomendadas','costo_compra'
        ]]
        self.df_dashboard_by_fuente = self.df_dashboard_by_fuente[[
            'fuente_suministro',
            'lead_time',
            'recomendacion',
            'demanda_mensual_usd'
        ]]
        # columns mapping
        display_columns = {
            'date': 'Fecha',
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
            'lt': 'Lead Time',
            'mean_margen': 'Margen',
            'meses_proteccion': 'Meses proteccion',
            'ultima_fecha': 'Ultima fecha',
            'costo_compra': 'Recomendacion USD'
            }
        display_columns_fuente = {
            'date': 'Fecha',
            'fuente_suministro': 'Fuente de Suministro',
            'lead_time': 'Lead Time',
            'recomendacion': 'Recomendacion USD',
            'demanda_mensual_usd': 'Demanda Mensual USD'
        }
        self.df_dashboard = self.df_dashboard.rename(columns=display_columns)
        self.df_download = self.df_download.rename(columns=display_columns)
        self.df_dashboard_by_fuente = self.df_dashboard_by_fuente.rename(columns=display_columns_fuente)
       
    def process_all(self) -> None:
        self.load_and_process_data()
        self.processing_price_usd()
        self.merge_dataframes()
        self.adding_final_variables()
        self.formatting()

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
    print(len(processor.df_download['Artículo'].unique()))
    processor.df_dashboard.to_csv(cleaned_path / 'dashboard.csv', index=False)
    processor.df_dashboard_by_fuente.to_csv(cleaned_path / 'dashboard_by_fuente.csv', index=False)
    processor.df_download.to_csv(cleaned_path / 'download.csv', index=False)

