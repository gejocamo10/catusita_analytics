import pandas as pd
import numpy as np
from datetime import datetime

class RFMProcessor:
    def __init__(self, df_catusita):
        self.df_catusita = df_catusita
        self.df_catusita['fecha'] = pd.to_datetime(self.df_catusita['fecha'])
        
    def get_top_codes(self, df, codes, var, var_threshold):
        """Get top codes based on variable threshold"""
        return list(df[df[var] > df[var].quantile(var_threshold)][codes].unique())

    def get_normalized(self, series):
        """Normalize series to 0-1 range"""
        return (series - min(series))/(max(series) - min(series))

    def get_transaction_summary(self):
        """Create transaction summary by article"""
        return self.df_catusita.groupby("articulo").agg({
            "codigo": "nunique",
            "cantidad": "sum",
            "venta_pen": "sum",
            "transacciones": "sum",
            "fecha": "nunique"
        }).reset_index()

    def get_monthly_sales(self):
        """Create monthly sales aggregation"""
        df_ventas = self.df_catusita.loc[:, ['fecha', 'articulo', 'venta_pen']].copy()
        df_ventas['year'] = df_ventas['fecha'].dt.year
        df_ventas['month'] = df_ventas['fecha'].dt.month
        df_ventas['YEAR-MONTH'] = df_ventas['year'].map(str) + '-' + df_ventas['month'].map(str)

        df_agg = pd.crosstab(
            df_ventas['YEAR-MONTH'], 
            df_ventas['articulo'],
            df_ventas['venta_pen'],
            aggfunc='sum'
        ).fillna(0).reset_index()

        df_agg[['year', 'month']] = df_agg['YEAR-MONTH'].str.split('-', n=1, expand=True)
        df_agg['year'] = df_agg['year'].astype(int)
        df_agg['month'] = df_agg['month'].astype(int)
        
        return df_agg.sort_values(['year', 'month']).reset_index(drop=True)

    def get_active_products(self):
        """Get products with 6+ months of sales and coefficient of variation > 0.05"""
        df_agg = self.get_monthly_sales()
        df_temp = df_agg.iloc[:, 1:df_agg.shape[1]-2]

        products_6m = df_temp.columns[df_temp.apply(lambda x: len(x[x > 0]) >= 6, axis=0)].tolist()

        products_cvar = df_temp.columns[(df_temp.std()/df_temp.mean() >= 0.05)].tolist()

        return list(set(products_6m) & set(products_cvar))

    def get_rfm_list(self, var_threshold=0.8):
        """Get final RFM list"""
        active_products = self.get_active_products()
        
        df_trans = self.get_transaction_summary()
        df_trans = df_trans[df_trans["articulo"].isin(active_products)]

        skus_fechas = self.get_top_codes(df_trans, "articulo", "fecha", var_threshold)
        skus_clientes = self.get_top_codes(df_trans, "articulo", "codigo", var_threshold)
        skus_cantidad = self.get_top_codes(df_trans, "articulo", "cantidad", var_threshold)
        skus_monto = self.get_top_codes(df_trans, "articulo", "venta_pen", var_threshold)
        skus_transacciones = self.get_top_codes(df_trans, "articulo", "transacciones", var_threshold)

        lista_skus_top = list(set(skus_fechas + skus_clientes + skus_cantidad + skus_monto + skus_transacciones))

        max_year = self.df_catusita['fecha'].dt.year.max()
        max_month = self.df_catusita[self.df_catusita['fecha'].dt.year == max_year]['fecha'].dt.month.max()
        current_skus = list(self.df_catusita[
            (self.df_catusita['fecha'].dt.year == max_year) & 
            (self.df_catusita['fecha'].dt.month == max_month)
        ]['articulo'].unique())

        df_rfm = pd.DataFrame(columns=["sku", "resencia", "fechas", "clientes", "cantidad", "monto", "transacciones"])
        
        for i, sku in enumerate(lista_skus_top):
            df_rfm.loc[i] = [
                sku,

                int(sku in current_skus),
                int(sku in skus_fechas),
                int(sku in skus_clientes),
                int(sku in skus_cantidad),
                int(sku in skus_monto),
                int(sku in skus_transacciones)
            ]

        df_rfm['rfm'] = df_rfm['resencia'] + df_rfm['fechas'] + df_rfm['monto']
        # return df_rfm[df_rfm.rfm == 3]['sku'].unique()
        dfm_rfm_final = df_rfm[['sku','rfm']]
        rfm_final_list = df_rfm[df_rfm.rfm == 3]['sku'].unique()
        return rfm_final_list, dfm_rfm_final

def process_rfm(df_catusita):
    """Main function to process RFM analysis"""
    processor = RFMProcessor(df_catusita)
    # lista_skus_rfm = processor.get_rfm_list()
    lista_skus_rfm, df_rfm = processor.get_rfm_list()
    
    sum_total_sells = processor.df_catusita["venta_pen"].sum()
    sum_rfm_sells = processor.df_catusita[
        processor.df_catusita["articulo"].isin(lista_skus_rfm)
    ]["venta_pen"].sum()
    
    print(f"RFM SKUs represent {sum_rfm_sells/sum_total_sells:.2%} of total sales")
    # return lista_skus_rfm
    return lista_skus_rfm, df_rfm