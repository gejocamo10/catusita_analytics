import pandas as pd
import numpy as np
from utils.process_data.config import DATA_PATHS
from datetime import datetime
from dateutil.relativedelta import relativedelta


class CorrelationsProcessor:
    def __init__(self, df_catusita, df_autos, df_partes):
        self.df_catusita = df_catusita
        self.df_autos = df_autos
        self.df_partes = df_partes
        self.skus_to_process = self._load_skus_to_process()

    def _load_skus_to_process(self):
        """Loads the SKUs to be processed from the specified file."""
        try:
            df_skus_rfm = pd.read_csv(DATA_PATHS['process'] / 'df_skus_rfm.csv')
            return df_skus_rfm['sku'].tolist()
        except FileNotFoundError:
            print(f"Warning: File {DATA_PATHS['process'] / 'df_skus_rfm.csv'} not found. Processing all SKUs.")
            return None  # Return None if the file is not found
        
    def get_max_min(self, lst):
        """Get maximum absolute value maintaining sign"""
        ans = [0] * len(lst)
        for i in range(len(lst)):
            if np.abs(lst[i]) == np.max(np.abs(lst)):
                ans[i] = lst[i]
        return ans

    def crosstab_df(self, df, column_year, column_month, values_cross, measure_column):
        """Create crosstab with year-month handling"""
        df["year-month"] = df[column_year].map(str) + "-" + df[column_month].map(str)
        df_crosstab = pd.crosstab(
            df["year-month"], 
            df[values_cross], 
            df[measure_column], 
            aggfunc=sum
        )
        df_crosstab = df_crosstab.fillna(0)
        df_crosstab = df_crosstab.reset_index()
        
        df_crosstab[["year", "mes"]] = df_crosstab["year-month"].str.split("-", n=1, expand=True)
        df_crosstab["year"] = df_crosstab["year"].astype(int)
        df_crosstab["month"] = df_crosstab["mes"].astype(int)  # Changed this line
        
        return df_crosstab.sort_values(['year','month']).reset_index(drop=True)

    def prepare_autos_data(self):
        """Prepare autos data for correlation analysis"""
        dic_month_to_number = {
            'Ene':1, 'Feb':2, 'Mar':3, 'Abr':4, 'May':5, 'Jun':6,
            'Jul':7, 'Ago':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dic':12
        }
        
        df_autos = self.df_autos.copy()
        df_autos['month'] = df_autos['MES'].map(dic_month_to_number)
        df_autos['YEAR-MONTH'] = df_autos['ANIO'].map(str) + '-' + df_autos['month'].map(str)
        
        df_autos_agg = pd.crosstab(
            df_autos['YEAR-MONTH'], 
            df_autos['TIPO'],
            df_autos['VENTAS'],
            aggfunc='sum'
        ).reset_index()
        
        df_autos_agg[['year', 'month']] = df_autos_agg['YEAR-MONTH'].str.split('-', n=1, expand=True)
        df_autos_agg['year'] = df_autos_agg['year'].astype(int)
        df_autos_agg['month'] = df_autos_agg['month'].astype(int)
        
        df_autos_tipos = df_autos_agg[['year', 'month', 'Hibridos y Electricos', 'Livianos', 
                                      'Menores', 'Pesados', 'Remolques y SemiR']]
        
        # Handle missing months
        min_year = df_autos_tipos['year'].min()
        min_month = df_autos_tipos.loc[df_autos_tipos['year'] == min_year, 'month'].min()

        # Determine the last month of the current year
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        # Calculate the last month using the date range and the current year
        max_year = current_year
        max_month = current_month
        
        date_range = pd.date_range(start=f'{min_year}-{min_month}', end=f'{max_year}-{max_month}', freq='MS')
        
        df_all_months = pd.DataFrame({'year': date_range.year, 'month': date_range.month})
        
        df_autos_tipos = pd.merge(df_all_months, df_autos_tipos, on=['year','month'], how='left')

        for col in df_autos_tipos.columns:
            if col not in ['year', 'month']:
                 df_autos_tipos[col] = df_autos_tipos[col].fillna(method='ffill')
        
        return df_autos_tipos.fillna(0)


    def prepare_partes_data(self):
        """Prepare partes data for correlation analysis"""
        df_partes = self.df_partes.copy()
        
        # Convert fecha to datetime if it's not already
        # print(df_partes)
        df_partes['fecha'] = pd.to_datetime(df_partes['fecha'])
        
        # Extract year and month from fecha
        df_partes['year'] = df_partes['fecha'].dt.year
        df_partes['mes'] = df_partes['fecha'].dt.month
        
        print("Initial partes data shape:", df_partes.shape)
        print("Columns in partes data:", df_partes.columns.tolist())
        
        # Create crosstab using the correct column names
        df_partes_tipos = self.crosstab_df(df_partes, "year", "mes", "Descripcion", "value")
        
        # Filter for years >= 2020
        df_partes_tipos = df_partes_tipos[df_partes_tipos["year"] >= 2020]
        
        if df_partes_tipos.empty:
            raise ValueError("No data available after filtering for years >= 2020")
        
        df_partes_tipos = df_partes_tipos.drop(columns=["year-month", "mes"])
        
        # Define expected columns based on your Descripcion values
        tipos_unicos = df_partes['Descripcion'].unique().tolist()
        columnas_partes = ['year', 'month'] + tipos_unicos
        
        # Ensure all expected columns exist
        for col in columnas_partes:
            if col not in df_partes_tipos.columns and col not in ['year', 'month']:
                df_partes_tipos[col] = 0
        
        df_partes_tipos = df_partes_tipos[columnas_partes]
        
        # Get min and max dates
        min_year = df_partes_tipos['year'].min()
        min_month = df_partes_tipos.loc[df_partes_tipos['year'] == min_year, 'month'].min()
        
        if pd.isna(min_year) or pd.isna(min_month):
            print("Current data state:")
            print(df_partes_tipos.head())
            print("Year range:", df_partes_tipos['year'].unique())
            raise ValueError("Invalid min_year or min_month values")
        
        # Current date boundaries
        current_date = datetime.now() - relativedelta(months=1)
        current_year = current_date.year
        current_month = current_date.month
        
        # Create date range
        all_dates = []
        for year in range(int(min_year), current_year + 1):
            for month in range(1, 13):
                if (year == current_year and month > current_month) or \
                (year == min_year and month < min_month):
                    continue
                all_dates.append({'year': year, 'month': month})
        
        df_all_months = pd.DataFrame(all_dates)
        
        # Merge with existing data
        df_partes_tipos = pd.merge(df_all_months, df_partes_tipos, on=['year','month'], how='left')
        
        # Forward fill missing values
        for col in df_partes_tipos.columns:
            if col not in ['year', 'month']:
                df_partes_tipos[col] = df_partes_tipos[col].fillna(method='ffill')
        
        return df_partes_tipos.fillna(0)
    
    def prepare_catusita_data(self):
        """Prepare Catusita data for correlation analysis"""
        df_catusita_ventas = self.df_catusita[['fecha', 'articulo', 'cantidad']].copy()
        
        df_catusita_ventas['year'] = df_catusita_ventas['fecha'].dt.year
        df_catusita_ventas['month'] = df_catusita_ventas['fecha'].dt.month
        df_catusita_ventas['YEAR-MONTH'] = df_catusita_ventas['year'].map(str) + '-' + df_catusita_ventas['month'].map(str)
        
        df_catusita_agg = pd.crosstab(
            df_catusita_ventas['YEAR-MONTH'],
            df_catusita_ventas['articulo'],
            df_catusita_ventas['cantidad'],
            aggfunc='sum'
        ).reset_index()
        
        df_catusita_agg[['year', 'month']] = df_catusita_agg['YEAR-MONTH'].str.split('-', n=1, expand=True)
        df_catusita_agg['year'] = df_catusita_agg['year'].astype(int)
        df_catusita_agg['month'] = df_catusita_agg['month'].astype(int)
        
        return df_catusita_agg.sort_values(['year', 'month']).reset_index(drop=True)

    def calculate_correlations(self, df_catusita_agg, df_tipos, correlation_type='autos'):
        """Calculate correlations with lags"""
        print(f"Calculating {correlation_type} correlations...")
        df_correlaciones = pd.DataFrame(columns=['lag', 'tipo', 'corr', 'sku'])
        
        all_skus = [col for col in df_catusita_agg.columns 
                    if col not in ['YEAR-MONTH', 'year', 'month', 'index']]

        if self.skus_to_process:
            skus_to_use = [sku for sku in all_skus if sku in self.skus_to_process]
            print(f"Processing {len(skus_to_use)} SKUs found in df_skus_rfm.csv")
        else:
            skus_to_use = all_skus
            print(f"Processing all {len(skus_to_use)} SKUs.")
            
        total_skus = len(skus_to_use)
        for idx, sku in enumerate(skus_to_use, 1):
            if idx % 10 == 0:  
                print(f"Processing SKU {idx}/{total_skus}")
                
            ventas_temp = df_catusita_agg[['year', 'month', sku]]
            temp_corr_lags = []
            
            for lag in range(8):
                df_tipos_lagged = df_tipos.copy()
                df_tipos_lagged.iloc[:, 2:] = df_tipos_lagged.iloc[:, 2:].shift(lag)
                df_tipos_lagged = df_tipos_lagged.dropna()
                
                temp_agg = pd.merge(ventas_temp, df_tipos_lagged, how='inner', on=['year', 'month'])
                temp_agg_series = temp_agg.drop(columns=['month', 'year'])
                temp_corr = temp_agg_series.corr().iloc[0]
                temp_corr_lags.append(temp_corr.to_list())
            
            temp_corr_lags = pd.DataFrame(temp_corr_lags)
            temp_corr_lags = temp_corr_lags.drop(columns=[0])
            temp_corr_lags.columns = df_tipos_lagged.columns.tolist()[2:]
            
            for col in temp_corr_lags.columns:
                temp_corr_lags[col] = self.get_max_min(temp_corr_lags[col])
            
            temp_corr_lags_unstacked = temp_corr_lags.stack().reset_index()
            temp_corr_lags_unstacked.columns = ['lag', 'tipo', 'corr']
            temp_corr_lags_unstacked['sku'] = sku
            
            df_correlaciones = pd.concat([df_correlaciones, temp_corr_lags_unstacked], ignore_index=True)
        
        return df_correlaciones

    def process_correlations(self):
        """Process all correlations"""
        print("Preparing data...")
        # Prepare data
        df_autos_tipos = self.prepare_autos_data()
        df_partes_tipos = self.prepare_partes_data()
        df_catusita_agg = self.prepare_catusita_data()
        
        # Create and save covariables
        print("Creating and saving covariables dataframe...")
        df_covariables = pd.merge(df_autos_tipos, df_partes_tipos, how='inner', on=['year', 'month'])
        df_covariables.to_csv(DATA_PATHS['process'] / 'df_covariables.csv', index=False)
        
        # Calculate correlations
        df_correlaciones_autos = self.calculate_correlations(
            df_catusita_agg, df_autos_tipos, 'autos'
        )
        df_correlaciones_partes = self.calculate_correlations(
            df_catusita_agg, df_partes_tipos, 'partes'
        )
        
        # Filter significant correlations
        print("Filtering significant correlations...")
        df_correlaciones_autos_sig = df_correlaciones_autos[abs(df_correlaciones_autos['corr']) > 0.3]
        df_correlaciones_partes_sig = df_correlaciones_partes[abs(df_correlaciones_partes['corr']) > 0.3]
        df_correlaciones_sig = pd.concat([df_correlaciones_autos_sig, df_correlaciones_partes_sig], 
                                       ignore_index=True)
        
        return df_correlaciones_autos_sig, df_correlaciones_partes_sig, df_correlaciones_sig

    def save_results(self, df_autos_sig, df_partes_sig, df_all_sig):
        """Save correlation results"""
        print("Saving results...")
        df_autos_sig.to_csv(DATA_PATHS['process'] / 'df_correlaciones_autos_sig.csv', index=False)
        df_partes_sig.to_csv(DATA_PATHS['process'] / 'df_correlaciones_partes_sig.csv', index=False)
        df_all_sig.to_csv(DATA_PATHS['process'] / 'df_correlaciones_sig.csv', index=False)


def process_correlations(df_catusita, df_autos, df_partes):
    """Main function to process correlations"""
    processor = CorrelationsProcessor(df_catusita, df_autos, df_partes)
    autos_sig, partes_sig, all_sig = processor.process_correlations()
    processor.save_results(autos_sig, partes_sig, all_sig)
    return autos_sig, partes_sig, all_sig