from utils.process_data.sunarp.sunarp_processor import SunarpProcessor
from utils.process_data.sunat.sunat_processor import SunatProcessor
from utils.process_data.catusita.catusita_processor import CatusitaProcessor
from utils.dashboard.dashboard import DataProcessor
from utils.predictions.predictor import Predictor
from utils.correlations.correlations_processor import process_correlations
from utils.process_data.config import DATA_PATHS
from utils.rfm.rfm_processor import process_rfm
import pandas as pd
from datetime import datetime

def process_data(date_filter=None):
    DATA_PATHS['process'].mkdir(parents=True, exist_ok=True)
    DATA_PATHS['cleaned'].mkdir(parents=True, exist_ok=True)
    
    print("Starting data processing...")
    
    if date_filter and isinstance(date_filter, str):
        date_filter = pd.to_datetime(date_filter)
    
    print("Processing SUNARP data...")
    sunarp_processor = SunarpProcessor()
    df_sunarp = sunarp_processor.process_all()
    if date_filter:
        df_sunarp['fecha'] = pd.to_datetime(df_sunarp['fecha'])
        df_sunarp = df_sunarp[df_sunarp['fecha'] < date_filter]
    df_sunarp.to_csv(DATA_PATHS['process'] / 'sunarp_consolidated.csv', index=False)
    
    print("Processing SUNAT data...")
    sunat_processor = SunatProcessor()
    df_sunat = sunat_processor.process_all()
    df_sunat = df_sunat[df_sunat['value'].notna()]
    if date_filter:
        df_sunat['fecha'] = pd.to_datetime(df_sunat['fecha'])
        df_sunat = df_sunat[df_sunat['fecha'] < date_filter]
    df_sunat.to_csv(DATA_PATHS['process'] / 'sunat_consolidated.csv', index=False)
    
    print("Processing Catusita data...")
    catusita_processor = CatusitaProcessor()
    df_catusita = catusita_processor.process_data()
    if date_filter:
        df_catusita['fecha'] = pd.to_datetime(df_catusita['fecha'])
        df_catusita = df_catusita[df_catusita['fecha'] < date_filter]
    catusita_processor.save_data(df_catusita)
    
    print("All processing completed successfully!")
    return df_sunarp, df_sunat, df_catusita

def load_processed_data(date_filter=None):
    print("Loading processed data...")
    
    if date_filter and isinstance(date_filter, str):
        date_filter = pd.to_datetime(date_filter)
        print(f"Applying date filter: up to {date_filter.strftime('%Y-%m-%d')}")
    
    try:
        df_sunarp = pd.read_csv(DATA_PATHS['process'] / 'sunarp_consolidated.csv')
        df_sunarp['fecha'] = pd.to_datetime(df_sunarp['fecha'])
        if date_filter:
            df_sunarp = df_sunarp[df_sunarp['fecha'] < date_filter]
        print(f"SUNARP data loaded successfully: {len(df_sunarp)} rows")
    except FileNotFoundError:
        print("Warning: SUNARP processed data not found")
        df_sunarp = None
        
    try:
        df_sunat = pd.read_csv(DATA_PATHS['process'] / 'sunat_consolidated.csv')
        df_sunat['fecha'] = pd.to_datetime(df_sunat['fecha'])
        if date_filter:
            df_sunat = df_sunat[df_sunat['fecha'] < date_filter]
        print(f"SUNAT data loaded successfully: {len(df_sunat)} rows")
    except FileNotFoundError:
        print("Warning: SUNAT processed data not found")
        df_sunat = None
        
    try:
        df_catusita = pd.read_csv(DATA_PATHS['process'] / 'catusita_consolidated.csv')
        df_catusita['transacciones'] = 1
        df_catusita['fecha'] = pd.to_datetime(df_catusita['fecha'])
        if date_filter:
            df_catusita = df_catusita[df_catusita['fecha'] < date_filter]
        print(f"Catusita data loaded successfully: {len(df_catusita)} rows")
    except FileNotFoundError:
        print("Warning: Catusita processed data not found")
        df_catusita = None
    
    return df_sunarp, df_sunat, df_catusita

def main(calculator=0, date_filter=None):
    if calculator == 1:
        df_sunarp, df_sunat, df_catusita = process_data(date_filter)
    else:
        df_sunarp, df_sunat, df_catusita = load_processed_data(date_filter)
    
    if df_catusita is not None:
        print("\nProcessing RFM analysis...")
        lista_skus_rfm = process_rfm(df_catusita)
        df_skus_rfm = pd.DataFrame({'sku': lista_skus_rfm})
        df_skus_rfm.to_csv(DATA_PATHS['process'] / 'df_skus_rfm.csv', index=False)
        print(f"RFM analysis completed. Found {len(lista_skus_rfm)} SKUs")

    if df_catusita is not None and df_sunarp is not None and df_sunat is not None:
        print("\nProcessing correlations for all SKUs...")
        autos_sig, partes_sig, all_sig = process_correlations(
            df_catusita, df_sunarp, df_sunat
        )
        print(f"Found significant correlations for {len(all_sig.sku.unique())} SKUs")
        
        print("\nProcessing predictions...")
        predictor = Predictor()
        predictions_df = predictor.process_predictions()
        if predictions_df is not None:
            predictions_df.to_csv(DATA_PATHS['cleaned'] / 'predictions.csv', index=False)
            print(f"Generated predictions for {len(predictions_df)} SKUs")
    
        print("\nProcessing dashboard data...")
        try:
            # Get the parent directory of 'data' folder
            base_path = str(DATA_PATHS['cleaned'].parent.parent)
            processor = DataProcessor(base_path)
            processor.process_all()
            if processor.dffinal2 is not None:
                processor.dffinal2.to_csv(DATA_PATHS['cleaned'] / 'dashboard.csv', index=False)
                print(f"Generated dashboard data for {len(processor.dffinal2)} SKUs")
            else:
                print("Warning: No dashboard data generated")
        except Exception as e:
            print(f"Error processing dashboard data: {str(e)}")
    
    return df_sunarp, df_sunat, df_catusita

if __name__ == "__main__":
    CALCULATOR = 1
    DATE_FILTER = '2024-12-01'
    df_sunarp, df_sunat, df_catusita = main(calculator=CALCULATOR, date_filter=DATE_FILTER)
    
    print("\nProcessed data information:")
    if df_sunarp is not None:
        print(f"SUNARP data shape: {df_sunarp.shape}")
    if df_sunat is not None:
        print(f"SUNAT data shape: {df_sunat.shape}")
    if df_catusita is not None:
        print(f"Catusita data shape: {df_catusita.shape}")