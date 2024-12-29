import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from utils.process_data.sunarp.sunarp_processor import SunarpProcessor
from utils.process_data.sunat.sunat_processor import SunatProcessor
from utils.process_data.catusita.catusita_processor import CatusitaProcessor
from utils.predictions.predictor import Predictor
from utils.correlations.correlations_processor import process_correlations
from utils.process_data.config import DATA_PATHS
from utils.rfm.rfm_processor import process_rfm
import pandas as pd

st.set_page_config(layout="wide", page_title="Procesamiento de Predicciones")

def process_data():
    """Process all raw data and save results"""
    from utils.process_data.config import DATA_PATHS
    
    DATA_PATHS['process'].mkdir(parents=True, exist_ok=True)
    DATA_PATHS['cleaned'].mkdir(parents=True, exist_ok=True)
    
    print("Starting data processing...")
    
    print("Processing SUNARP data...")
    sunarp_processor = SunarpProcessor()
    df_sunarp = sunarp_processor.process_all()
    df_sunarp.to_csv(DATA_PATHS['process'] / 'sunarp_consolidated.csv', index=False)
    
    print("Processing SUNAT data...")
    sunat_processor = SunatProcessor()
    df_sunat = sunat_processor.process_all()
    df_sunat=df_sunat[df_sunat['value'].notna()]
    df_sunat.to_csv(DATA_PATHS['process'] / 'sunat_consolidated.csv', index=False)
    
    print("Processing Catusita data...")
    catusita_processor = CatusitaProcessor()
    df_catusita = catusita_processor.process_data() 
    catusita_processor.save_data(df_catusita)
    
    print("All processing completed successfully!")
    return df_sunarp, df_sunat, df_catusita

def load_processed_data():
    """Load previously processed data"""
    from utils.process_data.config import DATA_PATHS
    
    print("Loading processed data...")
    
    try:
        df_sunarp = pd.read_csv(DATA_PATHS['process'] / 'sunarp_consolidated.csv')
        print("SUNARP data loaded successfully")
    except FileNotFoundError:
        print("Warning: SUNARP processed data not found")
        df_sunarp = None
        
    try:
        df_sunat = pd.read_csv(DATA_PATHS['process'] / 'sunat_consolidated.csv')
        print("SUNAT data loaded successfully")
    except FileNotFoundError:
        print("Warning: SUNAT processed data not found")
        df_sunat = None
        
    try:
        df_catusita = pd.read_csv(DATA_PATHS['process'] / 'catusita_consolidated.csv')
        df_catusita['transacciones'] = 1
        df_catusita['fecha'] = pd.to_datetime(df_catusita['fecha'])
        print("Catusita data loaded successfully")
    except FileNotFoundError:
        print("Warning: Catusita processed data not found")
        df_catusita = None
    
    return df_sunarp, df_sunat, df_catusita

def main(calculator=0):
    from utils.process_data.sunarp.sunarp_processor import SunarpProcessor
    from utils.process_data.sunat.sunat_processor import SunatProcessor
    from utils.process_data.catusita.catusita_processor import CatusitaProcessor
    from utils.predictions.predictor import Predictor
    from utils.correlations.correlations_processor import process_correlations
    from utils.process_data.config import DATA_PATHS
    from utils.rfm.rfm_processor import process_rfm
    
    if calculator == 1:
        df_sunarp, df_sunat, df_catusita = process_data()
    else:
        df_sunarp, df_sunat, df_catusita = load_processed_data()
    
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
    
    return df_sunarp, df_sunat, df_catusita

def run_predictions():
    try:
        with st.spinner('Procesando datos...'):
            df_sunarp, df_sunat, df_catusita = main(calculator=1)
            st.success('¡Procesamiento completado!')
            
            if df_sunarp is not None:
                st.write(f"Datos SUNARP procesados: {df_sunarp.shape[0]} registros")
            if df_sunat is not None:
                st.write(f"Datos SUNAT procesados: {df_sunat.shape[0]} registros")
            if df_catusita is not None:
                st.write(f"Datos Catusita procesados: {df_catusita.shape[0]} registros")
                
    except Exception as e:
        st.error(f"Error durante el procesamiento: {str(e)}")

st.title("Procesamiento de Predicciones")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    """)
    
    if st.button("Iniciar Procesamiento", key="start_processing"):
        run_predictions()