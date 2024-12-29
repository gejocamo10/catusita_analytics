import streamlit as st
import pandas as pd
from pathlib import Path
from utils.process_data.config import DATA_PATHS

st.set_page_config(layout="wide", page_title="Resultados")

def load_predictions():
    try:
        predictions_path = DATA_PATHS['process'] / 'predictions.csv'
        if predictions_path.exists():
            df = pd.read_csv(predictions_path)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error al cargar las predicciones: {str(e)}")
        return None

st.title("Resultados de Predicciones")

df_predictions = load_predictions()

if df_predictions is not None:
    st.write(f"Total de predicciones: {len(df_predictions)} SKUs")
    
    search_term = st.text_input("Buscar SKU:")
    
    if search_term:
        filtered_df = df_predictions[df_predictions['sku'].str.contains(search_term, case=False, na=False)]
    else:
        filtered_df = df_predictions
    
    st.dataframe(filtered_df)
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name="predicciones.csv",
        mime="text/csv"
    )
else:
    st.warning("No se encontraron predicciones. Por favor, ejecute primero el procesamiento en la página de Predicciones.")