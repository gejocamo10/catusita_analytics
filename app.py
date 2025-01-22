import streamlit as st
import pandas as pd
from pathlib import Path
import os
import re
from datetime import datetime

from utils.process_data.sunarp.sunarp_processor import SunarpProcessor
from utils.process_data.sunat.sunat_processor import SunatProcessor
from utils.process_data.catusita.catusita_processor import CatusitaProcessor
from utils.dashboard.dashboard import DataProcessor
from utils.predictions.predictor import Predictor
from utils.correlations.correlations_processor import process_correlations
from utils.process_data.config import DATA_PATHS
from utils.rfm.rfm_processor import process_rfm


# Configuration
st.set_page_config(layout="wide", page_title="Procesador de Datos")

# Constants
CATUSITA_FILES = {
    'inventory': 'inventory.xlsx',
    'backorder': 'backorder12_12.xlsx',
    'kits': 'KITS AISIN.xlsx',
    'saldos': 'saldo de todo 04.11.2024.2.xlsx',
    'blacklist': 'IMCASA SD 14.08.xlsx',
    'ventas': 'Data de venta 01.01.21 a 06.12.24.xlsx'
}

SUNARP_CATEGORIES = {
    'Menores': 'Vehículos_Menores',
    'Livianos': 'Vehículos_Livianos',
    'Pesados': 'Vehículos_Pesados',
    'Remolques': 'Remolques_SemiR',
    'Híbridos': 'Vehículos_Híbridos-Eléctricos'
}

# Setup directories
BASE_DIR = Path("data")
PATHS = {
    'catusita': BASE_DIR / 'raw'/ 'catusita',
    'sunarp': BASE_DIR / 'raw' / 'sunarp',
    'sunat': BASE_DIR / 'raw' / 'sunat'
}

# Your existing helper functions remain the same
def setup_directories():
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

def extract_date_and_category_sunarp(filename):
    try:
        date_match = re.search(r'(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre)_(\d{4})', filename)
        category = None
        for cat_key, cat_pattern in SUNARP_CATEGORIES.items():
            if cat_pattern in filename:
                category = cat_key
                break
        if date_match and category:
            return {'category': category, 'month': date_match.group(1), 'year': date_match.group(2)}
    except Exception as e:
        st.error(f"Error parsing SUNARP filename: {str(e)}")
    return None

def extract_date_sunat(filename):
    try:
        date_match = re.search(r'(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre)_(\d{4})', filename)
        if date_match:
            return {'month': date_match.group(1), 'year': date_match.group(2)}
    except Exception as e:
        st.error(f"Error parsing SUNAT filename: {str(e)}")
    return None

def process_file_upload(uploaded_file, file_type, category=None):
    try:
        if file_type == 'catusita':
            target_path = PATHS['catusita'] / CATUSITA_FILES[category]
            if target_path.exists():
                st.warning(f"Se reemplazará el archivo existente: {CATUSITA_FILES[category]}")
            with open(target_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            return True, f"Archivo guardado como {CATUSITA_FILES[category]}"

        elif file_type == 'sunarp':
            file_info = extract_date_and_category_sunarp(uploaded_file.name)
            if not file_info:
                return False, "Formato de nombre de archivo inválido"
            existing_files = list(PATHS['sunarp'].glob(f"*{file_info['category']}*{file_info['year']}*"))
            if existing_files:
                st.warning(f"Se reemplazará el archivo: {existing_files[0].name}")
                existing_files[0].unlink()
            target_path = PATHS['sunarp'] / uploaded_file.name
            with open(target_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            return True, "Archivo SUNARP procesado correctamente"

        elif file_type == 'sunat':
            file_info = extract_date_sunat(uploaded_file.name)
            if not file_info:
                return False, "Formato de nombre de archivo inválido"
            existing_files = list(PATHS['sunat'].glob(f"*{file_info['year']}*"))
            if existing_files:
                st.warning(f"Se reemplazará el archivo: {existing_files[0].name}")
                existing_files[0].unlink()
            target_path = PATHS['sunat'] / uploaded_file.name
            with open(target_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            return True, "Archivo SUNAT procesado correctamente"

    except Exception as e:
        return False, f"Error al procesar archivo: {str(e)}"

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

def main_processor(calculator=0, date_filter=None):
    if calculator == 1:
        df_sunarp, df_sunat, df_catusita = process_data(date_filter)
    else:
        df_sunarp, df_sunat, df_catusita = load_processed_data(date_filter)
    
    if df_catusita is not None:
        print("\nProcessing RFM analysis...")
        lista_skus_rfm, df_rfm = process_rfm(df_catusita)
        df_skus_rfm = pd.DataFrame({'sku': lista_skus_rfm})
        df_skus_rfm.to_csv(DATA_PATHS['process'] / 'df_skus_rfm.csv', index=False)
        df_rfm.to_csv(DATA_PATHS['process'] / 'df_rfm.csv', index=False)
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

def main():
    setup_directories()
    st.title("Carga de Archivos")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Carga de Archivos", "Predicciones", "Recomendación De Compras"])

    # Tab 1: File Uploads
    with tab1:
        # Create three columns for file uploads
        col1, col2, col3 = st.columns(3)

        # Column 1: Catusita Files
        with col1:
            st.header("Archivos Catusita")
            for category, filename in CATUSITA_FILES.items():
                st.subheader(f"{category.title()}")
                uploaded_file = st.file_uploader(
                    f"Seleccionar archivo para {filename}",
                    key=f"catusita_{category}",
                    type=['xlsx']
                )
                if uploaded_file:
                    success, message = process_file_upload(uploaded_file, 'catusita', category)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        # Column 2: SUNARP Files
        with col2:
            st.header("Archivos SUNARP")
            for category in SUNARP_CATEGORIES.keys():
                st.subheader(f"Vehículos {category}")
                uploaded_file = st.file_uploader(
                    f"Seleccionar archivo para {category}",
                    key=f"sunarp_{category}",
                    type=['xlsx']
                )
                if uploaded_file:
                    success, message = process_file_upload(uploaded_file, 'sunarp')
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        # Column 3: SUNAT Files
        with col3:
            st.header("Archivos SUNAT")
            sunat_files = st.file_uploader(
                "Seleccionar archivos SUNAT",
                accept_multiple_files=True,
                key="sunat",
                type=['xlsx']
            )
            if sunat_files:
                for file in sunat_files:
                    success, message = process_file_upload(file, 'sunat')
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # Tab 2: Predictions
    with tab2:
        st.header("Procesamiento de Predicciones")
        
        # Month and Year selection
        month_names = [
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
        ]
        selected_month = st.selectbox("Seleccionar mes:", month_names, index=datetime.now().month - 1)
        current_year = datetime.now().year
        year_options = list(range(current_year - 5, current_year + 6))
        selected_year = st.selectbox("Seleccionar año:", year_options, index=5)


        # Convert to the date
        month_number = month_names.index(selected_month) + 1
        date_filter_str = f"{selected_year}-{month_number:02}-01"
        date_filter_date = datetime.strptime(date_filter_str, "%Y-%m-%d").date()

        if st.button("Ejecutar Procesamiento de Datos"):
            with st.spinner("Procesando datos y generando predicciones..."):
                df_sunarp, df_sunat, df_catusita = main_processor(calculator=1, date_filter=str(date_filter_date))
            st.success("Procesamiento de datos y predicciones completado!")
            
            print("\nProcessed data information:")
            if df_sunarp is not None:
                print(f"SUNARP data shape: {df_sunarp.shape}")
            if df_sunat is not None:
                print(f"SUNAT data shape: {df_sunat.shape}")
            if df_catusita is not None:
                print(f"Catusita data shape: {df_catusita.shape}")

    
    # Tab 3: Recomendación De Compras
    with tab3:
        st.markdown("### Dashboard")
        
        try:
            # Load dashboard data
            dashboard_path = DATA_PATHS['cleaned'] / 'dashboard.csv'
            dashboard_df = pd.read_csv(dashboard_path)
            
            # Create filters
            with st.expander("Filtros"):
                fuente_suministro_list = ['todos'] + sorted(dashboard_df['fuente_suministro'].unique().tolist())
                
                filtro_fuente = st.selectbox(
                    "Selecciona Fuente de Suministro:",
                    fuente_suministro_list,
                    index=0
                )
                
                if filtro_fuente == 'todos':
                    articulo_list = ['todos'] + sorted(dashboard_df['articulo'].unique().tolist())
                else:
                    articulo_list = ['todos'] + sorted(
                        dashboard_df[dashboard_df['fuente_suministro'] == filtro_fuente]['articulo'].unique().tolist()
                    )
                
                filtro_articulo = st.selectbox(
                    "Selecciona Artículo:",
                    articulo_list,
                    index=0
                )
            
            # Apply filters
            filtered_df = dashboard_df.copy()
            if filtro_fuente != 'todos':
                filtered_df = filtered_df[filtered_df['fuente_suministro'] == filtro_fuente]
            if filtro_articulo != 'todos':
                filtered_df = filtered_df[filtered_df['articulo'] == filtro_articulo]
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stock Total", f"{filtered_df['stock'].sum():,.0f}")
            with col2:
                st.metric("Demanda Mensual Total", f"{filtered_df['demanda_mensual'].sum():,.0f}")
            with col3:
                st.metric("Total Backorder", f"{filtered_df['backorder'].sum():,.0f}")
            
            # Display columns mapping
            display_columns = {
                'articulo': 'Artículo',
                'stock': 'Stock',
                'compras_recomendadas': 'Compras Recomendadas',
                'demanda_mensual': 'Demanda Mensual',
                'meses_proteccion': 'Meses Protección',
                'index_riesgo': 'Índice Riesgo',
                'riesgo': 'Riesgo',
                'ranking_riesgo': 'Ranking de Riesgo',
                'lt_x': 'Lead Time',
                'mean_margen': 'Margen Promedio',
                'ultima_fecha': 'Última Fecha',
                'monto_usd': 'Monto USD',
                'ultima_compra': 'Última Compra',
                'costo_compra': 'Costo Compra',
                'fuente_suministro': 'Fuente Suministro',
                'rfm':'rfm',
                # 'urgency': 'Urgencia',
                # 'hierarchy': 'Jerarquía',
                'backorder': 'Backorder'
            }
            
            # Rename columns first
            filtered_df = filtered_df.rename(columns=display_columns)
            
            # Modified highlight function using new column names
            def highlight_risk(row):
                color_map = {
                    'Verde': '#b7f898',
                    'Amarillo': '#f6f69b',
                    'Naranja': '#f8dc98',
                    'Rojo': '#f69e9b'
                }
                
                risk_value = row['Riesgo']
                color = color_map.get(risk_value, '')
                
                return ['background-color: ' + color if color else ''] * len(row)
            
            # Apply styling with new column names
            styled_df = filtered_df.style.format({
                'Stock': '{:,.0f}',
                'Compras Recomendadas': '{:,.0f}',
                'Demanda Mensual': '{:,.0f}',
                'Meses Protección': '{:,.2f}',
                'Índice Riesgo': '{:,.2f}',
                'Margen Promedio': '{:,.2%}',
                'Monto USD': '{:,.2f}',
                'Última Compra': '{:,.0f}',
                'Costo Compra': '{:,.2f}',
                'Backorder': '{:,.0f}'
            }).apply(highlight_risk, axis=1)
            
            st.dataframe(styled_df)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Descargar Datos",
                data=csv,
                file_name="dashboard_filtered.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error al cargar o procesar los datos del dashboard: {str(e)}")
            st.write("Tipo de error:", type(e).__name__)
            import traceback
            st.write("Traceback completo:", traceback.format_exc())
        

if __name__ == "__main__":
    main()