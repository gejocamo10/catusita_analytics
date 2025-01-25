import streamlit as st
import pandas as pd
from pathlib import Path
import os
import re
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode, JsCode

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
        # df_skus_rfm = pd.DataFrame({'sku': lista_skus_rfm})
        # df_skus_rfm.to_csv(DATA_PATHS['process'] / 'df_skus_rfm.csv', index=False)
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
            if processor.dfdashboard is not None:
                processor.dfdashboard.to_csv(DATA_PATHS['cleaned'] / 'dashboard.csv', index=False)
                print(f"Generated dashboard data for {len(processor.dfdashboard)} SKUs")
            else:
                print("Warning: No dashboard data generated")
            if processor.dffinal3 is not None:
                processor.dffinal3.to_csv(DATA_PATHS['cleaned'] / 'dashboard_by_fuente.csv', index=False)
                print(f"Generated dashboard data by fuente for {len(processor.dffinal3)} fuentes")
            else:
                print("Warning: No dashboard data by fuente generated")
            if processor.dffinal2 is not None:
                processor.dffinal2.to_csv(DATA_PATHS['cleaned'] / 'download.csv', index=False)
                print(f"Generated download data for {len(processor.dffinal2)} SKUs")
            else:
                print("Warning: No download data generated")
        except Exception as e:
            print(f"Error processing dashboard datasets: {str(e)}")
    
    return df_sunarp, df_sunat, df_catusita

def main():
    setup_directories()
    st.title("Dashboard Catusita")
    
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

    with tab3:
        st.markdown("### Resultados de Prediccion")
        
        try:
            # Carga de datos
            dashboard_path = DATA_PATHS['cleaned'] / 'dashboard.csv'
            dashboard_path_by_fuente = DATA_PATHS['cleaned'] / 'dashboard_by_fuente.csv'
            download_path = DATA_PATHS['cleaned'] / 'download.csv'
            
            # Leer archivos CSV
            dashboard_df = pd.read_csv(dashboard_path)
            dashboard_df_by_fuente = pd.read_csv(dashboard_path_by_fuente)
            download_df = pd.read_csv(download_path)

            # Apply filters
            # if filtro_fuente != 'todos':
            #     dashboard_df = dashboard_df[dashboard_df['fuente_suministro'] == filtro_fuente]
            # if filtro_articulo != 'todos':
            #     dashboard_df = dashboard_df[dashboard_df['articulo'] == filtro_articulo]
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recomendacion Total", f"{dashboard_df_by_fuente['Recomendacion USD'].sum():,.0f}")
            with col2:
                st.metric("Demanda Mensual Total", f"{dashboard_df_by_fuente['Demanda Mensual USD'].sum():,.0f}")
            with col3:
                st.metric("xxx", f"0")

            
            # Modified highlight function using new column names
            def highlight_risk(row):
                color_map = {
                    'verde': '#b7f898',
                    'amarillo': '#f6f69b',
                    # 'naranja': '#f8dc98',
                    'rojo': '#f69e9b'
                }
                
                risk_value = row['Alerta']
                color = color_map.get(risk_value, '')
                
                return ['background-color: ' + color if color else ''] * len(row)
            
            def highlight_risk_fuente(row):
                color_map = {
                    'verde': '#b7f898',
                    'amarillo': '#f6f69b',
                    # 'naranja': '#f8dc98',
                    'rojo': '#f69e9b'
                }
                
                risk_value = row['Alerta']
                color = color_map.get(risk_value, '')
                
                return ['background-color: ' + color if color else ''] * len(row)

            styled_df_fuente = dashboard_df_by_fuente.style.format({
                'Lead Times': '{:,.0f}',
                'Recomendacion USD': '{:,.0f}',
                'Margen Promedio': '{:,.0%}' 
            }).apply(highlight_risk_fuente, axis=1)


            # dashboard_df_by_fuente['Recomendacion USD'] = pd.to_numeric(dashboard_df_by_fuente['Recomendacion USD'], errors='coerce')
            # dashboard_df_by_fuente['Demanda Mensual USD'] = pd.to_numeric(dashboard_df_by_fuente['Demanda Mensual USD'], errors='coerce')

            gb = GridOptionsBuilder.from_dataframe(dashboard_df_by_fuente)
            gb.configure_column(
                "Recomendacion USD",
                type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                valueFormatter="function(params) { return params.value ? Number(params.value).toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0}) : ''; }"
            )
            gb.configure_column(
                "Demanda Mensual USD",
                type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                valueFormatter="function(params) { return params.value ? Number(params.value).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}) : ''; }"
            )
            gb.configure_selection('single', use_checkbox=True)
            grid_options_fuente = gb.build()
            response = AgGrid(
                dashboard_df_by_fuente,
                gridOptions=grid_options_fuente,
                height=300,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=True
            )            
            selected_row = pd.DataFrame(response.get("selected_rows", []))  # Convertir a DataFrame si no lo es            
            if not selected_row.empty:
                # Acceder al valor de 'fuente_suministro'
                fuente_seleccionada = selected_row['Fuente de Suministro'].iloc[0]
                st.write(f"Fuente seleccionada: {fuente_seleccionada}")
                dashboard_df = dashboard_df[dashboard_df['Fuente Suministro'] == fuente_seleccionada].reset_index(drop=True)
                # Apply styling with new column names
                styled_df = dashboard_df.style.format({
                    'Inventario': '{:,.0f}',
                    'Compras Recomendadas': '{:,.0f}',
                    'Monto USD': '{:,.2f}',
                    'Última Compra': '{:,.0f}',
                    'Backorder': '{:,.0f}',
                    'Recomendacion USD': '{:,.0F}'
                }).apply(highlight_risk, axis=1)
                st.write(styled_df)
            else:
                st.write("Seleccionar alguna fila")

            # Download button
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="Descargar Datos",
                data=csv,
                file_name="dashboard_filtered.csv",
                mime="text/csv"
            )
        
            # Leyenda de colores para la variable Alerta con emojis
            st.markdown("### Leyenda de Alerta")
            st.markdown(
                """
                <div style="margin-bottom: 10px;">
                    <span style="font-size: 20px;">🔴</span> Rojo: Se quebró o se va a quebrar
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="font-size: 20px;">🟡</span> Amarillo: Se está consumiento el inventario de seguridad
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="font-size: 20px;">🟢</span> Verde: No se está consumiendo el inventario de seguridad
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error al cargar o procesar los datos del dashboard: {str(e)}")
            st.write("Tipo de error:", type(e).__name__)
            import traceback
            st.write("Traceback completo:", traceback.format_exc())


if __name__ == "__main__":
    main()