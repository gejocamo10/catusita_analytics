import streamlit as st
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import shutil
from utils.process_data.config import DATA_PATHS
from utils.process_data.sunarp.config import FILE_CATEGORIES
from utils.correlations.correlations_processor import process_correlations
from utils.predictions.predictor import Predictor

# Configure Streamlit page
st.set_page_config(layout="wide", page_title="Procesador de Datos")

# Custom CSS to make file uploaders more compact
st.markdown("""
    <style>
        .stFileUploader > div > div {
            padding: 15px;
        }
        .stFileUploader > div {
            padding: 5px;
        }
        .stFileUploader button {
            padding: 2px 10px;
            font-size: 14px;
        }
        .uploadStatusBox {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .small-text {
            font-size: 14px;
            color: #666;
            margin: 0;
            padding: 0;
        }
    </style>
""", unsafe_allow_html=True)

def setup_directories():
    for path in DATA_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

def process_uploaded_file(uploaded_file, category, subcategory=None):
    try:
        if category == 'sunarp':
            save_path = DATA_PATHS['raw_sunarp'] / uploaded_file.name
        elif category == 'sunat':
            save_path = DATA_PATHS['raw_sunat'] / uploaded_file.name
        else:  # catusita
            if subcategory:
                save_path = DATA_PATHS['raw_catusita'] / subcategory / uploaded_file.name
                save_path.parent.mkdir(exist_ok=True)
            else:
                save_path = DATA_PATHS['raw_catusita'] / uploaded_file.name

        file_size = uploaded_file.size
        chunk_size = min(file_size // 10, 5 * 1024 * 1024)
        
        progress_bar = st.progress(0)
        bytes_processed = 0

        with open(save_path, "wb") as f:
            while True:
                chunk = uploaded_file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_processed += len(chunk)
                progress = min(bytes_processed / file_size, 1.0)
                progress_bar.progress(progress)
        
        progress_bar.empty()
        return True, save_path
    except Exception as e:
        if 'progress_bar' in locals():
            progress_bar.empty()
        return False, str(e)

def get_file_year(filename):
    try:
        import re
        matches = re.findall(r'20\d{2}', filename)
        if matches:
            return int(matches[0])
    except:
        pass
    return None

def create_uploader_section(title, key, file_types, help_text=""):
    st.markdown(f"<p class='small-text'>{title}</p>", unsafe_allow_html=True)
    return st.file_uploader(
        "",
        type=file_types,
        accept_multiple_files=True,
        key=key,
        help=help_text
    )

def app():
    if 'upload_status' not in st.session_state:
        st.session_state.upload_status = {'success': [], 'error': []}
    
    setup_directories()

    st.markdown("### Aplicación de Procesamiento de Datos")
    
    # Create tabs for Upload and Predictions
    tab1, tab2 = st.tabs(["Cargar Archivos", "Predicciones"])
    
    with tab1:
        # Create three main columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Archivos Catusita")
            
            # Main Catusita files
            ventas_files = create_uploader_section(
                "Archivos de Ventas", 
                'catusita_ventas', 
                ['xlsx', 'csv'],
                "Cargar archivos de ventas de Catusita"
            )
            
            inventory_files = create_uploader_section(
                "Inventario", 
                'catusita_inventory', 
                ['xlsx', 'csv'],
                "Cargar archivo de inventario"
            )
            
            kits_files = create_uploader_section(
                "Kits", 
                'catusita_kits', 
                ['xlsx'],
                "Cargar archivo de kits"
            )
            
            blacklist_files = create_uploader_section(
                "Lista Negra", 
                'catusita_blacklist', 
                ['xlsx'],
                "Cargar archivo de lista negra"
            )
            
            for file_group in [ventas_files, inventory_files, kits_files, blacklist_files]:
                if file_group:
                    for file in file_group:
                        success, result = process_uploaded_file(file, 'catusita')
                        if success:
                            st.session_state.upload_status['success'].append(f"✓ {file.name}")
                        else:
                            st.session_state.upload_status['error'].append(f"✗ {file.name}: {result}")
        
        with col2:
            st.markdown("#### Archivos SUNARP")
            
            # SUNARP categories
            sunarp_categories = {
                'livianos': "Vehículos Livianos",
                'pesados': "Vehículos Pesados",
                'hibridos': "Vehículos Híbridos",
                'remolques': "Remolques",
                'menores': "Vehículos Menores"
            }
            
            for key, title in sunarp_categories.items():
                files = create_uploader_section(
                    title,
                    f'sunarp_{key}',
                    ['xlsx'],
                    f"Cargar archivos de {title}"
                )
                
                if files:
                    for file in files:
                        year = get_file_year(file.name)
                        if year:
                            success, result = process_uploaded_file(file, 'sunarp', key)
                            if success:
                                st.session_state.upload_status['success'].append(f"✓ {file.name}")
                            else:
                                st.session_state.upload_status['error'].append(f"✗ {file.name}: {result}")
                        else:
                            st.session_state.upload_status['error'].append(
                                f"✗ {file.name}: No se pudo determinar el año"
                            )
        
        with col3:
            st.markdown("#### Archivos SUNAT")
            sunat_files = create_uploader_section(
                "Archivos SUNAT",
                'sunat',
                ['xlsx'],
                "Cargar archivos SUNAT"
            )
            
            if sunat_files:
                for file in sunat_files:
                    year = get_file_year(file.name)
                    if year:
                        success, result = process_uploaded_file(file, 'sunat')
                        if success:
                            st.session_state.upload_status['success'].append(f"✓ {file.name}")
                        else:
                            st.session_state.upload_status['error'].append(f"✗ {file.name}: {result}")
                    else:
                        st.session_state.upload_status['error'].append(
                            f"✗ {file.name}: No se pudo determinar el año"
                        )

        # Status display
        if st.session_state.upload_status['success'] or st.session_state.upload_status['error']:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                with st.expander("Mostrar Estado de Carga"):
                    if st.session_state.upload_status['success']:
                        st.markdown("**Archivos cargados exitosamente:**")
                        for msg in st.session_state.upload_status['success'][-5:]:
                            st.text(msg)
                        if len(st.session_state.upload_status['success']) > 5:
                            st.text(f"...y {len(st.session_state.upload_status['success']) - 5} más")
                    
                    if st.session_state.upload_status['error']:
                        st.markdown("**Cargas fallidas:**")
                        for msg in st.session_state.upload_status['error']:
                            st.text(msg)
                    
                    if st.button("Limpiar Estado", key="clear_status"):
                        st.session_state.upload_status = {'success': [], 'error': []}
                        st.experimental_rerun()

    with tab2:
        st.markdown("### Predicciones")
        
        # Add date selector
        prediction_date = st.date_input(
            "Seleccione fecha hasta la cual usar datos para predicción",
            value=datetime.now(),
            help="Los datos posteriores a esta fecha no serán considerados para el entrenamiento"
        )
        
        if st.button("Ejecutar Predicciones"):
            with st.spinner("Procesando predicciones..."):
                try:
                    # Initialize predictor with selected date
                    predictor = Predictor(prediction_date=prediction_date)
                    
                    # Process predictions
                    results_df = predictor.process_predictions()
                    
                    if results_df is not None and not results_df.empty:
                        # Get unique fuente_suministro values for filtering
                        fuente_suministro_options = ['Todos'] + sorted(results_df['fuente_suministro'].unique().tolist())
                        
                        # Add filter in sidebar
                        selected_fuente = st.selectbox(
                            "Filtrar por Fuente de Suministro",
                            options=fuente_suministro_options
                        )
                        
                        # Filter results if needed
                        if selected_fuente != 'Todos':
                            filtered_results = results_df[results_df['fuente_suministro'] == selected_fuente]
                        else:
                            filtered_results = results_df
                        
                        # Display results
                        st.markdown("#### Resultados de Predicción")
                        st.dataframe(filtered_results)
                        
                        # Add download button
                        csv = filtered_results.to_csv(index=False)
                        st.download_button(
                            label="Descargar Resultados",
                            data=csv,
                            file_name=f"predicciones_{prediction_date.strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        
                        # Display predictions plot
                        st.markdown("#### Gráfico de Predicciones")
                        try:
                            predictions_df = pd.read_csv(DATA_PATHS['process'] / 'predictions.csv')
                            predictions_df['fecha'] = pd.to_datetime(predictions_df['fecha'])
                            
                            if selected_fuente != 'Todos':
                                predictions_df = predictions_df[
                                    predictions_df['fuente_suministro'] == selected_fuente
                                ]
                            
                            # Create plot using plotly
                            import plotly.express as px
                            fig = px.line(
                                predictions_df, 
                                x='fecha', 
                                y='prediction',
                                color='sku',
                                title='Predicciones por SKU'
                            )
                            st.plotly_chart(fig)
                            
                        except Exception as e:
                            st.error(f"Error al mostrar el gráfico: {str(e)}")
                    else:
                        st.warning("No se encontraron resultados para las predicciones")
                        
                except Exception as e:
                    st.error(f"Error al procesar predicciones: {str(e)}")

if __name__ == "__main__":
    app()