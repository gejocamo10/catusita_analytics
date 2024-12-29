from utils.process_data.config import DATA_PATHS
import os

PATHS = {
    'raw_data': os.path.join(DATA_PATHS['raw'], 'sunat'),
    'processed_data': os.path.join(DATA_PATHS['process'], 'sunat_consolidated.csv')
}

COLUMNS_MAPPING = {
    'standard': ["Partida", "Descripcion", "Enero", "Febrero", "Marzo", "Abril", 
                 "Mayo", "Junio", "Julio", "Agosto", "Setiembre", "Octubre", 
                 "Noviembre", "Diciembre"],
    'partial': ["Partida", "Descripcion", "Enero", "Febrero", "Marzo", "Abril", 
                "Mayo", "Junio", "Julio", "Agosto"]
}
#Jalar hasta diciembre y droppear columnas con vacíos

EXCLUDED_YEARS = [2017]
EXCLUDED_PARTIDAS = ["TOTAL"]