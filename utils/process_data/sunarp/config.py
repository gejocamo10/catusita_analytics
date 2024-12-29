from utils.process_data.config import DATA_PATHS
import os

PATHS = {
    'raw_data': os.path.join(DATA_PATHS['raw'], 'sunarp'),
    'processed_data': os.path.join(DATA_PATHS['process'], 'sunarp_consolidated.csv')
}

FILE_CATEGORIES = {
    'Menores': [],
    'Livianos': [],
    'Pesados': [],
    'Híbridos': [],
    'Remolques': []
}