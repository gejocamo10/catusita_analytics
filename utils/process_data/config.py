from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

DATA_PATHS = {
    'raw': ROOT_DIR / 'data' / 'raw' ,
    'cleaned': ROOT_DIR / 'data' / 'cleaned',
    'process': ROOT_DIR / 'data' / 'process',
    'raw_sunarp': ROOT_DIR / 'data' / 'raw' / 'sunarp',
    'raw_sunat': ROOT_DIR / 'data' / 'raw' / 'sunat',
    'raw_catusita': ROOT_DIR / 'data' / 'raw' / 'catusita'
}
