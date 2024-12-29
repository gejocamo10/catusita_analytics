PATHS = {
    'base_path': r'.',
    'input_file': r'/Data de venta 01.01.21 a 06.12.24.xls',
    'kits_file': r'KITS AISIN.xlsx',
    'blacklist_file': r'IMCASA SD 14.08.xls',
    'lt': r'/lt_catusita.csv',
    'process': r'\data\process',
    'output_file': r'/data/clean/df_catusita_cleaned.csv'
}

COLUMNS_TO_KEEP = [
    'fecha',
    'articulo',
    'codigo',
    'cantidad',
    'transacciones',
    'venta_pen',
    'fuente_suministro',
    'costo'
]

COLUMN_RENAME_MAPPING = {
    "razon_social_": "razon_social",
    "venta_$": "venta_usd",
    "venta_s/.": "venta_pen",
    "nombre_de_articulo": "nombre_articulo",
    "fuente_de_suministro": "fuente_suministro",
}

KITS_RENAME_MAPPING = {
    "Código KIT (Sin historial)": "cod_madre",
    "Código 1": "cod_1",
    "Código 2": "cod_2",
    "Código 3": "cod_3"
}

FILTER_COLUMNS = ["cantidad", "venta_pen"]
FILTER_DATE='2024/12/01'