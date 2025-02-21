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
    'cantidad',
    'transacciones',
    'venta_pen',
    'fuente_suministro',
    'costo'
]

COLUMN_RENAME_MAPPING = {
    "dateDocument": "fecha",
    "codeArticle": "articulo",
    "nameArticle": "descripcion",
    "nameSupply": "fuente_suministro",
    "quantity": "cantidad",
    "amountSOL": "venta_pen",
    "amountUSD": "venta_usd"
}

# dateDocument:fecha, codeArticle: articulo, nameArticle: nombre, nameSupply: fuente_suministro, quantity: cantidad, amountSOL: venta_pen, amountUSD: venta_usd
KITS_RENAME_MAPPING = {
    "Código KIT (Sin historial)": "cod_madre",
    "Código 1": "cod_1",
    "Código 2": "cod_2",
    "Código 3": "cod_3"
}

FILTER_COLUMNS = ["cantidad", "venta_pen"]
FILTER_DATE='2024/12/01'