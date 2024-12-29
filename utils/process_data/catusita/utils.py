import pandas as pd
from unidecode import unidecode

def format_column_names(df):
    """Format column names to lowercase, replace spaces with underscores and remove accents"""
    return df.rename(columns=lambda x: unidecode(str(x).lower().replace(" ", "_")))

def clean_string_columns(df):
    """Clean string columns by removing extra spaces and converting to lowercase"""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace("  ", " ")
    return df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

def clean_article_names(df):
    """Clean article names by removing special characters"""
    df["nombre_articulo"] = df["nombre_articulo"].str.lstrip("#")
    df["nombre_articulo"] = df["nombre_articulo"].str.lstrip('"')
    return df