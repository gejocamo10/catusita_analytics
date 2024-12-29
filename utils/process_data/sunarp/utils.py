import pandas as pd

def clean_dataframe(df, fill_columns, exclude_patterns={'OFICINA': 'Total', 'CLASE': 'Total'}):
    """Generic function to clean dataframes"""
    df.columns = df.columns.str.strip()
    df[fill_columns] = df[fill_columns].fillna(method='ffill')
    
    for column, pattern in exclude_patterns.items():
        if column in df.columns:
            df = df[~df[column].str.contains(pattern)]
    
    if 'Total' in df.columns:
        df = df.drop(columns="Total")
    
    return df

def melt_dataframe(df, id_vars, var_name="MES", value_name="VENTAS"):
    """Generic function to melt dataframes"""
    return pd.melt(df, id_vars=id_vars, var_name=var_name, value_name=value_name)

def get_year_from_filename(file_path):
    """Extract year from filename"""
    return file_path.split("_")[-1].split(".")[0]