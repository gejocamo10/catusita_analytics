import pandas as pd

def clean_dataframe(df, columns):
    """Clean and format dataframe"""
    df_cleaned = df.dropna(how='all')
    df_cleaned = df_cleaned.iloc[1:]
    df_cleaned.columns = columns
    return df_cleaned

def melt_dataframe(df, year):
    """Melt dataframe to long format"""
    df_long = pd.melt(df, 
                      id_vars=["Partida", "Descripcion"], 
                      var_name="mes", 
                      value_name="value")
    df_long["year"] = year
    return df_long