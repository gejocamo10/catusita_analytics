import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from utils.process_data.config import DATA_PATHS 

class CatusitaSalesProcessor:
    def __init__(self, auth_token: str = None):
        self.api_url = "http://api.catusita.com:8083/api/sales/forDate"
        self.auth_token = auth_token
        self.base_path = DATA_PATHS['process']

    def _get_full_path(self, relative_path: str) -> str:
        """Construye la ruta absoluta desde base_path y un path relativo."""
        return os.path.join(self.base_path, relative_path.lstrip('/'))

    def _get_yesterday_date_str(self):
        """Devuelve la fecha de ayer en formato YYYYMMDD"""
        yesterday = datetime.today() - timedelta(days=1)
        date_str = yesterday.strftime("%Y%m%d")
        return date_str, date_str  # start_date, end_date (mismo día)

    def _download_sales_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Descarga los datos de ventas desde la API"""
        params = {"Date1": start_date, "Date2": end_date}
        headers = {"Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            response = requests.get(self.api_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json().get("data", [])
            if isinstance(data, list):
                return pd.DataFrame(data)
            else:
                print("⚠️ Respuesta de API inesperada: 'data' no es una lista.")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error en la solicitud a la API: {e}")
            return pd.DataFrame()

    def _clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y transforma el DataFrame descargado"""
        column_mapping = {
            "dateDocument": "fecha",
            "document": "documento",
            "codeArticle": "articulo",
            "nameArticle": "nombre_articulo",
            "codeSupply": "codigo",
            "nameSupply": "fuente_suministro",
            "codeClient": "cliente",
            "nameClient": "nombre_cliente",
            "rucClient": "ruc_cliente",
            "codeSeller": "vendedor",
            "nameSeller": "nombre_vendedor",
            "quantity": "cantidad",
            "amountSOL": "venta_pen",
            "amountUSD": "venta_usd",
            "cost": "costo"
        }

        df.rename(columns=column_mapping, inplace=True)
        df["fecha"] = pd.to_datetime(df["fecha"], errors='coerce')
        df["ruc_cliente"] = df["ruc_cliente"].astype(str)

        # Marcar devoluciones
        df["tipo_transaccion"] = "venta"
        df.loc[df[["cantidad", "venta_pen", "venta_usd", "costo"]].lt(0).any(axis=1), "tipo_transaccion"] = "devolucion"

        # Reportar fechas inválidas si hay
        if df["fecha"].isna().any():
            print("⚠️ Advertencia: Fechas inválidas encontradas.")
            print(df[df["fecha"].isna()])

        return df

    def _enrich_with_metas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combina los datos de ventas con metas a partir de 'fuente_suministro'"""
        metas_path = os.path.abspath(os.path.join(self.base_path, "..", "raw", "catusita", "metas.xlsx"))

        if not os.path.exists(metas_path):
            raise FileNotFoundError(f"No se encontró el archivo de metas: {metas_path}")
        
        metas_df = pd.read_excel(metas_path)

        if "fuente_suministro" not in df.columns:
            raise KeyError("La columna 'fuente_suministro' no está en los datos de ventas.")
        if "fuente_suministro" not in metas_df.columns:
            raise KeyError("La columna 'fuente_suministro' no está en el archivo de metas.")

        print("🔄 Realizando merge con archivo de metas...")

        merged_df = df.merge(metas_df, on="fuente_suministro", how="left")

        # Rellenar valores faltantes
        for col in ["familia", "segmento", "marca", "gestor"]:
            merged_df[col] = merged_df[col].fillna("Desconocido")
        if "meta" in merged_df.columns:
            merged_df["meta"] = merged_df["meta"].fillna(0)

        return merged_df

    def run_daily(self):
        """Ejecuta el pipeline para el día anterior"""
        print("⏳ Iniciando procesamiento diario de Catusita...")

        start_date, end_date = self._get_yesterday_date_str()
        df_raw = pd.DataFrame()
        max_intentos = 5
        intentos = 0
        while df_raw.empty and intentos < max_intentos:
            df_raw = self._download_sales_data(start_date, end_date)
            if df_raw.empty:
                intentos += 1
                print(f"Intento {intentos}/{max_intentos}: No se obtuvieron datos. Reintentando en 2 segundos...")
                time.sleep(2)
        if df_raw.empty:
            print("No se obtuvieron datos después de 5 intentos.")
        else:
            print("Datos del api de ventas obtenidos exitosamente.")

        df_clean = self._clean_sales_data(df_raw)
        df_enriched = self._enrich_with_metas(df_clean)

        output_path = self._get_full_path("catusita_sales.csv")
        df_enriched.to_csv(output_path, index=False)
        print(f"✅ Proceso completado. Archivo final guardado en {output_path}")

if __name__ == "__main__":
    processor = CatusitaSalesProcessor(
        auth_token=None  # O token si es necesario
    )
    processor.run_daily()
