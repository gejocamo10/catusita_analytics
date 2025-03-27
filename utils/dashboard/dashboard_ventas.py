import pandas as pd
import os

class DataProcessorSales:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.sales_path = os.path.join(base_dir, "data", "process", "ventas_consolidated.csv")
        self.metas_path = os.path.join(base_dir, "data", "raw", "catusita", "metas.xlsx")
        self.output_path = os.path.join(base_dir, "data", "cleaned", "ventas_metas.csv")

    def load_data(self):
        if not os.path.exists(self.sales_path):
            raise FileNotFoundError(f"No se encontró el archivo de ventas: {self.sales_path}")
        if not os.path.exists(self.metas_path):
            raise FileNotFoundError(f"No se encontró el archivo de metas: {self.metas_path}")

        print("📥 Cargando archivos...")
        self.sales_df = pd.read_csv(self.sales_path, low_memory=False)
        self.metas_df = pd.read_excel(self.metas_path)

        if "fuente_suministro" not in self.sales_df.columns:
            raise KeyError("La columna 'fuente_suministro' no está presente en el archivo de ventas.")
        if "fuente_suministro" not in self.metas_df.columns:
            raise KeyError("La columna 'fuente_suministro' no está presente en el archivo de metas.")

    def merge_and_clean(self):
        print("🔄 Combinando y limpiando datos...")
        merged_df = self.sales_df.merge(self.metas_df, on="fuente_suministro", how="left")

        # Relleno de NaNs
        merged_df["familia"] = merged_df["familia"].fillna("Desconocido")
        merged_df["segmento"] = merged_df["segmento"].fillna("Desconocido")
        merged_df["marca"] = merged_df["marca"].fillna("Desconocido")
        merged_df["gestor"] = merged_df["gestor"].fillna("Desconocido")
        merged_df["meta"] = merged_df["meta"].fillna(0)

        self.result_df = merged_df

    def save_output(self):
        print(f"💾 Guardando archivo final en: {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.result_df.to_csv(self.output_path, index=False)
        print("✅ Archivo guardado exitosamente.")

    def run(self):
        try:
            self.load_data()
            self.merge_and_clean()
            self.save_output()
        except Exception as e:
            print(f"❌ Error durante el procesamiento: {e}")

if __name__ == "__main__":
    processor = DataProcessorSales()
    processor.run()
