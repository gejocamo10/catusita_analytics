import requests
import pandas as pd 

# URL de la API
url = "http://api.catusita.com:8083/api/sales/forDate"
# http://api.catusita.com:8083/api/sales/forDate?Date1=20250101&Date2=20250115

# Parámetros de la consulta
params = {
    "Date1": "20250101",
    "Date2": "20250131"
}

# Realizar la solicitud GET
response = requests.get(url, params=params)

# Verificar el estado de la respuesta
if response.status_code == 200:
    # Si la respuesta es exitosa, imprimir el contenido (JSON)
    data = response.json().get("data", [])
    data = pd.DataFrame(data)
    print(data)
else:
    print(f"Error en la solicitud: {response.status_code}")


data.to_csv("df_sales.csv")

