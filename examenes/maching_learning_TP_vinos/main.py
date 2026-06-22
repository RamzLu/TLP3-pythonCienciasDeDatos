#importo la libreria fastapi con todas las herramientas de la clase 
#principal que es FastAPI que seria como el molde para construir este servidor web
from fastapi import FastAPI
#jolib es una libr para empaquetar y desemp objet pesados de py (como el modelo de vinos)
import joblib

#-------------------------------------inicio de la app--------------------------------------------------
#creo el servidor con el molde del FastAPI y lo guardo. Doy un titulo y descripcion
# en una variable llamada "app". Todas las rutas, recepcion de datos y demas se 
# conectaran a esta variable
app = FastAPI(
    #detalles personalizados
    title= "API para la Prediccion de Vinos",
    description="Backend para el modelo de Machine Learning del TP final.",
    version= "1.0.0" #etiqueta p/llevar el control de versiones
)

#---------------------------------icarga de modelo y escalador-------------------------------------------
#funcion try...catch pero de python
try:
    #los .pkl son archivos binarios cerrados. No puedo abrirlos con un bloc de notas
    #la herramienta joblib.load() sirve para "abrir y leer" estos archivos especiales y traerlos a la memoria de tu servedr
    modelo = joblib.load("models/modelo_vino.pkl") # modelo es para cargar el cerebro que entreno Lu, lo q predice 
    escalador = joblib.load("models/scaler_vino.pkl") #escalador es para cargar la regla matmtica. Cuando el front mande datos de un nuevo vino
    #este escalador ajustara los numeros antes de pasarlo al brocere para evitar confuciones
    print("cerebro conectado: modelo y escalador conectados")
except Exception as e:
    print("error al cargar los archivos de ML")

#------------------------------------------endpoint-------------------------------------------
@app.get("/")
#cuando llegue una eticion devuelve este diccionario para que FastAPI empaquete esto
#y lo traduzca automat. a JSONpara el nav
def ruta_raiz():
    return{
        "mensaje" : "API del back funcionando"
    }